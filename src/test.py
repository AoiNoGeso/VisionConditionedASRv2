import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
from tqdm import tqdm
from pyctcdecode import build_ctcdecoder
import jiwer
from safetensors.torch import load_file
from audiomentations import Compose, AddGaussianSNR, AddBackgroundNoise, AddGaussianNoise

from model import VisionConditionedASR
# from purewav2vec2_train import PureWav2Vec2ASR
from dataloader import create_dataloader
from train import TrainingConfig


@dataclass
class TestConfig:
    checkpoint_dir: str = "checkpoints/DINOv2_model/epoch_10"
    model_type: str = "vision"  # "pure" or "vision"
    
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    vocab_size: Optional[int] = None
    num_heads: int = 8
    dropout: float = 0.1
    
    batch_size: int = 16
    num_workers: int = 4
    max_audio_length: float = 10.0
    validate_files: bool = True
    
    noise_type: str = "white"  # "none", "gaussian", "white", "background"
    noise_dir: Optional[str] = "../../Datasets/noise"
    gaussian_snr_db: tuple = (5, 20)
    white_noise_level: float = 0.01
    background_snr_db: tuple = (5, 20)
    
    use_beam_search: bool = True
    beam_width: int = 10
    use_image: bool = True
    
    device: str = "cuda:0"
    save_results: bool = True
    results_dir: str = "results/"


class NoiseAugmenter:
    def __init__(
        self,
        noise_type: str = "none",
        gaussian_snr_db: tuple = (5, 20),
        white_noise_level: float = 0.01,
        noise_dir: Optional[str] = None,
        background_snr_db: tuple = (5, 20),
        sample_rate: int = 16000
    ):
        self.noise_type = noise_type
        self.sample_rate = sample_rate
        
        print(f"\n[NoiseAugmenter] Type: {noise_type}")
        
        if noise_type == "none":
            self.augment = None
            print("  No noise augmentation")
        elif noise_type == "gaussian":
            self.augment = Compose([
                AddGaussianSNR(
                    min_snr_db=gaussian_snr_db[0],
                    max_snr_db=gaussian_snr_db[1],
                    p=1.0
                )
            ])
            print(f"  SNR: {gaussian_snr_db[0]}-{gaussian_snr_db[1]} dB")
        elif noise_type == "white":
            self.augment = Compose([
                AddGaussianNoise(
                    min_amplitude=white_noise_level,
                    max_amplitude=white_noise_level,
                    p=1.0
                )
            ])
            print(f"  Level: {white_noise_level}")
        elif noise_type == "background":
            if not noise_dir or not os.path.exists(noise_dir):
                raise ValueError(f"Background noise directory not found: {noise_dir}")
            noise_files = [f for f in os.listdir(noise_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
            if not noise_files:
                raise ValueError(f"No noise files in: {noise_dir}")
            self.augment = Compose([
                AddBackgroundNoise(
                    sounds_path=noise_dir,
                    min_snr_in_db=background_snr_db[0],
                    max_snr_in_db=background_snr_db[1],
                    p=1.0
                )
            ])
            print(f"  Files: {len(noise_files)}, SNR: {background_snr_db[0]}-{background_snr_db[1]} dB")
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        if self.augment is None:
            return audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return self.augment(samples=audio, sample_rate=self.sample_rate)


class CTCDecoder:
    def __init__(
        self,
        tokenizer,
        use_beam_search: bool = True,
        beam_width: int = 100
    ):
        self.tokenizer = tokenizer
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        
        vocab_list = []
        for i in range(tokenizer.vocab_size):
            token = tokenizer.convert_ids_to_tokens(i)
            if token is None:
                token = ""
            vocab_list.append(token)
        
        self.decoder = build_ctcdecoder(labels=vocab_list, kenlm_model_path=None)
        
        print(f"[CTCDecoder] Initialized")
        print(f"  Vocabulary size: {len(vocab_list)}")
        print(f"  Beam search: {use_beam_search}")
        if use_beam_search:
            print(f"  Beam width: {beam_width}")
    
    def decode(self, logits: torch.Tensor) -> List[str]:
        batch_size = logits.size(0)
        results = []
        
        for i in range(batch_size):
            logits_i = logits[i].cpu().numpy()
            if self.use_beam_search:
                text = self.decoder.decode(logits_i, beam_width=self.beam_width)
            else:
                text = self.decoder.decode(logits_i, beam_width=1)
            results.append(text)
        
        return results


def compute_wer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    output = jiwer.process_words(references, hypotheses)
    
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_hits = 0
    
    for alignment in output.alignments:
        for op in alignment:
            if op.type == 'substitute':
                total_substitutions += 1
            elif op.type == 'delete':
                total_deletions += 1
            elif op.type == 'insert':
                total_insertions += 1
            elif op.type == 'equal':
                total_hits += 1
    
    return {
        'wer': output.wer * 100,
        'mer': output.mer * 100,
        'wil': output.wil * 100,
        'substitutions': total_substitutions,
        'deletions': total_deletions,
        'insertions': total_insertions,
        'hits': total_hits
    }


def load_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    device: torch.device,
    model_type: str = "vision"
):
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    state_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}_state.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(state_path):
        raise FileNotFoundError(f"Checkpoint files incomplete in: {checkpoint_dir}")
    
    print(f"\n[Loading] From: {checkpoint_dir}")
    print(f"[Loading] Model type: {model_type}")
    
    state_dict = load_file(model_path, device=str(device))
    model.load_state_dict(state_dict)
    checkpoint_state = torch.load(state_path, map_location=device, weights_only=False)
    
    epoch = checkpoint_state.get('epoch', -1)
    train_loss = checkpoint_state.get('train_loss', 0.0)
    val_loss = checkpoint_state.get('val_loss', 0.0)
    
    print(f"[Loading] Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return epoch + 1


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    decoder: CTCDecoder,
    noise_augmenter: NoiseAugmenter,
    device: torch.device,
    config: TestConfig
):
    model.eval()
    
    hook_handle = None
    if config.model_type == "vision" and not config.use_image:
        print("\n[Evaluation] Image disabled (vision encoder output set to zero)")
        
        def zero_vision_output_hook(module, input, output):
            return torch.zeros_like(output)
        
        hook_handle = model.vision_encoder.register_forward_hook(zero_vision_output_hook)
    
    all_references = []
    all_hypotheses = []
    all_samples = []
    
    print(f"\n{'='*60}")
    print("Starting Evaluation")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batches: {len(dataloader)}")
    print(f"Noise: {config.noise_type}")
    if config.model_type == "vision":
        print(f"Use image: {config.use_image}")
    print(f"{'='*60}\n")
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                try:
                    if noise_augmenter.augment is not None:
                        batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]
                    
                    logits = model(batch)
                    hypotheses = decoder.decode(logits)
                    references = batch["text"]
                    
                    all_references.extend(references)
                    all_hypotheses.extend(hypotheses)
                    
                    if len(all_samples) < 100:
                        for ref, hyp in zip(references, hypotheses):
                            all_samples.append({'reference': ref, 'hypothesis': hyp})
                except Exception as e:
                    print(f"\n[Error] Batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    finally:
        if hook_handle is not None:
            hook_handle.remove()
            print("\n[Evaluation] Hook removed")
    
    print(f"\n{'='*60}\nComputing WER...\n{'='*60}")
    wer_metrics = compute_wer(all_references, all_hypotheses)
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Samples: {len(all_references)}")
    print(f"Noise: {config.noise_type}")
    if config.model_type == "vision":
        print(f"Use image: {config.use_image}")
    print(f"\nWER: {wer_metrics['wer']:.2f}%")
    print(f"MER: {wer_metrics['mer']:.2f}%")
    print(f"WIL: {wer_metrics['wil']:.2f}%")
    print(f"\nError Breakdown:")
    print(f"  Substitutions: {wer_metrics['substitutions']}")
    print(f"  Deletions:     {wer_metrics['deletions']}")
    print(f"  Insertions:    {wer_metrics['insertions']}")
    print(f"  Hits:          {wer_metrics['hits']}")
    print(f"{'='*60}\n")
    
    print(f"{'='*60}\nSample Predictions (first 5):\n{'='*60}")
    for i, sample in enumerate(all_samples[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Ref:  {sample['reference']}")
        print(f"  Hyp:  {sample['hypothesis']}")
    print(f"{'='*60}\n")
    
    return {
        'wer_metrics': wer_metrics,
        'num_samples': len(all_references),
        'references': all_references,
        'hypotheses': all_hypotheses,
        'samples': all_samples
    }


def save_results(
    results: Dict,
    config: TestConfig,
    checkpoint_epoch: int,
    model_type: str
):
    os.makedirs(config.results_dir, exist_ok=True)
    
    filename_parts = [f"wer_results_{model_type}_epoch_{checkpoint_epoch}"]
    if config.model_type == "vision" and not config.use_image:
        filename_parts.append("noimage")
    filename_parts.append(f"noise_{config.noise_type}")
    
    results_file = os.path.join(config.results_dir, "_".join(filename_parts) + ".txt")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"WER Evaluation Results ({model_type.upper()} Model)\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model Type: {model_type}\n")
        if config.model_type == "vision":
            f.write(f"Use Image: {config.use_image}\n")
        f.write(f"Checkpoint: {config.checkpoint_dir}\n")
        f.write(f"Epoch: {checkpoint_epoch}\n")
        f.write(f"Dataset: {config.val_json}\n")
        f.write(f"Noise Type: {config.noise_type}\n")
        if config.noise_type == "gaussian":
            f.write(f"Gaussian SNR: {config.gaussian_snr_db[0]}-{config.gaussian_snr_db[1]} dB\n")
        elif config.noise_type == "white":
            f.write(f"White Noise Level: {config.white_noise_level}\n")
        elif config.noise_type == "background":
            f.write(f"Background SNR: {config.background_snr_db[0]}-{config.background_snr_db[1]} dB\n")
        f.write(f"Beam Search: {config.use_beam_search}\n")
        if config.use_beam_search:
            f.write(f"Beam Width: {config.beam_width}\n")
        f.write(f"\nTotal samples: {results['num_samples']}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Metrics:\n")
        f.write("="*60 + "\n")
        f.write(f"WER: {results['wer_metrics']['wer']:.2f}%\n")
        f.write(f"MER: {results['wer_metrics']['mer']:.2f}%\n")
        f.write(f"WIL: {results['wer_metrics']['wil']:.2f}%\n")
        f.write(f"\nError Breakdown:\n")
        f.write(f"  Substitutions: {results['wer_metrics']['substitutions']}\n")
        f.write(f"  Deletions:     {results['wer_metrics']['deletions']}\n")
        f.write(f"  Insertions:    {results['wer_metrics']['insertions']}\n")
        f.write(f"  Hits:          {results['wer_metrics']['hits']}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Sample Predictions (first 20):\n")
        f.write("="*60 + "\n\n")
        
        for i, sample in enumerate(results['samples'][:20]):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  REF: {sample['reference']}\n")
            f.write(f"  HYP: {sample['hypothesis']}\n\n")
    
    print(f"[Results] Saved to: {results_file}\n")
    
    csv_filename_parts = [f"predictions_{model_type}_epoch_{checkpoint_epoch}"]
    if config.model_type == "vision" and not config.use_image:
        csv_filename_parts.append("noimage")
    csv_filename_parts.append(f"noise_{config.noise_type}")
    csv_file = os.path.join(config.results_dir, "_".join(csv_filename_parts) + ".csv")
    
    import csv
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Reference', 'Hypothesis'])
        for i, (ref, hyp) in enumerate(zip(results['references'], results['hypotheses'])):
            writer.writerow([i+1, ref, hyp])
    
    print(f"[Results] Predictions saved to: {csv_file}\n")


def main():
    config = TestConfig()
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Test Configuration")
    print(f"{'='*60}")
    print(f"Model Type: {config.model_type}")
    print(f"Device: {device}")
    print(f"Checkpoint: {config.checkpoint_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Noise: {config.noise_type}")
    if config.model_type == "vision":
        print(f"Use image: {config.use_image}")
    print(f"Beam search: {config.use_beam_search}")
    if config.use_beam_search:
        print(f"Beam width: {config.beam_width}")
    print(f"{'='*60}\n")
    
    noise_augmenter = NoiseAugmenter(
        noise_type=config.noise_type,
        gaussian_snr_db=config.gaussian_snr_db,
        white_noise_level=config.white_noise_level,
        noise_dir=config.noise_dir,
        background_snr_db=config.background_snr_db,
        sample_rate=16000
    )
    
    print("\n[Setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    print("[Setup] Initializing model...")
    if config.model_type == "pure":
        model = PureWav2Vec2ASR(device=device).to(device)
        print("[Model] Using PureWav2Vec2ASR")
    elif config.model_type == "vision":
        model = VisionConditionedASR(
            vocab_size=config.vocab_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            device=device
        ).to(device)
        print("[Model] Using VisionConditionedASR")
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    checkpoint_epoch = load_checkpoint(
        checkpoint_dir=config.checkpoint_dir,
        model=model,
        device=device,
        model_type=config.model_type
    )
    
    print("\n[Setup] Initializing decoder...")
    decoder = CTCDecoder(
        tokenizer=tokenizer,
        use_beam_search=config.use_beam_search,
        beam_width=config.beam_width
    )
    
    print("\n[Setup] Creating dataloader...")
    val_loader = create_dataloader(
        json_path=config.val_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        max_audio_length=config.max_audio_length,
        validate_files=config.validate_files
    )
    
    results = evaluate(
        model=model,
        dataloader=val_loader,
        decoder=decoder,
        noise_augmenter=noise_augmenter,
        device=device,
        config=config
    )
    
    if config.save_results:
        save_results(
            results=results,
            config=config,
            checkpoint_epoch=checkpoint_epoch,
            model_type=config.model_type
        )
    
    print("="*60 + "\nEvaluation Completed!\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()