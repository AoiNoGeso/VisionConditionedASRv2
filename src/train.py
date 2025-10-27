import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
import os
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from tqdm import tqdm
import wandb
from safetensors.torch import save_file, load_file
from audiomentations import Compose, AddGaussianSNR, AddBackgroundNoise, AddGaussianNoise

from model import VisionConditionedASR
from dataloader import create_dataloader


@dataclass
class TrainingConfig:
    train_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_train_fixed.json"
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    vocab_size: Optional[int] = None
    num_heads: int = 8
    dropout: float = 0.1
    
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    num_workers: int = 4
    max_audio_length: float = 10.0
    validate_files: bool = True
    
    freeze_audio_encoder: bool = False
    freeze_vision_encoder: bool = True
    freeze_cross_attention: bool = False

    noise_type: str = "none"  # "none", "gaussian", "white", "background"
    noise_dir: Optional[str] = "../../Datasets/noise"
    gaussian_snr_db: tuple = (5, 20)  # SNR range in dB
    white_noise_level: float = 0.01  # 0.0-1.0
    background_snr_db: tuple = (5, 20)  # SNR range in dB
    
    warmup_steps: int = 1000
    use_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    
    checkpoint_dir: str = "../checkpoints/DINOv2_model"
    save_epoch: int = 1
    resume_from: Optional[str] = None  # e.g., "../checkpoints/pure_cross/epoch_5"
    device: str = "cuda:1"  # "cuda:0", "cuda:1", "cpu"
    
    log_step: int = 50
    validate_epoch: int = 1
    use_wandb: bool = True
    wandb_project: str = "VisionConditionedASRv2"


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


def freeze_layers(model: VisionConditionedASR, config: TrainingConfig):
    print("\n" + "="*60)
    print("Layer Freeze Configuration")
    print("="*60)
    
    if config.freeze_audio_encoder:
        for param in model.audio_encoder.model.parameters():
            param.requires_grad = False
        print("✓ Audio Encoder:    FROZEN")
    else:
        print("✓ Audio Encoder:    Trainable")
    
    if config.freeze_vision_encoder:
        for param in model.vision_encoder.model.parameters():
            param.requires_grad = False
        print("✓ Vision Encoder:   FROZEN")
    else:
        print("✓ Vision Encoder:   Trainable")
    
    if config.freeze_cross_attention:
        for param in model.cross_attention.parameters():
            param.requires_grad = False
        print("✓ Cross Attention:  FROZEN")
    else:
        print("✓ Cross Attention:  Trainable")
    
    print("✓ Classifier:       Trainable")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*60}")
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Frozen:    {total-trainable:,} ({100*(total-trainable)/total:.2f}%)")
    print(f"Total:     {total:,}")
    print(f"{'='*60}\n")


def decode_predictions(
    predicted_ids: torch.Tensor,
    tokenizer,
    blank_token_id: int = 0
) -> List[str]:
    decoded_texts = []
    for pred_ids_seq in predicted_ids:
        pred_tokens = []
        prev_token = None
        for token_id in pred_ids_seq.tolist():
            if token_id == blank_token_id:
                prev_token = None
                continue
            if token_id != prev_token:
                pred_tokens.append(token_id)
            prev_token = token_id
        decoded_texts.append(tokenizer.decode(pred_tokens, skip_special_tokens=True))
    return decoded_texts


def compute_ctc_loss(
    logits: torch.Tensor,
    texts: List[str],
    tokenizer,
    wav_lengths: torch.Tensor
) -> torch.Tensor:
    batch_size = logits.size(0)
    device = logits.device
    
    target_ids = []
    target_lengths = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        target_ids.extend(tokens)
        target_lengths.append(len(tokens))
    
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=device)
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    try:
        loss = ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
    except RuntimeError as e:
        print(f"\n[Warning] CTC error: {e}")
        loss = torch.tensor(1e6, device=device, requires_grad=True)
    
    return loss


def train_one_epoch(
    model: VisionConditionedASR,
    noise_augmenter: NoiseAugmenter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    tokenizer,
    device: torch.device,
    epoch: int,
    config: TrainingConfig
):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    
    print(f"\n{'='*60}\nEpoch {epoch+1}/{config.num_epochs} - Training\n{'='*60}")
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Train", total=num_batches)
    
    for batch_idx, batch in enumerate(pbar):
        try:
            if noise_augmenter.augment is not None:
                batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]

            wav_lengths = batch["wav_lengths"].to(device)
            
            with autocast(device_type='cuda', enabled=config.use_amp, dtype=amp_dtype):
                logits = model(batch)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"\n🚨 Logits NaN/Inf at batch {batch_idx}, skipping")
                    continue
                loss = compute_ctc_loss(logits, batch["text"], tokenizer, wav_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n🚨 Loss NaN/Inf at batch {batch_idx}, skipping")
                continue
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            
            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_postfix(loss=f"{current_loss:.4f}")
            
            if (batch_idx + 1) % config.log_step == 0 or (batch_idx + 1) == num_batches:
                avg_loss = total_loss / (batch_idx + 1)
                if config.use_wandb:
                    wandb.log({
                        "train/loss_step": current_loss,
                        "train/avg_loss_step": avg_loss,
                        "train/scale": scaler.get_scale(),
                        "epoch": epoch
                    }, step=epoch * num_batches + batch_idx + 1)
            
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[Error] Batch {batch_idx}: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if config.use_wandb:
        wandb.log({"train/loss_epoch": avg_loss, "epoch": epoch})
    
    print(f"\n{'='*60}\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}\n{'='*60}\n")
    return avg_loss


def validate(
    model: VisionConditionedASR,
    noise_augmenter: NoiseAugmenter,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
    num_examples: int = 3
):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_references = []
    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    
    print(f"\n{'='*60}\nEpoch {epoch+1}/{config.num_epochs} - Validation\n{'='*60}")
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Val", total=len(dataloader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                if noise_augmenter.augment is not None:
                    batch["wav"] = [noise_augmenter.apply(wav) for wav in batch["wav"]]

                wav_lengths = batch["wav_lengths"].to(device)
                
                with autocast(device_type='cuda', enabled=config.use_amp, dtype=amp_dtype):
                    logits = model(batch)
                    loss = compute_ctc_loss(logits, batch["text"], tokenizer, wav_lengths)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    current_loss = loss.item()
                    total_loss += current_loss
                    num_batches += 1
                    pbar.set_postfix(loss=f"{current_loss:.4f}")
                
                predicted_ids = torch.argmax(logits, dim=-1)
                pred_texts = decode_predictions(predicted_ids, tokenizer, blank_token_id=0)
                all_predictions.extend(pred_texts)
                all_references.extend(batch["text"])
            except Exception as e:
                print(f"\n[Error] Val batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    print(f"\n{'='*60}\nPrediction Examples:\n{'='*60}")
    prediction_table = []
    for i in range(min(num_examples, len(all_predictions))):
        ref = all_references[i][:80]
        pred = all_predictions[i][:80]
        print(f"\nExample {i+1}:\n  Ref:  {ref}\n  Pred: {pred}")
        prediction_table.append([i+1, ref, pred])
        
    if config.use_wandb:
        wandb.log({
            "val/loss_epoch": avg_loss,
            "val/prediction_examples": wandb.Table(
                data=prediction_table,
                columns=["Example", "Reference", "Prediction"]
            ),
            "epoch": epoch
        })
    
    print(f"\n{'='*60}\nEpoch {epoch+1} Val Loss: {avg_loss:.4f}\n{'='*60}\n")
    return avg_loss


def save_checkpoint(
    model: VisionConditionedASR,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: TrainingConfig
):
    epoch_dir = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    model_path = os.path.join(epoch_dir, f"model_epoch_{epoch+1}.safetensors")
    save_file(model.state_dict(), model_path)
    
    state_path = os.path.join(epoch_dir, f"checkpoint_epoch_{epoch+1}_state.pt")
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }, state_path)
    
    print(f"[Checkpoint] Saved to: {model_path}")


def load_checkpoint(
    checkpoint_dir: str,
    model: VisionConditionedASR,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
):
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    state_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}_state.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(state_path):
        raise FileNotFoundError(f"Checkpoint files incomplete in: {checkpoint_dir}")
    
    print(f"\n{'='*60}\nResuming from: {checkpoint_dir}\n{'='*60}")
    
    model.load_state_dict(load_file(model_path, device=str(device)))
    checkpoint_state = torch.load(state_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint_state:
        scaler.load_state_dict(checkpoint_state['scaler_state_dict'])
    
    start_epoch = checkpoint_state['epoch'] + 1
    print(f"[Resume] Epoch {start_epoch}, Loss: {checkpoint_state.get('val_loss', 0):.4f}\n{'='*60}\n")
    return start_epoch, checkpoint_state.get('val_loss', 0.0)


def main():
    config = TrainingConfig()
    
    if config.use_wandb:
        wandb.init(project=config.wandb_project, config=config.__dict__)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nDevice: {device} | Batch: {config.batch_size} | LR: {config.learning_rate} | "
          f"Epochs: {config.num_epochs} | Noise: {config.noise_type} | AMP: {config.use_amp}\n{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    model = VisionConditionedASR(
        vocab_size=config.vocab_size,
        num_heads=config.num_heads,
        dropout=config.dropout,
        device=device
    ).to(device)
    
    freeze_layers(model, config)
    scaler = GradScaler(enabled=config.use_amp)
    
    train_loader = create_dataloader(
        json_path=config.train_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        max_audio_length=config.max_audio_length,
        validate_files=config.validate_files
    )
    
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
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    noise_augmenter = NoiseAugmenter(
        noise_type=config.noise_type,
        gaussian_snr_db=config.gaussian_snr_db,
        white_noise_level=config.white_noise_level,
        noise_dir=config.noise_dir,
        background_snr_db=config.background_snr_db
    )
    
    print("\n" + "="*60 + "\nStarting Training\n" + "="*60 + "\n")
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    if config.resume_from:
        start_epoch, best_val_loss = load_checkpoint(
            checkpoint_dir=config.resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )
    
    for epoch in range(start_epoch, config.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            noise_augmenter=noise_augmenter,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            tokenizer=tokenizer,
            device=device,
            epoch=epoch,
            config=config
        )
        
        val_loss = 0.0
        if (epoch + 1) % config.validate_epoch == 0:
            val_loss = validate(
                model=model,
                noise_augmenter=noise_augmenter,
                dataloader=val_loader,
                tokenizer=tokenizer,
                device=device,
                epoch=epoch,
                config=config
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\n✨ New best: {best_val_loss:.4f}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    config=config
                )
        
        if (epoch + 1) % config.save_epoch == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config
            )
    
    print("\n" + "="*60 + "\nTraining Completed!\n" + "="*60 + "\n")
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()