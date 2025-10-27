import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from tqdm import tqdm
import wandb
from safetensors.torch import save_file, load_file
from audiomentations import Compose, AddGaussianSNR, AddBackgroundNoise, AddGaussianNoise

from model_DINOv2 import VisionConditionedASR
from dataloader import create_dataloader


@dataclass
class TrainingConfig:
    """学習設定"""
    # データセットパス
    train_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_train_fixed.json"
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # モデル設定
    vocab_size: Optional[int] = None
    hidden_dim: int = 256
    num_heads: int = 4
    
    # 学習設定
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # データローダー設定
    num_workers: int = 4
    max_audio_length: float = 10.0
    validate_files: bool = True
    
    # 層凍結設定
    freeze_audio_encoder: bool = False
    freeze_vision_encoder: bool = True
    freeze_cross_attention: bool = False

    # ノイズ設定
    noise_type: str = "none"  # "none", "gaussian", "white", "background"
    noise_dir: Optional[str] = "../../Datasets/noise"  # バックグラウンドノイズ用ディレクトリ
    gaussian_snr_db: tuple = (5, 20)  # ガウシアンノイズのSNR範囲(dB)
    white_noise_level: float = 0.01  # ホワイトノイズのレベル (0.0-1.0)
    background_snr_db: tuple = (5, 20)  # バックグラウンドノイズのSNR範囲(dB)
    
    # 学習スケジュール
    warmup_steps: int = 1000
    
    # 半精度学習設定
    use_amp: bool = True  # Automatic Mixed Precision (AMP) の使用
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    
    # 保存設定
    checkpoint_dir: str = "../checkpoints/DINOv2_model"
    save_epoch: int = 1
    
    # 学習再開設定
    resume_from: Optional[str] = None # 再開するチェックポイントディレクトリ (例: "../checkpoints/pure_cross/epoch_5")
    
    # デバイス設定
    device: str = "cuda:1"  # "cuda:0", "cuda:1", "cpu"
    
    # ログ設定
    log_step: int = 50  # ステップごと
    validate_epoch: int = 1  # エポックごと
    use_wandb: bool = True  # wandbの使用/不使用
    wandb_project: str = "VisionConditionedASR"


class NoiseAugmenter:
    """音声にノイズを付加するクラス"""
    
    def __init__(self, noise_type: str = "none", gaussian_snr_db: tuple = (5, 20),
                 white_noise_level: float = 0.01, noise_dir: Optional[str] = None, 
                 background_snr_db: tuple = (5, 20), sample_rate: int = 16000):
        """
        Args:
            noise_type: ノイズタイプ ("none", "gaussian", "white", "background")
            gaussian_snr_db: ガウシアンノイズのSNR範囲(dB)
            white_noise_level: ホワイトノイズのレベル
            noise_dir: バックグラウンドノイズファイルのディレクトリ
            background_snr_db: バックグラウンドノイズのSNR範囲(dB)
            sample_rate: サンプリングレート
        """
        self.noise_type = noise_type
        self.sample_rate = sample_rate
        
        print(f"\n[NoiseAugmenter] Initializing with noise_type: {noise_type}")
        
        if noise_type == "none":
            self.augment = None
            print("  No noise augmentation")
            
        elif noise_type == "gaussian":
            self.augment = Compose([
                AddGaussianSNR(min_snr_db=gaussian_snr_db[0], max_snr_db=gaussian_snr_db[1], p=1.0)
            ])
            print(f"  Gaussian noise SNR: {gaussian_snr_db[0]}-{gaussian_snr_db[1]} dB")
            
        elif noise_type == "white":
            # ホワイトノイズは均一分布のガウシアンノイズとして実装
            self.augment = Compose([
                AddGaussianNoise(min_amplitude=white_noise_level, max_amplitude=white_noise_level, p=1.0)
            ])
            print(f"  White noise level: {white_noise_level}")
            
        elif noise_type == "background":
            if noise_dir is None or not os.path.exists(noise_dir):
                raise ValueError(f"Background noise directory not found: {noise_dir}")
            
            # ノイズファイルのリストを取得
            noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) 
                          if f.endswith(('.wav', '.mp3', '.flac'))]
            
            if len(noise_files) == 0:
                raise ValueError(f"No noise files found in: {noise_dir}")
            
            self.augment = Compose([
                AddBackgroundNoise(
                    sounds_path=noise_dir,
                    min_snr_in_db=background_snr_db[0],
                    max_snr_in_db=background_snr_db[1],
                    p=1.0
                )
            ])
            print(f"  Background noise from: {noise_dir}")
            print(f"  Found {len(noise_files)} noise files")
            print(f"  SNR range: {background_snr_db[0]}-{background_snr_db[1]} dB")
            
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        """
        音声にノイズを付加
        
        Args:
            audio: 音声データ (numpy array, shape: (T,))
        
        Returns:
            ノイズが付加された音声データ
        """
        if self.augment is None:
            return audio
        
        # audiomentationsはfloat32を期待
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # ノイズを付加
        augmented_audio = self.augment(samples=audio, sample_rate=self.sample_rate)
        
        return augmented_audio


def freeze_layers(model: VisionConditionedASR, config: TrainingConfig):
    """
    指定された層を凍結
    
    Args:
        model: VisionConditionedASRモデル
        config: 学習設定
    """
    print("\n" + "="*60)
    print("Layer Freeze Configuration")
    print("="*60)
    
    # Audio Encoderの凍結
    if config.freeze_audio_encoder:
        for param in model.audio_encoder.model.parameters():
            param.requires_grad = False
        print("✓ Audio Encoder (Wav2Vec2):    FROZEN")
    else:
        print("✓ Audio Encoder (Wav2Vec2):    Trainable")
    
    # Vision Encoderの凍結
    if config.freeze_vision_encoder:
        for param in model.vision_encoder.model.vision_model.parameters():
            param.requires_grad = False
        print("✓ Vision Encoder (CLIP):       FROZEN")
    else:
        print("✓ Vision Encoder (CLIP):       Trainable")
    
    # Cross Attentionの凍結
    if config.freeze_cross_attention:
        for param in model.cross_attention.parameters():
            param.requires_grad = False
        print("✓ Cross Attention:             FROZEN")
    else:
        print("✓ Cross Attention:             Trainable")
    
    # Classifierは常に学習可能
    print("✓ Classifier (Linear):         Trainable (always)")
    
    # 学習可能なパラメータ数を表示
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"\n{'='*60}")
    print("Parameter Statistics:")
    print(f"{'='*60}")
    print(f"Trainable parameters:  {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Frozen parameters:     {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
    print(f"Total parameters:      {total_params:,}")
    print(f"{'='*60}\n")


def decode_predictions(
    predicted_ids: torch.Tensor, 
    tokenizer,
    blank_token_id: int = 0
) -> List[str]:
    """
    CTCの予測結果をテキストにデコード
    
    Args:
        predicted_ids: 予測されたトークンID [batch, seq_len]
        tokenizer: トークナイザー
        blank_token_id: CTCのblankトークンID（通常は0）
        
    Returns:
        デコードされたテキストのリスト
    """
    decoded_texts = []
    
    for pred_ids_seq in predicted_ids:
        pred_tokens = []
        prev_token = None
        
        for token_id in pred_ids_seq.tolist():
            # blank tokenはスキップ
            if token_id == blank_token_id:
                prev_token = None
                continue
            
            # 連続する同じトークンは1つだけ保持
            if token_id != prev_token:
                pred_tokens.append(token_id)
            
            prev_token = token_id
        
        # トークンをテキストに変換
        decoded_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        decoded_texts.append(decoded_text)
    
    return decoded_texts


def compute_ctc_loss(
    logits: torch.Tensor,
    texts: List[str],
    tokenizer,
    wav_lengths: torch.Tensor
) -> torch.Tensor:
    """
    CTC損失を計算
    
    Args:
        logits: モデル出力 [B, T, vocab_size]
        texts: ターゲットテキスト [B]
        tokenizer: トークナイザー
        wav_lengths: 各音声の長さ [B]
    
    Returns:
        loss: CTC損失
    """
    batch_size = logits.size(0)
    device = logits.device
    
    # ターゲットのトークン化
    target_ids = []
    target_lengths = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        target_ids.extend(tokens)
        target_lengths.append(len(tokens))
    
    # Tensorに変換
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    
    # 入力長を計算
    input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=device)
    
    # CTCLossの計算
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs.transpose(0, 1)  # [T, B, vocab_size]
    
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    try:
        loss = ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
    except RuntimeError as e:
        print(f"\n[Warning] CTC Loss calculation error: {e}")
        print(f"  Input lengths: {input_lengths.tolist()}")
        print(f"  Target lengths: {target_lengths.tolist()}")
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
    """
    1エポック分の学習
    
    Args:
        model: モデル
        noise_augmenter: ノイズ付加器
        dataloader: データローダー
        optimizer: オプティマイザー
        scaler: GradScaler (AMP用)
        tokenizer: トークナイザー
        device: デバイス
        epoch: 現在のエポック番号
        config: 学習設定
    
    Returns:
        avg_loss: 平均損失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # amp_dtypeの設定
    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{config.num_epochs} - Training")
    print(f"{'='*60}")
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Train", total=num_batches)
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # ノイズを付加
            if noise_augmenter.augment is not None:
                noisy_wav = []
                for wav in batch["wav"]:
                    augmented = noise_augmenter.apply(wav)
                    noisy_wav.append(augmented)
                batch["wav"] = noisy_wav

            # データをデバイスに移動（wav_lengthsのみ）
            wav_lengths = batch["wav_lengths"].to(device)
            
            # Forward pass with autocast
            with autocast(device_type='cuda', enabled=config.use_amp, dtype=amp_dtype):
                logits = model(batch)  # [B, T, vocab_size]
                
                # NaN/Infチェック
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"\n🚨 CRITICAL: Logits contain NaN or Inf at batch {batch_idx}!")
                    print(f"  Skipping this batch...")
                    continue
                
                # CTC損失の計算
                loss = compute_ctc_loss(logits, batch["text"], tokenizer, wav_lengths)
            
            # NaN/Infチェック
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n🚨 CRITICAL: Loss is NaN or Inf at batch {batch_idx}!")
                print(f"  Skipping this batch...")
                continue
            
            # Backward pass with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # 損失の累積
            current_loss = loss.item()
            total_loss += current_loss
            
            # プログレスバーに損失を表示
            pbar.set_postfix(loss=f"{current_loss:.4f}")
            
            # ログ出力
            if (batch_idx + 1) % config.log_step == 0 or (batch_idx + 1) == num_batches:
                avg_loss = total_loss / (batch_idx + 1)

                # wandbにステップごとの損失をログ
                if config.use_wandb:
                    wandb.log({
                        "train/loss_step": current_loss,
                        "train/avg_loss_step": avg_loss,
                        "train/scale": scaler.get_scale(),
                        "epoch": epoch,
                    }, step=epoch * num_batches + batch_idx + 1)
            
            # メモリクリア
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"\n[Error] Exception at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # wandbにエポックごとの平均損失をログ
    if config.use_wandb:
        wandb.log({
            "train/loss_epoch": avg_loss,
            "epoch": epoch,
        })
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} Training Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"{'='*60}\n")
    
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
    """
    検証
    
    Args:
        model: モデル
        noise_augmenter: ノイズ付加器
        dataloader: 検証データローダー
        tokenizer: トークナイザー
        device: デバイス
        epoch: 現在のエポック番号
        config: 学習設定
        num_examples: 表示する予測例の数
    
    Returns:
        avg_loss: 平均損失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_references = []
    
    # amp_dtypeの設定
    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{config.num_epochs} - Validation")
    print(f"{'='*60}")
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Val", total=len(dataloader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                # ノイズを付加
                if noise_augmenter.augment is not None:
                    noisy_wav = []
                    for wav in batch["wav"]:
                        augmented = noise_augmenter.apply(wav)
                        noisy_wav.append(augmented)
                    batch["wav"] = noisy_wav

                # データをデバイスに移動
                wav_lengths = batch["wav_lengths"].to(device)
                
                # Forward pass with autocast
                with autocast(device_type='cuda', enabled=config.use_amp, dtype=amp_dtype):
                    logits = model(batch)
                    
                    # 損失計算
                    loss = compute_ctc_loss(logits, batch["text"], tokenizer, wav_lengths)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    current_loss = loss.item()
                    total_loss += current_loss
                    num_batches += 1
                    
                    # プログレスバーに損失を表示
                    pbar.set_postfix(loss=f"{current_loss:.4f}")
                
                # 予測のデコード
                predicted_ids = torch.argmax(logits, dim=-1)
                pred_texts = decode_predictions(predicted_ids, tokenizer, blank_token_id=0)
                
                # 結果の保存
                all_predictions.extend(pred_texts)
                all_references.extend(batch["text"])
                
            except Exception as e:
                print(f"\n[Error] Exception at validation batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 予測例を表示
    print(f"\n{'='*60}")
    print("Prediction Examples:")
    print(f"{'='*60}")
    
    prediction_table = []
    for i in range(min(num_examples, len(all_predictions))):
        ref = all_references[i][:80]
        pred = all_predictions[i][:80]
        print(f"\nExample {i+1}:")
        print(f"  Reference:  {ref}")
        print(f"  Prediction: {pred}")
        prediction_table.append([i+1, ref, pred])
        
    # wandbに検証結果をログ
    if config.use_wandb:
        wandb.log({
            "val/loss_epoch": avg_loss,
            "val/prediction_examples": wandb.Table(
                data=prediction_table, 
                columns=["Example", "Reference", "Prediction"]
            ),
            "epoch": epoch,
        })
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} Validation Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Total samples: {len(all_predictions)}")
    print(f"{'='*60}\n")
    
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
    """
    チェックポイントを保存
    
    Args:
        model: モデル
        optimizer: オプティマイザー
        scaler: GradScaler (AMP用)
        epoch: エポック番号
        train_loss: 訓練損失
        val_loss: 検証損失
        config: 学習設定
    """
    # エポックごとのディレクトリを作成
    epoch_dir = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # モデルの重みを.safetensors形式で保存
    model_path = os.path.join(epoch_dir, f"model_epoch_{epoch+1}.safetensors")
    save_file(model.state_dict(), model_path)
    
    # その他の学習状態を.pt形式で保存
    state_path = os.path.join(epoch_dir, f"checkpoint_epoch_{epoch+1}_state.pt")
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }, state_path)
    
    print(f"[Checkpoint] Model saved to: {model_path}")
    print(f"[Checkpoint] State saved to: {state_path}")


def load_checkpoint(
    checkpoint_dir: str,
    model: VisionConditionedASR,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
):
    """
    チェックポイントから学習を再開
    
    Args:
        checkpoint_dir: チェックポイントディレクトリのパス (例: ../checkpoints/epoch_3)
        model: モデルインスタンス
        optimizer: オプティマイザー
        scaler: GradScaler
        device: デバイス
    
    Returns:
        start_epoch: 再開するエポック番号
        best_val_loss: 保存されていた最良の検証損失
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # エポック番号を取得
    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    
    # モデルファイルのパス
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    state_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}_state.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"State file not found: {state_path}")
    
    print(f"\n{'='*60}")
    print("Resuming Training from Checkpoint")
    print(f"{'='*60}")
    print(f"Loading from: {checkpoint_dir}")
    
    # モデルの重みをロード（safetensors）
    state_dict = load_file(model_path, device=str(device))
    model.load_state_dict(state_dict)
    
    # 学習状態をロード（.pt）
    checkpoint_state = torch.load(state_path, map_location=device, weights_only=False)
    
    # オプティマイザーの状態を復元
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    
    # GradScalerの状態を復元
    if 'scaler_state_dict' in checkpoint_state:
        scaler.load_state_dict(checkpoint_state['scaler_state_dict'])
    
    # エポック情報を取得
    start_epoch = checkpoint_state['epoch'] + 1
    train_loss = checkpoint_state.get('train_loss', 0.0)
    val_loss = checkpoint_state.get('val_loss', 0.0)
    
    print(f"[Resume] Successfully loaded checkpoint")
    print(f"  Last completed epoch: {checkpoint_state['epoch'] + 1}")
    print(f"  Resuming from epoch: {start_epoch + 1}")
    print(f"  Last train loss: {train_loss:.4f}")
    print(f"  Last val loss: {val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return start_epoch, val_loss


def main():
    """メイン学習関数"""
    # 設定の初期化
    config = TrainingConfig()
    
    # wandbの初期化
    if config.use_wandb:
        print("[Setup] Initializing wandb...")
        wandb.init(
            project=config.wandb_project,
            config=config.__dict__
        )
    
    # デバイスの設定
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Num epochs: {config.num_epochs}")
    print(f"Noise type: {config.noise_type}")
    print(f"Use AMP: {config.use_amp}")
    if config.use_amp:
        print(f"AMP dtype: {config.amp_dtype}")
    print(f"Use wandb: {config.use_wandb}")
    print(f"{'='*60}\n")
    
    # トークナイザーの初期化
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデルの初期化
    print("[Setup] Initializing model...")
    model = VisionConditionedASR(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        device=device
    ).to(device)
    
    # 層の凍結
    freeze_layers(model, config)
    
    # GradScalerの初期化
    scaler = GradScaler(enabled=config.use_amp)
    
    # データローダーの作成
    print("[Setup] Creating dataloaders...")
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
    
    # オプティマイザーの設定
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # ノイズ付加器の初期化
    print("[Setup] Initializing noise augmenter...")
    noise_augmenter = NoiseAugmenter(
        noise_type=config.noise_type,
        gaussian_snr_db=config.gaussian_snr_db,
        white_noise_level=config.white_noise_level,
        noise_dir=config.noise_dir,
        background_snr_db=config.background_snr_db,
        sample_rate=16000
    )
    
    # 学習ループ
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # チェックポイントから再開する場合
    if config.resume_from is not None:
        start_epoch, best_val_loss = load_checkpoint(
            config.resume_from, model, optimizer, scaler, device
        )
        print(f"[Resume] Best validation loss so far: {best_val_loss:.4f}\n")
    
    for epoch in range(start_epoch, config.num_epochs):
        # 訓練
        train_loss = train_one_epoch(
            model, noise_augmenter, train_loader, optimizer, scaler, tokenizer, device, epoch, config
        )
        
        # 検証
        val_loss = 0.0
        if (epoch + 1) % config.validate_epoch == 0:
            val_loss = validate(
                model, noise_augmenter, val_loader, tokenizer, device, epoch, config
            )
            
            # ベストモデルの保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\n✨ New best validation loss: {best_val_loss:.4f}")
                save_checkpoint(
                    model, optimizer, scaler, epoch, train_loss, val_loss, config
                )
        
        # 定期的なチェックポイント保存
        if (epoch + 1) % config.save_epoch == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch, train_loss, val_loss, config
            )
    
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60 + "\n")
    
    # wandbの終了
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()