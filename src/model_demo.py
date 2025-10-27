from src.model import VisionConditionedASR
from transformers import AutoTokenizer
import torchaudio
import torch
from PIL import Image
import numpy as np
import os


def count_parameters(model, module_name="Model"):
    """
    モデルのパラメータ数をカウントして表示
    
    Args:
        model: PyTorchモデルまたはモジュール
        module_name: モジュール名（表示用）
    
    Returns:
        dict: パラメータ統計情報
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'name': module_name,
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def print_model_parameters(model):
    """
    モデル全体とコンポーネントごとのパラメータ数を表示
    
    Args:
        model: VisionConditionedASRモデル
    """
    print(f"\n{'='*70}")
    print("Model Parameter Statistics")
    print(f"{'='*70}")
    
    # 全体
    overall = count_parameters(model, "Overall Model")
    
    # 各コンポーネント
    audio_enc = count_parameters(model.audio_encoder, "Audio Encoder (Wav2Vec2)")
    vision_enc = count_parameters(model.vision_encoder, "Vision Encoder (DINOv2)")
    cross_attn = count_parameters(model.cross_attention, "Cross Attention")
    classifier = count_parameters(model.classifier, "Classifier")
    
    # 表示
    components = [overall, audio_enc, vision_enc, cross_attn, classifier]
    
    # ヘッダー
    print(f"\n{'Component':<35} {'Total':>12} {'Trainable':>12} {'Frozen':>12}")
    print(f"{'-'*35} {'-'*12} {'-'*12} {'-'*12}")
    
    # 各コンポーネント
    for comp in components:
        name = comp['name']
        total = comp['total']
        trainable = comp['trainable']
        frozen = comp['frozen']
        
        print(f"{name:<35} {total:>12,} {trainable:>12,} {frozen:>12,}")
    
    # フッター
    print(f"{'='*70}")
    
    # パーセンテージ表示
    print(f"\nTrainable ratio: {overall['trainable']/overall['total']*100:.2f}%")
    print(f"Frozen ratio:    {overall['frozen']/overall['total']*100:.2f}%")
    
    # コンポーネント別の割合
    print(f"\n{'='*70}")
    print("Parameter Distribution:")
    print(f"{'='*70}")
    for comp in components[1:]:  # overall以外
        ratio = comp['total'] / overall['total'] * 100
        print(f"  {comp['name']:<33} {comp['total']:>12,} ({ratio:>5.2f}%)")
    print(f"{'='*70}\n")


def demo():
    """デモ実行関数"""
    print("="*60)
    print("Vision-Conditioned ASR Demo")
    print("="*60)
    
    # モデルの初期化
    print("\n[1/5] Initializing model...")
    avsr = VisionConditionedASR()
    avsr.eval()  # 評価モードに設定
    
    # パラメータ数の表示
    print("\n[2/5] Counting parameters...")
    print_model_parameters(avsr)
    
    # データパス
    wav_path = "../Datasets/SpokenCOCO/wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav"
    img_path = "../Datasets/stair_captions/images/val2014/COCO_val2014_000000325114.jpg"
    
    # 音声データ読み込み
    print("\n[3/5] Loading audio data...")
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    print(f"  Audio shape: {wav.shape}, Sample rate: 16000Hz")
    
    # 画像データ読み込み
    print("\n[4/5] Loading image data...")
    image = Image.open(img_path).convert('RGB')
    print(f"  Image size: {image.size}, Mode: {image.mode}")
    
    # データセット作成（バッチ形式）
    sample = {
        "wav": [wav.numpy()],  # List形式
        "image": [image],        # List形式
        "text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE"
    }
    
    # 推論実行
    print("\n[5/5] Running inference...")
    with torch.no_grad():
        # 各コンポーネントの出力確認
        audio_outputs = avsr.audio_encoder(data=sample)
        vision_outputs = avsr.vision_encoder(data=sample)
        asr_outputs = avsr(data=sample)
    
    print(f"\n{'='*60}")
    print("Output Shapes:")
    print(f"{'='*60}")
    print(f"  Audio features:  {audio_outputs.shape}")
    print(f"  Vision features: {vision_outputs.shape}")
    print(f"  ASR logits:      {asr_outputs.shape}")
    
    # デコーディング処理
    print(f"\n{'='*60}")
    print("Decoding Results:")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # logitsから予測されたトークンIDを取得
    predicted_ids = torch.argmax(asr_outputs, dim=-1)  # [B, seq_len]
    
    # CTC blank tokenのID（通常は0）
    blank_token_id = 0
    
    for i, pred_ids in enumerate(predicted_ids):
        pred_ids = pred_ids.cpu().numpy()
        
        # CTCデコーディング
        pred_tokens = []
        prev_token = None
        
        for token_id in pred_ids:
            # blank tokenをスキップ
            if token_id == blank_token_id:
                prev_token = None
                continue
            
            # 連続する同じトークンは1つだけ保持
            if token_id != prev_token:
                pred_tokens.append(token_id)
            
            prev_token = token_id
        
        # デコード
        transcription = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        
        print(f"\nSample {i}:")
        print(f"  Ground Truth: {sample['text']}")
        print(f"  Prediction:   {transcription}")
        print(f"  Note: Model is untrained, output is random")
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print(f"{'='*60}\n")


def demo_batch():
    """バッチ処理によるデモ実行関数（複数の音声と画像を同時に処理）"""
    print("="*70)
    print("Vision-Conditioned ASR Batch Demo")
    print("="*70)
    
    # モデルの初期化
    print("\n[1/6] Initializing model...")
    avsr = VisionConditionedASR()
    avsr.eval()  # 評価モードに設定
    
    # パラメータ数の表示
    print("\n[2/6] Analyzing model parameters...")
    print_model_parameters(avsr)
    
    sample_files = [
        {
            "wav": "../Datasets/SpokenCOCO/wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav",
            "img": "../Datasets/stair_captions/images/val2014/COCO_val2014_000000325114.jpg",
            "text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE"
        },
        {
            "wav": "../Datasets/SpokenCOCO/wavs/val/0/m1a5mox83rrx60-3V5Q80FXIXRDGZWLAJ5EEBXFON723D_297698_737627.wav",
            "img": "../Datasets/stair_captions/images/val2014/COCO_val2014_000000297698.jpg",
            "text": "THE SKIER TAKES OFF DOWN THE STEEP HILL"
        }
    ]
    
    # ファイル存在チェックと音声/画像のロード
    wavs = []
    images = []
    texts = []
    
    print("\n[3/6] Loading samples...")
    for i, sample_file in enumerate(sample_files):
        try:
            # ファイル存在チェック
            if not os.path.exists(sample_file["wav"]):
                print(f"  ⚠️  Sample {i+1}: Audio file not found, skipping...")
                continue
            if not os.path.exists(sample_file["img"]):
                print(f"  ⚠️  Sample {i+1}: Image file not found, skipping...")
                continue
            
            # 音声読み込み
            wav, sr = torchaudio.load(sample_file["wav"])
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0).numpy()
            
            # 画像読み込み
            image = Image.open(sample_file["img"]).convert('RGB')
            
            wavs.append(wav)
            images.append(image)
            texts.append(sample_file["text"])
            
            # 各サンプルの情報を表示
            print(f"  ✓ Sample {i+1}: Audio shape={wav.shape}, Image size={image.size}")
            
        except Exception as e:
            print(f"  ✗ Sample {i+1}: Error loading - {e}")
            continue
    
    # ロードされたサンプルが0の場合はエラー
    if len(wavs) == 0:
        print("\n❌ No valid samples loaded. Please check file paths.")
        return
    
    # 音声長の統計情報を表示
    wav_lengths = [len(w) for w in wavs]
    print(f"\n[4/6] Audio length statistics:")
    print(f"  Min length: {min(wav_lengths):,} samples ({min(wav_lengths)/16000:.2f}s)")
    print(f"  Max length: {max(wav_lengths):,} samples ({max(wav_lengths)/16000:.2f}s)")
    print(f"  Mean length: {np.mean(wav_lengths):,.0f} samples ({np.mean(wav_lengths)/16000:.2f}s)")
    print(f"  → Padding will be applied to max length")
    
    # バッチデータセット作成
    batch_sample = {
        "wav": wavs,
        "image": images,
        "text": texts
    }
    
    # 推論実行
    print(f"\n[5/6] Running batch inference (Batch Size: {len(batch_sample['wav'])})...")
    try:
        with torch.no_grad():
            # 各コンポーネントの出力確認
            audio_outputs = avsr.audio_encoder(data=batch_sample)
            vision_outputs = avsr.vision_encoder(data=batch_sample)
            asr_outputs = avsr(data=batch_sample)
        
        # 出力形状の確認
        print(f"\n{'='*70}")
        print("Batch Output Shapes:")
        print(f"{'='*70}")
        print(f"  Audio features:  {audio_outputs.shape}  # [B, seq_len, audio_dim]")
        print(f"  Vision features: {vision_outputs.shape}  # [B, num_patches, vision_dim]")
        print(f"  ASR logits:      {asr_outputs.shape}    # [B, seq_len, vocab_size]")
        print(f"  Batch size (B):  {asr_outputs.shape[0]}")
        
        # NaN/Infチェック
        if torch.isnan(asr_outputs).any():
            print("\n⚠️  WARNING: NaN detected in ASR outputs!")
        if torch.isinf(asr_outputs).any():
            print("\n⚠️  WARNING: Inf detected in ASR outputs!")
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # デコーディング処理
    print(f"\n[6/6] Decoding Batch Results...")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # logitsから予測されたトークンIDを取得
    predicted_ids = torch.argmax(asr_outputs, dim=-1)  # [B, seq_len]
    
    # CTC blank tokenのID（通常は0）
    blank_token_id = 0
    
    for i, pred_ids in enumerate(predicted_ids):
        pred_ids = pred_ids.cpu().numpy()
        
        # CTCデコーディング
        pred_tokens = []
        prev_token = None
        
        for token_id in pred_ids:
            # blank tokenをスキップ
            if token_id == blank_token_id:
                prev_token = None
                continue
            
            # 連続する同じトークンは1つだけ保持
            if token_id != prev_token:
                pred_tokens.append(token_id)
            
            prev_token = token_id
        
        # デコード
        transcription = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        
        print(f"\nSample {i+1} / {len(predicted_ids)}:")
        print(f"  Audio length: {wav_lengths[i]:,} samples ({wav_lengths[i]/16000:.2f}s)")
        print(f"  Ground Truth: {batch_sample['text'][i]}")
        print(f"  Prediction:   {transcription}")
        print(f"  Note: Model is untrained, output is random")
    
    print(f"\n{'='*70}")
    print("Batch Demo completed successfully!")
    print(f"{'='*70}\n")


def compare_model_versions():
    """
    Ver2とVer3のパラメータ数を比較
    （Ver3がある場合）
    """
    print("="*70)
    print("Model Version Comparison")
    print("="*70)
    
    print("\n[1/2] Loading Ver2 (hidden_dim=256, heads=4)...")
    from model_ver2 import VisionConditionedASR as VisionConditionedASR_v2
    model_v2 = VisionConditionedASR_v2(vocab_size=32, hidden_dim=256, num_heads=4)
    
    try:
        print("\n[2/2] Loading Ver3 (dim=768, heads=8)...")
        from model_ver3 import VisionConditionedASR as VisionConditionedASR_v3
        model_v3 = VisionConditionedASR_v3(vocab_size=32, num_heads=8)
        
        print("\n" + "="*70)
        print("Ver2 Parameters:")
        print("="*70)
        print_model_parameters(model_v2)
        
        print("\n" + "="*70)
        print("Ver3 Parameters:")
        print("="*70)
        print_model_parameters(model_v3)
        
        # 差分表示
        v2_total = sum(p.numel() for p in model_v2.parameters())
        v3_total = sum(p.numel() for p in model_v3.parameters())
        
        v2_cross = sum(p.numel() for p in model_v2.cross_attention.parameters())
        v3_cross = sum(p.numel() for p in model_v3.cross_attention.parameters())
        
        print("\n" + "="*70)
        print("Comparison Summary:")
        print("="*70)
        print(f"{'Metric':<40} {'Ver2':>14} {'Ver3':>14}")
        print("-"*70)
        print(f"{'Total Parameters':<40} {v2_total:>14,} {v3_total:>14,}")
        print(f"{'Cross Attention Parameters':<40} {v2_cross:>14,} {v3_cross:>14,}")
        print(f"{'Difference (Ver3 - Ver2)':<40} {'':<14} {v3_total-v2_total:>+14,}")
        print(f"{'Cross Attn Diff (Ver3 - Ver2)':<40} {'':<14} {v3_cross-v2_cross:>+14,}")
        print("="*70 + "\n")
        
    except ImportError:
        print("\n⚠️  Ver3 (model_ver3.py) not found. Skipping comparison.")
        print("Showing Ver2 parameters only:\n")
        print_model_parameters(model_v2)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # 比較モード
        compare_model_versions()
    elif len(sys.argv) > 1 and sys.argv[1] == "single":
        # 単一データ版
        demo()
    else:
        # デフォルト: バッチ処理版
        demo_batch()