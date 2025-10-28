# VisionConditionedASRv2

視覚情報を条件付けとして利用する音声認識（ASR）モデル

## 概要

VisionConditionedASRv2は、音声信号と視覚情報を統合することで音声認識精度の向上を目指すマルチモーダルASRモデルです。視覚情報を音声認識の文脈情報として活用することで、特にノイズ環境下での認識性能改善を実現します。

### 主な特徴

- **マルチモーダル統合**: 音声（Wav2Vec2）と画像（DINOv2）の特徴を融合
- **Cross Attention機構**: 音声特徴が画像の全パッチに注意を向ける仕組み

## モデルアーキテクチャ

```
[Audio Input (16kHz)] → [Wav2Vec2-base] → [Audio Features: B×T×768]
                                                    ↓
                                            [Cross Attention]
                                                    ↑
[Image Input (RGB)]   → [DINOv2-base]   → [Vision Features: B×N×768]
                                                    ↓
                                            [Linear Classifier]
                                                    ↓
                                            [CTC Logits: B×T×vocab]
```

### コンポーネント詳細

| コンポーネント | モデル | 次元 |
|---|---|---|
| Audio Encoder | Wav2Vec2-base | 768 | 
| Vision Encoder | DINOv2-base | 768 |
| Cross Attention | Multi-head (8 heads) | 768 |
| Classifier | Linear | 768→32 |

## 環境構築

### 必要要件

- Python 3.10.13

### インストール

```bash
pip install -r requirements.txt
```

### 主要な依存ライブラリ

- torch >= 2.9.0
- torchaudio >= 2.9.0
- transformers >= 4.57.1
- Pillow >= 12.0.0
- audiomentations >= 0.43.1
- jiwer >= 4.0.0
- pyctcdecode >= 0.5.0
- wandb >= 0.22.2

## データセット

### SpokenCOCO形式

本モデルはSpokenCOCO形式のデータセットを使用します。

#### ディレクトリ構造

```
Datasets/
├── SpokenCOCO/
│   ├── SpokenCOCO_train_fixed.json
│   ├── SpokenCOCO_val_fixed.json
│   └── wavs/
│       ├── train/
│       │   └── *.wav
│       └── val/
│           └── *.wav
└── stair_captions/
    └── images/
        ├── train2014/
        │   └── COCO_train2014_*.jpg
        └── val2014/
            └── COCO_val2014_*.jpg
```

#### JSON形式

```json
{
  "data": [
    {
      "image": "val2014/COCO_val2014_000000325114.jpg",
      "captions": [
        {
          "wav": "wavs/val/0/sample_325114_629297.wav",
          "text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE"
        }
      ]
    }
  ]
}
```

## 使用方法

### 1. デモ実行

#### バッチ推論（デフォルト）

```bash
cd src
python model_demo.py
```

### 2. 学習

#### 基本的な学習

```bash
cd src
python train.py
```

#### 設定のカスタマイズ

`src/train.py`の`TrainingConfig`クラスを編集：

```python
@dataclass
class TrainingConfig:
    # データパス
    train_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_train_fixed.json"
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # モデル設定
    num_heads: int = 8              # Attention heads
    dropout: float = 0.1            # Dropout率
    
    # 学習設定
    batch_size: int = 16            # バッチサイズ
    num_epochs: int = 20            # エポック数
    learning_rate: float = 1e-5     # 学習率
    
    # レイヤーのfreeze設定
    freeze_audio_encoder: bool = False
    freeze_vision_encoder: bool = True
    freeze_cross_attention: bool = False
    
    # ノイズ増強
    noise_type: str = "none"        # "none", "gaussian", "white", "background"
    
    # Mixed Precision
    use_amp: bool = True
    amp_dtype: str = "float16"      # "float16" or "bfloat16"
    
    # チェックポイント
    checkpoint_dir: str = "checkpoints/DINOv2_model"
    resume_from: Optional[str] = None
    
    # デバイス
    device: str = "cuda:0"
```

#### チェックポイントから再開

```python
resume_from: Optional[str] = "checkpoints/DINOv2_model/epoch_10"
```

### 3. 評価・テスト

#### WER評価

```bash
cd src
python test.py
```

#### 設定のカスタマイズ

`src/test.py`の`TestConfig`クラスを編集：

```python
@dataclass
class TestConfig:
    # チェックポイント
    checkpoint_dir: str = "checkpoints/DINOv2_model/epoch_10"
    model_type: str = "vision"      # "vision" or "pure"
    
    # 評価設定
    use_image: bool = True          # 画像を使用するか
    use_beam_search: bool = True    # Beam Searchを使用
    beam_width: int = 10            # Beam幅
    
    # ノイズ設定
    noise_type: str = "white"       # テスト時のノイズ
    white_noise_level: float = 0.01
    
    # 結果保存
    save_results: bool = True
    results_dir: str = "results/"
```

## 主要機能

### ノイズ増強

学習・評価時に以下のノイズを付加可能：

#### 1. Gaussian SNRノイズ

```python
noise_type: str = "gaussian"
gaussian_snr_db: tuple = (5, 20)  # SNR範囲
```

#### 2. ホワイトノイズ

```python
noise_type: str = "white"
white_noise_level: float = 0.01   # ノイズレベル
```

#### 3. 背景ノイズ

```python
noise_type: str = "background"
noise_dir: str = "../../Datasets/noise"
background_snr_db: tuple = (5, 20)
```

### デコーディング

#### Greedy Decoding

```python
use_beam_search: bool = False
```

#### Beam Search Decoding

```python
use_beam_search: bool = True
beam_width: int = 10
```

### WER評価指標

- **WER (Word Error Rate)**: 単語誤り率
- **MER (Match Error Rate)**: マッチ誤り率
- **WIL (Word Information Lost)**: 単語情報損失
- **Error Breakdown**: Substitutions, Deletions, Insertions, Hits


## ファイル構成

```
VisionConditionedASRv2/
├── README.md
├── requirements.txt             
├── pyproject.toml              
└── src/
    ├── model.py                # モデル定義
    ├── dataloader.py           # データローダー
    ├── train.py                # 学習スクリプト
    ├── test.py                 # 評価スクリプト
    └── model_demo.py           # デモスクリプト
```

### 各ファイルの役割

#### `src/model.py`

- `AudioEncoder`: Wav2Vec2ベースの音声エンコーダー
- `VisionEncoder`: DINOv2ベースの画像エンコーダー
- `CrossAttention`: マルチヘッドクロスアテンション
- `VisionConditionedASR`: 統合モデル

#### `src/dataloader.py`

- `SpokenCOCODataset`: データセットクラス
- `spokenCOCO_collate`: バッチコレート関数
- `create_dataloader`: DataLoader作成ヘルパー

#### `src/train.py`

- `TrainingConfig`: 学習設定
- `NoiseAugmenter`: ノイズ増強
- `train_one_epoch()`: 1エポックの学習
- `validate()`: 検証
- `save_checkpoint()`: チェックポイント保存
- `load_checkpoint()`: チェックポイント読み込み

#### `src/test.py`

- `TestConfig`: テスト設定
- `CTCDecoder`: CTC/Beam Searchデコーダー
- `compute_wer()`: WER計算
- `evaluate()`: モデル評価
- `save_results()`: 結果保存

#### `src/model_demo.py`

- `demo()`: 単一サンプルデモ
- `demo_batch()`: バッチデモ
- `print_model_parameters()`: パラメータ数表示
- `compare_model_versions()`: バージョン比較
