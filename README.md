# CA-DETR: Cross-Attention DETR for Reference-Based Detection

Cross-Attention DETR cho bài toán reference-based object detection trong video drone surveillance.

## 🏗️ Kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGES                             │
│                                                                  │
│  Template (640×640)              Search (640×640)                │
│         │                              │                         │
└─────────┼──────────────────────────────┼─────────────────────────┘
          │                              │
          ▼                              ▼
    ┌─────────┐                    ┌─────────┐
    │Backbone │                    │Backbone │
    │ResNet-18│                    │(shared) │
    │  or     │                    │         │
    │EfficNet │                    │         │
    └────┬────┘                    └────┬────┘
         │                              │
         │ (B, 256, 20, 20)            │ (B, 256, 20, 20)
         │                              │
         ▼                              ▼
    ┌─────────────┐              ┌─────────────┐
    │  Positional │              │  Positional │
    │  Encoding   │              │  Encoding   │
    └──────┬──────┘              └──────┬──────┘
           │                            │
           │ Flatten: (B, 400, 256)    │ Flatten: (B, 400, 256)
           │                            │
           ▼                            ▼
    ┌─────────────┐              ┌─────────────┐
    │ Transformer │              │ Transformer │
    │  Encoder    │              │  Encoder    │
    │  (3 layers) │              │  (3 layers) │
    └──────┬──────┘              └──────┬──────┘
           │                            │
           │                            │
           ▼                            X (not used)
    ┌─────────────┐
    │  Template   │
    │   Memory    │
    │ (B,400,256) │
    └──────┬──────┘
           │
           │         ┌──────────────────┐
           │         │ Learnable Query  │
           │         │   (B, 1, 256)    │
           │         └────────┬─────────┘
           │                  │
           └──────────────────┼──────────┐
                              │          │
                              ▼          │
                       ┌──────────────┐  │
                       │ Transformer  │  │
                       │   Decoder    │◄─┘
                       │  (3 layers)  │
                       │              │
                       │ Self-Attn +  │
                       │ Cross-Attn   │
                       └──────┬───────┘
                              │
                              │ (B, 1, 256)
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌───────────────┐          ┌───────────────┐
        │Classification │          │   BBox Reg    │
        │     Head      │          │     Head      │
        │   (Linear)    │          │  (3-layer MLP)│
        └───────┬───────┘          └───────┬───────┘
                │                          │
                ▼                          ▼
         ┌─────────────┐            ┌─────────────┐
         │ Confidence  │            │   [cx,cy,   │
         │   (B,1,1)   │            │    w, h]    │
         │             │            │   (B,1,4)   │
         └─────────────┘            └─────────────┘
                │                          │
                └──────────┬───────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Final BBox  │
                    │  Prediction │
                    └─────────────┘
```

### Đặc điểm chính

- **Cross-Attention**: Decoder cross-attend vào template memory → focus vào object cần tìm
- **Dense Detection**: Predict continuous bbox (không bị giới hạn patch grid) → IoU cao (0.8-0.87)
- **Lightweight**: ResNet-18 (~24M) hoặc EfficientNet-B0 (~18M params)
- **GIoU Loss**: Tối ưu trực tiếp IoU metric

---

## 🚀 Cài đặt

```bash
cd CA-DETR
pip install -r requirements.txt
```

---

## 🏋️ Training

### Basic

```bash
python train.py \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --backbone resnet18 \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4
```

### Full Options

```bash
python train.py \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --backbone resnet18 \
  --hidden_dim 256 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --num_heads 8 \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --lr_schedule cosine \
  --loss_ce_weight 1.0 \
  --loss_bbox_weight 5.0 \
  --loss_giou_weight 2.0 \
  --workers 4
```

**Giải thích tham số:**

- `--data_dir`: Đường dẫn đến thư mục dataset (có cấu trúc train/val với templates và search images)
- `--output_dir`: Thư mục lưu checkpoints và logs
- `--backbone`: Backbone network - `resnet18` (tốt cho localization) hoặc `efficientnet_b0` (nhẹ hơn)
- `--hidden_dim`: Hidden dimension của transformer (256 = standard, 128 = lightweight, 512 = high capacity)
- `--num_encoder_layers`: Số layers của transformer encoder (2-4, thường 3)
- `--num_decoder_layers`: Số layers của transformer decoder (2-4, thường 3)
- `--num_heads`: Số attention heads (8 = standard, 4 = lightweight)
- `--batch_size`: Batch size (16 = standard, 8 = low memory, 32 = high memory)
- `--epochs`: Số epochs để train (50 = standard, 100 = high accuracy)
- `--lr`: Learning rate ban đầu (1e-4 = standard, 5e-5 = stable, 2e-4 = fast)
- `--min_lr`: Learning rate tối thiểu khi decay (dùng với scheduler)
- `--lr_schedule`: Lịch trình giảm LR - `cosine` (smooth decay), `linear` (linear decay), `constant` (không giảm)
- `--loss_ce_weight`: Trọng số cho classification loss (1.0 = default)
- `--loss_bbox_weight`: Trọng số cho L1 bbox loss (5.0 = default, tăng nếu IoU thấp)
- `--loss_giou_weight`: Trọng số cho GIoU loss (2.0 = default, tăng nếu IoU thấp)
- `--workers`: Số processes để load data (4 = standard, 8 = fast, 2 = low memory)

### Resume Training

```bash
python train.py \
  --checkpoint_path ./outputs/last_model_epoch_20.pth \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --backbone resnet18 \
  --epochs 50
```

---

## 🔍 Inference

```bash
python inference.py \
  --checkpoint_path ./outputs/best_model.pth \
  --data_dir /path/to/dataset \
  --split public_test \
  --output_dir ./predictions \
  --backbone resnet18 \
  --confidence_threshold 0.5
```

**Output**: `predictions/submission.json`

---

## ⚙️ Tham số quan trọng

### Model

- `--backbone`: `resnet18` (tốt cho localization) hoặc `efficientnet_b0` (nhẹ hơn)
- `--hidden_dim`: 256 (default)
- `--num_encoder_layers`: 3 (default, 2-4)
- `--num_decoder_layers`: 3 (default, 2-4)
- `--num_heads`: 8 (default)

### Training

- `--batch_size`: 16 (default, giảm nếu OOM)
- `--epochs`: 50 (default)
- `--lr`: 1e-4 (default)
- `--lr_schedule`: `cosine` (default), `linear`, `constant`

### Loss Weights

- `--loss_ce_weight`: 1.0 (classification)
- `--loss_bbox_weight`: 5.0 (L1 loss)
- `--loss_giou_weight`: 2.0 (GIoU loss)

---

## 📊 Model Variants

### Lightweight (~18M params)

```bash
--backbone efficientnet_b0 --num_encoder_layers 2 --num_decoder_layers 2 --batch_size 32
```

### Balanced (~24M params, recommended)

```bash
--backbone resnet18 --num_encoder_layers 3 --num_decoder_layers 3 --batch_size 16
```

### High Accuracy (~30M params)

```bash
--backbone resnet18 --num_encoder_layers 4 --num_decoder_layers 4 --batch_size 12
```

---

## 📈 Expected Results

Sau 50 epochs (Balanced config):

- **Train Loss**: ~1.0-1.5
- **Val Loss**: ~1.2-1.8
- **IoU**: **0.80-0.85**
- **Speed**: ~25 FPS

---

## 🐛 Troubleshooting

### Out of Memory

```bash
--batch_size 8 --workers 2
```

### IoU thấp

```bash
--loss_bbox_weight 7.0 --loss_giou_weight 3.0 --epochs 100
```

### Training không hội tụ

```bash
--lr 5e-5 --lr_schedule linear
```

---

## 📄 License

MIT License
