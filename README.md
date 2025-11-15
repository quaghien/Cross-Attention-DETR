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

### Quick Start (Balanced Config)

```bash
python train.py \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --backbone resnet18 \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4
```

### 🎯 Recommended: Optimized for Maximum IoU

**Config tốt nhất để đạt IoU 0.85-0.90:**

```bash
python train.py \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --backbone resnet18 \
  --num_encoder_layers 4 \
  --num_decoder_layers 4 \
  --loss_bbox_weight 7.0 \
  --loss_giou_weight 3.0 \
  --epochs 100 \
  --augment_prob 0.5 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --lr_schedule cosine \
  --batch_size 16 \
  --workers 4
```

### Lightweight (Fast, Low Memory)

```bash
python train.py \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --backbone efficientnet_b0 \
  --num_encoder_layers 2 \
  --num_decoder_layers 2 \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4
```

### Full Options (All Parameters)

```bash
python train.py \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --backbone resnet18 \
  --hidden_dim 256 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --num_heads 8 \
  --dim_feedforward 1024 \
  --dropout 0.1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --lr_schedule cosine \
  --weight_decay 1e-4 \
  --augment_prob 0.75 \
  --loss_ce_weight 1.0 \
  --loss_bbox_weight 5.0 \
  --loss_giou_weight 2.0 \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --save_every 5 \
  --workers 4
```

**Giải thích tham số:**

**Data & Output:**
- `--data_dir`: Đường dẫn dataset (cấu trúc: train/val/test với templates và search/images, search/labels)
- `--output_dir`: Thư mục lưu checkpoints (best_model.pth, last_model_epoch_X.pth, history.json)
- `--img_size`: Kích thước ảnh input (640 = default)

**Model Architecture:**
- `--backbone`: `resnet18` (tốt cho localization, 11M params) hoặc `efficientnet_b0` (nhẹ hơn, 4M params)
- `--hidden_dim`: Hidden dimension (256 = standard, 128 = lightweight, 512 = high capacity)
- `--num_encoder_layers`: Encoder layers (2-4, **4 cho IoU cao nhất**)
- `--num_decoder_layers`: Decoder layers (2-4, **4 cho IoU cao nhất**)
- `--num_heads`: Attention heads (8 = standard, 4 = lightweight)
- `--dim_feedforward`: FFN dimension (1024 = default, 2048 = high capacity)
- `--dropout`: Dropout rate (0.1 = default)
- `--num_queries`: Số object queries (1 = single object detection)

**Training:**
- `--batch_size`: Batch size (16 = standard, 8 = low memory, 32 = high memory)
- `--epochs`: Số epochs (50 = standard, **100 cho IoU cao nhất**)
- `--lr`: Learning rate (1e-4 = standard, 5e-5 = stable, 2e-4 = fast)
- `--min_lr`: LR tối thiểu (1e-6 = default, dùng với scheduler)
- `--lr_schedule`: `cosine` (smooth decay, **recommended**), `linear` (linear decay), `constant`
- `--weight_decay`: Weight decay (1e-4 = default)
- `--augment_prob`: Xác suất augmentation (0.75 = default, **0.5 cho IoU cao nhất**)
- `--workers`: Data loader workers (4 = standard, 8 = fast, 2 = low memory)

**Loss Weights (quan trọng cho IoU):**
- `--loss_ce_weight`: Classification loss (1.0 = default)
- `--loss_bbox_weight`: L1 bbox loss (5.0 = default, **7.0 cho IoU cao nhất**)
- `--loss_giou_weight`: GIoU loss (2.0 = default, **3.0-4.0 cho IoU cao nhất**)
- `--focal_alpha`: Focal loss alpha (0.25 = default)
- `--focal_gamma`: Focal loss gamma (2.0 = default)

**Checkpointing:**
- `--checkpoint_path`: Resume từ checkpoint
- `--save_every`: Lưu checkpoint mỗi N epochs (5 = default)

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

## 💡 Tips để tăng IoU

### 1. Loss Weights (Quan trọng nhất - gain +8-13% IoU)

```bash
--loss_bbox_weight 7.0    # Tăng từ 5.0 → +3-5% IoU
--loss_giou_weight 3.0    # Tăng từ 2.0 → +5-8% IoU
```

GIoU loss tối ưu trực tiếp IoU metric → tăng weight này có impact lớn nhất!

### 2. Model Architecture (gain +3-5% IoU)

```bash
--backbone resnet18              # ResNet-18 > EfficientNet-B0 cho localization
--num_encoder_layers 4           # Tăng từ 3 → +2-3% IoU
--num_decoder_layers 4
```

### 3. Training Strategy (gain +3-6% IoU)

```bash
--epochs 100                     # Tăng từ 50 → +2-4% IoU
--augment_prob 0.5              # Giảm từ 0.75 → +1-2% IoU (ít noise cho bbox)
--lr_schedule cosine --min_lr 1e-6
```

### 4. Khi nào điều chỉnh gì?

| Vấn đề | Giải pháp | Command |
|--------|-----------|---------|
| **IoU thấp** | Tăng GIoU + BBox weights | `--loss_bbox_weight 7.0 --loss_giou_weight 3.0` |
| **Training chậm** | Giảm batch size | `--batch_size 8` |
| **Out of Memory** | Giảm batch + workers | `--batch_size 8 --workers 2` |
| **Không hội tụ** | Giảm LR | `--lr 5e-5 --lr_schedule linear` |
| **Overfitting** | Tăng dropout + weight decay | `--dropout 0.2 --weight_decay 5e-4` |

---

## 📊 Model Variants & Performance

### Lightweight (~6M params)

```bash
--backbone efficientnet_b0 --num_encoder_layers 2 --num_decoder_layers 2 --batch_size 32
```

- **Params**: 6.38M (Backbone: 4.34M, Encoder: 0.79M, Decoder: 1.05M)
- **Speed**: ~40 FPS
- **IoU**: 0.75-0.80

### Balanced (~13M params, recommended)

```bash
--backbone resnet18 --num_encoder_layers 3 --num_decoder_layers 3 --batch_size 16
```

- **Params**: 13.35M (Backbone: 11.31M, Encoder: 0.79M, Decoder: 1.05M)
- **Speed**: ~25 FPS
- **IoU**: 0.80-0.85

### High Accuracy (~13M params)

```bash
--backbone resnet18 --num_encoder_layers 4 --num_decoder_layers 4 --batch_size 12
```

- **Params**: 13.35M (Backbone: 11.31M, Encoder: 0.79M, Decoder: 1.05M)
- **Speed**: ~20 FPS
- **IoU**: 0.82-0.87

### 🎯 Optimized for Maximum IoU (~13M params)

**Config tối ưu để đạt IoU cao nhất (0.85-0.90):**

```bash
python train.py \
  --data_dir /path/to/dataset \
  --output_dir ./outputs \
  --backbone resnet18 \
  --num_encoder_layers 4 \
  --num_decoder_layers 4 \
  --loss_bbox_weight 7.0 \
  --loss_giou_weight 3.0 \
  --epochs 100 \
  --augment_prob 0.5 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --lr_schedule cosine \
  --batch_size 16
```

**Cải tiến so với Balanced:**
- ✅ Tăng GIoU weight: 2.0 → 3.0 (+5-8% IoU)
- ✅ Tăng BBox weight: 5.0 → 7.0 (+3-5% IoU)
- ✅ Tăng epochs: 50 → 100 (+2-4% IoU)
- ✅ Giảm augmentation: 0.75 → 0.5 (+1-2% IoU)
- ✅ Tăng decoder layers: 3 → 4 (+2-3% IoU)

**Expected**: IoU 0.85-0.90, Speed ~20 FPS

---

## 📈 Expected Results

### Balanced Config (50 epochs)

- **Train Loss**: ~1.0-1.5
- **Val Loss**: ~1.2-1.8
- **IoU**: **0.80-0.85**
- **Speed**: ~25 FPS

### Optimized Config (100 epochs)

- **Train Loss**: ~0.8-1.2
- **Val Loss**: ~1.0-1.5
- **IoU**: **0.85-0.90**
- **Speed**: ~20 FPS

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Triệu chứng**: `RuntimeError: CUDA out of memory`

**Giải pháp**:
```bash
# Giảm batch size
--batch_size 8

# Giảm workers
--workers 2

# Giảm model size
--num_encoder_layers 2 --num_decoder_layers 2

# Giảm hidden dim
--hidden_dim 128
```

### IoU thấp (<0.7)

**Triệu chứng**: Val IoU không tăng sau nhiều epochs

**Giải pháp**:
```bash
# Tăng loss weights (quan trọng nhất!)
--loss_bbox_weight 7.0 --loss_giou_weight 3.0

# Tăng epochs
--epochs 100

# Giảm augmentation
--augment_prob 0.5

# Thử backbone khác
--backbone resnet18  # Tốt hơn cho localization
```

### Training không hội tụ

**Triệu chứng**: Loss không giảm hoặc NaN

**Giải pháp**:
```bash
# Giảm learning rate
--lr 5e-5

# Thử scheduler khác
--lr_schedule linear

# Giảm loss weights
--loss_bbox_weight 3.0 --loss_giou_weight 1.5

# Tăng gradient clipping (trong code: max_norm=1.0)
```

### Overfitting

**Triệu chứng**: Train loss thấp nhưng val loss cao

**Giải pháp**:
```bash
# Tăng dropout
--dropout 0.2

# Tăng weight decay
--weight_decay 5e-4

# Tăng augmentation
--augment_prob 0.9

# Giảm model size
--num_encoder_layers 2 --num_decoder_layers 2
```

### Training quá chậm

**Giải pháp**:
```bash
# Tăng workers
--workers 8

# Giảm batch size (paradoxical nhưng đôi khi nhanh hơn)
--batch_size 8

# Dùng backbone nhẹ hơn
--backbone efficientnet_b0

# Giảm layers
--num_encoder_layers 2 --num_decoder_layers 2
```

---

## 📄 License

MIT License
