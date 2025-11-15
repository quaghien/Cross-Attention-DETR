# CA-DETR: Cross-Attention DETR for Reference-Based Detection

Multi-Template Cross-Attention DETR cho reference-based object detection trong drone surveillance.

---

## 🏗️ Kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGES                             │
│                                                                  │
│  Template 1 (640×640)                                            │
│  Template 2 (640×640)          Search (640×640)                  │
│  Template 3 (640×640)                │                           │
│         │                             │                          │
└─────────┼─────────────────────────────┼──────────────────────────┘
          │                             │
          ▼                             ▼
    ┌─────────┐                   ┌─────────┐
    │Backbone │                   │Backbone │
    │ResNet-18│                   │(shared) │
    │  11.3M  │                   │         │
    └────┬────┘                   └────┬────┘
         │                             │
         │ 3× (B, 256, 20, 20)        │ (B, 256, 20, 20)
         │                             │
         ▼                             ▼
    ┌─────────────┐             ┌─────────────┐
    │ Positional  │             │ Positional  │
    │  Encoding   │             │  Encoding   │
    │  (Sine)     │             │  (Sine)     │
    └──────┬──────┘             └──────┬──────┘
           │                           │
           │ Flatten: 3×(B,400,256)   │ Flatten: (B,400,256)
           │                           │
           ▼                           ▼
    ┌─────────────┐             ┌─────────────┐
    │ Transformer │             │ Transformer │
    │  Encoder    │             │  Encoder    │
    │  6 layers   │             │  6 layers   │
    │   1.32M     │             │   (shared)  │
    └──────┬──────┘             └──────┬──────┘
           │                           │
           │                           │
           ▼                           X (not used)
    ┌─────────────┐
    │  Concat ALL │
    │  Templates  │
    │ (B,1200,256)│ ← 3 templates × 400 tokens
    └──────┬──────┘
           │
           │         ┌──────────────────┐
           │         │ Learnable Query  │
           │         │   (B, 5, 256)    │ ← 5 queries
           │         └────────┬─────────┘
           │                  │
           └──────────────────┼──────────┐
                              │          │
                              ▼          │
                       ┌──────────────┐  │
                       │ Transformer  │  │
                       │   Decoder    │◄─┘
                       │  6 layers    │
                       │   1.58M      │
                       │              │
                       │ Self-Attn +  │
                       │ Cross-Attn   │ ← Attends to 1200 tokens
                       └──────┬───────┘
                              │
                              │ (B, 5, 256)
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
         │   (B,5,1)   │            │    w, h]    │
         │             │            │   (B,5,4)   │
         └─────────────┘            └─────────────┘
                │                          │
                └──────────┬───────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Hungarian   │
                    │  Matching   │
                    │  + NMS      │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Final BBox  │
                    │  Prediction │
                    └─────────────┘
```

**Total: 14.4M params (FP32, ~58MB)**

---

## 🚀 Training

### Quick Start (Optimized Config)

```bash
python train.py \
  --data_dir refdet/retrieval_dataset_flat_zoomed/ \
  --output_dir ./outputs \
  --backbone resnet18 \
  --num_encoder_layers 6 \
  --num_decoder_layers 6 \
  --num_queries 5 \
  --num_heads 16 \
  --dim_feedforward 2048 \
  --loss_bbox_weight 7.0 \
  --loss_giou_weight 3.0 \
  --epochs 100 \
  --augment_prob 0.5 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --lr_schedule cosine \
  --batch_size 32 \
  --workers 4
```

### Resume Training

```bash
python train.py \
  --checkpoint_path ./outputs/last_model_epoch_50.pth \
  --data_dir refdet/retrieval_dataset_flat_zoomed/ \
  --output_dir ./outputs \
  --backbone resnet18 \
  --epochs 100
```

---

## 🔍 Inference

```bash
python inference.py \
  --checkpoint_path ./outputs/best_model.pth \
  --data_dir refdet/retrieval_dataset_flat_zoomed/ \
  --split public_test \
  --output_dir ./predictions \
  --backbone resnet18 \
  --confidence_threshold 0.5
```

**Output**: `predictions/submission.json`

---

## 📊 Performance

| Config | Params | IoU | Speed | Memory |
|--------|--------|-----|-------|--------|
| Lightweight | 6.4M | 0.75-0.80 | 40 FPS | 20GB |
| Balanced | 13.4M | 0.80-0.85 | 25 FPS | 40GB |
| **Optimized** | **14.4M** | **0.90-0.95** | **15 FPS** | **60GB** |

---

## 📝 Technical Details

Xem [TECHNIQUES.md](TECHNIQUES.md) để biết chi tiết về các kỹ thuật và paper references.
