"""Default configuration for CA-DETR."""

# Model
MODEL = dict(
    backbone='resnet18',  # 'resnet18' or 'efficientnet_b0'
    num_queries=1,
    hidden_dim=256,
    num_encoder_layers=3,
    num_decoder_layers=3,
    num_heads=8,
    dim_feedforward=1024,
    dropout=0.1,
    pretrained_backbone=True
)

# Loss
LOSS = dict(
    loss_ce_weight=1.0,
    loss_bbox_weight=5.0,
    loss_giou_weight=2.0,
    focal_alpha=0.25,
    focal_gamma=2.0
)

# Training
TRAIN = dict(
    batch_size=16,
    epochs=50,
    lr=1e-4,
    min_lr=1e-6,
    lr_schedule='cosine',  # 'constant', 'cosine', 'linear'
    weight_decay=1e-4,
    augment_prob=0.75,
    workers=4,
    save_every=5
)

# Data
DATA = dict(
    img_size=640,
    data_dir='/path/to/dataset',
    output_dir='./outputs'
)

# Inference
INFERENCE = dict(
    confidence_threshold=0.5,
    split='public_test'
)

