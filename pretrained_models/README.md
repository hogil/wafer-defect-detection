# Download Pretrained Models

This directory should contain the following pretrained models:

## Required Models

### 1. ConvNeXtV2 Classification Model
- **File**: `convnextv2_base.fcmae_ft_in22k_in1k_pretrained.pth`
- **Description**: Fine-tuned ConvNeXtV2 model for wafer defect classification
- **Size**: ~350MB
- **Download**: You need to train this model on your wafer dataset or obtain from your training pipeline

### 2. YOLO Detection Model  
- **File**: `yolo11x.pt`
- **Description**: YOLO11x model for object detection in ROI regions
- **Size**: ~130MB
- **Download**: 
  ```bash
  # Download official YOLO11x weights
  wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt
  ```

## Setup Instructions

1. **Download YOLO11x**:
   ```bash
   cd pretrained_models/
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt
   ```

2. **Add your trained ConvNeXtV2 model**:
   - Train ConvNeXtV2 on your wafer dataset
   - Save the model as `convnextv2_base.fcmae_ft_in22k_in1k_pretrained.pth`
   - Place it in this directory

## File Structure
```
pretrained_models/
├── README.md                                           # This file
├── convnextv2_base.fcmae_ft_in22k_in1k_pretrained.pth # Your trained model
└── yolo11x.pt                                          # YOLO11x weights
```

## Notes
- Model files are excluded from git tracking due to their large size
- Make sure to download/prepare these files before running the detection system
- The system will fail with clear error messages if models are missing
