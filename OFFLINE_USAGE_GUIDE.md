# 🌐 폐쇄망 환경 사용 가이드

## 📁 다운로드된 파일들
- `pretrained_models/convnextv2_base.fcmae_ft_in22k_in1k_pretrained.pth`: ConvNeXtV2 사전 훈련 가중치
- `pretrained_models/yolo11x.pt`: YOLO11x 사전 훈련 모델

## 🔧 폐쇄망에서 사용 방법

### 1️⃣ 파일 복사
```bash
# pretrained_models 폴더를 폐쇄망 서버로 복사
scp -r pretrained_models/ user@server:/path/to/wafer-defect-detection/
```

### 2️⃣ 코드 수정
폐쇄망에서 실행하기 전에 다음 파일들을 수정하세요:

#### train.py 수정:
```python
# 기존 코드
self.model = timm.create_model(
    self.config.CONVNEXT_MODEL_NAME,
    pretrained=False,  # False로 설정
    num_classes=self.num_classes
)

# 사전 훈련 가중치 로드 추가
pretrained_path = Path("pretrained_models") / f"{self.config.CONVNEXT_MODEL_NAME}_pretrained.pth"
if pretrained_path.exists():
    pretrained_weights = torch.load(pretrained_path, map_location=self.device)
    
    # 분류층 제외하고 로드
    model_dict = self.model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() 
                      if k in model_dict and 'head' not in k}
    model_dict.update(pretrained_dict)
    self.model.load_state_dict(model_dict)
    print(f"✅ 사전 훈련 가중치 로드: {pretrained_path}")
```

### 3️⃣ YOLO 모델 경로 설정
config.py에서 YOLO 모델 경로를 로컬 파일로 변경:
```python
DETECTION_MODEL: str = "pretrained_models/yolo11x.pt"
```

## 📝 주의사항
- 인터넷 연결 없이 사전 훈련된 가중치를 사용할 수 있습니다
- 첫 실행 시 약간의 성능 향상이 있을 수 있습니다
- 모든 필요한 Python 패키지도 미리 설치되어 있어야 합니다
