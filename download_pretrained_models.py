#!/usr/bin/env python3
"""
🌐 사전 훈련된 모델 다운로드 스크립트
폐쇄망 환경에서 사용하기 위해 필요한 모델들을 미리 다운로드합니다.
"""

import os
import torch
import timm
from pathlib import Path
from config import ConfigManager

def download_convnext_pretrained():
    """ConvNeXtV2 사전 훈련된 모델 다운로드"""
    print("🤖 ConvNeXtV2 사전 훈련된 모델 다운로드 중...")
    
    config = ConfigManager().get_config()
    model_name = config.CONVNEXT_MODEL_NAME
    
    # ImageFolder에서 클래스 수 자동 계산
    from torchvision import datasets
    if config.DATASET_ROOT and Path(config.DATASET_ROOT).exists():
        temp_dataset = datasets.ImageFolder(config.DATASET_ROOT)
        num_classes = len(temp_dataset.classes)
        print(f"📊 Dataset classes detected: {num_classes} ({temp_dataset.classes})")
    else:
        num_classes = 1000  # 기본값 (ImageNet)
        print(f"⚠️ Dataset not found, using default: {num_classes} classes")
    
    try:
        # pretrained=True로 모델 생성하여 가중치 다운로드
        model = timm.create_model(
            model_name,
            pretrained=True,  # 사전 훈련된 가중치 다운로드
            num_classes=num_classes  # 동적으로 계산된 클래스 수
        )
        
        # 모델 저장 디렉토리 생성
        save_dir = Path("pretrained_models")
        save_dir.mkdir(exist_ok=True)
        
        # 모델 가중치 저장
        model_path = save_dir / f"{model_name}_pretrained.pth"
        torch.save(model.state_dict(), model_path)
        
        print(f"✅ ConvNeXtV2 모델 저장 완료: {model_path}")
        print(f"  - 모델명: {model_name}")
        print(f"  - 파일 크기: {model_path.stat().st_size / (1024*1024):.1f} MB")
        
        return model_path
        
    except Exception as e:
        print(f"❌ ConvNeXtV2 다운로드 실패: {e}")
        return None

def download_yolo_pretrained():
    """YOLO 사전 훈련된 모델 다운로드"""
    print("\n🎯 YOLO 사전 훈련된 모델 다운로드 중...")
    
    config = ConfigManager().get_config()
    model_name = config.DETECTION_MODEL
    
    try:
        from ultralytics import YOLO
        
        # YOLO 모델 로드 (자동으로 다운로드됨)
        model = YOLO(model_name)
        
        # 현재 디렉토리에 다운로드된 파일 확인
        current_model_path = Path(model_name)
        
        # 저장 디렉토리로 이동
        save_dir = Path("pretrained_models")
        save_dir.mkdir(exist_ok=True)
        target_path = save_dir / model_name
        
        if current_model_path.exists():
            # pretrained_models 폴더로 복사
            import shutil
            shutil.copy2(current_model_path, target_path)
            print(f"✅ YOLO 모델 저장 완료: {target_path}")
            print(f"  - 모델명: {model_name}")
            print(f"  - 파일 크기: {target_path.stat().st_size / (1024*1024):.1f} MB")
            return target_path
        else:
            print(f"⚠️ YOLO 모델 파일을 찾을 수 없습니다: {model_name}")
            return None
            
    except Exception as e:
        print(f"❌ YOLO 다운로드 실패: {e}")
        return None

def create_offline_usage_guide():
    """폐쇄망 사용 가이드 생성"""
    guide_content = """# 🌐 폐쇄망 환경 사용 가이드

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
"""
    
    guide_path = Path("OFFLINE_USAGE_GUIDE.md")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"\n📝 폐쇄망 사용 가이드 생성: {guide_path}")

def main():
    """메인 함수"""
    print("🌐 사전 훈련된 모델 다운로드 시작")
    print("=" * 50)
    
    # 다운로드 결과 저장
    results = {}
    
    # ConvNeXtV2 다운로드
    convnext_path = download_convnext_pretrained()
    results['convnext'] = convnext_path
    
    # YOLO 다운로드
    yolo_path = download_yolo_pretrained()
    results['yolo'] = yolo_path
    
    # 결과 요약
    print("\n📊 다운로드 결과 요약")
    print("=" * 30)
    
    success_count = 0
    total_size = 0
    
    for model_type, path in results.items():
        if path and path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            print(f"✅ {model_type.upper()}: {path} ({size_mb:.1f} MB)")
            success_count += 1
            total_size += size_mb
        else:
            print(f"❌ {model_type.upper()}: 다운로드 실패")
    
    print(f"\n🎯 성공: {success_count}/2 모델")
    print(f"💾 총 크기: {total_size:.1f} MB")
    
    # 사용 가이드 생성
    create_offline_usage_guide()
    
    print("\n🎉 모든 작업 완료!")
    print("폐쇄망 환경에서 사용하려면 OFFLINE_USAGE_GUIDE.md를 참고하세요.")

if __name__ == "__main__":
    main() 