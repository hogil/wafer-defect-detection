#!/usr/bin/env python3
"""
사전 훈련된 가중치 로드 상태 테스트
"""

import torch
import timm
from pathlib import Path
from config import ConfigManager

def test_pretrained_loading():
    """사전 훈련된 가중치 로드 테스트"""
    
    print("🧪 Testing pretrained weight loading...")
    
    # 설정 로드
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 사전 훈련된 가중치 로드
    pretrained_path = Path(config.CONVNEXT_PRETRAINED_MODEL)
    pretrained_weights = torch.load(pretrained_path, map_location='cpu')
    
    # model. prefix 제거
    clean_pretrained_weights = {}
    for key, value in pretrained_weights.items():
        if key.startswith('model.'):
            new_key = key[6:]  # "model." 제거
            clean_pretrained_weights[new_key] = value
        else:
            clean_pretrained_weights[key] = value
    
    # 가중치에서 클래스 수 추출
    weight_num_classes = None
    for key in clean_pretrained_weights.keys():
        if key in ['head.fc.weight', 'classifier.weight']:
            weight_num_classes = clean_pretrained_weights[key].shape[0]
            break
    
    # 모델 생성 (가중치의 클래스 수로)
    print(f"🤖 Creating model: {config.CONVNEXT_MODEL_NAME}")
    model = timm.create_model(
        config.CONVNEXT_MODEL_NAME,
        pretrained=False,
        num_classes=weight_num_classes
    )
    
    # 분류 헤드를 테스트용 클래스 수로 교체
    if weight_num_classes != 4:
        # 기존 분류 헤드 제거
        if hasattr(model, 'head'):
            delattr(model, 'head')
        if hasattr(model, 'classifier'):
            delattr(model, 'classifier')
        
        # 새로운 분류 헤드 추가
        if hasattr(model, 'head'):
            model.head = torch.nn.Linear(model.head.in_features, 4)
        elif hasattr(model, 'classifier'):
            model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
        else:
            # ConvNeXtV2의 경우 head.fc를 직접 수정
            if hasattr(model, 'head') and hasattr(model.head, 'fc'):
                model.head.fc = torch.nn.Linear(model.head.fc.in_features, 4)
    
    # 가중치 로드 (분류 헤드 제외)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in clean_pretrained_weights.items() 
                      if k in model_dict and 'head' not in k and 'classifier' not in k}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    
    print(f"✅ Model created:")
    print(f"  - Architecture: {config.CONVNEXT_MODEL_NAME}")
    print(f"  - Weight classes: {weight_num_classes}")
    print(f"  - Test classes: 4")
    print(f"  - Image size: {config.CLASSIFICATION_SIZE}")
    print(f"  - Pretrained: Yes ({len(pretrained_dict)} layers)")
    
    return True

if __name__ == "__main__":
    test_pretrained_loading() 