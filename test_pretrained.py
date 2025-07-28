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
    
    # 모델 생성 (pretrained=False)
    print(f"🤖 Creating model: {config.CONVNEXT_MODEL_NAME}")
    model = timm.create_model(
        config.CONVNEXT_MODEL_NAME,
        pretrained=False,  # 별도 가중치 로드
        num_classes=4
    )
    
    # 사전 훈련된 가중치 로드
    pretrained_path = Path(config.CONVNEXT_PRETRAINED_MODEL)
    weights_loaded = False
    
    if pretrained_path.exists():
        print(f"🔄 Loading pretrained weights from: {pretrained_path}")
        pretrained_weights = torch.load(pretrained_path, map_location='cpu')
        
        # model. prefix 제거 (있을 경우)
        clean_pretrained_weights = {}
        for key, value in pretrained_weights.items():
            if key.startswith('model.'):
                new_key = key[6:]  # "model." 제거
                clean_pretrained_weights[new_key] = value
            else:
                clean_pretrained_weights[key] = value
        
        # head와 classifier 레이어 제외 (클래스 수가 다름)
        filtered_weights = {}
        for key, value in clean_pretrained_weights.items():
            if not key.startswith('head.') and not key.startswith('classifier.'):
                filtered_weights[key] = value
        
        # 가중치 로드 (strict=False로 head/classifier 제외)
        model.load_state_dict(filtered_weights, strict=False)
        print(f"✅ Pretrained weights loaded: {len(clean_pretrained_weights)} layers")
        weights_loaded = True
    else:
        print(f"⚠️ Pretrained weights not found: {pretrained_path}")
    
    print(f"✅ Model created:")
    print(f"  - Architecture: {config.CONVNEXT_MODEL_NAME}")
    print(f"  - Classes: 4")
    print(f"  - Image size: {config.CLASSIFICATION_SIZE}")
    print(f"  - Pretrained: {'Yes' if weights_loaded else 'No'}")
    
    return weights_loaded

if __name__ == "__main__":
    test_pretrained_loading() 