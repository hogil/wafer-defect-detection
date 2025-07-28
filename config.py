"""
🔧 Enhanced Wafer Defect Detection - 핵심 설정 관리
실제 사용되는 필수 설정들만 포함
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SystemConfig:
    # =====================================================
    # 📂 경로 설정
    # =====================================================
    DATASET_ROOT: str = "test_dataset"  # 테스트 데이터셋 경로
    OUTPUT_DIR: str = "enhanced_pipeline_output"
    
    # 모델 파일명
    DETECTION_MODEL: str = "pretrained_models/yolo11x.pt"  # YOLO 모델 (사전 훈련됨)
    CLASSIFICATION_MODEL_NAME: str = "best_classification_model.pth"  # 분류 모델
    CONVNEXT_MODEL_NAME: str = "convnextv2_base.fcmae_ft_in22k_in1k"  # ConvNeXtV2
    MAPPING_RESULTS_FILE: str = "discovered_mappings.json"  # ROI 매핑 결과
    
    # =====================================================
    # 🎯 핵심 임계값
    # =====================================================
    MAPPING_THRESHOLD: float = 0.6  # 60% 이상 출현시 ROI 매핑 생성
    F1_THRESHOLD: float = 0.75  # F1 < 0.75면 어려운 클래스
    CONFIDENCE_THRESHOLD: float = 0.75  # 이 값 미만이면 ROI 클래스 변경
    ROI_CONFIDENCE_BOOST: float = 0.2   # ROI 변경 성공시 추가 신뢰도
    ROI_CONFIDENCE_PENALTY: float = 0.15  # ROI 변경 실패시 신뢰도 감소
    OBJECT_CONFIDENCE_THRESHOLD: float = 0.6  # YOLO 탐지 최소 신뢰도
    
    # =====================================================
    # 🔍 ROI (Region of Interest) 설정
    # =====================================================
    ROI_METHOD: str = "wafer_center"  # ROI 검출 방법: 'center_square', 'wafer_center', 'brightness'
    YOLO_INPUT_SIZE: int = 1024  # YOLO 모델 입력 크기
    
    # ROI 검출 방법 설명:
    # - center_square: 이미지 중앙 기준 정사각형
    # - wafer_center: HoughCircles/밝기 분석으로 웨이퍼 중심 검출
    # - brightness: 가장 밝은 영역 기준
    
    # =====================================================
    # 🚀 훈련 및 모델 설정
    # =====================================================
    CLASSIFICATION_SIZE: int = 384   # ConvNeXtV2 입력 크기
    DEFAULT_EPOCHS: int = 50
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4


class ConfigManager:
    """🔧 간소화된 설정 관리자"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = SystemConfig()
        
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def update_dataset_path(self, dataset_path: str):
        """데이터셋 경로 업데이트"""
        self.config.DATASET_ROOT = str(Path(dataset_path).resolve())
        print(f"📂 Dataset path updated: {self.config.DATASET_ROOT}")
    
    def set_quick_test_mode(self):
        """빠른 테스트 모드"""
        self.config.DEFAULT_EPOCHS = 2
        self.config.BATCH_SIZE = 16
        print("⚡ Quick test mode activated")
    
    def set_production_mode(self):
        """프로덕션 모드"""
        self.config.DEFAULT_EPOCHS = 100
        self.config.BATCH_SIZE = 32
        print("🚀 Production mode activated")
    
    def get_config(self) -> SystemConfig:
        """현재 설정 반환"""
        return self.config
    
    def print_current_config(self):
        """현재 설정 출력"""
        print("🔧 Current Configuration")
        print("=" * 40)
        
        print(f"📂 Dataset: {self.config.DATASET_ROOT or 'Not set'}")
        print(f"📁 Output: {self.config.OUTPUT_DIR}")
        print(f"🎯 YOLO Model: {self.config.DETECTION_MODEL}")
        print(f"🤖 ConvNeXt: {self.config.CONVNEXT_MODEL_NAME}")
        
        print(f"\n📊 Thresholds:")
        print(f"  Mapping: {self.config.MAPPING_THRESHOLD:.1%}")
        print(f"  F1: {self.config.F1_THRESHOLD:.2f}")
        print(f"  Confidence: {self.config.CONFIDENCE_THRESHOLD:.2f}")
        print(f"  ROI Boost/Penalty: +{self.config.ROI_CONFIDENCE_BOOST:.2f} / -{self.config.ROI_CONFIDENCE_PENALTY:.2f}")
        
        print(f"\n🚀 Training:")
        print(f"  Epochs: {self.config.DEFAULT_EPOCHS}")
        print(f"  Batch Size: {self.config.BATCH_SIZE}")
        print(f"  Image Size: {self.config.CLASSIFICATION_SIZE}×{self.config.CLASSIFICATION_SIZE}")
    
    def save_to_file(self, filepath: str):
        """설정을 파일로 저장"""
        import json
        
        config_dict = {
            'dataset_root': self.config.DATASET_ROOT,
            'output_dir': self.config.OUTPUT_DIR,
            'mapping_threshold': self.config.MAPPING_THRESHOLD,
            'f1_threshold': self.config.F1_THRESHOLD,
            'confidence_threshold': self.config.CONFIDENCE_THRESHOLD,
            'roi_confidence_boost': self.config.ROI_CONFIDENCE_BOOST,
            'roi_confidence_penalty': self.config.ROI_CONFIDENCE_PENALTY,
            'classification_size': self.config.CLASSIFICATION_SIZE,
            'default_epochs': self.config.DEFAULT_EPOCHS,
            'batch_size': self.config.BATCH_SIZE,
            'learning_rate': self.config.LEARNING_RATE
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Configuration saved to: {filepath}")
    
    def load_from_file(self, filepath: str):
        """파일에서 설정 로드"""
        import json
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 설정 업데이트
            for key, value in config_dict.items():
                if hasattr(self.config, key.upper()):
                    setattr(self.config, key.upper(), value)
            
            print(f"📁 Configuration loaded from: {filepath}")
            
        except Exception as e:
            print(f"⚠️ Failed to load config: {e}")


# =====================================================
# 🎯 간편한 설정 프리셋들
# =====================================================

def get_quick_test_config(dataset_path: str) -> ConfigManager:
    """빠른 테스트용 설정"""
    config_manager = ConfigManager()
    config_manager.update_dataset_path(dataset_path)
    config_manager.set_quick_test_mode()
    return config_manager


def get_production_config(dataset_path: str) -> ConfigManager:
    """프로덕션용 설정"""
    config_manager = ConfigManager()
    config_manager.update_dataset_path(dataset_path)
    config_manager.set_production_mode()
    return config_manager


if __name__ == "__main__":
    print("🔧 Enhanced Wafer Defect Detection - Configuration")
    print("=" * 50)
    
    config_manager = ConfigManager()
    config_manager.print_current_config()
    
    print(f"\n💡 Usage:")
    print(f"  from config import get_production_config")
    print(f"  config = get_production_config('dataset_path')")
