#!/usr/bin/env python3
"""
🚀 Quick Start Example for Wafer Defect Detection
빠른 시작을 위한 예제 스크립트
"""

import os
import sys
from pathlib import Path
import logging

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from wafer_detector import WaferDetector, WaferDetectorError
from utils import setup_directories, log_system_info, create_sample_config

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_full_pipeline():
    """전체 파이프라인 실행 예제"""
    print("=" * 60)
    print("🎯 웨이퍼 결함 검출 시스템 - 전체 파이프라인 예제")
    print("=" * 60)
    
    # 시스템 정보 출력
    log_system_info()
    
    # 설정
    config = {
        'DATASET_ROOT': 'path/to/your/dataset',  # 실제 데이터셋 경로로 변경 필요
        'MODEL_PATH': 'pretrained_models/convnextv2_base.fcmae_ft_in22k_in1k_pretrained.pth',
        'YOLO_MODEL': 'pretrained_models/yolo11x.pt',
        'CLASSIFICATION_SIZE': 384,
        'YOLO_SIZE': 1024,
        'F1_THRESHOLD': 0.8,
        'CONFIDENCE_THRESHOLD': 0.7,
        'MAPPING_THRESHOLD': 0.3,
        'OUTPUT_DIR': 'outputs'
    }
    
    print("\n📋 사용 설정:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # 검출기 초기화
        print("\n🚀 검출기 초기화 중...")
        detector = WaferDetector(config)
        
        # 모델 로드 (실제 파일이 있어야 함)
        print("📦 모델 로드 중...")
        # detector.load_models(config['MODEL_PATH'], config['YOLO_MODEL'])
        
        # 데이터셋이 있는 경우 전체 파이프라인 실행
        if Path(config['DATASET_ROOT']).exists():
            print("📊 성능 분석 중...")
            # detector.analyze_performance(config['DATASET_ROOT'])
            
            print("🧠 ROI 패턴 학습 중...")
            # detector.learn_roi_patterns(config['DATASET_ROOT'])
            
            print("🎯 클래스-객체 매핑 생성 중...")
            # detector.create_mapping(config['DATASET_ROOT'])
            
            print("💾 결과 저장 중...")
            # detector.save_results(config['OUTPUT_DIR'])
            
            print("✅ 전체 파이프라인 완료!")
        else:
            print(f"⚠️ 데이터셋을 찾을 수 없습니다: {config['DATASET_ROOT']}")
            print("실제 데이터셋 경로를 설정해주세요.")
            
    except WaferDetectorError as e:
        print(f"❌ 검출기 오류: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")


def example_prediction():
    """예측 예제"""
    print("\n" + "=" * 60)
    print("🔍 단일 이미지 예측 예제")
    print("=" * 60)
    
    config = {
        'DATASET_ROOT': 'path/to/your/dataset',
        'MODEL_PATH': 'pretrained_models/convnextv2_base.fcmae_ft_in22k_in1k_pretrained.pth',
        'YOLO_MODEL': 'pretrained_models/yolo11x.pt',
        'CLASSIFICATION_SIZE': 384,
        'YOLO_SIZE': 1024,
        'F1_THRESHOLD': 0.8,
        'CONFIDENCE_THRESHOLD': 0.7,
        'MAPPING_THRESHOLD': 0.3,
        'OUTPUT_DIR': 'outputs'
    }
    
    try:
        # 검출기 초기화
        detector = WaferDetector(config)
        
        # 예측할 이미지 경로 (실제 파일로 변경 필요)
        image_path = "path/to/your/test/image.jpg"
        
        if Path(image_path).exists():
            print(f"📸 이미지 예측 중: {image_path}")
            
            # 클래스 로드 (필요한 경우)
            if Path(config['DATASET_ROOT']).exists():
                detector.load_classes(config['DATASET_ROOT'])
            
            # 예측 수행
            # result = detector.predict_image(image_path)
            
            # 결과 출력
            print("🎯 예측 결과:")
            # print(f"   예측 클래스: {result['predicted_class']}")
            # print(f"   신뢰도: {result['confidence']:.3f}")
            # print(f"   방법: {result['method']}")
            
        else:
            print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {image_path}")
            print("실제 이미지 경로를 설정해주세요.")
            
    except Exception as e:
        print(f"❌ 예측 오류: {e}")


def example_configuration():
    """설정 예제"""
    print("\n" + "=" * 60)
    print("⚙️ 설정 사용 예제")
    print("=" * 60)
    
    # 샘플 설정 생성
    sample_configs = create_sample_config()
    
    print("📄 사용 가능한 설정 모드:")
    for mode, settings in sample_configs.items():
        if isinstance(settings, dict):
            print(f"\n🔧 {mode}:")
            for key, value in settings.items():
                print(f"   {key}: {value}")
    
    # 설정 파일 저장 예제
    import json
    config_path = Path("example_config.json")
    
    with open(config_path, 'w') as f:
        json.dump(sample_configs, f, indent=2)
    
    print(f"\n💾 설정 파일 저장됨: {config_path}")
    print("이 파일을 수정하여 사용자 정의 설정을 만들 수 있습니다.")


def example_directory_setup():
    """디렉토리 설정 예제"""
    print("\n" + "=" * 60)
    print("📁 출력 디렉토리 설정 예제")
    print("=" * 60)
    
    output_dir = Path("example_outputs")
    
    # 디렉토리 구조 생성
    dirs = setup_directories(output_dir)
    
    print("✅ 다음 디렉토리가 생성되었습니다:")
    for name, path in dirs.items():
        print(f"   {name}: {path}")
    
    print("\n이 구조를 사용하여 결과를 체계적으로 저장할 수 있습니다.")


def show_usage_examples():
    """사용법 예제 출력"""
    print("\n" + "=" * 60)
    print("📚 명령어 사용법 예제")
    print("=" * 60)
    
    examples = [
        ("전체 파이프라인 실행", "python main.py /path/to/dataset"),
        ("단일 이미지 예측", "python main.py --predict /path/to/image.jpg"),
        ("폴더 배치 예측", "python main.py --predict /path/to/images/"),
        ("사용자 정의 설정", "python main.py /path/to/dataset --config custom.json"),
        ("출력 디렉토리 지정", "python main.py /path/to/dataset --output-dir results/"),
        ("상세 로그", "python main.py /path/to/dataset --verbose"),
    ]
    
    for description, command in examples:
        print(f"\n🔸 {description}:")
        print(f"   {command}")
    
    print("\n" + "=" * 60)
    print("💡 팁:")
    print("   - 먼저 데이터셋으로 학습을 실행하여 ROI 패턴과 매핑을 생성하세요")
    print("   - 그 다음 예측 모드에서 학습된 패턴을 활용할 수 있습니다")
    print("   - config.json 파일을 수정하여 임계값을 조정할 수 있습니다")
    print("=" * 60)


def main():
    """메인 함수"""
    print("🎯 웨이퍼 결함 검출 시스템 - 빠른 시작 가이드")
    print("=" * 60)
    
    # 사용법 예제 출력
    show_usage_examples()
    
    # 설정 예제
    example_configuration()
    
    # 디렉토리 설정 예제
    example_directory_setup()
    
    print("\n" + "=" * 60)
    print("🚀 시작하기:")
    print("1. 데이터셋을 ImageFolder 구조로 준비")
    print("2. 사전 훈련된 모델 파일 다운로드")
    print("3. config.json에서 경로 설정")
    print("4. python main.py로 전체 파이프라인 실행")
    print("5. --predict 옵션으로 새 이미지 예측")
    print("=" * 60)
    
    # 환경 확인
    print("\n🔍 환경 확인:")
    
    # Python 버전
    print(f"   Python: {sys.version}")
    
    # 필수 패키지 확인
    required_packages = ['torch', 'torchvision', 'PIL', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}: 설치됨")
        except ImportError:
            print(f"   ❌ {package}: 설치 필요")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 다음 패키지를 설치해주세요:")
        print(f"   pip install -r requirements.txt")
    else:
        print(f"\n✅ 모든 필수 패키지가 설치되어 있습니다!")


if __name__ == "__main__":
    main()
