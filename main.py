#!/usr/bin/env python3
"""
🎯 Wafer Defect Detection - Main Entry Point
지능형 2단계 웨이퍼 결함 검출 시스템 실행 스크립트
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

from wafer_detector import WaferDetector, WaferDetectorError

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wafer_detection.log')
    ]
)
logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_CONFIG = {
    'DATASET_ROOT': os.getenv('DATASET_ROOT', ''),  
    'MODEL_PATH': 'pretrained_models/convnextv2_base.fcmae_ft_in22k_in1k_pretrained.pth',
    'YOLO_MODEL': 'pretrained_models/yolo11x.pt',
    'CLASSIFICATION_SIZE': 384,
    'YOLO_SIZE': 1024,
    'PRECISION_THRESHOLD': 0.8,
    'CONFIDENCE_THRESHOLD': 0.7,
    'MAPPING_THRESHOLD': 0.3,
    'OUTPUT_DIR': 'outputs'
}


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # 기본 설정과 병합
            config = DEFAULT_CONFIG.copy()
            config.update(user_config)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    설정 파일 저장
    
    Args:
        config: 설정 딕셔너리
        config_path: 저장 경로
    """
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save config: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    설정 유효성 검증
    
    Args:
        config: 설정 딕셔너리
        
    Raises:
        ValueError: 설정이 유효하지 않은 경우
    """
    required_keys = ['MODEL_PATH', 'YOLO_MODEL', 'CLASSIFICATION_SIZE', 'YOLO_SIZE']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # 모델 파일 존재 확인
    model_path = Path(config['MODEL_PATH'])
    yolo_path = Path(config['YOLO_MODEL'])
    
    if not model_path.exists():
        raise ValueError(f"Classification model not found: {model_path}")
    if not yolo_path.exists():
        raise ValueError(f"YOLO model not found: {yolo_path}")
    
    # 임계값 범위 확인
    thresholds = ['PRECISION_THRESHOLD', 'CONFIDENCE_THRESHOLD', 'MAPPING_THRESHOLD']
    for key in thresholds:
        if key in config:
            value = config[key]
            if not 0 <= value <= 1:
                raise ValueError(f"{key} must be between 0 and 1, got {value}")


def run_full_pipeline(detector: WaferDetector, dataset_root: str, config: Dict[str, Any]) -> None:
    """
    전체 파이프라인 실행 (학습 모드)
    
    Args:
        detector: 웨이퍼 검출기 인스턴스
        dataset_root: 데이터셋 루트 경로
        config: 설정 딕셔너리
    """
    try:
        logger.info("Starting full pipeline...")
        
        # Stage 1: 성능 분석
        logger.info("Stage 1: Performance Analysis")
        f1_scores = detector.analyze_performance(dataset_root)
        
        # Stage 2: ROI 패턴 학습
        logger.info("Stage 2: ROI Pattern Learning")
        detector.learn_roi_patterns(dataset_root)
        
        # Stage 3: 클래스-객체 매핑 생성
        logger.info("Stage 3: Class-Object Mapping")
        detector.create_mapping(dataset_root)
        
        # Stage 4: 결과 저장
        logger.info("Stage 4: Saving Results")
        detector.save_results(config['OUTPUT_DIR'])
        
        # 통계 출력
        stats = detector.get_stats()
        logger.info("Pipeline Statistics:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
        
        logger.info("Full pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def run_prediction_mode(
    detector: WaferDetector, 
    predict_path: str, 
    config: Dict[str, Any]
) -> None:
    """
    예측 모드 실행
    
    Args:
        detector: 웨이퍼 검출기 인스턴스
        predict_path: 예측 대상 경로
        config: 설정 딕셔너리
    """
    try:
        predict_path = Path(predict_path)
        
        if not predict_path.exists():
            raise FileNotFoundError(f"Prediction path not found: {predict_path}")
        
        # 기존 결과 로드 시도
        output_dir = Path(config['OUTPUT_DIR'])
        if output_dir.exists():
            try:
                detector.load_results(output_dir)
                logger.info("Loaded existing ROI patterns and mappings")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        
        if predict_path.is_file():
            # 단일 이미지 예측
            logger.info(f"Predicting single image: {predict_path}")
            
            if not detector.classes:
                if not config['DATASET_ROOT']:
                    raise ValueError("DATASET_ROOT required for class loading")
                detector.load_classes(config['DATASET_ROOT'])
            
            result = detector.predict_image(str(predict_path))
            
            # 결과 출력
            logger.info("Prediction Result:")
            logger.info(f"   Image: {result['image_path']}")
            logger.info(f"   Predicted Class: {result['predicted_class']}")
            logger.info(f"   Confidence: {result['confidence']:.3f}")
            logger.info(f"   Method: {result['method']}")
            
            if result['method'] == 'roi_enhanced':
                logger.info(f"   Detected Object: {result.get('detected_object', 'N/A')}")
                logger.info(f"   Object Counts: {result.get('object_counts', {})}")
            
        elif predict_path.is_dir():
            # 폴더 예측
            logger.info(f"Predicting folder: {predict_path}")
            
            # 이미지 파일 수집
            image_files = []
            subdirs = [d for d in predict_path.iterdir() if d.is_dir()]
            
            if subdirs:
                # ImageFolder 구조
                logger.info("Detected ImageFolder structure")
                detector.load_classes(str(predict_path))
                for subdir in subdirs:
                    image_files.extend(subdir.glob("*.jpg"))
                    image_files.extend(subdir.glob("*.png"))
                    image_files.extend(subdir.glob("*.jpeg"))
            else:
                # 단순 폴더 구조
                logger.info("Detected simple folder structure")
                if not detector.classes:
                    if not config['DATASET_ROOT']:
                        raise ValueError("DATASET_ROOT required for class loading")
                    detector.load_classes(config['DATASET_ROOT'])
                
                image_files = list(predict_path.glob("*.jpg"))
                image_files.extend(predict_path.glob("*.png"))
                image_files.extend(predict_path.glob("*.jpeg"))
            
            if not image_files:
                logger.warning("No image files found")
                return
            
            logger.info(f"Found {len(image_files)} images to process")
            
            # 배치 예측
            results = []
            method_counts = {'classification_only': 0, 'roi_enhanced': 0}
            
            for i, img_path in enumerate(image_files):
                try:
                    result = detector.predict_image(str(img_path))
                    results.append(result)
                    method_counts[result['method']] += 1
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(image_files)} images")
                    
                except Exception as e:
                    logger.warning(f"Failed to predict {img_path}: {e}")
                    continue
            
            # 결과 저장
            output_dir = Path(config['OUTPUT_DIR'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = output_dir / 'prediction_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # 통계 출력
            logger.info("Batch Prediction Results:")
            logger.info(f"   Total Images: {len(image_files)}")
            logger.info(f"   Successful Predictions: {len(results)}")
            logger.info(f"   Classification Only: {method_counts['classification_only']}")
            logger.info(f"   ROI Enhanced: {method_counts['roi_enhanced']}")
            logger.info(f"   Results saved to: {results_file}")
            
            # 클래스별 예측 분포
            class_counts = {}
            for result in results:
                class_name = result['predicted_class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            logger.info("   Class Distribution:")
            for class_name, count in sorted(class_counts.items()):
                logger.info(f"     {class_name}: {count}")
        
        else:
            raise ValueError(f"Invalid prediction path: {predict_path}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Wafer Defect Detection with ROI Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (training mode)
    python main.py /path/to/dataset
    
    # Predict single image
    python main.py --predict /path/to/image.jpg
    
    # Predict folder
    python main.py --predict /path/to/images/
    
    # Custom config and output
    python main.py /path/to/dataset --config config.json --output-dir results/
        """
    )
    
    parser.add_argument(
        "dataset_path", 
        nargs='?', 
        help="Dataset root path (for training mode)"
    )
    parser.add_argument(
        "--predict", 
        help="Predict single image or folder"
    )
    parser.add_argument(
        "--config", 
        default="config.json",
        help="Configuration file path (default: config.json)"
    )
    parser.add_argument(
        "--output-dir", 
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--load-results",
        action="store_true",
        help="Load existing results before prediction"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # 로깅 레벨 조정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 설정 로드
        config_path = Path(args.config)
        config = load_config(config_path)
        
        # 명령행 인수로 설정 오버라이드
        if args.dataset_path:
            config['DATASET_ROOT'] = args.dataset_path
        if args.output_dir:
            config['OUTPUT_DIR'] = args.output_dir
        
        # 설정 유효성 검증
        validate_config(config)
        
        # 설정 저장 (업데이트된 경우)
        save_config(config, config_path)
        
        logger.info("Starting Wafer Defect Detection System")
        logger.info(f"Configuration: {config}")
        
        # 검출기 초기화
        detector = WaferDetector(config)
        detector.load_models(config['MODEL_PATH'], config['YOLO_MODEL'])
        
        # 실행 모드 결정
        if args.predict:
            # 예측 모드
            run_prediction_mode(detector, args.predict, config)
        else:
            # 학습 모드
            if not config['DATASET_ROOT']:
                raise ValueError("Dataset path required for training mode. "
                               "Provide as argument or set DATASET_ROOT in config.")
            run_full_pipeline(detector, config['DATASET_ROOT'], config)
        
        logger.info("Process completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
