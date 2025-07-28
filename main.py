#!/usr/bin/env python3
"""
🎯 Simplified Main - 간소화된 실행 로직
"""

import os
import json
import argparse
from pathlib import Path

from wafer_detector import WaferDetector

# 기본 설정
CONFIG = {
    'DATASET_ROOT': os.getenv('DATASET_ROOT', ''),  
    'MODEL_PATH': 'pretrained_models/convnextv2_base.fcmae_ft_in22k_in1k_pretrained.pth',
    'YOLO_MODEL': 'pretrained_models/yolo11x.pt',
    'CLASSIFICATION_SIZE': 384,
    'YOLO_SIZE': 1024,
    'F1_THRESHOLD': 0.8,
    'CONFIDENCE_THRESHOLD': 0.7,
    'MAPPING_THRESHOLD': 0.3,
    'OUTPUT_DIR': 'outputs'
}


def run_pipeline(detector: WaferDetector, dataset_root: str):
    """전체 파이프라인 실행"""
    print("🎯 Running full pipeline...")
    
    # Stage 1-3 순차 실행
    detector.analyze_performance(dataset_root)
    detector.learn_roi_patterns(dataset_root)
    detector.create_mapping(dataset_root)
    detector.save_results(CONFIG['OUTPUT_DIR'])
    
    print("✅ Pipeline completed!")


def run_prediction(detector: WaferDetector, predict_path: str):
    """예측 실행"""
    predict_path = Path(predict_path)
    
    if predict_path.is_file():
        # 단일 이미지
        if not detector.classes:
            detector.load_classes(CONFIG['DATASET_ROOT'])
        
        result = detector.predict_image(str(predict_path))
        print(f"🎯 Result: {result}")
        
    elif predict_path.is_dir():
        # 폴더 예측
        subdirs = [d for d in predict_path.iterdir() if d.is_dir()]
        
        if subdirs:
            # ImageFolder 구조
            detector.load_classes(str(predict_path))
            image_files = []
            for subdir in subdirs:
                image_files.extend(subdir.glob("*.jpg"))
                image_files.extend(subdir.glob("*.png"))
        else:
            # 단순 폴더
            if not detector.classes:
                detector.load_classes(CONFIG['DATASET_ROOT'])
            image_files = list(predict_path.glob("*.jpg")) + list(predict_path.glob("*.png"))
        
        # 배치 예측
        results = []
        for img_path in image_files:
            result = detector.predict_image(str(img_path))
            results.append(result)
            print(f"🎯 {img_path.name}: {result['predicted_class']} ({result['confidence']:.3f})")
        
        # 결과 저장
        output_path = Path(CONFIG['OUTPUT_DIR']) / 'prediction_results.json'
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {output_path}")
    
    else:
        raise FileNotFoundError(f"Path not found: {predict_path}")


def main():
    parser = argparse.ArgumentParser(description="Wafer Defect Detection")
    parser.add_argument("dataset_path", nargs='?', help="Dataset root path")
    parser.add_argument("--predict", help="Predict single image or folder")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # 설정 업데이트
    if args.dataset_path:
        CONFIG['DATASET_ROOT'] = args.dataset_path
    if args.output_dir:
        CONFIG['OUTPUT_DIR'] = args.output_dir
    
    # 검출기 초기화
    detector = WaferDetector(CONFIG)
    detector.load_models(CONFIG['MODEL_PATH'], CONFIG['YOLO_MODEL'])
    
    # 실행
    if args.predict:
        run_prediction(detector, args.predict)
    else:
        if not CONFIG['DATASET_ROOT']:
            raise ValueError("Dataset path required. Set DATASET_ROOT or provide as argument.")
        run_pipeline(detector, CONFIG['DATASET_ROOT'])


if __name__ == "__main__":
    main()
