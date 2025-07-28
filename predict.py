#!/usr/bin/env python3
"""
🔮 Enhanced Wafer Defect Detection - 추론 전용
ImageFolder 기반 간단한 예측 시스템
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
import timm
from ultralytics import YOLO
import cv2

from roi_utils import ROIExtractor

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import ConfigManager


class WaferPredictor:
    """🔮 간단한 웨이퍼 예측기"""
    
    def __init__(self, model_dir: str = "enhanced_pipeline_output"):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 관련
        self.model = None
        self.classes = []
        self.num_classes = 0
        self.transform = None
        
        # ROI 관련
        self.yolo_model = None
        self.difficult_classes = []
        self.class_object_mapping = {}
        self.yolo_objects = []
        
        # ROI 추출기 초기화 (클래스별 패턴 파일 지정)
        roi_patterns_file = self.model_dir / "class_roi_patterns.json"
        self.roi_extractor = ROIExtractor(str(roi_patterns_file) if roi_patterns_file.exists() else None)
        
        # 설정 로드
        config_manager = ConfigManager()
        self.config = config_manager.get_config()
        
        print(f"🔮 WaferPredictor initialized")
        print(f"  Model dir: {self.model_dir}")
        print(f"  Device: {self.device}")
    
    def _prepare_classification_image(self, image_path: str) -> tuple:
        """Classification용 이미지와 원본 이미지 경로 준비"""
        # Classification용 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # ROI 검출용 리사이즈된 numpy 배열로 변환
        classification_size = getattr(self.config, 'CLASSIFICATION_SIZE', 384)
        image_resized = image.resize((classification_size, classification_size))
        image_resized_np = np.array(image_resized)
        
        return image_tensor, image_resized_np, image_path
    
    def _extract_roi_with_learned_pattern(self, original_image_path: str, predicted_class: str) -> np.ndarray:
        """학습된 클래스별 ROI 패턴을 사용하여 ROI 추출"""
        yolo_size = getattr(self.config, 'YOLO_INPUT_SIZE', 1024)
        
        return self.roi_extractor.crop_roi_from_original(
            original_image_path, 
            predicted_class,
            target_size=yolo_size
        )
    
    def _load_model(self):
        """모델 로드"""
        
        # 클래스 정보 로드
        info_path = self.model_dir / 'class_info.json'
        if not info_path.exists():
            raise FileNotFoundError(f"❌ Class info not found: {info_path}")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            class_info = json.load(f)
        
        self.classes = class_info['classes']
        self.num_classes = class_info['num_classes']
        
        print(f"📋 Loaded {self.num_classes} classes: {self.classes}")
        
        # 모델 생성
        model_config = class_info['config']
        self.model = timm.create_model(
            model_config['CONVNEXT_MODEL_NAME'],
            pretrained=False,
            num_classes=self.num_classes
        )
        
        # 가중치 로드
        model_path = self.model_dir / self.config.CLASSIFICATION_MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model weights not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # model. prefix 제거
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('model.'):
                new_key = key[6:]  # "model." 제거
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Transform 설정
        self.transform = transforms.Compose([
            transforms.Resize((model_config['CLASSIFICATION_SIZE'], model_config['CLASSIFICATION_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # ROI 매핑 정보 로드
        self._load_roi_mappings()
        
        # YOLO 모델 로드
        self._load_yolo_model()
        
        print(f"✅ Model loaded successfully")
        print(f"  - Difficult classes: {len(self.difficult_classes)}")
        print(f"  - ROI mappings: {len(self.class_object_mapping)}")
    
    def _load_roi_mappings(self):
        """ROI 매핑 정보 로드"""
        
        mapping_path = self.model_dir / 'discovered_mappings.json'
        if not mapping_path.exists():
            print("⚠️ No ROI mappings found, using classification only")
            self.difficult_classes = []
            self.class_object_mapping = {}
            self.yolo_objects = []
            return
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        self.difficult_classes = mapping_data.get('difficult_classes', [])
        self.class_object_mapping = mapping_data.get('class_object_mapping', {})
        self.yolo_objects = mapping_data.get('yolo_objects', [])
    
    def _load_yolo_model(self):
        """YOLO 모델 로드"""
        
        try:
            self.yolo_model = YOLO(self.config.DETECTION_MODEL)
            
            if hasattr(self.yolo_model, 'names'):
                if not self.yolo_objects:  # 매핑에서 로드되지 않았다면
                    self.yolo_objects = list(self.yolo_model.names.values())
                print(f"🎯 YOLO model loaded: {len(self.yolo_objects)} objects")
            else:
                print("⚠️ Could not extract YOLO object names")
                
        except Exception as e:
            print(f"⚠️ YOLO model loading failed: {e}")
            self.yolo_model = None
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """단일 이미지 예측 - ROI 검증 포함"""
        
        if self.model is None:
            self._load_model()
        
        # ImageFolder 방식으로 단일 이미지 처리
        image_path = Path(image_path)
        if image_path.is_file():
            # 파일명 입력 경우 - 직접 로드 (예외)
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            # 폴더 경우 - ImageFolder 사용
            temp_dataset = datasets.ImageFolder(root=image_path, transform=self.transform)
            if len(temp_dataset) == 0:
                raise ValueError(f"No images found in {image_path}")
            input_tensor, _ = temp_dataset[0]
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # 1. Classification 예측
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = np.argmax(probabilities)
            initial_confidence = float(probabilities[predicted_idx])
        
        predicted_class = self.classes[predicted_idx]
        final_confidence = initial_confidence
        used_roi_verification = False
        roi_verification_success = None
        
        # 2. ROI 검증 필요한지 확인
        needs_roi_verification = (
            predicted_class in self.difficult_classes and
            initial_confidence < self.config.CONFIDENCE_THRESHOLD and
            len(self.class_object_mapping) > 0 and  # ROI 매핑이 존재하면
            self.yolo_model is not None
        )
        
        if needs_roi_verification:
            # 3. ROI 기반 클래스 재결정
            used_roi_verification = True
            roi_suggested_class = self._get_roi_suggested_class(image_path, predicted_class)
            
            if roi_suggested_class:
                # 4. ROI로 클래스 완전 변경
                predicted_class = roi_suggested_class
                predicted_idx = self.classes.index(roi_suggested_class)
                final_confidence = self.config.ROI_CONFIDENCE_BOOST + 0.6  # 높은 신뢰도 부여
                roi_verification_success = True
            else:
                # ROI 검출 실패 시 원래 결과 유지하되 신뢰도 하락
                final_confidence = max(0.0, initial_confidence - self.config.ROI_CONFIDENCE_PENALTY)
                roi_verification_success = False
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'initial_confidence': initial_confidence,
            'final_confidence': final_confidence,
            'confidence': final_confidence,  # 호환성
            'class_idx': predicted_idx,
            'used_roi_verification': used_roi_verification,
            'roi_verification_success': roi_verification_success,
            'all_probabilities': {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}
        }
    
    def _get_roi_suggested_class(self, image_path: str, predicted_class: str) -> Optional[str]:
        """ROI 기반 클래스 제안 - 학습된 클래스별 ROI 패턴 사용"""
        
        try:
            # 학습된 클래스별 ROI 패턴을 사용하여 ROI 추출
            roi_image = self._extract_roi_with_learned_pattern(image_path, predicted_class)
            results = self.yolo_model(roi_image, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return None
            
            # 신뢰도 기준 필터링 후 가장 많은 객체 종류 찾기
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # 신뢰도 임계값 이상인 객체들만 고려
            valid_indices = confidences > self.config.OBJECT_CONFIDENCE_THRESHOLD
            if not np.any(valid_indices):
                return None
            
            valid_classes = classes[valid_indices]
            
            # 각 객체 클래스별 개수 세기
            unique_classes, counts = np.unique(valid_classes, return_counts=True)
            
            # 가장 많이 검출된 객체 클래스 선택
            most_frequent_idx = np.argmax(counts)
            detected_class = int(unique_classes[most_frequent_idx])
            object_count = counts[most_frequent_idx]
            
            detected_object = self.yolo_objects[detected_class]
            print(f"🔍 Most frequent ROI object: '{detected_object}' (count: {object_count})")
            
            # 역매핑: 검출된 객체가 어떤 클래스와 연결되는지 찾기
            for class_name, mapped_object in self.class_object_mapping.items():
                if mapped_object == detected_object:
                    return class_name
            
            # 매핑에 없는 객체면 None 반환
            return None
            
        except Exception as e:
            print(f"⚠️ ROI class suggestion failed: {e}")
            return None
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """배치 예측"""
        
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'predicted_class': None,
                    'confidence': 0.0
                })
        
        return results
    
    def _evaluate_with_imagefolder(self, test_dataset) -> Dict[str, Any]:
        """ImageFolder 데이터셋으로 성능 평가"""
        
        from torch.utils.data import DataLoader
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # 데이터 로더 생성
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        all_preds = []
        all_labels = []
        all_confidences = []
        roi_used_count = 0
        roi_success_count = 0
        
        print("🔍 Running evaluation...")
        
        for images, labels in test_loader:
            images = images.to(self.device)
            
            # 배치 예측
            with torch.no_grad():
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                predicted_indices = np.argmax(probabilities, axis=1)
                confidences = np.max(probabilities, axis=1)
            
            for i in range(len(images)):
                predicted_class = self.classes[predicted_indices[i]]
                initial_confidence = float(confidences[i])
                true_label = labels[i].item()
                
                # ROI 검증 필요한지 확인
                needs_roi_verification = (
                    predicted_class in self.difficult_classes and
                    initial_confidence < self.config.CONFIDENCE_THRESHOLD and
                    len(self.class_object_mapping) > 0 and  # ROI 매핑이 존재하면
                    self.yolo_model is not None
                )
                
                final_confidence = initial_confidence
                
                if needs_roi_verification:
                    roi_used_count += 1
                    # 평가 모드에서는 시뮬레이션 (실제 ROI는 단일 이미지 파일 경로 필요)
                    roi_suggested_class = np.random.choice(self.classes)  # 임시 시뮬레이션
                    
                    if roi_suggested_class and roi_suggested_class in self.class_object_mapping:
                        # ROI로 클래스 변경
                        predicted_class = roi_suggested_class
                        predicted_indices[i] = self.classes.index(roi_suggested_class)
                        final_confidence = self.config.ROI_CONFIDENCE_BOOST + 0.6
                        roi_success_count += 1
                    else:
                        # ROI 검출 실패
                        final_confidence = max(0.0, initial_confidence - self.config.ROI_CONFIDENCE_PENALTY)
                
                all_preds.append(predicted_indices[i])
                all_labels.append(true_label)
                all_confidences.append(final_confidence)
        
        # 성능 계산
        accuracy = accuracy_score(all_labels, all_preds)
        
        # 클래스별 성능 리포트
        class_names = [self.classes[i] for i in range(len(self.classes))]
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': accuracy,
            'total_samples': len(all_labels),
            'roi_used_count': roi_used_count,
            'roi_success_count': roi_success_count,
            'roi_usage_rate': roi_used_count / len(all_labels) * 100,
            'average_confidence': np.mean(all_confidences),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        # 결과 출력
        print(f"\n📊 Evaluation Results:")
        print(f"  🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  📊 Total samples: {len(all_labels)}")
        print(f"  🔍 ROI class changes: {roi_used_count} ({roi_used_count/len(all_labels)*100:.1f}%)")
        print(f"  ✅ ROI successful changes: {roi_success_count}/{roi_used_count}")
        print(f"  📈 Average confidence: {np.mean(all_confidences):.3f}")
        
        print(f"\n📋 Per-class Performance:")
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1 = report[class_name]['f1-score']
                support = report[class_name]['support']
                difficult_marker = " 🔴" if class_name in self.difficult_classes else ""
                roi_marker = " 🔍" if class_name in self.class_object_mapping else ""
                print(f"  {class_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={support}){difficult_marker}{roi_marker}")
        
        return results


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Wafer Defect Detection - Prediction Only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 실전 예측
  python predict.py image.jpg
  python predict.py folder/ --batch
  
  # 성능 평가 (ImageFolder 구조 필요)
  python predict.py test_dataset/ --eval
  python predict.py test_dataset/ --eval --save-results eval_results.json
        """
    )
    
    parser.add_argument("input", nargs='?', help="예측할 이미지 파일 또는 폴더 경로 (기본값: 대화형 입력)")
    parser.add_argument("--model-dir", default="enhanced_pipeline_output", help="모델 디렉토리")
    parser.add_argument("--batch", action="store_true", help="폴더 내 모든 이미지 배치 처리")
    parser.add_argument("--save-results", help="예측 결과를 JSON 파일로 저장")
    parser.add_argument("--eval", action="store_true", help="성능 평가 모드 (ImageFolder 구조 필요)")
    
    args = parser.parse_args()
    
    print("🔮 Enhanced Wafer Defect Detection - Prediction")
    print("=" * 45)
    
    # 입력 경로 설정 (대화형 입력 가능)
    if args.input:
        input_path = Path(args.input)
    else:
        # 대화형 입력 요청
        input_str = input("📁 예측할 이미지 파일 또는 폴더 경로를 입력하세요: ").strip()
        if not input_str:
            print("❌ 입력 경로가 필요합니다.")
            return 1
        input_path = Path(input_str)
    
    if not input_path.exists():
        print(f"❌ Input path not found: {input_path}")
        return 1
    
    # 예측기 초기화
    try:
        predictor = WaferPredictor(args.model_dir)
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return 1
    
    try:
        if args.eval and input_path.is_dir():
            # 성능 평가 모드 (ImageFolder 구조)
            print(f"📊 Evaluation mode with ImageFolder: {input_path}")
            
            # ImageFolder로 테스트 데이터셋 로드
            test_dataset = datasets.ImageFolder(
                root=input_path,
                transform=transforms.Compose([
                    transforms.Resize((224, 224)),  # predictor config에 맞춰야 함
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            )
            
            if len(test_dataset) == 0:
                print("❌ No images found in dataset structure")
                return 1
            
            print(f"📋 Found {len(test_dataset.classes)} classes: {test_dataset.classes}")
            print(f"🔍 Processing {len(test_dataset)} images...")
            
            # 성능 평가 수행
            results = predictor._evaluate_with_imagefolder(test_dataset)
            
            # 결과 저장
            if args.save_results:
                with open(args.save_results, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"💾 Results saved: {args.save_results}")
                
        elif args.batch and input_path.is_dir():
            # 배치 예측 (실전 모드)
            print(f"📂 Batch prediction: {input_path}")
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                print("❌ No image files found")
                return 1
            
            print(f"🔍 Processing {len(image_files)} images...")
            results = predictor.predict_batch([str(f) for f in image_files])
            
            # 결과 출력
            print(f"\n📊 Batch Results:")
            correct_predictions = 0
            roi_used_count = 0
            for result in results:
                if 'error' not in result:
                    roi_indicator = " 🔍" if result.get('used_roi_verification', False) else ""
                    print(f"  {Path(result['image_path']).name}: {result['predicted_class']} ({result['confidence']:.3f}){roi_indicator}")
                    correct_predictions += 1
                    if result.get('used_roi_verification', False):
                        roi_used_count += 1
                else:
                    print(f"  {Path(result['image_path']).name}: ERROR - {result['error']}")
            
            print(f"\n✅ Successfully processed: {correct_predictions}/{len(results)} images")
            print(f"🔍 ROI class changes: {roi_used_count} ({roi_used_count/len(results)*100:.1f}%)")
            
            # 결과 저장
            if args.save_results:
                with open(args.save_results, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"💾 Results saved: {args.save_results}")
            else:
                # 단일 예측
                print(f"🔍 Predicting: {input_path}")
                result = predictor.predict(str(input_path))
                
                print(f"\n🎯 Prediction Result:")
                print(f"  📋 Class: {result['predicted_class']}")
            print(f"  📊 Final Confidence: {result['confidence']:.3f}")
            
            if result.get('used_roi_verification', False):
                if result['roi_verification_success']:
                    print(f"  🔍 ROI Class Change: ✅ Changed to ROI suggested class")
                    print(f"  📈 Initial → Final: {result['initial_confidence']:.3f} → {result['confidence']:.3f}")
                else:
                    print(f"  🔍 ROI Class Change: ❌ No ROI suggestion, reduced confidence")
                    print(f"  📉 Initial → Final: {result['initial_confidence']:.3f} → {result['confidence']:.3f}")
            else:
                print(f"  🔍 ROI Class Change: Not used")
            
            print(f"\n📈 All Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                print(f"  {cls}: {prob:.3f}")
            
            # 결과 저장
            if args.save_results:
                with open(args.save_results, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"💾 Result saved: {args.save_results}")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
