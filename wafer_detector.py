#!/usr/bin/env python3
"""
🎯 WaferDetector - 웨이퍼 결함 검출 시스템
지능형 2단계 검출: 기본 분류 + ROI 기반 정밀 검증
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import timm
from ultralytics import YOLO
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from gradcam_utils import GradCAMAnalyzer, extract_roi_from_heatmap

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WaferDetectorError(Exception):
    """웨이퍼 검출기 관련 예외"""
    pass


class WaferDetector:
    """
    지능형 웨이퍼 결함 검출기
    
    2단계 검출 시스템:
    1. 기본 분류 (모든 이미지)
    2. ROI 기반 정밀 검증 (어려운 클래스 + 낮은 신뢰도)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classification_model = None
        self.yolo_model = None
        self.gradcam_analyzer = None
        
        # 클래스 및 패턴 정보
        self.classes: List[str] = []
        self.difficult_classes: List[str] = []
        self.class_object_mapping: Dict[str, str] = {}
        self.roi_patterns: Dict[str, Dict[str, float]] = {}
        self.f1_scores: Optional[np.ndarray] = None
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((config['CLASSIFICATION_SIZE'], config['CLASSIFICATION_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"🚀 WaferDetector initialized - Device: {self.device}")
    
    def load_models(self, model_path: Union[str, Path], yolo_path: Union[str, Path]) -> None:
        """
        모델 로드
        
        Args:
            model_path: ConvNeXtV2 모델 경로
            yolo_path: YOLO 모델 경로
            
        Raises:
            WaferDetectorError: 모델 로드 실패시
        """
        try:
            model_path = Path(model_path)
            yolo_path = Path(yolo_path)
            
            # 파일 존재 확인
            if not model_path.exists():
                raise WaferDetectorError(f"Classification model not found: {model_path}")
            if not yolo_path.exists():
                raise WaferDetectorError(f"YOLO model not found: {yolo_path}")
            
            # 1. ConvNeXtV2 모델 생성
            logger.info("Loading ConvNeXtV2 model...")
            self.classification_model = timm.create_model(
                'convnextv2_base.fcmae_ft_in22k_in1k', 
                pretrained=True
            )
            
            # 2. 가중치 로드 및 prefix 제거
            state_dict = torch.load(model_path, map_location="cpu")
            cleaned_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
            # 3. 분류기 교체
            if 'head.fc.weight' not in cleaned_state_dict:
                raise WaferDetectorError("Invalid model weights: missing head.fc.weight")
                
            num_classes = cleaned_state_dict['head.fc.weight'].shape[0]
            self.classification_model.head.fc = nn.Linear(
                self.classification_model.head.fc.in_features, 
                num_classes
            )
            
            # 4. 가중치 로드
            self.classification_model.load_state_dict(cleaned_state_dict, strict=True)
            self.classification_model.to(self.device).eval()
            
            # 5. GradCAM 및 YOLO 초기화
            self.gradcam_analyzer = GradCAMAnalyzer(self.classification_model)
            self.yolo_model = YOLO(str(yolo_path))
            
            logger.info(f"✅ Models loaded successfully - Classes: {num_classes}")
            
        except Exception as e:
            raise WaferDetectorError(f"Failed to load models: {str(e)}")
    
    def load_classes(self, dataset_root: Union[str, Path]) -> None:
        """
        클래스 정보 로드
        
        Args:
            dataset_root: 데이터셋 루트 경로
            
        Raises:
            WaferDetectorError: 데이터셋 로드 실패시
        """
        try:
            dataset_root = Path(dataset_root)
            if not dataset_root.exists():
                raise WaferDetectorError(f"Dataset root not found: {dataset_root}")
            
            dataset = datasets.ImageFolder(str(dataset_root))
            self.classes = dataset.classes
            
            if not self.classes:
                raise WaferDetectorError("No classes found in dataset")
                
            logger.info(f"📋 Classes loaded: {self.classes}")
            
        except Exception as e:
            raise WaferDetectorError(f"Failed to load classes: {str(e)}")
    
    def analyze_performance(self, dataset_root: Union[str, Path]) -> np.ndarray:
        """
        성능 분석 및 어려운 클래스 식별
        
        Args:
            dataset_root: 데이터셋 루트 경로
            
        Returns:
            F1 스코어 배열
            
        Raises:
            WaferDetectorError: 성능 분석 실패시
        """
        try:
            if self.classification_model is None:
                raise WaferDetectorError("Classification model not loaded")
                
            dataset_root = Path(dataset_root)
            dataset = datasets.ImageFolder(str(dataset_root), transform=self.transform)
            dataloader = DataLoader(
                dataset, 
                batch_size=32, 
                shuffle=False, 
                num_workers=min(2, torch.get_num_threads())
            )
            self.classes = dataset.classes
            
            logger.info(f"📊 Analyzing performance on {len(dataset)} samples...")
            
            # 예측 수행
            all_preds, all_labels = [], []
            self.classification_model.eval()
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(dataloader):
                    images = images.to(self.device)
                    outputs = self.classification_model(images)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Processed {batch_idx * 32}/{len(dataset)} samples")
            
            # F1 score 계산
            _, _, f1_scores, _ = precision_recall_fscore_support(
                all_labels, all_preds, average=None, zero_division=0
            )
            self.f1_scores = f1_scores
            
            # 어려운 클래스 식별
            self.difficult_classes = [
                self.classes[i] for i, f1 in enumerate(f1_scores) 
                if f1 < self.config['F1_THRESHOLD']
            ]
            
            # 결과 출력
            logger.info("Performance Analysis Results:")
            for i, f1 in enumerate(f1_scores):
                status = "⚠️" if f1 < self.config['F1_THRESHOLD'] else "✅"
                logger.info(f"   {status} {self.classes[i]}: F1={f1:.3f}")
            
            logger.info(f"Identified {len(self.difficult_classes)} difficult classes")
            
            return f1_scores
            
        except Exception as e:
            raise WaferDetectorError(f"Performance analysis failed: {str(e)}")
    
    def learn_roi_patterns(self, dataset_root: Union[str, Path], max_samples: int = 10) -> None:
        """
        ROI 패턴 학습
        
        Args:
            dataset_root: 데이터셋 루트 경로
            max_samples: 클래스당 최대 샘플 수
            
        Raises:
            WaferDetectorError: ROI 패턴 학습 실패시
        """
        try:
            if not self.difficult_classes:
                logger.info("ℹ️ No difficult classes found, skipping ROI pattern learning")
                return
                
            if self.gradcam_analyzer is None:
                raise WaferDetectorError("GradCAM analyzer not initialized")
            
            dataset_root = Path(dataset_root)
            logger.info(f"🧠 Learning ROI patterns for {len(self.difficult_classes)} classes...")
            
            for class_name in self.difficult_classes:
                try:
                    class_dir = dataset_root / class_name
                    if not class_dir.exists():
                        logger.warning(f"Class directory not found: {class_dir}")
                        continue
                    
                    image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    if not image_files:
                        logger.warning(f"No images found in {class_dir}")
                        continue
                    
                    roi_coords_list = []
                    processed_count = 0
                    
                    for img_path in image_files[:max_samples]:
                        try:
                            image = Image.open(img_path).convert('RGB')
                            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                            
                            class_idx = self.classes.index(class_name)
                            heatmap = self.gradcam_analyzer.generate_gradcam(input_tensor, class_idx)
                            roi_coords = extract_roi_from_heatmap(heatmap)
                            roi_coords_list.append(roi_coords)
                            processed_count += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to process {img_path}: {str(e)}")
                            continue
                    
                    if roi_coords_list:
                        # 중앙값으로 대표 ROI 계산
                        roi_array = np.array(roi_coords_list)
                        representative_roi = np.median(roi_array, axis=0)
                        
                        self.roi_patterns[class_name] = {
                            'x1': float(representative_roi[0]), 
                            'y1': float(representative_roi[1]),
                            'x2': float(representative_roi[2]), 
                            'y2': float(representative_roi[3])
                        }
                        
                        logger.info(f"📍 {class_name}: ROI learned from {processed_count} samples")
                    else:
                        logger.warning(f"Failed to learn ROI pattern for {class_name}")
                        
                except Exception as e:
                    logger.error(f"Error learning ROI for {class_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            raise WaferDetectorError(f"ROI pattern learning failed: {str(e)}")
    
    def create_mapping(self, dataset_root: Union[str, Path], max_samples: int = 30) -> None:
        """
        클래스-객체 매핑 생성
        
        Args:
            dataset_root: 데이터셋 루트 경로
            max_samples: 클래스당 최대 샘플 수
            
        Raises:
            WaferDetectorError: 매핑 생성 실패시
        """
        try:
            if not self.difficult_classes or not self.roi_patterns:
                logger.info("ℹ️ No difficult classes or ROI patterns, skipping mapping creation")
                return
                
            if self.yolo_model is None:
                raise WaferDetectorError("YOLO model not loaded")
            
            dataset_root = Path(dataset_root)
            logger.info(f"🎯 Creating object mappings...")
            
            class_object_counts = {}
            
            for class_name in self.difficult_classes:
                if class_name not in self.roi_patterns:
                    logger.warning(f"No ROI pattern for {class_name}, skipping")
                    continue
                
                try:
                    class_dir = dataset_root / class_name
                    image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    object_counts = {}
                    processed_count = 0
                    
                    for img_path in image_files[:max_samples]:
                        try:
                            # ROI 추출
                            image = Image.open(img_path).convert('RGB')
                            w, h = image.size
                            roi = self.roi_patterns[class_name]
                            
                            x1 = max(0, int(roi['x1'] * w))
                            y1 = max(0, int(roi['y1'] * h))
                            x2 = min(w, int(roi['x2'] * w))
                            y2 = min(h, int(roi['y2'] * h))
                            
                            # 유효한 ROI 확인
                            if x2 <= x1 or y2 <= y1:
                                logger.warning(f"Invalid ROI for {img_path}")
                                continue
                            
                            roi_image = image.crop((x1, y1, x2, y2)).resize(
                                (self.config['YOLO_SIZE'], self.config['YOLO_SIZE'])
                            )
                            
                            # YOLO 검출
                            results = self.yolo_model(np.array(roi_image), verbose=False)
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                confidences = results[0].boxes.conf.cpu().numpy()
                                classes = results[0].boxes.cls.cpu().numpy()
                                
                                for conf, cls in zip(confidences, classes):
                                    if conf > 0.5:
                                        obj_name = self.yolo_model.names[int(cls)]
                                        object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                                        
                            processed_count += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to process {img_path}: {str(e)}")
                            continue
                    
                    class_object_counts[class_name] = object_counts
                    logger.info(f"Processed {processed_count} samples for {class_name}")
                    
                except Exception as e:
                    logger.error(f"Error creating mapping for {class_name}: {str(e)}")
                    continue
            
            # 매핑 생성
            mapping_created = 0
            for class_name, obj_counts in class_object_counts.items():
                if obj_counts:
                    best_obj, count = max(obj_counts.items(), key=lambda x: x[1])
                    total = sum(obj_counts.values())
                    ratio = count / total
                    
                    if ratio >= self.config['MAPPING_THRESHOLD']:
                        self.class_object_mapping[class_name] = best_obj
                        mapping_created += 1
                        logger.info(f"🎯 {class_name} → {best_obj} ({ratio:.2f})")
                    else:
                        logger.warning(f"Low confidence mapping for {class_name}: {ratio:.2f}")
                        
            logger.info(f"Created {mapping_created} class-object mappings")
            
        except Exception as e:
            raise WaferDetectorError(f"Mapping creation failed: {str(e)}")
    
    def predict_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        단일 이미지 예측
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            예측 결과 딕셔너리
            
        Raises:
            WaferDetectorError: 예측 실패시
        """
        try:
            if not self.classes:
                raise WaferDetectorError("Classes not loaded. Call load_classes() first.")
            if self.classification_model is None:
                raise WaferDetectorError("Classification model not loaded")
                
            image_path = Path(image_path)
            if not image_path.exists():
                raise WaferDetectorError(f"Image not found: {image_path}")
            
            # 기본 분류
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.classification_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_idx = np.argmax(probabilities)
                confidence = float(probabilities[predicted_idx])
                predicted_class = self.classes[predicted_idx]
            
            result = {
                'image_path': str(image_path),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'method': 'classification_only'
            }
            
            # ROI 검증 조건 확인
            needs_roi = (
                predicted_class in self.difficult_classes and
                confidence < self.config['CONFIDENCE_THRESHOLD'] and
                predicted_class in self.class_object_mapping and
                predicted_class in self.roi_patterns
            )
            
            if needs_roi:
                try:
                    # ROI에서 객체 검출
                    w, h = image.size
                    roi = self.roi_patterns[predicted_class]
                    
                    x1 = max(0, int(roi['x1'] * w))
                    y1 = max(0, int(roi['y1'] * h))
                    x2 = min(w, int(roi['x2'] * w))
                    y2 = min(h, int(roi['y2'] * h))
                    
                    if x2 > x1 and y2 > y1:
                        roi_image = image.crop((x1, y1, x2, y2)).resize(
                            (self.config['YOLO_SIZE'], self.config['YOLO_SIZE'])
                        )
                        yolo_results = self.yolo_model(np.array(roi_image), verbose=False)
                        
                        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                            object_counts = {}
                            confidences = yolo_results[0].boxes.conf.cpu().numpy()
                            classes = yolo_results[0].boxes.cls.cpu().numpy()
                            
                            for conf, cls in zip(confidences, classes):
                                if conf > 0.5:
                                    obj_name = self.yolo_model.names[int(cls)]
                                    object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
                            
                            if object_counts:
                                most_detected_obj = max(object_counts.items(), key=lambda x: x[1])[0]
                                
                                # 역매핑
                                for mapped_class, mapped_obj in self.class_object_mapping.items():
                                    if mapped_obj == most_detected_obj:
                                        result.update({
                                            'predicted_class': mapped_class,
                                            'confidence': 0.9,
                                            'method': 'roi_enhanced',
                                            'detected_object': most_detected_obj,
                                            'object_counts': object_counts,
                                            'roi_coordinates': roi
                                        })
                                        break
                        
                except Exception as e:
                    logger.warning(f"ROI processing failed: {str(e)}")
            
            return result
            
        except Exception as e:
            raise WaferDetectorError(f"Image prediction failed: {str(e)}")
    
    def load_results(self, output_dir: Union[str, Path]) -> None:
        """
        저장된 결과 로드
        
        Args:
            output_dir: 결과 디렉토리 경로
            
        Raises:
            WaferDetectorError: 결과 로드 실패시
        """
        try:
            output_dir = Path(output_dir)
            
            # ROI 패턴 로드
            roi_path = output_dir / 'roi_patterns.json'
            if roi_path.exists():
                with open(roi_path, 'r') as f:
                    self.roi_patterns = json.load(f)
                logger.info(f"Loaded ROI patterns for {len(self.roi_patterns)} classes")
            
            # 클래스 매핑 로드
            mapping_path = output_dir / 'class_mapping.json'
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.difficult_classes = mapping_data.get('difficult_classes', [])
                    self.class_object_mapping = mapping_data.get('class_object_mapping', {})
                logger.info(f"Loaded mappings for {len(self.class_object_mapping)} classes")
                
        except Exception as e:
            raise WaferDetectorError(f"Failed to load results: {str(e)}")
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        결과 저장
        
        Args:
            output_dir: 출력 디렉토리 경로
            
        Raises:
            WaferDetectorError: 저장 실패시
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # ROI 패턴 저장
            roi_file = output_path / 'roi_patterns.json'
            with open(roi_file, 'w') as f:
                json.dump(self.roi_patterns, f, indent=2)
            
            # 클래스 매핑 저장
            mapping_data = {
                'difficult_classes': self.difficult_classes,
                'class_object_mapping': self.class_object_mapping,
                'f1_scores': self.f1_scores.tolist() if self.f1_scores is not None else None,
                'config': self.config
            }
            mapping_file = output_path / 'class_mapping.json'
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            
            logger.info(f"💾 Results saved to {output_path}")
            
        except Exception as e:
            raise WaferDetectorError(f"Failed to save results: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        현재 상태 통계 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        return {
            'total_classes': len(self.classes),
            'difficult_classes': len(self.difficult_classes),
            'roi_patterns': len(self.roi_patterns),
            'class_mappings': len(self.class_object_mapping),
            'device': str(self.device),
            'models_loaded': {
                'classification': self.classification_model is not None,
                'yolo': self.yolo_model is not None,
                'gradcam': self.gradcam_analyzer is not None
            }
        }
