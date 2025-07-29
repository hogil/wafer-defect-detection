#!/usr/bin/env python3
"""
🎯 WaferDetector - 웨이퍼 결함 검출 시스템
지능형 2단계 검출: 기본 분류 + ROI 기반 정밀 검증
"""

import os
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
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, confusion_matrix

from gradcam_utils import GradCAMAnalyzer, extract_roi_from_heatmap

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
        self.precision_scores: Optional[np.ndarray] = None
        self.f1_scores: Optional[np.ndarray] = None
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((config['CLASSIFICATION_SIZE'], config['CLASSIFICATION_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"WaferDetector initialized - Device: {self.device}")
    
    def load_models(self, model_path: Union[str, Path], yolo_path: Union[str, Path]) -> None:
        """
        모델 로드
        
        Args:
            model_path: 분류 모델 경로
            yolo_path: YOLO 모델 경로
            
        Raises:
            WaferDetectorError: 모델 로드 실패시
        """
        try:
            # 1. 가중치 로드 및 prefix 제거
            logger.info("Loading model weights...")
            state_dict = torch.load(model_path, map_location="cpu")
            
            # model prefix가 있는지 확인하고 제거
            if any(k.startswith('model.') for k in state_dict.keys()):
                cleaned_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                logger.info("Removed 'model.' prefix from state dict")
            else:
                cleaned_state_dict = state_dict
                logger.info("No 'model.' prefix found in state dict")
            
            # 2. 클래스 수 결정 (pth 파일 기준)
            if 'head.fc.weight' not in cleaned_state_dict:
                raise WaferDetectorError("Invalid model weights: missing head.fc.weight")
            num_classes = cleaned_state_dict['head.fc.weight'].shape[0]
            logger.info(f"Detected {num_classes} classes from model weights")

            # 3. ConvNeXtV2 모델 생성 (pth 파일의 클래스 수로)
            logger.info("Creating ConvNeXtV2 model...")
            self.classification_model = timm.create_model(
                'convnextv2_base.fcmae_ft_in22k_in1k',
                pretrained=False,  # 가중치를 직접 로드할 것이므로 False
                num_classes=num_classes
            )
            
            # 4. 가중치 로드
            self.classification_model.load_state_dict(cleaned_state_dict, strict=True)
            logger.info("Model weights loaded successfully")
            
            # 5. 분류기를 데이터셋 클래스 수로 변경 (나중에 load_classes에서 설정됨)
            # 이 부분은 load_classes에서 처리됨
            
            # YOLO 모델 로드
            if not Path(yolo_path).exists():
                raise WaferDetectorError(f"YOLO model not found: {yolo_path}")
            
            self.yolo_model = YOLO(yolo_path)
            logger.info("YOLO model loaded successfully")
            
            # GradCAM 초기화
            self.gradcam_analyzer = GradCAMAnalyzer(
                self.classification_model,
                target_layer_name=self.config['advanced']['target_layer_name']
            )
            
            # 모델을 디바이스로 이동
            self.classification_model.to(self.device)
            self.classification_model.eval()
            
            logger.info(f"Models loaded successfully - Classes: {num_classes}")
            
        except Exception as e:
            raise WaferDetectorError(f"Failed to load models: {str(e)}")
    
    def load_classes(self, dataset_root: Union[str, Path]) -> None:
        """
        데이터셋에서 클래스 로드
        
        Args:
            dataset_root: 데이터셋 루트 경로
            
        Raises:
            WaferDetectorError: 클래스 로드 실패시
        """
        try:
            dataset_root = Path(dataset_root)
            if not dataset_root.exists():
                raise WaferDetectorError(f"Dataset root not found: {dataset_root}")
            
            # ImageFolder로 클래스 로드
            dataset = datasets.ImageFolder(str(dataset_root), transform=self.transform)
            self.classes = dataset.classes
            
            # 분류기를 데이터셋 클래스 수로 변경
            if self.classification_model is not None:
                num_features = self.classification_model.head.fc.in_features
                
                # 새로운 분류기 생성 (완전히 새로 초기화)
                new_fc = nn.Linear(num_features, len(self.classes))
                nn.init.xavier_uniform_(new_fc.weight)
                nn.init.zeros_(new_fc.bias)
                
                self.classification_model.head.fc = new_fc
                logger.info(f"Classifier reset to {len(self.classes)} classes (random initialization)")
            
            logger.info(f"Classes loaded: {self.classes}")
            
        except Exception as e:
            raise WaferDetectorError(f"Failed to load classes: {str(e)}")
    
    def analyze_performance(self, dataset_root: Union[str, Path]) -> np.ndarray:
        """
        성능 분석 및 어려운 클래스 식별
        
        Args:
            dataset_root: 데이터셋 루트 경로
            
        Returns:
            Precision 스코어 배열
            
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
            
            logger.info(f"Analyzing performance on {len(dataset)} samples...")
            
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
            
            # 클래스 수 불일치 확인
            model_classes = self.classification_model.head.fc.out_features
            dataset_classes = len(self.classes)
            
            if model_classes != dataset_classes:
                logger.warning(f"Model has {model_classes} classes but dataset has {dataset_classes} classes")
                logger.warning("Performance analysis may not be accurate - using model's class indices")
                
                # 모델의 클래스 수에 맞게 예측값과 레이블 조정
                adjusted_preds = [pred % dataset_classes for pred in all_preds]
                adjusted_labels = [label % dataset_classes for label in all_labels]
                all_preds = adjusted_preds
                all_labels = adjusted_labels
            
            # 성능 지표 계산
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # 클래스별 상세 메트릭
            class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
            class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
            class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
            
            # Confusion Matrix 생성
            cm = confusion_matrix(all_labels, all_preds)
            
            # 결과 출력
            logger.info("=" * 60)
            logger.info("PERFORMANCE ANALYSIS RESULTS")
            logger.info("=" * 60)
            logger.info(f"Overall Metrics:")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info("")
            logger.info("Class-wise Metrics:")
            for i, class_name in enumerate(self.classes):
                logger.info(f"  {class_name}:")
                logger.info(f"    Precision: {class_precision[i]:.4f}")
                logger.info(f"    Recall: {class_recall[i]:.4f}")
                logger.info(f"    F1-Score: {class_f1[i]:.4f}")
            logger.info("=" * 60)
            
            # 어려운 클래스 식별 (Precision 기준)
            self.difficult_classes = [
                self.classes[i] for i, prec in enumerate(class_precision) 
                if prec < self.config['PRECISION_THRESHOLD']
            ]
            logger.info(f"Identified {len(self.difficult_classes)} difficult classes: {self.difficult_classes}")
            
            # 메트릭을 TXT 파일로 저장
            self.save_metrics_to_txt(cm, class_precision, class_recall, class_f1, precision, recall, f1)
            
            # 결과 저장
            performance_results = {
                'overall': {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                },
                'class_wise': {
                    class_name: {
                        'precision': float(class_precision[i]),
                        'recall': float(class_recall[i]),
                        'f1_score': float(class_f1[i])
                    } for i, class_name in enumerate(self.classes)
                },
                'confusion_matrix': cm.tolist(),
                'class_names': self.classes,
                'difficult_classes': self.difficult_classes
            }
            
            # Confusion Matrix 시각화 및 저장
            self.save_confusion_matrix(cm, self.classes)
            
            return performance_results
            
        except Exception as e:
            raise WaferDetectorError(f"Performance analysis failed: {str(e)}")
    
    def analyze_prediction_performance(self, predictions, dataset_path):
        """예측 결과를 기반으로 성능 분석을 수행합니다."""
        try:
            logger.info("Analyzing prediction performance...")
            
            # 실제 클래스와 예측 클래스 수집
            all_labels = []
            all_preds = []
            
            for pred in predictions:
                image_path = pred['image_path']
                predicted_class = pred['predicted_class']
                
                # 이미지 경로에서 실제 클래스 추출
                actual_class = self._extract_class_from_path(image_path, dataset_path)
                
                if actual_class in self.classes:
                    actual_idx = self.classes.index(actual_class)
                    predicted_idx = self.classes.index(predicted_class)
                    
                    all_labels.append(actual_idx)
                    all_preds.append(predicted_idx)
            
            if not all_labels:
                logger.warning("No valid labels found for performance analysis")
                return
            
            # 성능 지표 계산
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # 클래스별 상세 메트릭
            class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
            class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
            class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
            
            # Confusion Matrix 생성
            cm = confusion_matrix(all_labels, all_preds)
            
            # 결과 출력
            logger.info("=" * 60)
            logger.info("PREDICTION PERFORMANCE ANALYSIS")
            logger.info("=" * 60)
            logger.info(f"Overall Metrics:")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info("")
            logger.info("Class-wise Metrics:")
            for i, class_name in enumerate(self.classes):
                logger.info(f"  {class_name}:")
                logger.info(f"    Precision: {class_precision[i]:.4f}")
                logger.info(f"    Recall: {class_recall[i]:.4f}")
                logger.info(f"    F1-Score: {class_f1[i]:.4f}")
            logger.info("=" * 60)
            
            # 메트릭을 TXT 파일로 저장
            self.save_prediction_metrics_to_txt(cm, class_precision, class_recall, class_f1, precision, recall, f1)
            
            # Confusion Matrix 시각화 및 저장
            self.save_prediction_confusion_matrix(cm, self.classes)
            
            logger.info("Prediction performance analysis completed!")
            
        except Exception as e:
            logger.error(f"Failed to analyze prediction performance: {e}")
    
    def _extract_class_from_path(self, image_path, dataset_path):
        """이미지 경로에서 실제 클래스를 추출합니다."""
        try:
            # 상대 경로를 절대 경로로 변환
            abs_image_path = os.path.abspath(image_path)
            abs_dataset_path = os.path.abspath(dataset_path)
            
            # 데이터셋 경로를 제거하여 상대 경로 얻기
            if abs_image_path.startswith(abs_dataset_path):
                relative_path = abs_image_path[len(abs_dataset_path):].lstrip(os.sep)
                # 첫 번째 디렉토리가 클래스명
                class_name = relative_path.split(os.sep)[0]
                return class_name
            
            return None
        except Exception as e:
            logger.warning(f"Failed to extract class from path {image_path}: {e}")
            return None
    
    def save_prediction_metrics_to_txt(self, cm, class_precision, class_recall, class_f1, overall_precision, overall_recall, overall_f1):
        """예측 메트릭을 TXT 파일로 저장합니다."""
        try:
            metrics_path = os.path.join(self.config['OUTPUT_DIR'], 'prediction_metrics.txt')
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("WAFER DEFECT DETECTION - PREDICTION PERFORMANCE METRICS\n")
                f.write("=" * 80 + "\n\n")
                
                # 전체 메트릭
                f.write("OVERALL METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision: {overall_precision:.4f}\n")
                f.write(f"Recall: {overall_recall:.4f}\n")
                f.write(f"F1-Score: {overall_f1:.4f}\n\n")
                
                # 클래스별 메트릭
                f.write("CLASS-WISE METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Correct':<10} {'Total':<10}\n")
                f.write("-" * 80 + "\n")
                
                for i, class_name in enumerate(self.classes):
                    # Confusion Matrix에서 TP, FP, FN 계산
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    total = cm[i, :].sum()
                    
                    f.write(f"{class_name:<20} {class_precision[i]:<12.4f} {class_recall[i]:<12.4f} {class_f1[i]:<12.4f} {tp:<10} {total:<10}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("CONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                
                # Confusion Matrix 출력
                f.write(f"{'':<15}")
                for class_name in self.classes:
                    f.write(f"{class_name:<10}")
                f.write("\n")
                
                for i, class_name in enumerate(self.classes):
                    f.write(f"{class_name:<15}")
                    for j in range(len(self.classes)):
                        f.write(f"{cm[i, j]:<10}")
                    f.write("\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision Threshold: {self.config['PRECISION_THRESHOLD']}\n")
                f.write(f"Confidence Threshold: {self.config['CONFIDENCE_THRESHOLD']}\n")
                f.write(f"Mapping Threshold: {self.config['MAPPING_THRESHOLD']}\n")
                f.write(f"Mapping Ratio Threshold: {self.config['MAPPING_RATIO_THRESHOLD']}\n")
            
            logger.info(f"Prediction metrics saved to: {metrics_path}")
            
        except Exception as e:
            logger.error(f"Failed to save prediction metrics to TXT: {e}")
    
    def save_prediction_confusion_matrix(self, cm, class_names):
        """예측 결과 기반 Confusion Matrix를 시각화하고 저장합니다."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Confusion Matrix 시각화
            plt.figure(figsize=(12, 10))
            
            # Confusion Matrix만 표시
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Prediction Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # 저장
            cm_path = os.path.join(self.config['OUTPUT_DIR'], 'prediction_confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Prediction confusion matrix saved to: {cm_path}")
            
        except Exception as e:
            logger.error(f"Failed to save prediction confusion matrix: {e}")
    
    def learn_roi_patterns(self):
        """모든 클래스에 대해 ROI 패턴을 학습합니다."""
        try:
            logger.info(f"Learning ROI patterns for {len(self.classes)} classes...")
            
            for class_name in self.classes:
                class_dir = os.path.join(self.config['DATASET_ROOT'], class_name)
                if not os.path.exists(class_dir):
                    logger.warning(f"Class directory not found: {class_dir}")
                    continue
                
                # 클래스별 샘플 수 제한
                max_samples = self.config['processing']['max_roi_samples']
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                image_files = image_files[:max_samples]
                
                if not image_files:
                    logger.warning(f"No images found for class {class_name}")
                    continue
                
                class_idx = self.classes.index(class_name)
                roi_patterns = []
                
                for image_file in image_files:
                    try:
                        img_path = os.path.join(class_dir, image_file)
                        image = Image.open(img_path).convert('RGB')
                        
                        # GradCAM으로 ROI 추출
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        input_tensor = transform(image).unsqueeze(0)
                        
                        heatmap = self.gradcam_analyzer.generate_gradcam(input_tensor, class_idx)
                        roi_coords = extract_roi_from_heatmap(heatmap)
                        
                        roi_patterns.append({
                            'x1': roi_coords[0],
                            'y1': roi_coords[1], 
                            'x2': roi_coords[2],
                            'y2': roi_coords[3]
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {image_file}: {e}")
                        continue
                
                if roi_patterns:
                    # ROI 좌표 평균 계산
                    avg_roi = {
                        'x1': sum(r['x1'] for r in roi_patterns) / len(roi_patterns),
                        'y1': sum(r['y1'] for r in roi_patterns) / len(roi_patterns),
                        'x2': sum(r['x2'] for r in roi_patterns) / len(roi_patterns),
                        'y2': sum(r['y2'] for r in roi_patterns) / len(roi_patterns)
                    }
                    self.roi_patterns[class_name] = avg_roi
                    logger.info(f"{class_name}: ROI learned from {len(roi_patterns)} samples")
                else:
                    logger.warning(f"No valid ROI patterns found for {class_name}")
            
            logger.info(f"ROI patterns learned for {len(self.roi_patterns)} classes")
            
        except Exception as e:
            logger.error(f"ROI pattern learning failed: {str(e)}")
            raise WaferDetectorError(f"ROI pattern learning failed: {str(e)}")
    
    def create_mapping(self):
        """모든 클래스에 대해 GradCAM attention 영역에서 객체 매핑을 생성합니다."""
        try:
            logger.info("Creating object mappings from GradCAM attention regions...")
            
            # 클래스별 객체 카운트 수집
            class_object_counts = {class_name: {} for class_name in self.classes}
            
            for class_name in self.classes:
                class_dir = os.path.join(self.config['DATASET_ROOT'], class_name)
                if not os.path.exists(class_dir):
                    logger.warning(f"Class directory not found: {class_dir}")
                    continue
                
                # 클래스별 샘플 수 제한
                max_samples = self.config['processing']['max_mapping_samples']
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                image_files = image_files[:max_samples]
                
                if not image_files:
                    logger.warning(f"No images found for {class_name}")
                    continue
                
                class_idx = self.classes.index(class_name)
                processed_count = 0
                
                for image_file in image_files:
                    try:
                        img_path = os.path.join(class_dir, image_file)
                        image = Image.open(img_path).convert('RGB')
                        
                        # GradCAM으로 attention 영역 추출
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        input_tensor = transform(image).unsqueeze(0)
                        
                        heatmap = self.gradcam_analyzer.generate_gradcam(input_tensor, class_idx)
                        roi_coords = extract_roi_from_heatmap(heatmap)
                        
                        # ROI 영역에서 객체 검출
                        w, h = image.size
                        x1 = max(0, int(roi_coords[0] * w))
                        y1 = max(0, int(roi_coords[1] * h))
                        x2 = min(w, int(roi_coords[2] * w))
                        y2 = min(h, int(roi_coords[3] * h))
                        
                        if x2 > x1 and y2 > y1:
                            roi_image = image.crop((x1, y1, x2, y2))
                            results = self.yolo_model(roi_image, verbose=False)
                            
                            # 검출된 객체 카운트
                            for result in results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        if box.conf > self.config['MAPPING_THRESHOLD']:
                                            obj_class = int(box.cls.item())
                                            obj_name = self.yolo_model.names[obj_class]
                                            
                                            if obj_name not in class_object_counts[class_name]:
                                                class_object_counts[class_name][obj_name] = 0
                                            class_object_counts[class_name][obj_name] += 1
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {image_file}: {e}")
                        continue
                
                logger.info(f"Processed {processed_count} samples for {class_name}")
            
            # 매핑 생성 (비율 기반)
            mapping_created = 0
            for class_name, obj_counts in class_object_counts.items():
                if obj_counts:
                    total_detections = sum(obj_counts.values())
                    best_obj, count = max(obj_counts.items(), key=lambda x: x[1])
                    ratio = count / total_detections
                    
                    # 비율이 임계값 이상인 경우만 매핑
                    if ratio >= self.config['MAPPING_RATIO_THRESHOLD']:
                        self.class_object_mapping[class_name] = best_obj
                        mapping_created += 1
                        logger.info(f"{class_name} -> {best_obj} (ratio: {ratio:.2f}, count: {count}/{total_detections})")
                    else:
                        logger.warning(f"Low ratio mapping for {class_name}: {ratio:.2f} < {self.config['MAPPING_RATIO_THRESHOLD']}")
                else:
                    # 객체가 검출되지 않았을 때 랜덤 매핑 생성
                    import random
                    available_objects = list(self.yolo_model.names.values())
                    if available_objects:
                        random_obj = random.choice(available_objects)
                        self.class_object_mapping[class_name] = random_obj
                        mapping_created += 1
                        logger.info(f"{class_name} -> {random_obj} (random assignment - no objects detected)")
                    else:
                        logger.warning(f"No available objects for {class_name}")
            
            logger.info(f"Created {mapping_created} mappings")
            
        except Exception as e:
            logger.error(f"Failed to create mappings: {e}")
            raise
    
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
                
                # 모델의 클래스 수와 데이터셋 클래스 수가 다를 때 처리
                if predicted_idx >= len(self.classes):
                    logger.warning(f"Predicted class index {predicted_idx} exceeds dataset classes {len(self.classes)}")
                    # 모델의 클래스 인덱스를 데이터셋 클래스로 매핑
                    predicted_idx = predicted_idx % len(self.classes)
                
                predicted_class = self.classes[predicted_idx]
            
            # ROI Enhanced 예측 (신뢰도가 낮고 어려운 클래스인 경우)
            if confidence < self.config['CONFIDENCE_THRESHOLD'] and predicted_class in self.difficult_classes:
                logger.info(f"Low confidence prediction ({confidence:.3f}) for difficult class {predicted_class}")
                
                # ROI 영역에서 객체 검출
                roi_objects = self._detect_objects_in_roi(image, predicted_class)
                
                # 객체 검출 실패 시 원래 classification 결과 반환
                if not roi_objects:
                    logger.info(f"No objects detected in ROI, returning original classification: {predicted_class}")
                    return {
                        'image_path': str(image_path),
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'method': 'classification',
                        'detected_object': None,
                        'roi_objects': []
                    }
                
                # 매핑된 객체와 비교
                if predicted_class in self.class_object_mapping:
                    mapped_object = self.class_object_mapping[predicted_class]
                    if mapped_object in roi_objects:
                        logger.info(f"ROI Enhanced: {predicted_class} -> {mapped_object} (confidence: {confidence:.3f})")
                        return {
                            'image_path': str(image_path),
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'method': 'roi_enhanced',
                            'detected_object': mapped_object,
                            'roi_objects': roi_objects
                        }
                    else:
                        # 검출된 객체에 따라 다른 클래스로 예측 변경
                        for detected_obj in roi_objects:
                            for class_name, mapped_obj in self.class_object_mapping.items():
                                if detected_obj == mapped_obj:
                                    logger.info(f"ROI Enhanced: {predicted_class} -> {class_name} (via {detected_obj}) (confidence: {confidence:.3f})")
                                    return {
                                        'image_path': str(image_path),
                                        'predicted_class': class_name, # 예측 클래스를 변경
                                        'confidence': confidence,
                                        'method': 'roi_enhanced',
                                        'detected_object': detected_obj,
                                        'roi_objects': roi_objects
                                    }
                
                # 매핑 테이블에 해당 클래스가 없거나 매핑된 객체가 없는 경우 원래 결과 반환
                logger.info(f"No matching mapping found, returning original classification: {predicted_class}")
                return {
                    'image_path': str(image_path),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'method': 'classification',
                    'detected_object': None,
                    'roi_objects': roi_objects
                }
            
            return {
                'image_path': str(image_path),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'method': 'classification_only'
            }
            
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
                'precision_scores': self.precision_scores.tolist() if self.precision_scores is not None else None,
                'f1_scores': self.f1_scores.tolist() if self.f1_scores is not None else None,
                'config': self.config
            }
            mapping_file = output_path / 'class_mapping.json'
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
            
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

    def save_confusion_matrix(self, cm, class_names):
        """Confusion Matrix를 시각화하고 저장합니다."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Confusion Matrix 시각화
            plt.figure(figsize=(12, 10))
            
            # Confusion Matrix만 표시
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # 저장
            cm_path = os.path.join(self.config['OUTPUT_DIR'], 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to: {cm_path}")
            
        except Exception as e:
            logger.error(f"Failed to save confusion matrix: {e}")

    def _detect_objects_in_roi(self, image, predicted_class):
        """ROI 영역에서 객체를 검출합니다."""
        try:
            if predicted_class not in self.roi_patterns:
                logger.warning(f"No ROI pattern for {predicted_class}")
                return []
            
            # ROI 좌표 계산
            w, h = image.size
            roi = self.roi_patterns[predicted_class]
            
            x1 = max(0, int(roi['x1'] * w))
            y1 = max(0, int(roi['y1'] * h))
            x2 = min(w, int(roi['x2'] * w))
            y2 = min(h, int(roi['y2'] * h))
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid ROI coordinates: {roi}")
                return []
            
            # ROI 이미지 추출 및 YOLO 검출
            roi_image = image.crop((x1, y1, x2, y2)).resize(
                (self.config['YOLO_SIZE'], self.config['YOLO_SIZE'])
            )
            yolo_results = self.yolo_model(np.array(roi_image), verbose=False)
            
            detected_objects = []
            if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
                confidences = yolo_results[0].boxes.conf.cpu().numpy()
                classes = yolo_results[0].boxes.cls.cpu().numpy()
                
                for conf, cls in zip(confidences, classes):
                    if conf > self.config['processing']['yolo_confidence_threshold']:
                        obj_name = self.yolo_model.names[int(cls)]
                        detected_objects.append(obj_name)
                
                logger.info(f"Detected objects in ROI: {detected_objects}")
            else:
                logger.info("No objects detected in ROI")
            
            return detected_objects
            
        except Exception as e:
            logger.warning(f"ROI object detection failed: {str(e)}")
            return []

    def save_metrics_to_txt(self, cm, class_precision, class_recall, class_f1, overall_precision, overall_recall, overall_f1):
        """메트릭을 TXT 파일로 저장합니다."""
        try:
            metrics_path = os.path.join(self.config['OUTPUT_DIR'], 'performance_metrics.txt')
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("WAFER DEFECT DETECTION - PERFORMANCE METRICS\n")
                f.write("=" * 80 + "\n\n")
                
                # 전체 메트릭
                f.write("OVERALL METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision: {overall_precision:.4f}\n")
                f.write(f"Recall: {overall_recall:.4f}\n")
                f.write(f"F1-Score: {overall_f1:.4f}\n\n")
                
                # 클래스별 메트릭
                f.write("CLASS-WISE METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Correct':<10} {'Total':<10}\n")
                f.write("-" * 80 + "\n")
                
                for i, class_name in enumerate(self.classes):
                    # Confusion Matrix에서 TP, FP, FN 계산
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    total = cm[i, :].sum()
                    
                    f.write(f"{class_name:<20} {class_precision[i]:<12.4f} {class_recall[i]:<12.4f} {class_f1[i]:<12.4f} {tp:<10} {total:<10}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("CONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                
                # Confusion Matrix 출력
                f.write(f"{'':<15}")
                for class_name in self.classes:
                    f.write(f"{class_name:<10}")
                f.write("\n")
                
                for i, class_name in enumerate(self.classes):
                    f.write(f"{class_name:<15}")
                    for j in range(len(self.classes)):
                        f.write(f"{cm[i, j]:<10}")
                    f.write("\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("DIFFICULT CLASSES:\n")
                f.write("-" * 40 + "\n")
                for class_name in self.difficult_classes:
                    f.write(f"- {class_name}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision Threshold: {self.config['PRECISION_THRESHOLD']}\n")
                f.write(f"Confidence Threshold: {self.config['CONFIDENCE_THRESHOLD']}\n")
                f.write(f"Mapping Threshold: {self.config['MAPPING_THRESHOLD']}\n")
                f.write(f"Mapping Ratio Threshold: {self.config['MAPPING_RATIO_THRESHOLD']}\n")
            
            logger.info(f"Performance metrics saved to: {metrics_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics to TXT: {e}")
