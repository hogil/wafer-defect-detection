#!/usr/bin/env python3
"""
🎯 Enhanced Wafer Defect Detection - 통합 훈련+추론 시스템
torchvision.datasets.ImageFolder 기반 단순화
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import cv2

from roi_utils import ROIExtractor
from gradcam_utils import GradCAMAnalyzer

# 성능 분석을 위한 추가 import
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import ConfigManager, get_production_config, get_quick_test_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaferTrainer:
    """🎯 웨이퍼 불량 검출 시스템 (추론 전용)"""
    
    def __init__(self, dataset_root: str, config_manager: ConfigManager = None):
        self.dataset_root = Path(dataset_root)
        
        # 설정 관리자
        if config_manager is None:
            self.config_manager = ConfigManager()
            self.config_manager.update_dataset_path(str(dataset_root))
        else:
            self.config_manager = config_manager
        
        self.config = self.config_manager.get_config()
        # device 설정 (항상 GPU 사용, 없으면 에러)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise RuntimeError('CUDA(GPU)가 필요합니다. GPU 환경에서 실행하세요!')
        
        # 출력 디렉토리 생성
        self.output_dir = Path(self.config.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 관련
        self.model = None
        self.classes = []
        self.num_classes = 0
        self.class_to_idx = {}
        
        # ROI 관련
        self.yolo_model = None
        self.difficult_classes = []
        self.class_object_mapping = {}
        self.yolo_objects = []
        self.num_yolo_objects = 0
        
        # ROI 추출기 초기화 (클래스별 패턴 파일 지정)
        roi_patterns_file = self.output_dir / "class_roi_patterns.json"
        self.roi_extractor = ROIExtractor(str(roi_patterns_file))
        
        # Grad-CAM 분석기 (나중에 초기화)
        self.gradcam_analyzer = None
        
        logger.info("🎯 WaferTrainer initialized (Inference Only)")
        logger.info(f"  Dataset: {self.dataset_root}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
    
    def _create_datasets(self, train_ratio: float = 0.7):
        """ImageFolder로 데이터셋 생성"""
        
        print("📂 Creating datasets with ImageFolder...")
        
        # 전체 데이터셋 로드
        full_dataset = datasets.ImageFolder(
            root=self.dataset_root,
            transform=transforms.Compose([
                transforms.Resize((self.config.CLASSIFICATION_SIZE, self.config.CLASSIFICATION_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        
        # 클래스 정보 추출
        self.classes = full_dataset.classes
        self.num_classes = len(self.classes)
        self.class_to_idx = full_dataset.class_to_idx
        
        print(f"✅ Discovered {self.num_classes} classes: {self.classes}")
        
        # Train/Val 분할
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # 데이터 로더 생성 (추론용, augmentation 없음)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,  # 추론이므로 shuffle=False
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE * 2, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        print(f"📊 Dataset split:")
        print(f"  - Train: {train_size} samples")
        print(f"  - Val: {val_size} samples")
        
        return train_dataset, val_dataset
    
    def _create_model(self):
        """ConvNeXtV2 모델 생성"""
        
        print(f"🤖 Creating ConvNeXtV2 model...")
        
        self.model = timm.create_model(
            self.config.CONVNEXT_MODEL_NAME,
            pretrained=False,  # 별도 가중치 로드
            num_classes=self.num_classes
        )
        
        # 사전 훈련된 가중치 로드
        pretrained_path = Path(self.config.CONVNEXT_PRETRAINED_MODEL)
        weights_loaded = False
        
        if pretrained_path.exists():
            print(f"🔄 Loading pretrained weights from: {pretrained_path}")
            pretrained_weights = torch.load(pretrained_path, map_location=self.device)
            
            # model. prefix 제거 (있을 경우)
            clean_pretrained_weights = {}
            for key, value in pretrained_weights.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # "model." 제거
                    clean_pretrained_weights[new_key] = value
                else:
                    clean_pretrained_weights[key] = value
            
            # 전체 레이어 strict=True로 로드 (헤드 포함)
            self.model.load_state_dict(clean_pretrained_weights, strict=True)
            print(f"✅ Pretrained weights loaded: {len(clean_pretrained_weights)} layers")
            weights_loaded = True
        else:
            print(f"⚠️ Pretrained weights not found: {pretrained_path}")
            print("   Cannot proceed without pretrained weights!")
            return False
        
        self.model.to(self.device)
        self.model.eval()  # 추론 모드로 설정
        
        print(f"✅ Model created:")
        print(f"  - Architecture: {self.config.CONVNEXT_MODEL_NAME}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Image size: {self.config.CLASSIFICATION_SIZE}")
        print(f"  - Pretrained: {'Yes' if weights_loaded else 'No'}")
        
        return True
    
    def _learn_class_roi_patterns(self):
        """Grad-CAM을 사용하여 각 클래스별 ROI 패턴 학습"""
        
        print("\n🔍 Learning class-specific ROI patterns using Grad-CAM...")
        
        # Grad-CAM 분석기 초기화
        if self.gradcam_analyzer is None:
            # ConvNeXtV2의 마지막 feature layer 지정
            target_layer = "stages.3.blocks.2.norm"  # ConvNeXtV2 마지막 norm layer
            self.gradcam_analyzer = GradCAMAnalyzer(self.model, target_layer_name=target_layer)
        
        # 각 클래스별로 ROI 패턴 분석
        class_roi_patterns = self.gradcam_analyzer.analyze_class_attention_patterns(
            self.val_loader, 
            self.classes, 
            num_samples_per_class=10
        )
        
        # 각 클래스의 대표 ROI 계산 및 저장
        for class_name, roi_patterns in class_roi_patterns.items():
            if roi_patterns:
                # 중간값(median) 방식으로 대표 ROI 계산
                representative_roi = self.gradcam_analyzer.get_representative_roi_for_class(
                    roi_patterns, method='median'
                )
                
                # 상대 좌표로 변환 (classification 이미지 크기 기준)
                classification_size = getattr(self.config, 'CLASSIFICATION_SIZE', 384)
                x1_ratio = representative_roi[0] / classification_size
                y1_ratio = representative_roi[1] / classification_size
                x2_ratio = representative_roi[2] / classification_size
                y2_ratio = representative_roi[3] / classification_size
                
                # ROI 패턴 저장
                self.roi_extractor.set_class_roi_pattern(
                    class_name, (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
                )
                
                print(f"📍 Learned ROI for '{class_name}': ({x1_ratio:.3f},{y1_ratio:.3f}) to ({x2_ratio:.3f},{y2_ratio:.3f})")
        
        # 학습된 패턴을 파일에 저장
        roi_patterns_file = self.output_dir / "class_roi_patterns.json"
        self.roi_extractor.save_class_roi_patterns(str(roi_patterns_file))
        
        print(f"✅ ROI patterns saved to {roi_patterns_file}")
    
    def _extract_roi_with_learned_pattern(self, original_image_path: str, predicted_class: str) -> np.ndarray:
        """학습된 클래스별 ROI 패턴을 사용하여 ROI 추출"""
        yolo_size = getattr(self.config, 'YOLO_INPUT_SIZE', 1024)
        
        return self.roi_extractor.crop_roi_from_original(
            original_image_path, 
            predicted_class,
            target_size=yolo_size
        )
    
    def _load_yolo_model(self):
        """YOLO 모델 로드 및 객체 종류 추출"""
        
        try:
            self.yolo_model = YOLO(self.config.DETECTION_MODEL)
            
            if hasattr(self.yolo_model, 'names'):
                self.yolo_objects = list(self.yolo_model.names.values())
                self.num_yolo_objects = len(self.yolo_objects)
                print(f"🎯 YOLO model loaded: {self.config.DETECTION_MODEL}")
                print(f"  Objects: {self.num_yolo_objects} classes")
                print(f"  Examples: {self.yolo_objects[:10]}...")  # 처음 10개만 표시
            else:
                print("⚠️ Could not extract YOLO object names")
                self.yolo_objects = []
                self.num_yolo_objects = 0
                
        except Exception as e:
            print(f"⚠️ YOLO model loading failed: {e}")
            self.yolo_model = None
            self.yolo_objects = []
            self.num_yolo_objects = 0
    
    def run_inference_pipeline(self):
        """🎯 추론 파이프라인 실행 (학습 없음)"""
        
        print("\n🎯 Enhanced Wafer Defect Detection - Inference Pipeline")
        print("=" * 60)
        
        # 1. 모델 생성 및 가중치 로드
        if not self._create_model():
            return
        
        # 2. 데이터셋 생성
        self._create_datasets()
        
        # 3. Classification 전수 실행 및 성능 분석
        print("\n📊 STAGE 1: Classification Only Performance Analysis")
        print("-" * 50)
        
        train_cls_metrics = self._evaluate_dataset(self.train_loader, "Train-ClassificationOnly")
        val_cls_metrics = self._evaluate_dataset(self.val_loader, "Validation-ClassificationOnly")
        
        # 4. 어려운 클래스 식별
        self._identify_difficult_classes(val_cls_metrics)
        
        # 5. Grad-CAM으로 attention map 학습
        print("\n🧠 STAGE 2: Grad-CAM Attention Pattern Learning")
        print("-" * 50)
        self._learn_class_roi_patterns()
        
        # 6. YOLO 모델 로드
        self._load_yolo_model()
        
        # 7. ROI 매핑 생성
        print("\n🔗 STAGE 3: ROI Object Mapping Creation")
        print("-" * 50)
        self._create_roi_mappings()
        
        # 8. ROI Enhanced 성능 분석
        print("\n📊 STAGE 4: ROI Enhanced Performance Analysis")
        print("-" * 50)
        
        train_roi_metrics = self._evaluate_dataset_with_roi(self.train_loader, "Train-ROIEnhanced")
        val_roi_metrics = self._evaluate_dataset_with_roi(self.val_loader, "Validation-ROIEnhanced")
        
        # 9. 성능 비교 및 리포트 저장
        print("\n📋 STAGE 5: Performance Comparison & Report")
        print("-" * 50)
        
        self._save_comprehensive_performance_report(
            train_cls_metrics, val_cls_metrics,
            train_roi_metrics, val_roi_metrics
        )
        
        # 10. 클래스 정보 저장
        self._save_class_info()
        
        print("\n🎉 Inference Pipeline Completed!")
        print("=" * 60)
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Performance report: {self.output_dir}/performance_report.json")
        print(f"🧠 ROI patterns: {self.output_dir}/class_roi_patterns.json")
        print(f"🔗 Object mappings: {self.output_dir}/discovered_mappings.json")
    
    def _save_class_info(self):
        """클래스 정보 저장"""
        
        class_info = {
            'classes': self.classes,
            'num_classes': self.num_classes,
            'class_to_idx': self.class_to_idx,
            'config': {
                'CLASSIFICATION_SIZE': self.config.CLASSIFICATION_SIZE,
                'CONVNEXT_MODEL_NAME': self.config.CONVNEXT_MODEL_NAME,
                'F1_THRESHOLD': self.config.F1_THRESHOLD
            }
        }
        
        info_path = self.output_dir / 'class_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(class_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Class info saved: {info_path}")
    
    def _identify_difficult_classes(self, val_metrics: Dict[str, Any]):
        """어려운 클래스 식별"""
        
        print("\n🎯 Identifying difficult classes...")
        
        f1_scores = val_metrics.get('class_f1_scores', [])
        self.difficult_classes = []
        
        for i, f1 in enumerate(f1_scores):
            if f1 < self.config.F1_THRESHOLD:
                class_name = self.classes[i]
                self.difficult_classes.append(class_name)
                print(f"  ⚠️ Difficult class: {class_name} (F1 = {f1:.3f})")
        
        print(f"✅ Found {len(self.difficult_classes)} difficult classes")
    
    def _analyze_validation_performance(self):
        """2단계 성능 분석: Classification Only → ROI Enhanced"""
        
        print("📊 Comprehensive Performance Analysis (2-Stage)")
        print("=" * 60)
        
        # === 1단계: Classification Only 성능 ===
        print("\n🤖 STAGE 1: Classification Only Performance")
        print("-" * 50)
        
        train_metrics_cls = self._evaluate_dataset(self.train_loader, "Train-ClassificationOnly")
        val_metrics_cls = self._evaluate_dataset(self.val_loader, "Validation-ClassificationOnly")
        
        # 어려운 클래스 식별 (Validation F1 기준)
        self.difficult_classes = []
        for i, f1 in enumerate(val_metrics_cls['f1_scores']):
            if f1 < self.config.F1_THRESHOLD:
                self.difficult_classes.append(self.classes[i])
        
        print(f"\n🎯 Difficult classes identified: {len(self.difficult_classes)}")
        for class_name in self.difficult_classes:
            idx = self.classes.index(class_name)
            print(f"  - {class_name}: F1 = {val_metrics_cls['f1_scores'][idx]:.3f}")
        
        # === 2단계: ROI Enhanced 성능 (YOLO 로드 후) ===
        print(f"\n🔍 STAGE 2: ROI Enhanced Performance")
        print("-" * 50)
        
        # 클래스별 ROI 패턴 학습 (Grad-CAM 기반)
        print(f"\n🧠 Learning class-specific ROI patterns...")
        self._learn_class_roi_patterns()
        
        # YOLO 모델 로드 및 ROI 매핑 생성
        self._load_yolo_model()
        self._create_roi_mappings()
        
        if len(self.class_object_mapping) > 0:
            print(f"✅ ROI mappings found: {len(self.class_object_mapping)} classes")
            
            # ROI 적용된 성능 분석
            train_metrics_roi = self._evaluate_dataset_with_roi(self.train_loader, "Train-ROIEnhanced")
            val_metrics_roi = self._evaluate_dataset_with_roi(self.val_loader, "Validation-ROIEnhanced")
            
            # 성능 향상 비교
            self._compare_performance(val_metrics_cls, val_metrics_roi)
            
            # 틀린 예측 상세 분석 및 저장
            self._analyze_and_save_incorrect_predictions(val_metrics_roi)
            
            # 전체 성능 리포트 저장 (Both stages)
            self._save_comprehensive_performance_report(
                train_metrics_cls, val_metrics_cls,
                train_metrics_roi, val_metrics_roi
            )
        else:
            print("⚠️ No ROI mappings created - using Classification only")
            self._save_performance_report(train_metrics_cls, val_metrics_cls)
    
    def _evaluate_dataset(self, dataloader: DataLoader, split_name: str) -> Dict[str, Any]:
        """데이터셋 성능 평가 및 시각화"""
        
        print(f"\n📈 Evaluating {split_name} Dataset...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 메트릭 계산
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1_scores, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Weighted averages 계산
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1_scores, weights=support)
        
        # 클래스별 정답/전체 개수 계산
        class_corrects = []
        class_totals = []
        for i in range(len(self.classes)):
            class_mask = np.array(all_labels) == i
            class_total = np.sum(class_mask)
            class_correct = np.sum((np.array(all_preds)[class_mask]) == i)
            class_corrects.append(class_correct)
            class_totals.append(class_total)
        
        # Confusion Matrix 생성 및 저장
        cm = confusion_matrix(all_labels, all_preds)
        self._plot_confusion_matrix(cm, split_name)
        
        # 클래스별 상세 리포트 출력
        print(f"\n📊 {split_name} Performance Summary:")
        print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Weighted F1-Score: {weighted_f1:.4f}")
        
        print(f"\n📋 {split_name} Class-wise Metrics:")
        print("-" * 90)
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Correct':<8} {'Total':<8} {'Support':<8}")
        print("-" * 90)
        
        for i, class_name in enumerate(self.classes):
            # 배열 크기 안전 체크
            prec = precision[i] if i < len(precision) else 0.0
            rec = recall[i] if i < len(recall) else 0.0
            f1 = f1_scores[i] if i < len(f1_scores) else 0.0
            sup = support[i] if i < len(support) else 0
            
            print(f"{class_name:<20} {prec:<10.3f} {rec:<10.3f} "
                  f"{f1:<10.3f} {class_corrects[i]:<8} {class_totals[i]:<8} {sup:<8}")
        
        print("-" * 90)
        print(f"{'Weighted Avg':<20} {weighted_precision:<10.3f} {weighted_recall:<10.3f} "
              f"{weighted_f1:<10.3f} {sum(class_corrects):<8} {sum(class_totals):<8} {np.sum(support):<8}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_scores': f1_scores,
            'support': support,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'class_corrects': class_corrects,
            'class_totals': class_totals,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    def _evaluate_dataset_with_roi(self, dataloader: DataLoader, split_name: str) -> Dict[str, Any]:
        """ROI를 적용한 데이터셋 성능 평가"""
        
        print(f"\n📈 Evaluating {split_name} Dataset (with ROI)...")
        
        self.model.eval()
        all_preds_initial = []  # Classification만의 예측
        all_preds_final = []    # ROI 적용 후 최종 예측
        all_labels = []
        roi_usage_stats = {'used': 0, 'success': 0, 'total': 0}
        detailed_predictions = []  # 상세 예측 정보
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                for i in range(len(images)):
                    # 1. Classification 예측
                    single_image = images[i].unsqueeze(0)
                    outputs = self.model(single_image)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    initial_pred_idx = np.argmax(probabilities)
                    initial_confidence = float(probabilities[initial_pred_idx])
                    initial_pred_class = self.classes[initial_pred_idx]
                    
                    true_label_idx = labels[i].item()
                    true_class = self.classes[true_label_idx]
                    
                    # 2. ROI 클래스 변경 적용
                    final_pred_class = initial_pred_class
                    final_pred_idx = initial_pred_idx
                    detected_object = None
                    roi_used = False
                    roi_success = False
                    
                    # ROI 조건 확인
                    needs_roi = (
                        initial_confidence < self.config.CONFIDENCE_THRESHOLD and
                        len(self.class_object_mapping) > 0
                    )
                    
                    if needs_roi:
                        roi_used = True
                        roi_usage_stats['used'] += 1
                        
                        # 이미지를 PIL로 변환하여 ROI 영역만 YOLO 실행
                        image_np = single_image[0].permute(1, 2, 0).cpu().numpy()
                        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                        
                        try:
                            # ROI 영역만 추출해서 YOLO 실행
                            roi_image = self._extract_roi_region(image_np)
                            results = self.yolo_model(roi_image, verbose=False)
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                confidences = results[0].boxes.conf.cpu().numpy()
                                classes = results[0].boxes.cls.cpu().numpy()
                                
                                valid_indices = confidences > self.config.OBJECT_CONFIDENCE_THRESHOLD
                                if np.any(valid_indices):
                                    # 신뢰도 기준 필터링된 객체들 중에서 개수가 가장 많은 종류 선택
                                    valid_classes = classes[valid_indices]
                                    
                                    # 각 객체 클래스별 개수 세기
                                    unique_classes, counts = np.unique(valid_classes, return_counts=True)
                                    
                                    # 가장 많이 검출된 객체 클래스 선택
                                    most_frequent_idx = np.argmax(counts)
                                    detected_class_idx = int(unique_classes[most_frequent_idx])
                                    object_count = counts[most_frequent_idx]
                                    
                                    if detected_class_idx < len(self.yolo_objects):
                                        detected_object = self.yolo_objects[detected_class_idx]
                                        print(f"🎯 Most frequent object: '{detected_object}' (count: {object_count})")
                                        
                                        # 역매핑으로 클래스 찾기
                                        for class_name, mapped_object in self.class_object_mapping.items():
                                            if mapped_object == detected_object:
                                                final_pred_class = class_name
                                                final_pred_idx = self.classes.index(class_name)
                                                roi_success = True
                                                roi_usage_stats['success'] += 1
                                                break
                        except Exception:
                            pass
                    
                    # 결과 저장
                    all_preds_initial.append(initial_pred_idx)
                    all_preds_final.append(final_pred_idx)
                    all_labels.append(true_label_idx)
                    roi_usage_stats['total'] += 1
                    
                    # 상세 정보 저장
                    detailed_predictions.append({
                        'batch_idx': batch_idx,
                        'image_idx': i,
                        'true_class': true_class,
                        'initial_pred_class': initial_pred_class,
                        'initial_confidence': initial_confidence,
                        'detected_object': detected_object,
                        'final_pred_class': final_pred_class,
                        'roi_used': roi_used,
                        'roi_success': roi_success,
                        'correct_initial': (initial_pred_idx == true_label_idx),
                        'correct_final': (final_pred_idx == true_label_idx)
                    })
        
        # 최종 예측 기준으로 메트릭 계산
        accuracy = accuracy_score(all_labels, all_preds_final)
        precision, recall, f1_scores, support = precision_recall_fscore_support(
            all_labels, all_preds_final, average=None, zero_division=0
        )
        
        # Weighted averages 계산
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1_scores, weights=support)
        
        # 클래스별 정답/전체 개수 계산
        class_corrects = []
        class_totals = []
        for i in range(len(self.classes)):
            class_mask = np.array(all_labels) == i
            class_total = np.sum(class_mask)
            class_correct = np.sum((np.array(all_preds_final)[class_mask]) == i)
            class_corrects.append(class_correct)
            class_totals.append(class_total)
        
        # Confusion Matrix 생성 및 저장
        cm = confusion_matrix(all_labels, all_preds_final)
        self._plot_confusion_matrix(cm, split_name)
        
        # ROI 통계 출력
        print(f"\n📊 {split_name} Performance Summary:")
        print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Weighted F1-Score: {weighted_f1:.4f}")
        print(f"  ROI Usage: {roi_usage_stats['used']}/{roi_usage_stats['total']} ({roi_usage_stats['used']/roi_usage_stats['total']*100:.1f}%)")
        print(f"  ROI Success: {roi_usage_stats['success']}/{roi_usage_stats['used']} ({roi_usage_stats['success']/roi_usage_stats['used']*100:.1f}% of used)" if roi_usage_stats['used'] > 0 else "  ROI Success: 0%")
        
        # 클래스별 상세 리포트 출력
        print(f"\n📋 {split_name} Class-wise Metrics:")
        print("-" * 90)
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Correct':<8} {'Total':<8} {'Support':<8}")
        print("-" * 90)
        
        for i, class_name in enumerate(self.classes):
            # 배열 크기 안전 체크
            prec = precision[i] if i < len(precision) else 0.0
            rec = recall[i] if i < len(recall) else 0.0
            f1 = f1_scores[i] if i < len(f1_scores) else 0.0
            sup = support[i] if i < len(support) else 0
            
            print(f"{class_name:<20} {prec:<10.3f} {rec:<10.3f} "
                  f"{f1:<10.3f} {class_corrects[i]:<8} {class_totals[i]:<8} {sup:<8}")
        
        print("-" * 90)
        print(f"{'Weighted Avg':<20} {weighted_precision:<10.3f} {weighted_recall:<10.3f} "
              f"{weighted_f1:<10.3f} {sum(class_corrects):<8} {sum(class_totals):<8} {np.sum(support):<8}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_scores': f1_scores,
            'support': support,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'class_corrects': class_corrects,
            'class_totals': class_totals,
            'confusion_matrix': cm.tolist(),
            'predictions_initial': all_preds_initial,
            'predictions_final': all_preds_final,
            'true_labels': all_labels,
            'roi_usage_stats': roi_usage_stats,
            'detailed_predictions': detailed_predictions
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, split_name: str):
        """Confusion Matrix 시각화 및 저장"""
        
        plt.figure(figsize=(12, 10))
        
        # 정규화된 confusion matrix 계산
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Heatmap 생성
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            cbar_kws={'label': 'Normalized Accuracy'}
        )
        
        plt.title(f'{split_name} Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 저장
        save_path = self.output_dir / f'{split_name.lower()}_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Confusion matrix saved: {save_path}")
    
    def _compare_performance(self, cls_metrics: Dict, roi_metrics: Dict):
        """Classification Only vs ROI Enhanced 성능 비교"""
        
        print(f"\n📈 Performance Comparison")
        print("=" * 60)
        
        cls_acc = cls_metrics['accuracy']
        roi_acc = roi_metrics['accuracy']
        cls_f1 = cls_metrics['weighted_f1']
        roi_f1 = roi_metrics['weighted_f1']
        
        acc_improvement = roi_acc - cls_acc
        f1_improvement = roi_f1 - cls_f1
        
        print(f"📊 Overall Performance:")
        print(f"  Classification Only Accuracy:  {cls_acc:.4f} ({cls_acc*100:.2f}%)")
        print(f"  ROI Enhanced Accuracy:         {roi_acc:.4f} ({roi_acc*100:.2f}%)")
        print(f"  Accuracy Improvement:          {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
        
        print(f"\n📊 Weighted F1-Score:")
        print(f"  Classification Only F1:        {cls_f1:.4f}")
        print(f"  ROI Enhanced F1:               {roi_f1:.4f}")
        print(f"  F1 Improvement:                {f1_improvement:+.4f}")
        
        # 클래스별 개선 효과
        print(f"\n📋 Class-wise Improvements:")
        print("-" * 70)
        print(f"{'Class':<20} {'Cls-Only F1':<12} {'ROI-Enh F1':<12} {'Improvement':<12}")
        print("-" * 70)
        
        for i, class_name in enumerate(self.classes):
            cls_f1_class = cls_metrics['f1_scores'][i]
            roi_f1_class = roi_metrics['f1_scores'][i]
            improvement = roi_f1_class - cls_f1_class
            
            print(f"{class_name:<20} {cls_f1_class:<12.3f} {roi_f1_class:<12.3f} {improvement:+12.3f}")
        
        print("-" * 70)
        
        # ROI 사용 통계
        roi_stats = roi_metrics['roi_usage_stats']
        print(f"\n🔍 ROI Usage Statistics:")
        print(f"  Total Images:       {roi_stats['total']}")
        print(f"  ROI Used:           {roi_stats['used']} ({roi_stats['used']/roi_stats['total']*100:.1f}%)")
        print(f"  ROI Success:        {roi_stats['success']} ({roi_stats['success']/roi_stats['used']*100:.1f}% of used)" if roi_stats['used'] > 0 else "  ROI Success:        0%")
        
        # 개선 효과 요약
        if acc_improvement > 0:
            print(f"\n✅ ROI Enhancement Result: +{acc_improvement*100:.2f}% accuracy improvement!")
        elif acc_improvement == 0:
            print(f"\n➖ ROI Enhancement Result: No accuracy change")
        else:
            print(f"\n❌ ROI Enhancement Result: {acc_improvement*100:.2f}% accuracy decrease")
    
    def _analyze_and_save_incorrect_predictions(self, roi_metrics: Dict):
        """틀린 예측들 상세 분석 및 이미지 저장"""
        
        print(f"\n🔍 Analyzing Incorrect Predictions...")
        
        # 틀린 예측들만 필터링
        detailed_preds = roi_metrics['detailed_predictions']
        incorrect_preds = [pred for pred in detailed_preds if not pred['correct_final']]
        
        if len(incorrect_preds) == 0:
            print("🎉 Perfect accuracy! No incorrect predictions to analyze.")
            return
        
        print(f"📊 Found {len(incorrect_preds)} incorrect predictions out of {len(detailed_preds)} total")
        
        # 오류 분석 디렉토리 생성
        error_analysis_dir = self.output_dir / "error_analysis"
        error_analysis_dir.mkdir(exist_ok=True)
        
        # 클래스별 오류 디렉토리 생성
        for class_name in self.classes:
            class_dir = error_analysis_dir / f"true_{class_name}"
            class_dir.mkdir(exist_ok=True)
        
        # 오류 분석 데이터
        error_analysis = {
            'total_errors': len(incorrect_preds),
            'total_samples': len(detailed_preds),
            'error_rate': len(incorrect_preds) / len(detailed_preds),
            'class_wise_errors': {},
            'roi_impact_analysis': {
                'classification_only_errors': 0,
                'roi_corrected_errors': 0,
                'roi_caused_errors': 0,
                'roi_unchanged_errors': 0
            },
            'error_details': []
        }
        
        # 클래스별 오류 초기화
        for class_name in self.classes:
            error_analysis['class_wise_errors'][class_name] = {
                'total_samples': 0,
                'errors': 0,
                'error_rate': 0.0,
                'common_wrong_predictions': {}
            }
        
        print(f"💾 Saving error analysis images and metadata...")
        
        # 각 틀린 예측에 대해 분석
        for idx, pred_info in enumerate(incorrect_preds):
            try:
                # 상세 정보 생성
                true_class = pred_info['true_class']
                initial_pred = pred_info['initial_pred_class']
                final_pred = pred_info['final_pred_class']
                detected_obj = pred_info['detected_object'] or "None"
                
                filename = f"error_{idx:04d}_true-{true_class}_initial-{initial_pred}_final-{final_pred}_obj-{detected_obj}.json"
                
                # 상세 정보를 JSON으로 저장
                error_detail = {
                    'error_id': idx,
                    'true_class': true_class,
                    'initial_prediction': {
                        'class': initial_pred,
                        'confidence': pred_info['initial_confidence']
                    },
                    'detected_object': detected_obj,
                    'final_prediction': {
                        'class': final_pred,
                        'roi_used': pred_info['roi_used'],
                        'roi_success': pred_info['roi_success']
                    },
                    'analysis': {
                        'correct_initial': pred_info['correct_initial'],
                        'correct_final': pred_info['correct_final'],
                        'roi_impact': self._analyze_roi_impact(pred_info)
                    }
                }
                
                # JSON 파일로 저장
                json_path = error_analysis_dir / f"true_{true_class}" / filename
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(error_detail, f, indent=2, ensure_ascii=False)
                
                # 오류 분석 업데이트
                error_analysis['error_details'].append(error_detail)
                
                # 클래스별 통계 업데이트
                class_stats = error_analysis['class_wise_errors'][true_class]
                class_stats['errors'] += 1
                
                if final_pred not in class_stats['common_wrong_predictions']:
                    class_stats['common_wrong_predictions'][final_pred] = 0
                class_stats['common_wrong_predictions'][final_pred] += 1
                
                # ROI 영향 분석
                roi_impact = error_detail['analysis']['roi_impact']
                error_analysis['roi_impact_analysis'][roi_impact] += 1
                
            except Exception as e:
                print(f"⚠️ Error processing prediction {idx}: {e}")
        
        # 전체 샘플에 대한 클래스별 통계 완성
        for pred_info in detailed_preds:
            true_class = pred_info['true_class']
            error_analysis['class_wise_errors'][true_class]['total_samples'] += 1
        
        # 오류율 계산
        for class_name in self.classes:
            class_stats = error_analysis['class_wise_errors'][class_name]
            if class_stats['total_samples'] > 0:
                class_stats['error_rate'] = class_stats['errors'] / class_stats['total_samples']
        
        # 종합 오류 분석 저장
        analysis_summary_path = error_analysis_dir / "error_analysis_summary.json"
        with open(analysis_summary_path, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)
        
        # 오류 분석 결과 출력
        print(f"\n📋 Error Analysis Summary:")
        print(f"  Total Errors: {error_analysis['total_errors']}/{error_analysis['total_samples']} ({error_analysis['error_rate']*100:.2f}%)")
        
        print(f"\n📊 ROI Impact on Errors:")
        roi_impact = error_analysis['roi_impact_analysis']
        print(f"  Classification Only Errors: {roi_impact['classification_only_errors']}")
        print(f"  ROI Corrected Errors:       {roi_impact['roi_corrected_errors']}")
        print(f"  ROI Caused Errors:          {roi_impact['roi_caused_errors']}")
        print(f"  ROI Unchanged Errors:       {roi_impact['roi_unchanged_errors']}")
        
        print(f"\n📁 Error analysis saved to: {error_analysis_dir}")
        print(f"   - Individual error JSONs: {len(incorrect_preds)} files")
        print(f"   - Summary: error_analysis_summary.json")
    
    def _analyze_roi_impact(self, pred_info: Dict) -> str:
        """ROI의 영향 분석"""
        
        correct_initial = pred_info['correct_initial']
        correct_final = pred_info['correct_final']
        roi_used = pred_info['roi_used']
        
        if not roi_used:
            return 'classification_only_errors'
        elif correct_initial and not correct_final:
            return 'roi_caused_errors'
        elif not correct_initial and correct_final:
            return 'roi_corrected_errors'
        else:
            return 'roi_unchanged_errors'
    
    def _save_performance_report(self, train_metrics: Dict, val_metrics: Dict):
        """성능 리포트를 JSON으로 저장"""
        
        report = {
            'timestamp': str(Path().cwd()),
            'config': {
                'epochs': self.config.DEFAULT_EPOCHS,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'f1_threshold': self.config.F1_THRESHOLD
            },
            'classes': self.classes,
            'num_classes': self.num_classes,
            'train_metrics': {
                'accuracy': float(train_metrics['accuracy']),
                'weighted_precision': float(train_metrics['weighted_precision']),
                'weighted_recall': float(train_metrics['weighted_recall']),
                'weighted_f1': float(train_metrics['weighted_f1']),
                'class_wise': {
                    self.classes[i]: {
                        'precision': float(train_metrics['precision'][i]),
                        'recall': float(train_metrics['recall'][i]),
                        'f1_score': float(train_metrics['f1_scores'][i]),
                        'correct': int(train_metrics['class_corrects'][i]),
                        'total': int(train_metrics['class_totals'][i]),
                        'support': int(train_metrics['support'][i])
                    } for i in range(len(self.classes))
                },
                'confusion_matrix': train_metrics['confusion_matrix']
            },
            'validation_metrics': {
                'accuracy': float(val_metrics['accuracy']),
                'weighted_precision': float(val_metrics['weighted_precision']),
                'weighted_recall': float(val_metrics['weighted_recall']),
                'weighted_f1': float(val_metrics['weighted_f1']),
                'class_wise': {
                    self.classes[i]: {
                        'precision': float(val_metrics['precision'][i]),
                        'recall': float(val_metrics['recall'][i]),
                        'f1_score': float(val_metrics['f1_scores'][i]),
                        'correct': int(val_metrics['class_corrects'][i]),
                        'total': int(val_metrics['class_totals'][i]),
                        'support': int(val_metrics['support'][i])
                    } for i in range(len(self.classes))
                },
                'confusion_matrix': val_metrics['confusion_matrix']
            },
            'difficult_classes': self.difficult_classes
        }
        
        # JSON 저장
        report_path = self.output_dir / 'performance_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Performance report saved: {report_path}")
        
        # 요약 통계 출력
        print(f"\n🎯 Training Summary:")
        print(f"  Train Accuracy:     {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
        print(f"  Val Accuracy:       {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
        print(f"  Train Weighted F1:  {train_metrics['weighted_f1']:.4f}")
        print(f"  Val Weighted F1:    {val_metrics['weighted_f1']:.4f}")
        print(f"  Difficult Classes:  {len(self.difficult_classes)}/{len(self.classes)}")
    
    def _save_comprehensive_performance_report(self, train_cls: Dict, val_cls: Dict, 
                                              train_roi: Dict, val_roi: Dict):
        """2단계 성능을 포함한 종합 리포트 저장"""
        
        comprehensive_report = {
            'timestamp': str(Path().cwd()),
            'config': {
                'epochs': self.config.DEFAULT_EPOCHS,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'f1_threshold': self.config.F1_THRESHOLD,
                'confidence_threshold': self.config.CONFIDENCE_THRESHOLD
            },
            'classes': self.classes,
            'num_classes': self.num_classes,
            'difficult_classes': self.difficult_classes,
            
            # Stage 1: Classification Only
            'stage1_classification_only': {
                'train_metrics': {
                    'accuracy': float(train_cls['accuracy']),
                    'weighted_precision': float(train_cls['weighted_precision']),
                    'weighted_recall': float(train_cls['weighted_recall']),
                    'weighted_f1': float(train_cls['weighted_f1']),
                    'class_wise': {
                        self.classes[i]: {
                            'precision': float(train_cls['precision'][i]),
                            'recall': float(train_cls['recall'][i]),
                            'f1_score': float(train_cls['f1_scores'][i]),
                            'correct': int(train_cls['class_corrects'][i]),
                            'total': int(train_cls['class_totals'][i]),
                            'support': int(train_cls['support'][i])
                        } for i in range(len(self.classes))
                    },
                    'confusion_matrix': train_cls['confusion_matrix']
                },
                'validation_metrics': {
                    'accuracy': float(val_cls['accuracy']),
                    'weighted_precision': float(val_cls['weighted_precision']),
                    'weighted_recall': float(val_cls['weighted_recall']),
                    'weighted_f1': float(val_cls['weighted_f1']),
                    'class_wise': {
                        self.classes[i]: {
                            'precision': float(val_cls['precision'][i]),
                            'recall': float(val_cls['recall'][i]),
                            'f1_score': float(val_cls['f1_scores'][i]),
                            'correct': int(val_cls['class_corrects'][i]),
                            'total': int(val_cls['class_totals'][i]),
                            'support': int(val_cls['support'][i])
                        } for i in range(len(self.classes))
                    },
                    'confusion_matrix': val_cls['confusion_matrix']
                }
            },
            
            # Stage 2: ROI Enhanced
            'stage2_roi_enhanced': {
                'roi_mappings': self.class_object_mapping,
                'train_metrics': {
                    'accuracy': float(train_roi['accuracy']),
                    'weighted_precision': float(train_roi['weighted_precision']),
                    'weighted_recall': float(train_roi['weighted_recall']),
                    'weighted_f1': float(train_roi['weighted_f1']),
                    'roi_usage_stats': train_roi['roi_usage_stats'],
                    'class_wise': {
                        self.classes[i]: {
                            'precision': float(train_roi['precision'][i]),
                            'recall': float(train_roi['recall'][i]),
                            'f1_score': float(train_roi['f1_scores'][i]),
                            'correct': int(train_roi['class_corrects'][i]),
                            'total': int(train_roi['class_totals'][i]),
                            'support': int(train_roi['support'][i])
                        } for i in range(len(self.classes))
                    },
                    'confusion_matrix': train_roi['confusion_matrix']
                },
                'validation_metrics': {
                    'accuracy': float(val_roi['accuracy']),
                    'weighted_precision': float(val_roi['weighted_precision']),
                    'weighted_recall': float(val_roi['weighted_recall']),
                    'weighted_f1': float(val_roi['weighted_f1']),
                    'roi_usage_stats': val_roi['roi_usage_stats'],
                    'class_wise': {
                        self.classes[i]: {
                            'precision': float(val_roi['precision'][i]),
                            'recall': float(val_roi['recall'][i]),
                            'f1_score': float(val_roi['f1_scores'][i]),
                            'correct': int(val_roi['class_corrects'][i]),
                            'total': int(val_roi['class_totals'][i]),
                            'support': int(val_roi['support'][i])
                        } for i in range(len(self.classes))
                    },
                    'confusion_matrix': val_roi['confusion_matrix']
                }
            },
            
            # Performance Comparison
            'performance_comparison': {
                'accuracy_improvement': float(val_roi['accuracy'] - val_cls['accuracy']),
                'weighted_f1_improvement': float(val_roi['weighted_f1'] - val_cls['weighted_f1']),
                'class_wise_improvements': {
                    self.classes[i]: {
                        'f1_improvement': float(val_roi['f1_scores'][i] - val_cls['f1_scores'][i])
                    } for i in range(len(self.classes))
                }
            }
        }
        
        # JSON 저장
        report_path = self.output_dir / 'comprehensive_performance_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Comprehensive performance report saved: {report_path}")
        
        # 요약 통계 출력
        print(f"\n🎯 Final Training Summary:")
        print(f"  Stage 1 - Classification Only:")
        print(f"    Train Accuracy:     {train_cls['accuracy']:.4f} ({train_cls['accuracy']*100:.2f}%)")
        print(f"    Val Accuracy:       {val_cls['accuracy']:.4f} ({val_cls['accuracy']*100:.2f}%)")
        print(f"    Val Weighted F1:    {val_cls['weighted_f1']:.4f}")
        
        print(f"  Stage 2 - ROI Enhanced:")
        print(f"    Train Accuracy:     {train_roi['accuracy']:.4f} ({train_roi['accuracy']*100:.2f}%)")
        print(f"    Val Accuracy:       {val_roi['accuracy']:.4f} ({val_roi['accuracy']*100:.2f}%)")
        print(f"    Val Weighted F1:    {val_roi['weighted_f1']:.4f}")
        
        acc_improvement = val_roi['accuracy'] - val_cls['accuracy']
        f1_improvement = val_roi['weighted_f1'] - val_cls['weighted_f1']
        
        print(f"  Performance Improvement:")
        print(f"    Accuracy:           {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
        print(f"    Weighted F1:        {f1_improvement:+.4f}")
        print(f"    Difficult Classes:  {len(self.difficult_classes)}/{len(self.classes)}")
    
    def _create_roi_mappings(self):
        """ROI 매핑 생성 - 전체 validation 데이터로 클래스별 YOLO 객체 매핑"""
        
        if self.yolo_model is None:
            print("⚠️ YOLO model not loaded, skipping ROI mappings")
            return
        
        print("🔗 Creating ROI mappings with full validation dataset...")
        
        # 각 클래스별 객체 출현 횟수 카운터
        class_object_counts = {cls: {} for cls in self.classes}
        class_total_counts = {cls: 0 for cls in self.classes}
        
        total_processed = 0
        
        # 원본 이미지에서 직접 ROI 매핑 생성
        # validation dataset의 이미지 파일들을 직접 순회
        val_dataset_root = self.dataset_root
        
        for class_name in self.classes:
            class_dir = val_dataset_root / class_name
            if not class_dir.exists():
                continue
                
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(class_dir.glob(f"*{ext}"))
                image_files.extend(class_dir.glob(f"*{ext.upper()}"))
            
            print(f"  Processing class '{class_name}': {len(image_files)} images")
            
            for img_path in image_files:
                class_total_counts[class_name] += 1
                total_processed += 1
                
                try:
                    # 학습된 클래스별 ROI 패턴을 사용하여 ROI 추출
                    roi_image = self._extract_roi_with_learned_pattern(str(img_path), class_name)
                    results = self.yolo_model(roi_image, verbose=False)
                    
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        # 모든 검출된 객체 중 가장 신뢰도 높은 객체 선택
                        confidences = results[0].boxes.conf.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        
                        # 신뢰도 임계값 이상인 객체들만 고려
                        valid_indices = confidences > self.config.OBJECT_CONFIDENCE_THRESHOLD
                        
                        if np.any(valid_indices):
                            valid_classes = classes[valid_indices]
                            
                            # 각 객체 클래스별 개수 세기
                            unique_classes, counts = np.unique(valid_classes, return_counts=True)
                            
                            # 가장 많이 검출된 객체 클래스 선택
                            most_frequent_idx = np.argmax(counts)
                            detected_class = int(unique_classes[most_frequent_idx])
                            object_count = counts[most_frequent_idx]
                            
                            if detected_class < len(self.yolo_objects):
                                object_name = self.yolo_objects[detected_class]
                                print(f"🔍 [{class_name}] Most frequent object: '{object_name}' (count: {object_count})")
                                
                                # 클래스별 객체 카운트 증가
                                if object_name not in class_object_counts[class_name]:
                                    class_object_counts[class_name][object_name] = 0
                                class_object_counts[class_name][object_name] += 1
                
                except Exception as e:
                    # 개별 이미지 처리 실패는 무시하고 계속 진행
                    continue
        
        # 통계 및 매핑 생성
        print(f"\n📊 ROI mapping statistics:")
        print(f"  Total processed: {total_processed} images")
        
        self.class_object_mapping = {}
        mapping_details = {}
        
        for class_name in self.classes:
            class_total = class_total_counts[class_name]
            object_counts = class_object_counts[class_name]
            
            print(f"\n🔍 Class '{class_name}': {class_total} images")
            
            if not object_counts or class_total == 0:
                print(f"    ❌ No objects detected")
                continue
            
            # 객체별 출현 비율 계산 (전체 클래스 이미지 대비)
            object_ratios = {}
            for object_name, count in object_counts.items():
                ratio = count / class_total
                object_ratios[object_name] = ratio
                print(f"    📦 {object_name}: {count}/{class_total} = {ratio:.3f} ({ratio*100:.1f}%)")
            
            # 가장 높은 비율의 객체 찾기
            if object_ratios:
                best_object = max(object_ratios.items(), key=lambda x: x[1])
                best_object_name, best_ratio = best_object
                
                # 임계값 이상이면 매핑 생성
                if best_ratio >= self.config.MAPPING_THRESHOLD:
                    self.class_object_mapping[class_name] = best_object_name
                    print(f"    ✅ Mapping created: {class_name} → {best_object_name} ({best_ratio:.3f})")
                else:
                    print(f"    ❌ Best ratio {best_ratio:.3f} < threshold {self.config.MAPPING_THRESHOLD}")
                
                # 상세 정보 저장
                mapping_details[class_name] = {
                    'total_images': class_total,
                    'object_counts': object_counts,
                    'object_ratios': object_ratios,
                    'best_object': best_object_name,
                    'best_ratio': best_ratio,
                    'mapping_created': best_ratio >= self.config.MAPPING_THRESHOLD
                }
        
        print(f"\n✅ ROI mappings created: {len(self.class_object_mapping)}/{len(self.classes)}")
        for cls, obj in self.class_object_mapping.items():
            ratio = mapping_details[cls]['best_ratio']
            print(f"  🎯 {cls} → {obj} ({ratio:.3f})")
        
        # 매핑 결과 저장 (상세 정보 포함)
        mapping_path = self.output_dir / self.config.MAPPING_RESULTS_FILE
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'difficult_classes': self.difficult_classes,
                'class_object_mapping': self.class_object_mapping,
                'yolo_objects': self.yolo_objects,
                'mapping_details': mapping_details,
                'total_processed_images': total_processed,
                'class_total_counts': class_total_counts,
                'mapping_threshold': self.config.MAPPING_THRESHOLD
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 ROI mappings saved: {mapping_path}")
    
    def _load_model_for_inference(self):
        """추론용 모델 로드"""
        
        # 클래스 정보 로드
        info_path = self.output_dir / 'class_info.json'
        if not info_path.exists():
            raise FileNotFoundError(f"Class info not found: {info_path}")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            class_info = json.load(f)
        
        self.classes = class_info['classes']
        self.num_classes = class_info['num_classes']
        self.class_to_idx = class_info['class_to_idx']
        
        # 모델 생성
        self.model = timm.create_model(
            class_info['config']['CONVNEXT_MODEL_NAME'],
            pretrained=False,
            num_classes=self.num_classes
        )
        
        # 가중치 로드
        model_path = self.output_dir / self.config.CLASSIFICATION_MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
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
        
        # ROI 매핑 정보 로드
        self._load_roi_mappings()
        
        # YOLO 모델 로드
        if not self.yolo_model:
            self._load_yolo_model()
        
        print(f"✅ Model loaded for inference:")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Architecture: {class_info['config']['CONVNEXT_MODEL_NAME']}")
        print(f"  - Difficult classes: {len(self.difficult_classes)}")
        print(f"  - ROI mappings: {len(self.class_object_mapping)}")
    
    def _load_roi_mappings(self):
        """ROI 매핑 정보 로드"""
        
        mapping_path = self.output_dir / self.config.MAPPING_RESULTS_FILE
        if not mapping_path.exists():
            print("⚠️ No ROI mappings found, using classification only")
            self.difficult_classes = []
            self.class_object_mapping = {}
            return
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        self.difficult_classes = mapping_data.get('difficult_classes', [])
        self.class_object_mapping = mapping_data.get('class_object_mapping', {})
        self.yolo_objects = mapping_data.get('yolo_objects', [])
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """단일 이미지 예측 - ROI 검증 포함"""
        
        if self.model is None:
            self._load_model_for_inference()
        
        # ImageFolder 방식으로 단일 이미지 처리
        image_path = Path(image_path)
        if image_path.is_file():
            # 파일명 입력 경우 - 직접 로드 (예외)
            transform = transforms.Compose([
                transforms.Resize((self.config.CLASSIFICATION_SIZE, self.config.CLASSIFICATION_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)
        else:
            # 폴더 경우 - ImageFolder 사용
            transform = transforms.Compose([
                transforms.Resize((self.config.CLASSIFICATION_SIZE, self.config.CLASSIFICATION_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            temp_dataset = datasets.ImageFolder(root=image_path, transform=transform)
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
            roi_suggested_class = self._get_roi_suggested_class(image_path)
            
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
    
    def _get_roi_suggested_class(self, image_path: str) -> Optional[str]:
        """ROI 기반 클래스 제안"""
        
        try:
            # 원본 이미지를 ROI crop한 후 YOLO 검출
            original_image = Image.open(image_path).convert('RGB')
            image_np = np.array(original_image)
            roi_image = self._extract_roi_region(image_np)
            results = self.yolo_model(roi_image, verbose=False)
            
            if len(results) == 0 or len(results[0].boxes) == 0:
                return None
            
            # 가장 신뢰도 높은 객체 찾기
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            max_conf_idx = np.argmax(confidences)
            if confidences[max_conf_idx] < self.config.OBJECT_CONFIDENCE_THRESHOLD:
                return None
            
            # 검출된 객체 이름
            detected_class = int(classes[max_conf_idx])
            detected_object = self.yolo_objects[detected_class]
            
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
    
    def evaluate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """ImageFolder 구조의 데이터셋을 평가하여 성능 분석 결과 반환
        
        Args:
            dataset_path: ImageFolder 구조의 데이터셋 경로
            
        Returns:
            Dict: 정확도, F1 점수, 클래스별 성능, 예측 결과 등
        """
        from torchvision import datasets, transforms
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
        
        # 모델 로드
        self._load_model_for_inference()
        
        # 데이터셋 로드 (ImageFolder 사용)
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 예측 수행
        all_predictions = []
        all_true_labels = []
        all_confidences = []
        all_image_paths = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                # 배치 결과 저장
                for i in range(len(images)):
                    img_idx = batch_idx * dataloader.batch_size + i
                    if img_idx < len(dataset.samples):
                        image_path, true_label = dataset.samples[img_idx]
                        
                        all_predictions.append(predicted_classes[i].cpu().item())
                        all_true_labels.append(labels[i].cpu().item())
                        all_confidences.append(confidences[i].cpu().item())
                        all_image_paths.append(image_path)
        
        # 성능 계산
        accuracy = accuracy_score(all_true_labels, all_predictions)
        weighted_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
        
        # 클래스별 성능
        precision, recall, f1, support = precision_recall_fscore_support(
            all_true_labels, all_predictions, average=None, zero_division=0
        )
        
        class_names = list(dataset.class_to_idx.keys())
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_metrics[class_name] = {
                'precision': precision[i] if i < len(precision) else 0.0,
                'recall': recall[i] if i < len(recall) else 0.0,
                'f1': f1[i] if i < len(f1) else 0.0,
                'support': support[i] if i < len(support) else 0
            }
        
        # 예측 결과 상세 정보
        sample_predictions = []
        for i in range(len(all_predictions)):
            pred_class_name = class_names[all_predictions[i]]
            true_class_name = class_names[all_true_labels[i]]
            
            sample_predictions.append({
                'image_path': all_image_paths[i],
                'predicted_class': pred_class_name,
                'true_class': true_class_name,
                'confidence': all_confidences[i],
                'correct': all_predictions[i] == all_true_labels[i]
            })
        
        return {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'class_metrics': class_metrics,
            'sample_predictions': sample_predictions,
            'total_samples': len(all_predictions)
        }


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Wafer Defect Detection - Training & Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 훈련 (기본 동작)
  python train.py
  python train.py dataset_path/
  python train.py --epochs 50 --mode production
  
  # 예측
  python train.py --predict image.jpg        # 단일 이미지
  python train.py --predict folder/          # 단순 폴더 배치 처리
  python train.py --predict test_dataset/    # ImageFolder 구조 성능 평가
        """
    )
    
    parser.add_argument("dataset_path", nargs='?', help="데이터셋 루트 경로 (기본값: config.py의 DATASET_ROOT)")
    
    # 작업 모드 (기본값: 훈련)
    parser.add_argument("--predict", help="이미지/폴더 예측 (지정하지 않으면 자동으로 훈련 실행)")
    
    # 훈련 옵션
    parser.add_argument("--epochs", type=int, help="훈련 에포크 수")
    parser.add_argument("--mode", choices=['quick-test', 'production'], default='production', help="훈련 모드")
    
    # 예측 옵션
    parser.add_argument("--batch", action="store_true", help="[더 이상 필요 없음] 폴더는 자동으로 배치 처리됨")
    parser.add_argument("--output-dir", help="모델 출력 디렉토리")
    
    args = parser.parse_args()
    
    print("🎯 Enhanced Wafer Defect Detection")
    print("=" * 40)
    
    # 데이터셋 경로 설정 (config에서 기본값 사용 가능)
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        # config에서 기본 데이터셋 경로 가져오기
        from config import ConfigManager
        temp_config = ConfigManager()
        if not temp_config.get_config().DATASET_ROOT:
            print("❌ 데이터셋 경로를 지정하거나 config.py의 DATASET_ROOT를 설정해주세요.")
            print("   사용법: python train.py your_dataset_path/ --train")
            print("   또는 config.py에서 DATASET_ROOT = 'your_dataset_path' 설정")
            return 1
        dataset_path = Path(temp_config.get_config().DATASET_ROOT)
        print(f"📂 Using dataset from config: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1
    
    # 설정 적용
    if args.mode == 'quick-test':
        config_manager = get_quick_test_config(str(dataset_path))
    else:
        config_manager = get_production_config(str(dataset_path))
    
    if args.output_dir:
        config_manager.get_config().OUTPUT_DIR = args.output_dir
    
    # 훈련기 초기화
    trainer = WaferTrainer(str(dataset_path), config_manager)
    
    try:
        if args.predict:
            # 예측 모드
            predict_path = Path(args.predict)
            
            if not predict_path.exists():
                print(f"❌ Path not found: {predict_path}")
                return 1
            
            if predict_path.is_dir():
                # 폴더 예측 (ImageFolder 구조로 처리)
                print(f"📁 Folder detected: {predict_path}")
                
                # ImageFolder 구조인지 확인 (하위에 클래스 폴더들이 있는지)
                subfolders = [d for d in predict_path.iterdir() if d.is_dir()]
                
                if subfolders:
                    # ImageFolder 구조: 라벨이 있는 데이터셋으로 처리
                    print(f"🏷️ ImageFolder structure detected with {len(subfolders)} classes")
                    print(f"📊 Classes found: {[d.name for d in subfolders]}")
                    
                    # 데이터셋 평가 모드로 처리
                    results = trainer.evaluate_dataset(str(predict_path))
                    
                    # 성능 분석 결과 출력
                    print(f"\n📊 Dataset Evaluation Results:")
                    print("=" * 60)
                    print(f"🎯 Overall Accuracy: {results['accuracy']:.3f}")
                    print(f"📊 Weighted F1 Score: {results['weighted_f1']:.3f}")
                    
                    print(f"\n📋 Class-wise Performance:")
                    print("-" * 40)
                    for class_name, metrics in results['class_metrics'].items():
                        print(f"  {class_name}:")
                        print(f"    Precision: {metrics['precision']:.3f}")
                        print(f"    Recall:    {metrics['recall']:.3f}")
                        print(f"    F1 Score:  {metrics['f1']:.3f}")
                    
                    print(f"\n🔍 Sample Predictions:")
                    print("-" * 40)
                    for i, pred in enumerate(results['sample_predictions'][:10]):  # 처음 10개만 표시
                        rel_path = Path(pred['image_path']).relative_to(predict_path)
                        status = "✅" if pred['predicted_class'] == pred['true_class'] else "❌"
                        print(f"  {status} {rel_path}: {pred['predicted_class']} (conf: {pred['confidence']:.3f}) [True: {pred['true_class']}]")
                    
                    if len(results['sample_predictions']) > 10:
                        print(f"  ... and {len(results['sample_predictions']) - 10} more predictions")
                
                else:
                    # 단순 폴더: 라벨 없는 배치 처리
                    print("📁 Simple folder structure (no class labels)")
                    
                    # 모든 이미지 파일 찾기
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                    image_files = []
                    
                    for ext in image_extensions:
                        image_files.extend(predict_path.glob(f"*{ext}"))
                        image_files.extend(predict_path.glob(f"*{ext.upper()}"))
                    
                    if not image_files:
                        print("❌ No image files found in folder")
                        return 1
                    
                    print(f"🔍 Found {len(image_files)} images for batch prediction")
                    results = trainer.predict_batch([str(f) for f in image_files])
                    
                    # 결과 출력
                    print(f"\n📊 Batch Prediction Results:")
                    print("=" * 50)
                    for result in results:
                        if 'error' not in result:
                            print(f"  {Path(result['image_path']).name}: {result['predicted_class']} ({result['confidence']:.3f})")
                        else:
                            print(f"  {Path(result['image_path']).name}: ERROR - {result['error']}")
            
            else:
                # 단일 파일 예측
                print(f"🔍 Single image prediction: {predict_path}")
                result = trainer.predict(str(predict_path))
                
                print(f"\n🎯 Prediction Result:")
                print(f"  - Class: {result['predicted_class']}")
                print(f"  - Confidence: {result['confidence']:.3f}")
                print(f"  - All probabilities:")
                for cls, prob in result['all_probabilities'].items():
                    print(f"    {cls}: {prob:.3f}")
        else:
            # 추론 파이프라인 실행 (기본 동작)
            trainer.run_inference_pipeline()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 