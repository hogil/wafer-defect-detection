#!/usr/bin/env python3
"""
🎯 WaferDetector - 간소화된 핵심 로직
"""

import json
from pathlib import Path
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


class WaferDetector:
    """간소화된 웨이퍼 결함 검출기 - 실패시 즉시 에러"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classification_model = None
        self.yolo_model = None
        self.gradcam_analyzer = None
        self.classes = []
        self.difficult_classes = []
        self.class_object_mapping = {}
        self.roi_patterns = {}
        
        # 공통 transform
        self.transform = transforms.Compose([
            transforms.Resize((config['CLASSIFICATION_SIZE'], config['CLASSIFICATION_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"🚀 Device: {self.device}")
    
    def load_models(self, model_path: str, yolo_path: str):
        """모델 로드 - 실패시 에러"""
        # 1. ConvNeXtV2 모델 생성
        self.classification_model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=True)
        
        # 2. 가중치 로드 및 prefix 제거
        state_dict = torch.load(model_path, map_location="cpu")
        cleaned_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # 3. 분류기 교체
        num_classes = cleaned_state_dict['head.fc.weight'].shape[0]
        self.classification_model.head.fc = nn.Linear(self.classification_model.head.fc.in_features, num_classes)
        
        # 4. 가중치 로드
        self.classification_model.load_state_dict(cleaned_state_dict, strict=True)
        self.classification_model.to(self.device).eval()
        
        # 5. GradCAM 및 YOLO 초기화
        self.gradcam_analyzer = GradCAMAnalyzer(self.classification_model)
        self.yolo_model = YOLO(yolo_path)
        
        print(f"✅ Models loaded - Classes: {num_classes}")
    
    def load_classes(self, dataset_root: str):
        """클래스 정보만 로드"""
        dataset = datasets.ImageFolder(dataset_root)
        self.classes = dataset.classes
        print(f"📋 Classes: {self.classes}")
    
    def analyze_performance(self, dataset_root: str):
        """성능 분석"""
        dataset = datasets.ImageFolder(dataset_root, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        self.classes = dataset.classes
        
        print(f"📊 Analyzing {len(dataset)} samples...")
        
        # 예측 수행
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.classification_model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        # F1 score 계산
        _, _, f1_scores, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        
        # 어려운 클래스 식별
        self.difficult_classes = [self.classes[i] for i, f1 in enumerate(f1_scores) if f1 < self.config['F1_THRESHOLD']]
        
        for i, f1 in enumerate(f1_scores):
            status = "⚠️" if f1 < self.config['F1_THRESHOLD'] else "✅"
            print(f"   {status} {self.classes[i]}: F1={f1:.3f}")
        
        return f1_scores
    
    def learn_roi_patterns(self, dataset_root: str):
        """ROI 패턴 학습"""
        if not self.difficult_classes:
            print("ℹ️ No difficult classes")
            return
        
        print(f"🧠 Learning ROI patterns for {len(self.difficult_classes)} classes...")
        
        for class_name in self.difficult_classes:
            class_dir = Path(dataset_root) / class_name
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            roi_coords_list = []
            for img_path in image_files[:10]:  # 클래스당 10개
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                class_idx = self.classes.index(class_name)
                heatmap = self.gradcam_analyzer.generate_gradcam(input_tensor, class_idx)
                roi_coords = extract_roi_from_heatmap(heatmap)
                roi_coords_list.append(roi_coords)
            
            # 중앙값으로 대표 ROI 계산
            roi_array = np.array(roi_coords_list)
            representative_roi = np.median(roi_array, axis=0)
            
            self.roi_patterns[class_name] = {
                'x1': float(representative_roi[0]), 'y1': float(representative_roi[1]),
                'x2': float(representative_roi[2]), 'y2': float(representative_roi[3])
            }
            
            print(f"📍 {class_name}: ROI learned")
    
    def create_mapping(self, dataset_root: str):
        """클래스-객체 매핑 생성"""
        if not self.difficult_classes or not self.roi_patterns:
            print("ℹ️ No classes or ROI patterns for mapping")
            return
        
        print(f"🎯 Creating mappings...")
        
        class_object_counts = {}
        
        for class_name in self.difficult_classes:
            if class_name not in self.roi_patterns:
                continue
            
            class_dir = Path(dataset_root) / class_name
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            object_counts = {}
            
            for img_path in image_files[:30]:  # 클래스당 30개
                # ROI 추출
                image = Image.open(img_path).convert('RGB')
                w, h = image.size
                roi = self.roi_patterns[class_name]
                x1, y1 = int(roi['x1'] * w), int(roi['y1'] * h)
                x2, y2 = int(roi['x2'] * w), int(roi['y2'] * h)
                
                roi_image = image.crop((x1, y1, x2, y2)).resize((self.config['YOLO_SIZE'], self.config['YOLO_SIZE']))
                
                # YOLO 검출
                results = self.yolo_model(np.array(roi_image), verbose=False)
                if len(results) > 0 and len(results[0].boxes) > 0:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    for conf, cls in zip(confidences, classes):
                        if conf > 0.5:
                            obj_name = self.yolo_model.names[int(cls)]
                            object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
            
            class_object_counts[class_name] = object_counts
        
        # 매핑 생성
        for class_name, obj_counts in class_object_counts.items():
            if obj_counts:
                best_obj, count = max(obj_counts.items(), key=lambda x: x[1])
                total = sum(obj_counts.values())
                ratio = count / total
                
                if ratio >= self.config['MAPPING_THRESHOLD']:
                    self.class_object_mapping[class_name] = best_obj
                    print(f"🎯 {class_name} → {best_obj} ({ratio:.2f})")
    
    def predict_image(self, image_path: str) -> dict:
        """단일 이미지 예측"""
        if not self.classes:
            raise RuntimeError("Classes not loaded. Call load_classes() or analyze_performance() first.")
        
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
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'method': 'classification_only'
        }
        
        # ROI 검증 조건
        needs_roi = (
            predicted_class in self.difficult_classes and
            confidence < self.config['CONFIDENCE_THRESHOLD'] and
            predicted_class in self.class_object_mapping and
            predicted_class in self.roi_patterns
        )
        
        if needs_roi:
            # ROI에서 객체 검출
            w, h = image.size
            roi = self.roi_patterns[predicted_class]
            x1, y1 = int(roi['x1'] * w), int(roi['y1'] * h)
            x2, y2 = int(roi['x2'] * w), int(roi['y2'] * h)
            
            roi_image = image.crop((x1, y1, x2, y2)).resize((self.config['YOLO_SIZE'], self.config['YOLO_SIZE']))
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
                                'object_counts': object_counts
                            })
                            break
        
        return result
    
    def save_results(self, output_dir: str):
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / 'roi_patterns.json', 'w') as f:
            json.dump(self.roi_patterns, f, indent=2)
        
        mapping_data = {
            'difficult_classes': self.difficult_classes,
            'class_object_mapping': self.class_object_mapping
        }
        with open(output_path / 'class_mapping.json', 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")
