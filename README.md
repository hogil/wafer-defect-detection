# 🎯 Wafer Defect Detection with ROI Enhancement

**ConvNeXtV2 + YOLO + Grad-CAM을 결합한 2단계 웨이퍼 결함 검출 시스템**

---

## 💡 핵심 개념

### **문제 정의**
일반적인 분류 모델은 모든 클래스를 동일하게 처리하여, 특정 "어려운 클래스"에서 낮은 성능을 보입니다. 이 시스템은 **"어려운 클래스는 더 정밀하게"** 처리하는 선택적 접근법을 구현합니다.

### **핵심 아이디어**
```
기존 방식: 모든 클래스 → 동일한 분류 모델 → 결과
새로운 방식: 쉬운 클래스 → 기본 분류 → 결과
           어려운 클래스 → 기본 분류 + ROI 강화 → 재분류 → 결과
```

---

## 🏗️ 시스템 아키텍처

### **2단계 검출 파이프라인**

#### **Stage 1: 성능 기반 클래스 분류**
```python
# 1. 전체 데이터셋으로 성능 분석
for each_class:
    f1_score = evaluate_classification_performance(class)
    if f1_score < threshold:  # 기본값: 0.8
        difficult_classes.append(class)

# 결과: ['crack', 'contamination'] ← F1 < 0.8인 클래스들
```

#### **Stage 2: ROI 기반 재분류 (어려운 클래스만)**
```python
# 2. 어려운 클래스별 ROI 패턴 학습
for difficult_class in ['crack', 'contamination']:
    roi_patterns[difficult_class] = learn_gradcam_roi_pattern(difficult_class)

# 3. 클래스-객체 매핑 구축
for difficult_class in ['crack', 'contamination']:
    roi_region = extract_roi_using_pattern(image, difficult_class)
    detected_objects = yolo_detect(roi_region)
    class_object_mapping[difficult_class] = most_frequent_object

# 예시: {'crack': 'line', 'contamination': 'blob'}
```

---

## 🔍 핵심 구현 로직

### **예측 시 동작 흐름**
```python
def predict_image(image_path):
    # 1. 기본 ConvNeXtV2 분류
    predicted_class, confidence = classify_with_convnext(image)
    
    # 2. ROI 검증 필요 조건 (3가지 모두 만족)
    needs_roi = (
        predicted_class in difficult_classes and      # F1 < 0.8
        confidence < confidence_threshold and         # 낮은 신뢰도 (0.7)
        predicted_class in class_object_mapping       # 매핑 존재
    )
    
    # 3. ROI 강화 재분류 (조건 만족시)
    if needs_roi:
        roi_image = extract_roi_using_learned_pattern(image, predicted_class)
        detected_objects = yolo_detect(roi_image)
        most_common_object = max(detected_objects, key=count)
        
        # 역매핑으로 최종 클래스 결정
        final_class = reverse_mapping[most_common_object]
        return final_class, high_confidence
    
    return predicted_class, confidence
```

### **ROI 패턴 학습 과정**
```python
def learn_roi_patterns():
    for class_name in difficult_classes:
        roi_coordinates = []
        
        # 클래스당 10개 샘플로 Grad-CAM 분석
        for sample_image in class_samples[:10]:
            heatmap = gradcam.generate_gradcam(sample_image, class_name)
            roi_coords = extract_top_80_percent_region(heatmap)
            roi_coordinates.append(roi_coords)
        
        # 중앙값으로 대표 ROI 좌표 계산
        representative_roi = median(roi_coordinates)
        roi_patterns[class_name] = representative_roi
```

---

## 📊 구체적 동작 예시

### **Case 1: 쉬운 클래스 (Stage 1만 사용)**
```python
# 입력: normal_wafer.jpg
predicted_class = "normal"
confidence = 0.92
f1_score = 0.95  # > 0.8 (쉬운 클래스)

# 결과: ROI 검증 불필요, 바로 반환
result = {
    'predicted_class': 'normal',
    'confidence': 0.92,
    'method': 'classification_only'
}
```

### **Case 2: 어려운 클래스 (Stage 2 적용)**
```python
# 입력: crack_wafer.jpg
predicted_class = "crack"
confidence = 0.65  # < 0.7 (낮은 신뢰도)
f1_score = 0.75    # < 0.8 (어려운 클래스)

# ROI 검증 실행
roi_region = extract_roi(image, roi_patterns['crack'])
detected_objects = yolo_detect(roi_region)
# → {'line': 5개, 'blob': 1개}

most_common = 'line'
mapped_class = reverse_mapping['line']  # 'crack'

result = {
    'predicted_class': 'crack',
    'confidence': 0.9,
    'method': 'roi_enhanced',
    'detected_object': 'line'
}
```

---

## 🧠 Grad-CAM ROI 추출 개념

### **모델 Attention 기반 ROI**
```python
# 기존 방식: 고정된 중앙 영역 사용
roi = image[center-100:center+100, center-100:center+100]

# 새로운 방식: 모델이 실제로 보는 영역 사용
heatmap = gradcam.generate(image, target_class='crack')
roi_coords = extract_attention_region(heatmap, top_80_percent)
roi = image[roi_coords.y1:roi_coords.y2, roi_coords.x1:roi_coords.x2]
```

### **클래스별 다른 ROI 패턴**
```python
# 학습된 ROI 패턴 예시
roi_patterns = {
    'crack': {'x1': 0.25, 'y1': 0.15, 'x2': 0.75, 'y2': 0.85},      # 세로로 긴 영역
    'contamination': {'x1': 0.10, 'y1': 0.20, 'x2': 0.90, 'y2': 0.80}, # 넓은 영역  
    'scratch': {'x1': 0.30, 'y1': 0.30, 'x2': 0.70, 'y2': 0.70}     # 중앙 영역
}
```

---

## 📈 클래스-객체 매핑 구축

### **데이터 기반 매핑 생성**
```python
# 각 어려운 클래스의 ROI에서 객체 검출 통계
for class_name in ['crack', 'contamination']:
    object_counts = {}
    
    for sample_image in class_samples:
        roi_image = extract_roi(sample_image, class_name)
        detected_objects = yolo_detect(roi_image)
        
        for obj in detected_objects:
            object_counts[obj.name] += 1
    
    # 가장 빈번한 객체로 매핑 (임계값 30% 이상)
    most_frequent = max(object_counts.items())
    if most_frequent.ratio > 0.3:
        class_object_mapping[class_name] = most_frequent.object

# 결과 예시:
# {'crack': 'line', 'contamination': 'blob'}
```

### **역매핑을 통한 재분류**
```python
# 예측 시 객체 → 클래스 역매핑
detected_object = 'line'
reverse_mapping = {'line': 'crack', 'blob': 'contamination'}
final_class = reverse_mapping[detected_object]  # 'crack'
```

---

## 🔧 코드 구조

### **파일별 역할**
```
main.py (80줄)
├── 파이프라인 실행 제어
├── 명령행 인자 처리  
└── 예측/학습 모드 분기

wafer_detector.py (150줄)
├── WaferDetector 클래스 (핵심 로직)
├── 모델 로딩 및 성능 분석
├── ROI 패턴 학습
├── 클래스-객체 매핑 생성
└── 이미지 예측 (2단계 로직)

gradcam_utils.py (60줄)  
├── GradCAMAnalyzer 클래스
├── Hook 기반 gradient 추출
├── 히트맵 생성 및 ROI 좌표 계산
└── 최소한의 구현 (실패시 즉시 에러)
```

### **주요 설정값**
```python
CONFIG = {
    'F1_THRESHOLD': 0.8,           # 어려운 클래스 판정 기준
    'CONFIDENCE_THRESHOLD': 0.7,   # ROI 검증 사용 기준
    'MAPPING_THRESHOLD': 0.3,      # 클래스-객체 매핑 신뢰도
    'CLASSIFICATION_SIZE': 384,    # ConvNeXtV2 입력 크기
    'YOLO_SIZE': 1024             # YOLO 입력 크기
}
```

---

## 🎯 실행 방법

### **전체 파이프라인 (학습 + 분석)**
```bash
python main.py dataset_path/

# 실행 순서:
# 1. 성능 분석 → difficult_classes 식별
# 2. ROI 패턴 학습 → roi_patterns.json 생성
# 3. 클래스-객체 매핑 → class_mapping.json 생성
```

### **예측만 실행**
```bash
# 단일 이미지
python main.py --predict wafer.jpg

# 폴더 배치 예측  
python main.py --predict test_images/
```

### **데이터셋 구조**
```
dataset/
├── normal/          # F1 > 0.8 (쉬운 클래스)
├── crack/           # F1 < 0.8 (어려운 클래스)
├── contamination/   # F1 < 0.8 (어려운 클래스)  
└── scratch/         # F1 값에 따라 분류됨
```

---

## 💻 핵심 알고리즘 요약

### **선택적 정밀도 검출**
1. **성능 기반 분류**: F1 score로 어려운 클래스 자동 식별
2. **Attention 기반 ROI**: Grad-CAM으로 모델이 실제 보는 영역 추출  
3. **객체 기반 재분류**: ROI에서 YOLO 검출 → 통계 기반 매핑
4. **조건부 적용**: 3가지 조건 만족시에만 ROI 검증 실행

### **효율성 원칙**
- **쉬운 클래스**: 빠른 기본 분류로 충분
- **어려운 클래스**: 정밀한 2단계 검출 적용
- **실패시 즉시 에러**: 과도한 방어 코드 제거로 290줄 달성

이 시스템은 **모든 클래스를 동일하게 처리하는 기존 방식의 한계**를 극복하고, **필요한 곳에만 정밀 분석을 적용**하는 지능형 접근법을 구현합니다.
