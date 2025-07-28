import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Dict, List
import timm
from PIL import Image


class GradCAMAnalyzer:
    """Classification 모델이 실제로 어디를 보고 예측했는지 분석하는 Grad-CAM 구현"""
    
    def __init__(self, model, target_layer_name: str = None):
        """
        Args:
            model: 훈련된 Classification 모델
            target_layer_name: Grad-CAM을 적용할 레이어 이름 (None이면 자동 선택)
        """
        self.model = model
        self.model.eval()
        
        # ConvNeXtV2의 마지막 conv layer 찾기
        if target_layer_name is None:
            target_layer_name = self._find_target_layer()
        
        self.target_layer = self._get_layer_by_name(target_layer_name)
        self.gradients = None
        self.activations = None
        
        # Hook 등록
        self._register_hooks()
    
    def _find_target_layer(self) -> str:
        """ConvNeXtV2 모델에서 적절한 target layer 찾기"""
        # ConvNeXtV2의 마지막 feature extraction layer
        for name, module in self.model.named_modules():
            if 'stages' in name and 'blocks' in name and len(list(module.children())) == 0:
                continue
        
        # 일반적으로 마지막 convolutional stage
        return "stages.3"  # ConvNeXtV2의 마지막 stage
    
    def _get_layer_by_name(self, layer_name: str):
        """레이어 이름으로 실제 레이어 객체 가져오기"""
        names = layer_name.split('.')
        layer = self.model
        for name in names:
            layer = getattr(layer, name)
        return layer
    
    def _register_hooks(self):
        """Forward/Backward hook 등록"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_gradcam(self, image_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        특정 클래스에 대한 Grad-CAM 생성
        
        Args:
            image_tensor: 입력 이미지 텐서 (1, C, H, W)
            target_class: 분석할 클래스 (None이면 예측된 클래스)
            
        Returns:
            np.ndarray: Grad-CAM heatmap (H, W)
        """
        # 모델을 train 모드로 설정 (gradient 계산을 위해)
        original_mode = self.model.training
        self.model.train()
        
        # Forward pass
        image_tensor.requires_grad_(True)
        output = self.model(image_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        
        # gradient 계산 확인
        if not class_score.requires_grad:
            print(f"⚠️ Warning: class_score does not require grad")
            # 모델을 원래 모드로 복원
            self.model.train(original_mode)
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]))
        
        class_score.backward()
        
        # Grad-CAM 계산
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination
        gradcam = torch.zeros(activations.shape[1:])  # (H, W)
        for i, weight in enumerate(weights):
            gradcam += weight * activations[i]
        
        # ReLU
        gradcam = F.relu(gradcam)
        
        # Normalize
        if gradcam.max() > 0:
            gradcam = gradcam / gradcam.max()
        
        # 모델을 원래 모드로 복원
        self.model.train(original_mode)
        
        return gradcam.detach().cpu().numpy()
    
    def get_roi_from_gradcam(self, gradcam: np.ndarray, threshold: float = 0.5, 
                           min_area_ratio: float = 0.1) -> Tuple[int, int, int, int]:
        """
        Grad-CAM에서 ROI 영역 추출
        
        Args:
            gradcam: Grad-CAM heatmap
            threshold: 중요도 임계값 (0.0~1.0)
            min_area_ratio: 최소 ROI 면적 비율
            
        Returns:
            tuple: (x1, y1, x2, y2) ROI 좌표
        """
        # 원본 이미지 크기로 리사이즈
        h, w = gradcam.shape
        
        # 임계값 이상인 영역 찾기
        mask = (gradcam >= threshold).astype(np.uint8)
        
        # 연결된 컴포넌트 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 임계값을 낮춰서 다시 시도
            threshold = 0.3
            mask = (gradcam >= threshold).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 여전히 없으면 중앙 영역 반환
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 2
            return max(0, center_x - size), max(0, center_y - size), \
                   min(w, center_x + size), min(h, center_y + size)
        
        # 가장 큰 contour 선택
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_roi, h_roi = cv2.boundingRect(largest_contour)
        
        # 최소 면적 체크
        min_area = min_area_ratio * w * h
        if w_roi * h_roi < min_area:
            # 너무 작으면 확장
            center_x, center_y = x + w_roi // 2, y + h_roi // 2
            expand_size = int(np.sqrt(min_area) // 2)
            x = max(0, center_x - expand_size)
            y = max(0, center_y - expand_size)
            w_roi = min(w - x, expand_size * 2)
            h_roi = min(h - y, expand_size * 2)
        
        return x, y, x + w_roi, y + h_roi
    
    def analyze_class_attention_patterns(self, dataloader, class_names: List[str], 
                                       num_samples_per_class: int = 10) -> Dict[str, List[Tuple]]:
        """
        각 클래스별로 모델이 주로 보는 영역 패턴 분석
        
        Args:
            dataloader: 데이터로더
            class_names: 클래스 이름 리스트
            num_samples_per_class: 클래스당 분석할 샘플 수
            
        Returns:
            dict: {class_name: [(x1, y1, x2, y2), ...]} 클래스별 ROI 패턴
        """
        class_roi_patterns = {name: [] for name in class_names}
        class_sample_counts = {name: 0 for name in class_names}
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                for i, (image, label) in enumerate(zip(images, labels)):
                    class_name = class_names[label.item()]
                    
                    # 이미 충분한 샘플을 수집했으면 스킵
                    if class_sample_counts[class_name] >= num_samples_per_class:
                        continue
                    
                    # Grad-CAM 생성
                    image_tensor = image.unsqueeze(0)
                    gradcam = self.generate_gradcam(image_tensor, label.item())
                    
                    # ROI 추출
                    roi_coords = self.get_roi_from_gradcam(gradcam)
                    class_roi_patterns[class_name].append(roi_coords)
                    class_sample_counts[class_name] += 1
                    
                    print(f"📊 Analyzed {class_name}: {class_sample_counts[class_name]}/{num_samples_per_class}")
                
                # 모든 클래스에서 충분한 샘플을 수집했으면 종료
                if all(count >= num_samples_per_class for count in class_sample_counts.values()):
                    break
        
        return class_roi_patterns
    
    def get_representative_roi_for_class(self, roi_patterns: List[Tuple], 
                                       method: str = 'median') -> Tuple[int, int, int, int]:
        """
        클래스의 대표 ROI 영역 계산
        
        Args:
            roi_patterns: 해당 클래스의 ROI 패턴 리스트
            method: 'median', 'mean', 'most_common'
            
        Returns:
            tuple: (x1, y1, x2, y2) 대표 ROI 좌표
        """
        if not roi_patterns:
            return 0, 0, 100, 100  # 기본값
        
        if method == 'median':
            x1_values = [roi[0] for roi in roi_patterns]
            y1_values = [roi[1] for roi in roi_patterns]
            x2_values = [roi[2] for roi in roi_patterns]
            y2_values = [roi[3] for roi in roi_patterns]
            
            x1 = int(np.median(x1_values))
            y1 = int(np.median(y1_values))
            x2 = int(np.median(x2_values))
            y2 = int(np.median(y2_values))
            
            return x1, y1, x2, y2
        
        elif method == 'mean':
            x1 = int(np.mean([roi[0] for roi in roi_patterns]))
            y1 = int(np.mean([roi[1] for roi in roi_patterns]))
            x2 = int(np.mean([roi[2] for roi in roi_patterns]))
            y2 = int(np.mean([roi[3] for roi in roi_patterns]))
            
            return x1, y1, x2, y2
        
        # 추가 방법들은 필요시 구현
        return roi_patterns[0]  # 첫 번째 패턴 반환


def visualize_gradcam(image: np.ndarray, gradcam: np.ndarray, 
                     roi_coords: Tuple[int, int, int, int] = None) -> np.ndarray:
    """
    Grad-CAM 시각화
    
    Args:
        image: 원본 이미지 (H, W, 3)
        gradcam: Grad-CAM heatmap (H, W)
        roi_coords: ROI 좌표 (선택사항)
        
    Returns:
        np.ndarray: 시각화된 이미지
    """
    # Grad-CAM을 원본 이미지 크기로 리사이즈
    h, w = image.shape[:2]
    gradcam_resized = cv2.resize(gradcam, (w, h))
    
    # Heatmap 생성
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_resized), cv2.COLORMAP_JET)
    
    # 원본 이미지와 블렌드
    if len(image.shape) == 3:
        superimposed = heatmap * 0.4 + image * 0.6
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        superimposed = heatmap * 0.4 + image_rgb * 0.6
    
    superimposed = np.uint8(superimposed)
    
    # ROI 박스 그리기
    if roi_coords:
        x1, y1, x2, y2 = roi_coords
        cv2.rectangle(superimposed, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return superimposed 