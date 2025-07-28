#!/usr/bin/env python3
"""
🎯 GradCAM Utils - 간단하고 안정적인 GradCAM 구현
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class GradCAMError(Exception):
    """GradCAM 관련 예외"""
    pass


class GradCAMAnalyzer:
    """
    간단하고 안정적인 GradCAM 분석기
    ConvNeXtV2 모델에 최적화됨
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str = "stages.3"):
        """
        초기화
        
        Args:
            model: 타겟 모델
            target_layer_name: 타겟 레이어 이름 (점으로 구분된 경로)
            
        Raises:
            GradCAMError: 초기화 실패시
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer_name = target_layer_name
        
        try:
            # 타겟 레이어 찾기
            layer = model
            for name in target_layer_name.split('.'):
                if not hasattr(layer, name):
                    raise GradCAMError(f"Layer '{name}' not found in model")
                layer = getattr(layer, name)
            
            self.target_layer = layer
            
            # Hook 등록
            self.forward_hook = layer.register_forward_hook(self._forward_hook)
            self.backward_hook = layer.register_full_backward_hook(self._backward_hook)
            
            logger.info(f"GradCAM initialized for layer: {target_layer_name}")
            
        except Exception as e:
            raise GradCAMError(f"Failed to initialize GradCAM: {str(e)}")
    
    def _forward_hook(self, module, input, output):
        """Forward hook - activation 저장"""
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Backward hook - gradient 저장"""
        self.gradients = grad_output[0]
    
    def generate_gradcam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        GradCAM 생성
        
        Args:
            input_tensor: 입력 텐서 (B, C, H, W)
            target_class: 타겟 클래스 인덱스 (None이면 예측 클래스 사용)
            
        Returns:
            GradCAM 히트맵 (H, W)
            
        Raises:
            GradCAMError: GradCAM 생성 실패시
        """
        try:
            if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
                raise GradCAMError("Input tensor must be 4D with batch size 1")
            
            self.model.eval()
            input_tensor.requires_grad_(True)
            
            # Forward pass
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            if target_class >= output.size(1):
                raise GradCAMError(f"Target class {target_class} out of range")
            
            # Backward pass
            self.model.zero_grad()
            score = output[0, target_class]
            score.backward()
            
            # GradCAM 계산 검증
            if self.gradients is None or self.activations is None:
                raise GradCAMError("Gradients or activations not captured")
            
            gradients = self.gradients[0]  # (C, H, W)
            activations = self.activations[0]  # (C, H, W)
            
            if gradients.shape != activations.shape:
                raise GradCAMError("Gradients and activations shape mismatch")
            
            # 가중치 계산 (Global Average Pooling)
            weights = torch.mean(gradients, dim=(1, 2))  # (C,)
            
            # Weighted sum
            gradcam = torch.sum(
                weights.unsqueeze(1).unsqueeze(2) * activations, 
                dim=0
            )  # (H, W)
            
            # ReLU 적용
            gradcam = F.relu(gradcam)
            
            # 정규화
            if gradcam.max() > 0:
                gradcam = gradcam / gradcam.max()
            
            # 입력 크기로 리사이즈
            target_size = (input_tensor.shape[2], input_tensor.shape[3])
            gradcam = F.interpolate(
                gradcam.unsqueeze(0).unsqueeze(0), 
                size=target_size, 
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # 유효성 검증
            gradcam_np = gradcam.detach().cpu().numpy()
            if np.any(np.isnan(gradcam_np)):
                logger.warning("NaN values detected in GradCAM, using default")
                gradcam_np = np.ones(target_size) * 0.5
            
            return gradcam_np
            
        except Exception as e:
            logger.error(f"GradCAM generation failed: {str(e)}")
            # 기본 히트맵 반환
            target_size = (input_tensor.shape[2], input_tensor.shape[3])
            return np.ones(target_size) * 0.5
    
    def cleanup(self):
        """Hook 정리"""
        try:
            if hasattr(self, 'forward_hook'):
                self.forward_hook.remove()
            if hasattr(self, 'backward_hook'):
                self.backward_hook.remove()
            logger.info("GradCAM hooks cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup hooks: {str(e)}")
    
    def __del__(self):
        """소멸자"""
        self.cleanup()


def extract_roi_from_heatmap(
    heatmap: np.ndarray, 
    percentile: float = 80.0,
    min_area_ratio: float = 0.01
) -> Tuple[float, float, float, float]:
    """
    히트맵에서 ROI 추출
    
    Args:
        heatmap: 입력 히트맵 (H, W)
        percentile: ROI 임계값 백분위수
        min_area_ratio: 최소 ROI 면적 비율
        
    Returns:
        (x1, y1, x2, y2) 정규화된 좌표 (0~1)
    """
    try:
        if heatmap.ndim != 2:
            raise ValueError("Heatmap must be 2D")
        
        h, w = heatmap.shape
        min_area = int(h * w * min_area_ratio)
        
        # 임계값 계산
        threshold = np.percentile(heatmap, percentile)
        
        # 이진화
        binary_mask = heatmap >= threshold
        coords = np.where(binary_mask)
        
        # 유효한 영역이 없거나 너무 작은 경우
        if len(coords[0]) == 0 or len(coords[0]) < min_area:
            logger.warning(f"Insufficient ROI area, using default center region")
            return 0.25, 0.25, 0.75, 0.75
        
        # 바운딩 박스 계산
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        
        # 정규화 및 경계 확인
        x1_norm = max(0.0, min(1.0, x1 / w))
        y1_norm = max(0.0, min(1.0, y1 / h))
        x2_norm = max(0.0, min(1.0, x2 / w))
        y2_norm = max(0.0, min(1.0, y2 / h))
        
        # 유효성 검증
        if x2_norm <= x1_norm or y2_norm <= y1_norm:
            logger.warning("Invalid ROI coordinates, using default")
            return 0.25, 0.25, 0.75, 0.75
        
        # 최소 크기 보장
        width = x2_norm - x1_norm
        height = y2_norm - y1_norm
        min_size = 0.1  # 최소 10%
        
        if width < min_size:
            center_x = (x1_norm + x2_norm) / 2
            x1_norm = max(0.0, center_x - min_size/2)
            x2_norm = min(1.0, center_x + min_size/2)
        
        if height < min_size:
            center_y = (y1_norm + y2_norm) / 2
            y1_norm = max(0.0, center_y - min_size/2)
            y2_norm = min(1.0, center_y + min_size/2)
        
        return x1_norm, y1_norm, x2_norm, y2_norm
        
    except Exception as e:
        logger.error(f"ROI extraction failed: {str(e)}")
        return 0.25, 0.25, 0.75, 0.75


def visualize_gradcam_overlay(
    image: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.4
) -> np.ndarray:
    """
    이미지와 GradCAM 히트맵 오버레이
    
    Args:
        image: 원본 이미지 (H, W, 3) 또는 (H, W)
        heatmap: GradCAM 히트맵 (H, W)
        alpha: 히트맵 투명도
        
    Returns:
        오버레이된 이미지 (H, W, 3)
    """
    try:
        # 이미지 전처리
        if image.ndim == 3:
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = np.stack([image] * 3, axis=-1).astype(np.float32) / 255.0
        
        # 히트맵을 컬러맵으로 변환 (빨간색 계열)
        import matplotlib.cm as cm
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap)[:, :, :3]  # RGB만 사용
        
        # 오버레이
        overlay = (1 - alpha) * image_norm + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
