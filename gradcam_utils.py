#!/usr/bin/env python3
"""
🎯 Simple GradCAM - 최소한의 코드로 동작 보장
"""

import torch
import torch.nn.functional as F
import numpy as np


class GradCAMAnalyzer:
    """간단한 GradCAM - 실패시 즉시 에러"""
    
    def __init__(self, model, target_layer_name: str = "stages.3"):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # 타겟 레이어 찾기
        layer = model
        for name in target_layer_name.split('.'):
            layer = getattr(layer, name)
        
        # Hook 등록
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))
    
    def generate_gradcam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """GradCAM 생성 - 실패시 에러"""
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # GradCAM 계산
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        gradcam = torch.sum(weights.unsqueeze(1).unsqueeze(2) * activations, dim=0)  # (H, W)
        gradcam = F.relu(gradcam)
        
        if gradcam.max() > 0:
            gradcam = gradcam / gradcam.max()
        
        # 입력 크기로 리사이즈
        gradcam = F.interpolate(
            gradcam.unsqueeze(0).unsqueeze(0), 
            size=(input_tensor.shape[2], input_tensor.shape[3]), 
            mode='bilinear'
        ).squeeze()
        
        return gradcam.detach().cpu().numpy()


def extract_roi_from_heatmap(heatmap: np.ndarray) -> tuple:
    """히트맵에서 ROI 추출"""
    threshold = np.percentile(heatmap, 80)
    coords = np.where(heatmap >= threshold)
    
    if len(coords[0]) == 0:
        return 0.25, 0.25, 0.75, 0.75  # 기본 중앙 영역
    
    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()
    h, w = heatmap.shape
    
    return x1/w, y1/h, x2/w, y2/h
