import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, List
import json
from pathlib import Path


class ROIExtractor:
    """실제 Classification 모델의 attention 기반 ROI 추출 시스템
    
    각 클래스별로 모델이 실제로 어떤 영역을 보고 예측했는지 분석하여,
    그 영역을 ROI로 사용하는 정확한 접근 방식
    """
    
    def __init__(self, class_roi_patterns_file: str = None):
        """
        Args:
            class_roi_patterns_file: 클래스별 ROI 패턴이 저장된 JSON 파일 경로
        """
        self.class_roi_patterns = {}
        self.class_roi_patterns_file = class_roi_patterns_file
        
        # 기존 패턴 로드
        if class_roi_patterns_file and Path(class_roi_patterns_file).exists():
            self.load_class_roi_patterns(class_roi_patterns_file)
    
    def load_class_roi_patterns(self, patterns_file: str):
        """클래스별 ROI 패턴을 JSON 파일에서 로드"""
        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.class_roi_patterns = data.get('class_roi_patterns', {})
            print(f"✅ Loaded ROI patterns for {len(self.class_roi_patterns)} classes")
        except Exception as e:
            print(f"⚠️ Failed to load ROI patterns: {e}")
            self.class_roi_patterns = {}
    
    def save_class_roi_patterns(self, patterns_file: str):
        """클래스별 ROI 패턴을 JSON 파일에 저장"""
        try:
            data = {
                'class_roi_patterns': self.class_roi_patterns,
                'description': 'Each class contains representative ROI coordinates based on model attention analysis'
            }
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved ROI patterns for {len(self.class_roi_patterns)} classes")
        except Exception as e:
            print(f"⚠️ Failed to save ROI patterns: {e}")
    
    def set_class_roi_pattern(self, class_name: str, roi_coords: Tuple[int, int, int, int]):
        """특정 클래스의 ROI 패턴 설정"""
        self.class_roi_patterns[class_name] = {
            'x1_ratio': roi_coords[0],
            'y1_ratio': roi_coords[1], 
            'x2_ratio': roi_coords[2],
            'y2_ratio': roi_coords[3]
        }
        print(f"📍 Set ROI pattern for '{class_name}': {roi_coords}")
    
    def get_roi_for_class(self, class_name: str, image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """특정 클래스에 대한 ROI 영역을 상대 좌표로 반환
        
        Args:
            class_name: 클래스 이름
            image_size: (width, height) 이미지 크기
            
        Returns:
            tuple: (x1_ratio, y1_ratio, x2_ratio, y2_ratio) - 0.0~1.0 범위의 상대 좌표
        """
        if class_name not in self.class_roi_patterns:
            # 패턴이 없으면 중앙 영역 반환
            print(f"⚠️ No ROI pattern for '{class_name}', using center region")
            return self._get_default_roi(image_size)
        
        pattern = self.class_roi_patterns[class_name]
        x1_ratio = pattern['x1_ratio']
        y1_ratio = pattern['y1_ratio']
        x2_ratio = pattern['x2_ratio']
        y2_ratio = pattern['y2_ratio']
        
        print(f"🎯 Using learned ROI for '{class_name}': ({x1_ratio:.3f},{y1_ratio:.3f}) to ({x2_ratio:.3f},{y2_ratio:.3f})")
        
        return x1_ratio, y1_ratio, x2_ratio, y2_ratio
    
    def _get_default_roi(self, image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """기본 ROI (중앙 영역) 반환"""
        width, height = image_size
        
        # 중앙 70% 영역
        margin_ratio = 0.15  # 양쪽 15%씩 여백
        x1_ratio = margin_ratio
        y1_ratio = margin_ratio
        x2_ratio = 1.0 - margin_ratio
        y2_ratio = 1.0 - margin_ratio
        
        return x1_ratio, y1_ratio, x2_ratio, y2_ratio
    
    def crop_roi_from_original(self, original_image_path: str, class_name: str, target_size: int = 1024) -> np.ndarray:
        """클래스별 학습된 ROI 패턴을 사용하여 원본 이미지에서 정사각형 ROI 추출
        
        Args:
            original_image_path: 원본 이미지 파일 경로
            class_name: 클래스 이름 (ROI 패턴 결정용)
            target_size: 최종 출력 크기
            
        Returns:
            np.ndarray: 정사각형으로 리사이즈된 ROI 이미지
        """
        # 원본 이미지 로드
        original_image = Image.open(original_image_path).convert('RGB')
        original_np = np.array(original_image)
        orig_height, orig_width = original_np.shape[:2]
        
        # 클래스별 ROI 패턴 가져오기
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = self.get_roi_for_class(class_name, (orig_width, orig_height))
        
        # 상대 좌표를 원본 이미지 크기에 맞게 변환
        x1 = int(x1_ratio * orig_width)
        y1 = int(y1_ratio * orig_height)
        x2 = int(x2_ratio * orig_width)
        y2 = int(y2_ratio * orig_height)
        
        # 정사각형 ROI 좌표 계산
        square_x1, square_y1, square_x2, square_y2 = self._make_square_roi(
            x1, y1, x2, y2, orig_width, orig_height
        )
        
        # 정사각형 ROI 영역 crop
        roi_image = original_np[square_y1:square_y2, square_x1:square_x2]
        
        # 최종 크기로 resize (이미 정사각형이므로 비율 유지됨)
        roi_resized = cv2.resize(roi_image, (target_size, target_size))
        
        print(f"🔍 Square ROI extracted for '{class_name}': original({orig_width}×{orig_height}) -> crop({square_x2-square_x1}×{square_y2-square_y1}) -> resize({target_size}×{target_size})")
        
        return roi_resized
    
    def _make_square_roi(self, x1: int, y1: int, x2: int, y2: int, 
                        img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """ROI를 정사각형으로 만들되, 이미지 경계를 벗어나면 반대 방향으로 확장
        
        Args:
            x1, y1, x2, y2: 원본 ROI 좌표
            img_width, img_height: 이미지 크기
            
        Returns:
            Tuple[int, int, int, int]: 정사각형 ROI 좌표 (sx1, sy1, sx2, sy2)
        """
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        # 큰 쪽에 맞춰 정사각형 크기 결정
        square_size = max(roi_width, roi_height)
        
        # ROI 중심점 계산
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 이상적인 정사각형 좌표 (중심 기준)
        ideal_x1 = center_x - square_size // 2
        ideal_y1 = center_y - square_size // 2
        ideal_x2 = center_x + square_size // 2
        ideal_y2 = center_y + square_size // 2
        
        # 경계 조정하여 최종 정사각형 좌표 계산
        final_x1, final_y1, final_x2, final_y2 = self._adjust_square_boundaries(
            ideal_x1, ideal_y1, ideal_x2, ideal_y2, 
            square_size, img_width, img_height
        )
        
        return final_x1, final_y1, final_x2, final_y2
    
    def _adjust_square_boundaries(self, x1: int, y1: int, x2: int, y2: int,
                                 square_size: int, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """정사각형이 이미지 경계를 벗어나면 반대 방향으로 이동시켜 조정
        
        Args:
            x1, y1, x2, y2: 이상적인 정사각형 좌표
            square_size: 정사각형 한 변의 길이
            img_width, img_height: 이미지 크기
            
        Returns:
            Tuple[int, int, int, int]: 조정된 정사각형 좌표
        """
        # X축 경계 조정
        if x1 < 0:
            # 왼쪽 경계 벗어남 → 오른쪽으로 이동
            x1 = 0
            x2 = min(square_size, img_width)
        elif x2 > img_width:
            # 오른쪽 경계 벗어남 → 왼쪽으로 이동
            x2 = img_width
            x1 = max(0, img_width - square_size)
        
        # Y축 경계 조정
        if y1 < 0:
            # 위쪽 경계 벗어남 → 아래쪽으로 이동
            y1 = 0
            y2 = min(square_size, img_height)
        elif y2 > img_height:
            # 아래쪽 경계 벗어남 → 위쪽으로 이동
            y2 = img_height
            y1 = max(0, img_height - square_size)
        
        # 최종 안전장치 (경계 체크)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        return x1, y1, x2, y2
    
    def has_roi_pattern(self, class_name: str) -> bool:
        """특정 클래스에 대한 ROI 패턴이 존재하는지 확인"""
        return class_name in self.class_roi_patterns
    
    def get_available_classes(self) -> List[str]:
        """ROI 패턴이 있는 클래스 목록 반환"""
        return list(self.class_roi_patterns.keys()) 