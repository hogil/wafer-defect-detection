#!/usr/bin/env python3
"""
🧪 Test Suite for Wafer Defect Detection System
시스템 구성 요소들의 단위 테스트 및 통합 테스트
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import torch
from PIL import Image

# 테스트 대상 모듈들
from wafer_detector import WaferDetector, WaferDetectorError
from gradcam_utils import GradCAMAnalyzer, extract_roi_from_heatmap, GradCAMError
from utils import (
    setup_directories, validate_dataset_structure, 
    create_sample_config, analyze_class_distribution
)


class TestWaferDetector(unittest.TestCase):
    """WaferDetector 클래스 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        self.config = {
            'CLASSIFICATION_SIZE': 224,
            'YOLO_SIZE': 640,
            'PRECISION_THRESHOLD': 0.8,
            'CONFIDENCE_THRESHOLD': 0.7,
            'MAPPING_THRESHOLD': 0.3,
            'OUTPUT_DIR': 'test_output'
        }
        self.detector = WaferDetector(self.config)
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(len(self.detector.classes), 0)
        self.assertEqual(len(self.detector.difficult_classes), 0)
        self.assertIsNotNone(self.detector.transform)
    
    def test_config_validation(self):
        """설정 유효성 테스트"""
        # 잘못된 설정
        invalid_config = {'CLASSIFICATION_SIZE': 'invalid'}
        
        with self.assertRaises(Exception):
            detector = WaferDetector(invalid_config)
            # transform 생성 시 오류 발생할 것
    
    def test_get_stats(self):
        """통계 정보 테스트"""
        stats = self.detector.get_stats()
        
        self.assertIn('total_classes', stats)
        self.assertIn('difficult_classes', stats)
        self.assertIn('device', stats)
        self.assertIn('models_loaded', stats)
        
        # 초기 상태 확인
        self.assertEqual(stats['total_classes'], 0)
        self.assertEqual(stats['difficult_classes'], 0)


class TestGradCAMUtils(unittest.TestCase):
    """GradCAM 유틸리티 테스트"""
    
    def test_extract_roi_from_heatmap(self):
        """ROI 추출 테스트"""
        # 테스트 히트맵 생성
        heatmap = np.zeros((100, 100))
        heatmap[25:75, 25:75] = 1.0  # 중앙에 높은 값
        
        roi = extract_roi_from_heatmap(heatmap)
        
        # ROI 좌표 확인
        self.assertEqual(len(roi), 4)
        x1, y1, x2, y2 = roi
        
        # 좌표 범위 확인
        self.assertTrue(0 <= x1 < x2 <= 1)
        self.assertTrue(0 <= y1 < y2 <= 1)
    
    def test_extract_roi_empty_heatmap(self):
        """빈 히트맵 ROI 추출 테스트"""
        heatmap = np.zeros((100, 100))
        
        roi = extract_roi_from_heatmap(heatmap)
        
        # 기본값 반환 확인
        self.assertEqual(roi, (0.25, 0.25, 0.75, 0.75))
    
    def test_extract_roi_invalid_input(self):
        """잘못된 입력 ROI 추출 테스트"""
        # 1D 배열
        invalid_heatmap = np.zeros(100)
        
        roi = extract_roi_from_heatmap(invalid_heatmap)
        
        # 기본값 반환 확인
        self.assertEqual(roi, (0.25, 0.25, 0.75, 0.75))


class TestUtils(unittest.TestCase):
    """유틸리티 함수 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_setup_directories(self):
        """디렉토리 설정 테스트"""
        output_dir = self.temp_path / 'output'
        
        dirs = setup_directories(output_dir)
        
        # 디렉토리 생성 확인
        self.assertTrue(output_dir.exists())
        self.assertIn('root', dirs)
        self.assertIn('results', dirs)
        self.assertIn('visualizations', dirs)
        
        # 실제 디렉토리 존재 확인
        for dir_path in dirs.values():
            self.assertTrue(dir_path.exists())
    
    def test_create_sample_config(self):
        """샘플 설정 생성 테스트"""
        config = create_sample_config()
        
        self.assertIn('high_speed_mode', config)
        self.assertIn('high_accuracy_mode', config)
        self.assertIn('balanced_mode', config)
        
        # 각 모드별 설정 확인
        for mode in ['high_speed_mode', 'high_accuracy_mode', 'balanced_mode']:
            mode_config = config[mode]
            self.assertIn('PRECISION_THRESHOLD', mode_config)
            self.assertIn('CONFIDENCE_THRESHOLD', mode_config)
    
    def test_validate_dataset_structure_empty(self):
        """빈 데이터셋 구조 검증 테스트"""
        result = validate_dataset_structure(self.temp_path)
        
        self.assertFalse(result['valid'])
        self.assertIn('error', result)
    
    def test_validate_dataset_structure_valid(self):
        """유효한 데이터셋 구조 검증 테스트"""
        # 테스트 데이터셋 구조 생성
        class_dirs = ['normal', 'defect']
        
        for class_name in class_dirs:
            class_dir = self.temp_path / class_name
            class_dir.mkdir()
            
            # 더미 이미지 파일 생성
            for i in range(3):
                img_path = class_dir / f'image_{i}.jpg'
                
                # 더미 이미지 생성
                img = Image.new('RGB', (100, 100), color='white')
                img.save(img_path)
        
        result = validate_dataset_structure(self.temp_path)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['total_classes'], 2)
        self.assertEqual(len(result['classes']), 2)


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def setUp(self):
        """테스트 준비"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.config = {
            'CLASSIFICATION_SIZE': 224,
            'YOLO_SIZE': 640,
            'PRECISION_THRESHOLD': 0.8,
            'CONFIDENCE_THRESHOLD': 0.7,
            'MAPPING_THRESHOLD': 0.3,
            'OUTPUT_DIR': str(self.temp_path / 'output')
        }
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_config_save_load(self):
        """설정 저장/로드 테스트"""
        config_path = self.temp_path / 'test_config.json'
        
        # 설정 저장
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 설정 로드
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(self.config, loaded_config)
    
    @patch('wafer_detector.timm.create_model')
    @patch('wafer_detector.torch.load')
    @patch('wafer_detector.YOLO')
    def test_detector_initialization_with_mocks(self, mock_yolo, mock_torch_load, mock_timm):
        """Mock을 사용한 검출기 초기화 테스트"""
        # Mock 설정
        mock_model = Mock()
        mock_model.head.fc.in_features = 1024
        mock_timm.return_value = mock_model
        
        mock_state_dict = {
            'head.fc.weight': torch.randn(5, 1024),  # 5 클래스
            'head.fc.bias': torch.randn(5)
        }
        mock_torch_load.return_value = mock_state_dict
        
        mock_yolo_instance = Mock()
        mock_yolo.return_value = mock_yolo_instance
        
        # 테스트
        detector = WaferDetector(self.config)
        
        # 가짜 모델 경로로 로드 시도
        detector.load_models('fake_model.pth', 'fake_yolo.pt')
        
        # Mock 호출 확인
        mock_timm.assert_called_once()
        mock_torch_load.assert_called_once()
        mock_yolo.assert_called_once()


class TestErrorHandling(unittest.TestCase):
    """오류 처리 테스트"""
    
    def test_wafer_detector_error(self):
        """WaferDetectorError 테스트"""
        with self.assertRaises(WaferDetectorError):
            raise WaferDetectorError("Test error")
    
    def test_gradcam_error(self):
        """GradCAMError 테스트"""
        with self.assertRaises(GradCAMError):
            raise GradCAMError("Test error")
    
    def test_detector_without_models(self):
        """모델 없이 검출기 사용 테스트"""
        config = {
            'CLASSIFICATION_SIZE': 224,
            'YOLO_SIZE': 640,
            'PRECISION_THRESHOLD': 0.8,
            'CONFIDENCE_THRESHOLD': 0.7,
            'MAPPING_THRESHOLD': 0.3
        }
        detector = WaferDetector(config)
        
        # 클래스 로드 없이 예측 시도
        with self.assertRaises(WaferDetectorError):
            detector.predict_image('fake_image.jpg')


def create_test_dataset(root_path: Path, num_classes: int = 3, images_per_class: int = 5):
    """
    테스트용 데이터셋 생성
    
    Args:
        root_path: 데이터셋 루트 경로
        num_classes: 클래스 수
        images_per_class: 클래스당 이미지 수
    """
    class_names = [f'class_{i}' for i in range(num_classes)]
    
    for class_name in class_names:
        class_dir = root_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(images_per_class):
            # 랜덤 컬러 이미지 생성
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            img_path = class_dir / f'image_{i:03d}.jpg'
            img.save(img_path)


def run_performance_test():
    """성능 테스트 실행"""
    import time
    
    # 더미 데이터로 성능 측정
    config = {
        'CLASSIFICATION_SIZE': 224,
        'YOLO_SIZE': 640,
        'PRECISION_THRESHOLD': 0.8,
        'CONFIDENCE_THRESHOLD': 0.7,
        'MAPPING_THRESHOLD': 0.3
    }
    
    detector = WaferDetector(config)
    
    # Transform 성능 테스트
    img = Image.new('RGB', (1024, 1024), color='white')
    
    start_time = time.time()
    for _ in range(100):
        tensor = detector.transform(img)
    transform_time = time.time() - start_time
    
    print(f"Transform 100 images: {transform_time:.3f}s")
    print(f"Average per image: {transform_time/100*1000:.2f}ms")


if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)
