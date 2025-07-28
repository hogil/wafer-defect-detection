#!/usr/bin/env python3
"""
🛠️ Utility Functions for Wafer Defect Detection
이미지 처리, 시각화, 성능 평가 등 유틸리티 함수 모음
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)


def setup_directories(output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    출력 디렉토리 구조 설정
    
    Args:
        output_dir: 기본 출력 디렉토리
        
    Returns:
        생성된 디렉토리 경로들
    """
    output_dir = Path(output_dir)
    
    dirs = {
        'root': output_dir,
        'models': output_dir / 'models',
        'results': output_dir / 'results',
        'visualizations': output_dir / 'visualizations',
        'logs': output_dir / 'logs',
        'heatmaps': output_dir / 'visualizations' / 'heatmaps',
        'roi_images': output_dir / 'visualizations' / 'roi_images',
        'confusion_matrices': output_dir / 'visualizations' / 'confusion_matrices'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Directory structure created at {output_dir}")
    return dirs


def save_prediction_visualization(
    image_path: Union[str, Path],
    result: Dict[str, Any],
    output_path: Union[str, Path],
    show_roi: bool = True
) -> None:
    """
    예측 결과 시각화 저장
    
    Args:
        image_path: 원본 이미지 경로
        result: 예측 결과 딕셔너리
        output_path: 저장 경로
        show_roi: ROI 영역 표시 여부
    """
    try:
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 이미지 표시
        ax.imshow(image)
        
        # 예측 정보 텍스트
        title = f"Predicted: {result['predicted_class']} ({result['confidence']:.3f})"
        if result['method'] == 'roi_enhanced':
            title += f"\nMethod: ROI Enhanced ({result.get('detected_object', 'N/A')})"
        else:
            title += "\nMethod: Classification Only"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # ROI 영역 표시
        if show_roi and 'roi_coordinates' in result:
            roi = result['roi_coordinates']
            w, h = image.size
            
            x1 = roi['x1'] * w
            y1 = roi['y1'] * h
            x2 = roi['x2'] * w
            y2 = roi['y2'] * h
            
            # ROI 박스 그리기
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 10, 'ROI', color='red', fontweight='bold', fontsize=12)
        
        ax.axis('off')
        plt.tight_layout()
        
        # 저장
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")


def create_confusion_matrix_plot(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: Union[str, Path],
    normalize: bool = True
) -> None:
    """
    혼동 행렬 플롯 생성 및 저장
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 목록
        output_path: 저장 경로
        normalize: 정규화 여부
    """
    try:
        # 혼동 행렬 계산
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # 플롯 생성
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 저장
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create confusion matrix: {e}")


def generate_performance_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    성능 리포트 생성 및 저장
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 목록
        output_path: 저장 경로
        
    Returns:
        성능 메트릭 딕셔너리
    """
    try:
        # 성능 메트릭 계산
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # 전체 평균
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # 클래스별 성능
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        # 전체 성능
        overall_metrics = {
            'precision_weighted': float(precision_avg),
            'recall_weighted': float(recall_avg),
            'f1_weighted': float(f1_avg),
            'accuracy': float(np.mean(np.array(y_true) == np.array(y_pred)))
        }
        
        # 성능 리포트
        performance_report = {
            'overall_metrics': overall_metrics,
            'class_metrics': class_metrics,
            'classification_report': classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
        }
        
        # JSON 저장
        with open(output_path, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        logger.info(f"Performance report saved to {output_path}")
        return performance_report
        
    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")
        return {}


def visualize_roi_patterns(
    roi_patterns: Dict[str, Dict[str, float]],
    output_dir: Union[str, Path],
    image_size: Tuple[int, int] = (384, 384)
) -> None:
    """
    ROI 패턴 시각화
    
    Args:
        roi_patterns: ROI 패턴 딕셔너리
        output_dir: 출력 디렉토리
        image_size: 기준 이미지 크기
    """
    try:
        output_dir = Path(output_dir)
        
        # 전체 ROI 패턴 비교
        fig, axes = plt.subplots(
            2, (len(roi_patterns) + 1) // 2, 
            figsize=(15, 10)
        )
        axes = axes.flatten() if len(roi_patterns) > 1 else [axes]
        
        for i, (class_name, roi) in enumerate(roi_patterns.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 배경 이미지 (회색)
            background = np.ones((*image_size, 3)) * 0.8
            ax.imshow(background)
            
            # ROI 영역 표시
            w, h = image_size
            x1 = roi['x1'] * w
            y1 = roi['y1'] * h
            x2 = roi['x2'] * w
            y2 = roi['y2'] * h
            
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='red', facecolor='red', alpha=0.3
            )
            ax.add_patch(rect)
            
            ax.set_title(f"{class_name}\nROI Pattern", fontweight='bold')
            ax.axis('off')
        
        # 빈 subplot 숨기기
        for i in range(len(roi_patterns), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('ROI Patterns for Difficult Classes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 저장
        plt.savefig(output_dir / 'roi_patterns_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROI patterns visualization saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to visualize ROI patterns: {e}")


def analyze_class_distribution(
    dataset_root: Union[str, Path],
    output_path: Union[str, Path]
) -> Dict[str, int]:
    """
    데이터셋 클래스 분포 분석
    
    Args:
        dataset_root: 데이터셋 루트 경로
        output_path: 결과 저장 경로
        
    Returns:
        클래스별 샘플 수
    """
    try:
        dataset_root = Path(dataset_root)
        class_counts = {}
        
        # 클래스별 파일 수 계산
        for class_dir in dataset_root.iterdir():
            if class_dir.is_dir():
                image_files = (
                    list(class_dir.glob("*.jpg")) + 
                    list(class_dir.glob("*.png")) + 
                    list(class_dir.glob("*.jpeg"))
                )
                class_counts[class_dir.name] = len(image_files)
        
        # 시각화
        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = plt.bar(classes, counts, color='skyblue', alpha=0.7)
        plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=14)
        plt.ylabel('Number of Samples', fontsize=14)
        plt.xticks(rotation=45)
        
        # 값 표시
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + max(counts) * 0.01,
                str(count), 
                ha='center', 
                va='bottom',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class distribution analysis saved to {output_path}")
        return class_counts
        
    except Exception as e:
        logger.error(f"Failed to analyze class distribution: {e}")
        return {}


def validate_dataset_structure(dataset_root: Union[str, Path]) -> Dict[str, Any]:
    """
    데이터셋 구조 유효성 검증
    
    Args:
        dataset_root: 데이터셋 루트 경로
        
    Returns:
        검증 결과
    """
    try:
        dataset_root = Path(dataset_root)
        
        if not dataset_root.exists():
            return {'valid': False, 'error': 'Dataset root does not exist'}
        
        # 클래스 디렉토리 확인
        class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
        
        if not class_dirs:
            return {'valid': False, 'error': 'No class directories found'}
        
        validation_result = {
            'valid': True,
            'total_classes': len(class_dirs),
            'classes': [],
            'total_images': 0,
            'issues': []
        }
        
        # 각 클래스 검증
        for class_dir in class_dirs:
            image_files = (
                list(class_dir.glob("*.jpg")) + 
                list(class_dir.glob("*.png")) + 
                list(class_dir.glob("*.jpeg"))
            )
            
            class_info = {
                'name': class_dir.name,
                'image_count': len(image_files),
                'valid_images': 0,
                'invalid_images': []
            }
            
            # 이미지 파일 검증
            for img_file in image_files[:10]:  # 처음 10개만 샘플 검증
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                    class_info['valid_images'] += 1
                except Exception as e:
                    class_info['invalid_images'].append(str(img_file))
            
            validation_result['classes'].append(class_info)
            validation_result['total_images'] += class_info['image_count']
            
            # 이슈 체크
            if class_info['image_count'] == 0:
                validation_result['issues'].append(f"No images in {class_dir.name}")
            elif class_info['image_count'] < 10:
                validation_result['issues'].append(f"Very few images in {class_dir.name}: {class_info['image_count']}")
        
        return validation_result
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def create_sample_config() -> Dict[str, Any]:
    """
    샘플 설정 생성
    
    Returns:
        샘플 설정 딕셔너리
    """
    return {
        "description": "Sample configuration for different use cases",
        
        "high_speed_mode": {
            "F1_THRESHOLD": 0.9,
            "CONFIDENCE_THRESHOLD": 0.8,
            "MAPPING_THRESHOLD": 0.4,
            "max_roi_samples": 5,
            "max_mapping_samples": 15
        },
        
        "high_accuracy_mode": {
            "F1_THRESHOLD": 0.7,
            "CONFIDENCE_THRESHOLD": 0.6,
            "MAPPING_THRESHOLD": 0.2,
            "max_roi_samples": 20,
            "max_mapping_samples": 50
        },
        
        "balanced_mode": {
            "F1_THRESHOLD": 0.8,
            "CONFIDENCE_THRESHOLD": 0.7,
            "MAPPING_THRESHOLD": 0.3,
            "max_roi_samples": 10,
            "max_mapping_samples": 30
        }
    }


def log_system_info() -> None:
    """시스템 정보 로깅"""
    try:
        import torch
        import platform
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"GPU {i}: {gpu_name}")
        
        logger.info("=" * 30)
        
    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")
