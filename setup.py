#!/usr/bin/env python3
"""
Setup script for Wafer Defect Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# 현재 디렉토리
HERE = Path(__file__).parent

# README 파일 읽기
long_description = (HERE / "README.md").read_text(encoding='utf-8')

# requirements.txt에서 의존성 읽기
def read_requirements():
    """requirements.txt에서 의존성 목록 읽기"""
    requirements_file = HERE / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        requirements = []
        for line in lines:
            line = line.strip()
            # 주석과 빈 줄 제외
            if line and not line.startswith('#'):
                # 개발 의존성 제외
                if 'pytest' not in line and 'black' not in line and 'flake8' not in line and 'mypy' not in line:
                    requirements.append(line)
        
        return requirements
    else:
        # 기본 의존성
        return [
            'torch>=2.0.0,<3.0.0',
            'torchvision>=0.15.0,<1.0.0',
            'numpy>=1.24.0,<2.0.0',
            'opencv-python>=4.8.0,<5.0.0',
            'Pillow>=9.5.0,<11.0.0',
            'ultralytics>=8.0.0,<9.0.0',
            'timm>=0.9.2,<1.0.0',
            'scikit-learn>=1.3.0,<2.0.0',
            'matplotlib>=3.7.0,<4.0.0',
            'tqdm>=4.65.0,<5.0.0'
        ]

setup(
    name="wafer-defect-detection",
    version="1.0.0",
    author="Wafer Detection Team",
    author_email="team@waferdetection.com",
    description="지능형 2단계 웨이퍼 결함 검출 시스템",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/wafer-defect-detection",
    
    packages=find_packages(),
    py_modules=["wafer_detector", "gradcam_utils", "utils", "main"],
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "full": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ]
    },
    
    entry_points={
        "console_scripts": [
            "wafer-detect=main:main",
        ],
    },
    
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/yourorg/wafer-defect-detection/issues",
        "Source": "https://github.com/yourorg/wafer-defect-detection",
        "Documentation": "https://github.com/yourorg/wafer-defect-detection/wiki",
    },
    
    keywords="wafer defect detection computer-vision deep-learning pytorch yolo gradcam roi",
    
    zip_safe=False,
)
