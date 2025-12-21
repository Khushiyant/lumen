"""
Setup script for Lumen Python package
"""

from setuptools import setup, find_packages
import os
import sys

# Read README
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

# Version
VERSION = "0.1.0"

setup(
    name="lumen-ml",
    version=VERSION,
    author="Lumen Contributors",
    description="Intelligent Heterogeneous Deep Learning Runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lumen",
    
    # Python package (the wrapper module)
    packages=find_packages(),
    package_dir={"": "."},
    
    # Binary extension (built by CMake, needs to be in package)
    package_data={
        "lumen": ["*.so", "*.pyd", "*.dylib"],
    },
    
    # Dependencies
    install_requires=[
        "numpy>=1.19.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "mypy>=0.900",
        ],
        "onnx": [
            "onnx>=1.10.0",
            "onnxruntime>=1.10.0",
        ],
    },
    
    python_requires=">=3.7",
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
    
    keywords="deep-learning machine-learning gpu cuda metal heterogeneous inference",
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "lumen-info=lumen.cli:print_info",
            "lumen-benchmark=lumen.cli:benchmark",
        ],
    },
)