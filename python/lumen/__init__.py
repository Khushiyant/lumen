"""
Lumen: Intelligent Heterogeneous Deep Learning Runtime

A lightweight, efficient deep learning inference engine with:
- Multi-backend support (CPU, CUDA, Metal)
- Intelligent operation routing
- Graph optimization and fusion
- ONNX model import
- Zero-copy numpy integration

Quick Start:
    >>> import lumen
    >>> lumen.print_system_info()
    >>> x = lumen.randn(4, 4)
    >>> y = lumen.randn(4, 4)
    >>> z = lumen.matmul(x, y)
    >>> print(z.numpy())

For ONNX models:
    >>> model = lumen.load_onnx('model.onnx')
    >>> model.compile()
    >>> output = model(input_tensor)
"""

__version__ = "0.1.0"

# Import C++ extension
try:
    import lumen_py as _lumen_py
    _has_cpp_ext = True
except ImportError as e:
    _has_cpp_ext = False
    _import_error = str(e)

if not _has_cpp_ext:
    raise ImportError(
        f"Failed to import lumen_py C++ extension: {_import_error}\n"
        "Make sure you've built the project with CMake:\n"
        "  ./run.sh\n"
        "Or add the build directory to PYTHONPATH:\n"
        "  export PYTHONPATH=$PYTHONPATH:./build/lib"
    )

# Core low-level API (direct C++ bindings)
from lumen_py import (
    Runtime,
    Buffer,
    Event,
    Graph,
    ExecutableGraph,
    OpAttributes,
    DataType,
    ONNXImporter,
    get_available_backends,
    supports_cuda,
    supports_metal,
    create_conv2d_attrs,
    create_pool2d_attrs,
)

# High-level Python API
from .core import (
    # Tensor class
    Tensor,
    
    # Tensor creation
    tensor,
    zeros,
    ones,
    randn,
    empty,
    
    # Operations
    add,
    mul,
    matmul,
    relu,
    sigmoid,
    softmax,
    conv2d,
    
    # Model loading
    Model,
    load_onnx,
    
    # Backend management
    get_default_runtime,
    set_backend,
    get_backend,
    available_backends,
    
    # Utilities
    benchmark_backends,
    print_system_info,
)

# Submodules
from . import nn  # Neural network layers
# from . import optim  # Optimizers (future)
# from . import utils  # Utilities

__all__ = [
    # Core types
    'Runtime',
    'Buffer',
    'Event',
    'Tensor',
    'Graph',
    'ExecutableGraph',
    'Model',
    'OpAttributes',
    'DataType',
    
    # Tensor creation
    'tensor',
    'zeros',
    'ones',
    'randn',
    'empty',
    
    # Operations
    'add',
    'mul',
    'matmul',
    'relu',
    'sigmoid',
    'softmax',
    'conv2d',
    
    # Model loading
    'load_onnx',
    'ONNXImporter',
    
    # Backend management
    'set_backend',
    'get_backend',
    'available_backends',
    'get_default_runtime',
    
    # Utilities
    'benchmark_backends',
    'print_system_info',
    'supports_cuda',
    'supports_metal',
    
    # Helpers
    'create_conv2d_attrs',
    'create_pool2d_attrs',
]