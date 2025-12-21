"""
Lumen: Intelligent Heterogeneous Deep Learning Runtime
High-level Python API with PyTorch-like interface
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import lumen_py

__version__ = "0.1.0"

# Re-export core classes
Runtime = lumen_py.Runtime
Buffer = lumen_py.Buffer
Graph = lumen_py.Graph
ExecutableGraph = lumen_py.ExecutableGraph
OpAttributes = lumen_py.OpAttributes
DataType = lumen_py.DataType
ONNXImporter = lumen_py.ONNXImporter


class Tensor:
    """
    High-level tensor wrapper with automatic memory management.
    Provides numpy-like interface with automatic backend execution.
    """
    
    def __init__(self, data: Union[np.ndarray, List, Tuple, lumen_py.Buffer], 
                 runtime: Optional[Runtime] = None):
        """
        Create a tensor from numpy array, list, or existing buffer.
        
        Args:
            data: Input data as numpy array, list, or Buffer
            runtime: Runtime instance (creates default if None)
        """
        if runtime is None:
            runtime = get_default_runtime()
        self.runtime = runtime
        
        if isinstance(data, lumen_py.Buffer):
            self.buffer = data
        else:
            # Convert to numpy array
            arr = np.asarray(data, dtype=np.float32)
            
            # Allocate buffer
            self.buffer = runtime.alloc(list(arr.shape))
            
            # Copy data
            self.buffer.data()[:] = arr
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return tuple(self.buffer.shape)
    
    @property
    def size(self) -> int:
        """Get total number of elements."""
        return self.buffer.size
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self.buffer.shape)
    
    @property
    def dtype(self):
        """Get data type (always float32 for now)."""
        return np.float32
    
    def numpy(self) -> np.ndarray:
        """Get numpy array view of data (zero-copy)."""
        return self.buffer.data()
    
    def __array__(self) -> np.ndarray:
        """Allow numpy operations on tensor."""
        return self.numpy()
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"
    
    # Arithmetic operations
    def __add__(self, other):
        """Element-wise addition."""
        return add(self, other)
    
    def __mul__(self, other):
        """Element-wise multiplication."""
        return mul(self, other)
    
    def __matmul__(self, other):
        """Matrix multiplication."""
        return matmul(self, other)
    
    def __getitem__(self, key):
        """Get numpy view and index it."""
        return self.numpy()[key]
    
    def __setitem__(self, key, value):
        """Set values through numpy view."""
        self.numpy()[key] = value


# Global default runtime
_default_runtime: Optional[Runtime] = None


def get_default_runtime() -> Runtime:
    """Get or create default runtime instance."""
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = Runtime()
    return _default_runtime


def set_backend(backend: str):
    """Set backend for default runtime."""
    get_default_runtime().set_backend(backend)


def get_backend() -> str:
    """Get current backend of default runtime."""
    return get_default_runtime().current_backend()


def available_backends() -> List[str]:
    """Get list of available backends."""
    return lumen_py.get_available_backends()


def tensor(data: Union[np.ndarray, List, Tuple], 
           dtype: Optional[np.dtype] = None,
           runtime: Optional[Runtime] = None) -> Tensor:
    """
    Create a tensor from data.
    
    Args:
        data: Input data as numpy array or list
        dtype: Data type (numpy dtype)
        runtime: Runtime instance (uses default if None)
    
    Returns:
        Tensor object
    
    Example:
        >>> import lumen
        >>> x = lumen.tensor([[1, 2], [3, 4]])
        >>> print(x.shape)
        (2, 2)
    """
    if dtype is not None:
        data = np.asarray(data, dtype=dtype)
    return Tensor(data, runtime)


def zeros(shape: Tuple[int, ...], runtime: Optional[Runtime] = None) -> Tensor:
    """Create tensor filled with zeros."""
    return Tensor(np.zeros(shape, dtype=np.float32), runtime)


def ones(shape: Tuple[int, ...], runtime: Optional[Runtime] = None) -> Tensor:
    """Create tensor filled with ones."""
    return Tensor(np.ones(shape, dtype=np.float32), runtime)


def randn(*shape: int, runtime: Optional[Runtime] = None) -> Tensor:
    """Create tensor with random normal values."""
    return Tensor(np.random.randn(*shape).astype(np.float32), runtime)


def empty(shape: Tuple[int, ...], runtime: Optional[Runtime] = None) -> Tensor:
    """Create uninitialized tensor."""
    if runtime is None:
        runtime = get_default_runtime()
    return Tensor(runtime.alloc(list(shape)), runtime)


# ============================================================================
# OPERATIONS
# ============================================================================

def _ensure_tensor(x: Union[Tensor, np.ndarray, List]) -> Tensor:
    """Convert input to Tensor if needed."""
    if isinstance(x, Tensor):
        return x
    return tensor(x)


def add(a: Union[Tensor, np.ndarray], b: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Element-wise addition with broadcasting.
    
    Example:
        >>> x = lumen.tensor([1, 2, 3])
        >>> y = lumen.tensor([4, 5, 6])
        >>> z = lumen.add(x, y)
    """
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    rt = a.runtime
    
    # Compute output shape (simplified - assumes compatible shapes)
    out_shape = list(a.shape if len(a.shape) >= len(b.shape) else b.shape)
    out = empty(tuple(out_shape), rt)
    
    rt.execute("add", [a.buffer, b.buffer], out.buffer)
    rt.submit()
    rt.wait_all()
    
    return out


def mul(a: Union[Tensor, np.ndarray], b: Union[Tensor, np.ndarray]) -> Tensor:
    """Element-wise multiplication with broadcasting."""
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    rt = a.runtime
    
    out_shape = list(a.shape if len(a.shape) >= len(b.shape) else b.shape)
    out = empty(tuple(out_shape), rt)
    
    rt.execute("mul", [a.buffer, b.buffer], out.buffer)
    rt.submit()
    rt.wait_all()
    
    return out


def matmul(a: Union[Tensor, np.ndarray], b: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Matrix multiplication.
    
    Example:
        >>> x = lumen.randn(4, 8)
        >>> w = lumen.randn(8, 10)
        >>> y = lumen.matmul(x, w)
        >>> print(y.shape)
        (4, 10)
    """
    a, b = _ensure_tensor(a), _ensure_tensor(b)
    rt = a.runtime
    
    # Compute output shape
    out_shape = list(a.shape[:-1]) + [b.shape[-1]]
    out = empty(tuple(out_shape), rt)
    
    rt.execute("matmul", [a.buffer, b.buffer], out.buffer)
    rt.submit()
    rt.wait_all()
    
    return out


def relu(x: Union[Tensor, np.ndarray]) -> Tensor:
    """Apply ReLU activation."""
    x = _ensure_tensor(x)
    rt = x.runtime
    
    out = empty(x.shape, rt)
    rt.execute("relu", [x.buffer], out.buffer)
    rt.submit()
    rt.wait_all()
    
    return out


def sigmoid(x: Union[Tensor, np.ndarray]) -> Tensor:
    """Apply sigmoid activation."""
    x = _ensure_tensor(x)
    rt = x.runtime
    
    out = empty(x.shape, rt)
    rt.execute("sigmoid", [x.buffer], out.buffer)
    rt.submit()
    rt.wait_all()
    
    return out


def softmax(x: Union[Tensor, np.ndarray], axis: int = -1) -> Tensor:
    """Apply softmax along specified axis."""
    x = _ensure_tensor(x)
    rt = x.runtime
    
    out = empty(x.shape, rt)
    rt.execute("softmax", [x.buffer], out.buffer)
    rt.submit()
    rt.wait_all()
    
    return out


def conv2d(input: Tensor, weight: Tensor, 
           stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0)) -> Tensor:
    """
    2D convolution operation.
    
    Args:
        input: Input tensor [N, C_in, H, W]
        weight: Weight tensor [C_out, C_in, K_h, K_w]
        stride: Stride for convolution
        padding: Padding for convolution
    
    Returns:
        Output tensor [N, C_out, H_out, W_out]
    """
    rt = input.runtime
    
    # Compute output shape
    N, C_in, H, W = input.shape
    C_out, _, K_h, K_w = weight.shape
    
    H_out = (H + 2 * padding[0] - K_h) // stride[0] + 1
    W_out = (W + 2 * padding[1] - K_w) // stride[1] + 1
    
    out = empty((N, C_out, H_out, W_out), rt)
    
    attrs = lumen_py.create_conv2d_attrs(
        stride=list(stride),
        padding=list(padding)
    )
    
    rt.execute("conv2d", [input.buffer, weight.buffer], out.buffer, attrs)
    rt.submit()
    rt.wait_all()
    
    return out


# ============================================================================
# MODEL LOADING
# ============================================================================

class Model:
    """
    High-level model wrapper for inference.
    """
    
    def __init__(self, graph: Graph, runtime: Runtime):
        """Initialize model from graph and runtime."""
        self.graph = graph
        self.runtime = runtime
        self.executable: Optional[ExecutableGraph] = None
    
    def compile(self, optimize: bool = True):
        """
        Compile the model for execution.
        
        Args:
            optimize: Run optimization passes
        """
        if optimize:
            self.graph.optimize()
        self.executable = self.graph.compile(self.runtime)
    
    def __call__(self, *inputs: Tensor) -> List[Tensor]:
        """
        Run inference on inputs.
        
        Args:
            inputs: Input tensors
        
        Returns:
            List of output tensors
        """
        if self.executable is None:
            self.compile()
        
        # Convert inputs to buffers
        input_buffers = [inp.buffer for inp in inputs]
        
        # Execute
        output_buffers = self.executable.execute(input_buffers)
        
        # Wrap outputs as tensors
        return [Tensor(buf, self.runtime) for buf in output_buffers]
    
    def profile(self, *inputs: Tensor) -> List[Dict[str, Any]]:
        """
        Profile model execution.
        
        Returns:
            List of profiling data for each operation
        """
        if self.executable is None:
            self.compile()
        
        input_buffers = [inp.buffer for inp in inputs]
        prof_data = self.executable.profile(input_buffers)
        
        return [
            {
                'node_name': p.node_name,
                'op_type': p.op_type,
                'time_ms': p.time_ms,
                'backend': p.backend
            }
            for p in prof_data
        ]
    
    @property
    def memory_usage(self) -> int:
        """Get peak memory usage in bytes."""
        if self.executable is None:
            self.compile()
        return self.executable.memory_usage


def load_onnx(model_path: str, runtime: Optional[Runtime] = None) -> Model:
    """
    Load ONNX model and return high-level Model object.
    
    Args:
        model_path: Path to .onnx file
        runtime: Runtime instance (uses default if None)
    
    Returns:
        Model object ready for inference
    
    Example:
        >>> import lumen
        >>> model = lumen.load_onnx('resnet18.onnx')
        >>> model.compile()
        >>> x = lumen.randn(1, 3, 224, 224)
        >>> y = model(x)
    """
    if runtime is None:
        runtime = get_default_runtime()
    
    graph = ONNXImporter.import_model(model_path)
    return Model(graph, runtime)


# ============================================================================
# UTILITIES
# ============================================================================

def benchmark_backends(operation: str = "matmul", size: int = 1024, 
                      iterations: int = 10) -> Dict[str, float]:
    """
    Benchmark all available backends for a specific operation.
    
    Args:
        operation: Operation name to benchmark
        size: Problem size (dimension for matmul)
        iterations: Number of iterations to average
    
    Returns:
        Dictionary mapping backend name to average time in milliseconds
    """
    import time
    
    results = {}
    
    for backend in available_backends():
        rt = Runtime()
        try:
            rt.set_backend(backend)
        except:
            continue
        
        # Create test data
        if operation == "matmul":
            a = rt.alloc([size, size])
            b = rt.alloc([size, size])
            c = rt.alloc([size, size])
            inputs = [a, b]
            output = c
        else:
            a = rt.alloc([size, size])
            b = rt.alloc([size, size])
            c = rt.alloc([size, size])
            inputs = [a, b]
            output = c
        
        # Warmup
        rt.execute(operation, inputs, output)
        rt.submit()
        rt.wait_all()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            rt.execute(operation, inputs, output)
            rt.submit()
            rt.wait_all()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        results[backend] = sum(times) / len(times)
    
    return results


def print_system_info():
    """Print information about available backends and system."""
    print("Lumen Deep Learning Runtime")
    print(f"Version: {__version__}")
    print(f"\nAvailable backends: {', '.join(available_backends())}")
    print(f"CUDA support: {lumen_py.supports_cuda()}")
    print(f"Metal support: {lumen_py.supports_metal()}")
    print(f"Current backend: {get_backend()}")


if __name__ == "__main__":
    print_system_info()