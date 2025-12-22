Lumen is an intelligent, heterogeneous deep learning inference runtime designed for high-performance execution across various hardware backends, including CPU, CUDA, and Metal. It features a multi-layered architecture encompassing a low-level runtime for asynchronous execution, a middle-tier Graph IR for optimizations, and a high-level Python API with a PyTorch-like interface.

### Project Analysis

The Lumen project is organized into several key modules:

* **Core Runtime**: Manages device memory via an optimized memory pool and handles asynchronous execution through event tracking.
* **Intelligent Router**: Dynamically selects the most efficient backend (CPU, CUDA, or Metal) for specific operations based on kernel latency and throughput metrics.
* **Graph IR and Optimization**: Represents computations as directed graphs, allowing for optimization passes such as operation fusion (e.g., fusing Conv2d and ReLU into a single kernel) and dead code elimination to reduce computational overhead.
* **Memory Management**: Implements a memory planner that analyzes tensor liveness to reuse buffers, significantly reducing peak memory consumption.
* **ONNX Integration**: Provides a native importer to convert ONNX models into Lumen’s internal Graph IR.
* **Python Bindings**: Uses pybind11 to provide a high-level API with zero-copy NumPy integration, allowing for seamless transitions between Python-based model definitions and high-performance C++ execution.

Benchmarking results indicate that the runtime's memory planner can achieve up to 50% VRAM savings by optimizing buffer reuse. Performance comparisons show significant speedups on GPU backends, with Metal completing matrix multiplications nearly twice as fast as the CPU reference.

---

# Lumen: Intelligent Heterogeneous Deep Learning Runtime

Lumen is a lightweight, efficient deep learning inference engine designed to provide high-performance execution across CPU, NVIDIA CUDA, and Apple Metal backends.

## Key Features

* **Multi-Backend Support**: Seamless execution across CPU (using Accelerate or OpenBLAS), CUDA, and Metal.
* **Intelligent Routing**: Automatic selection of the optimal hardware backend for each operation based on real-time performance metrics and input size.
* **Graph Optimization**: Built-in compiler passes for operation fusion and dead code elimination to maximize execution speed.
* **Memory Efficiency**: An advanced memory planner that reduces peak memory usage through tensor liveness analysis and buffer reuse.
* **ONNX Support**: Native importer for loading and executing ONNX models directly.
* **Python Integration**: A high-level Python API with zero-copy NumPy integration for ease of use.

## Architecture

Lumen's architecture is divided into three primary layers:

1. **Core Layer**: Foundational components for memory pooling, buffer management, and cross-backend synchronization.
2. **Graph Layer**: A middle-tier Graph IR that handles shape inference, optimization, and compilation into an executable plan.
3. **Backend Layer**: Specialized implementations for different hardware, providing high-performance kernels for various neural network operations.

## Building the Project

Lumen uses CMake for its build system. To build the project and its Python bindings, ensure you have the necessary dependencies (Protobuf, ONNX, and a supported GPU toolkit) and run:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

```

Alternatively, use the provided helper script:

```bash
./run.sh

```

This script will detect your platform, install requirements like pybind11, and run the test suite.

## Usage

### Python API

The Python API provides a familiar interface for tensor operations and model inference.

```python
import lumen
import numpy as np

# Create tensors
x = lumen.randn(1, 3, 224, 224)
w = lumen.randn(64, 3, 3, 3)

# Run operations
y = lumen.conv2d(x, w, stride=(1, 1), padding=(1, 1))
z = lumen.relu(y)

# Load ONNX models
model = lumen.load_onnx('model.onnx')
model.compile()
output = model(x)

```

### C++ API

For low-level control, the C++ API allows manual graph construction and execution.

```cpp
lumen::Runtime rt;
lumen::Graph graph;

auto *in = graph.add_input("input", {1, 10});
auto *w = graph.add_weight("weight", {10, 5});
auto *out = graph.add_op("matmul", {in, w});

graph.mark_output(out);
graph.optimize();

auto *executable = graph.compile(&rt);
auto input_buf = rt.alloc({1, 10});
auto outputs = executable->execute({input_buf});

```

## Performance Benchmarks

Initial benchmarks on an Apple M-series system demonstrate the runtime's efficiency:

* **MatMul (4096x4096)**: Metal latency of 26.132 ms vs CPU latency of 44.371 ms.
* **Memory Planner**: Achieved a 50% reduction in peak memory usage by optimizing buffer reuse in a 20-layer MLP.
* **Model Fusion**: Successfully fused Conv2d and ReLU operations into a single kernel execution, reducing dispatch overhead.