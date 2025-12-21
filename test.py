"""
Lumen Python API Examples
Demonstrates high-level and low-level usage patterns
"""

import numpy as np
import sys
import os

# Add build directory to path
sys.path.append(os.path.abspath("./build/lib"))
import lumen

# ============================================================================
# Example 1: Basic Tensor Operations (High-Level API)
# ============================================================================

def example_basic_operations():
    print("\n" + "="*60)
    print("Example 1: Basic Tensor Operations")
    print("="*60)
    
    # Create tensors
    x = lumen.tensor([[1, 2, 3], 
                      [4, 5, 6]], dtype=np.float32)
    y = lumen.tensor([[10, 20, 30], 
                      [40, 50, 60]], dtype=np.float32)
    
    print(f"x shape: {x.shape}")
    print(f"x data:\n{x.numpy()}")
    
    # Addition
    z = lumen.add(x, y)
    print(f"\nx + y =\n{z.numpy()}")
    
    # Or use operator overloading
    z2 = x + y
    print(f"\nUsing operator: x + y =\n{z2.numpy()}")
    
    # Element-wise multiplication
    w = x * y
    print(f"\nx * y =\n{w.numpy()}")


# ============================================================================
# Example 2: Matrix Multiplication and Linear Layer
# ============================================================================

def example_linear_layer():
    print("\n" + "="*60)
    print("Example 2: Linear Layer (Matrix Multiplication)")
    print("="*60)
    
    lumen.set_backend('cpu')  # Use CPU backend
    
    # Input: batch_size=2, features=4
    x = lumen.randn(2, 4)
    
    # Weight: 4 input features -> 3 output features
    W = lumen.randn(4, 3)
    
    # Bias
    b = lumen.ones((3,))
    
    # Linear transformation: y = xW + b
    y = lumen.matmul(x, W) + b
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {W.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output:\n{y.numpy()}")


# ============================================================================
# Example 3: Backend Selection and Benchmarking
# ============================================================================

def example_backend_selection():
    print("\n" + "="*60)
    print("Example 3: Backend Selection")
    print("="*60)
    
    print(f"Available backends: {lumen.available_backends()}")
    
    # Test each backend
    size = 512
    for backend in lumen.available_backends():
        try:
            lumen.set_backend(backend)
            print(f"\n[{backend.upper()}]")
            
            # Create test matrices
            A = lumen.randn(size, size)
            B = lumen.randn(size, size)
            
            import time
            start = time.perf_counter()
            C = lumen.matmul(A, B)
            end = time.perf_counter()
            
            print(f"  MatMul ({size}x{size}): {(end-start)*1000:.2f} ms")
            
        except Exception as e:
            print(f"  Backend {backend} failed: {e}")


# ============================================================================
# Example 4: Convolution Operation
# ============================================================================

def example_convolution():
    print("\n" + "="*60)
    print("Example 4: 2D Convolution")
    print("="*60)
    
    # Input: 1 image, 3 channels, 32x32
    x = lumen.randn(1, 3, 32, 32)
    
    # Weight: 16 output channels, 3 input channels, 3x3 kernel
    w = lumen.randn(16, 3, 3, 3)
    
    # Convolution with stride=1, padding=1
    y = lumen.conv2d(x, w, stride=(1, 1), padding=(1, 1))
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {w.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected: (1, 16, 32, 32) - same spatial size due to padding")


# ============================================================================
# Example 5: Building and Executing a Graph (Low-Level API)
# ============================================================================

def example_graph_api():
    print("\n" + "="*60)
    print("Example 5: Graph IR (Low-Level API)")
    print("="*60)
    
    rt = lumen.Runtime()
    graph = lumen.Graph()
    
    # Build a simple MLP: input -> linear -> relu -> linear -> softmax
    x = graph.add_input('input', [1, 784])
    
    # First layer: 784 -> 128
    w1 = graph.add_weight('w1', [784, 128], 
                          data=np.random.randn(784, 128).astype(np.float32) * 0.01)
    h1 = graph.add_op('matmul', [x, w1])
    h1_relu = graph.add_op('relu', [h1])
    
    # Second layer: 128 -> 10
    w2 = graph.add_weight('w2', [128, 10],
                          data=np.random.randn(128, 10).astype(np.float32) * 0.01)
    h2 = graph.add_op('matmul', [h1_relu, w2])
    output = graph.add_op('softmax', [h2])
    
    graph.mark_output(output)
    
    # Print graph structure
    graph.print_summary()
    
    # Optimize and compile
    graph.optimize()
    executable = graph.compile(rt)
    
    # Run inference
    input_buffer = rt.alloc([1, 784])
    input_buffer.data()[:] = np.random.randn(1, 784).astype(np.float32)
    
    outputs = executable.execute([input_buffer])
    result = outputs[0].data()
    
    print(f"\nOutput shape: {result.shape}")
    print(f"Output (first 10 values): {result[0, :10]}")
    print(f"Sum of probabilities: {result.sum():.4f} (should be ~1.0)")
    
    print(f"\nPeak memory usage: {executable.memory_usage / 1024 / 1024:.2f} MB")


# ============================================================================
# Example 6: Loading and Running ONNX Model
# ============================================================================

def example_onnx_inference():
    print("\n" + "="*60)
    print("Example 6: ONNX Model Inference")
    print("="*60)
    
    model_path = "./models/mnist.onnx"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run test_model.py to generate it first")
        return
    
    # Load model
    model = lumen.load_onnx(model_path)
    model.compile(optimize=True)
    
    print(f"Model loaded and compiled")
    print(f"Peak memory: {model.memory_usage / 1024 / 1024:.2f} MB")
    
    # Create dummy input (1 image, 1 channel, 28x28)
    x = lumen.randn(1, 1, 28, 28)
    
    # Run inference
    outputs = model(x)
    result = outputs[0].numpy()
    
    print(f"\nOutput shape: {result.shape}")
    print(f"Predicted class: {result.argmax()}")
    print(f"Class probabilities: {result[0]}")
    
    # Profile execution
    print("\nProfiling model execution:")
    prof = model.profile(x)
    
    total_time = sum(p['time_ms'] for p in prof)
    print(f"Total time: {total_time:.3f} ms")
    
    for p in prof:
        pct = (p['time_ms'] / total_time) * 100
        print(f"  {p['op_type']:20s} {p['time_ms']:8.3f} ms ({pct:5.1f}%) [{p['backend']}]")


# ============================================================================
# Example 7: Broadcasting
# ============================================================================

def example_broadcasting():
    print("\n" + "="*60)
    print("Example 7: Broadcasting")
    print("="*60)
    
    # Matrix + vector (broadcasts vector to match matrix)
    x = lumen.ones((3, 4))
    y = lumen.tensor([10, 20, 30, 40])
    
    z = x + y
    
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"z = x + y shape: {z.shape}")
    print(f"Result:\n{z.numpy()}")


# ============================================================================
# Example 8: Performance Benchmarking
# ============================================================================

def example_benchmarking():
    print("\n" + "="*60)
    print("Example 8: Backend Performance Comparison")
    print("="*60)
    
    results = lumen.benchmark_backends(operation='matmul', size=1024, iterations=5)
    
    print("\nMatMul (1024x1024) - Average of 5 runs:")
    for backend, time_ms in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {backend:10s}: {time_ms:8.2f} ms")
    
    if len(results) > 1:
        fastest = min(results.values())
        print("\nSpeedup vs CPU:")
        for backend, time_ms in results.items():
            if backend != 'cpu':
                speedup = results.get('cpu', time_ms) / time_ms
                print(f"  {backend:10s}: {speedup:.2f}x faster")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LUMEN PYTHON API EXAMPLES")
    print("="*60)
    
    lumen.print_system_info()
    
    try:
        example_basic_operations()
        example_linear_layer()
        example_backend_selection()
        example_convolution()
        example_graph_api()
        example_broadcasting()
        example_benchmarking()
        example_onnx_inference()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)