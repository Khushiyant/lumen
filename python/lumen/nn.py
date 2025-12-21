"""
Neural network layers and modules (PyTorch-like interface)
"""

import numpy as np
from typing import Optional, Tuple
from .core import Tensor, matmul, add, conv2d, relu, sigmoid, softmax


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self._parameters = {}
    
    def __call__(self, *args, **kwargs):
        """Forward pass."""
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Define the forward computation. Must be overridden."""
        raise NotImplementedError
    
    def parameters(self):
        """Return iterator over module parameters."""
        return iter(self._parameters.values())
    
    def named_parameters(self):
        """Return iterator over (name, parameter) pairs."""
        return iter(self._parameters.items())


class Linear(Module):
    """
    Fully connected layer: y = xW + b
    
    Args:
        in_features: Size of input
        out_features: Size of output
        bias: Whether to include bias term
    
    Example:
        >>> layer = lumen.nn.Linear(784, 128)
        >>> x = lumen.randn(32, 784)
        >>> y = layer(x)
        >>> print(y.shape)  # (32, 128)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with Xavier/Glorot initialization
        from .core import tensor
        scale = np.sqrt(2.0 / (in_features + out_features))
        weight_data = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.weight = tensor(weight_data)
        self._parameters['weight'] = self.weight
        
        if bias:
            self.bias = tensor(np.zeros(out_features, dtype=np.float32))
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        y = matmul(x, self.weight)
        if self.bias is not None:
            y = add(y, self.bias)
        return y
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class Conv2d(Module):
    """
    2D Convolution layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Zero padding added to input
    
    Example:
        >>> conv = lumen.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        >>> x = lumen.randn(1, 3, 224, 224)
        >>> y = conv(x)
        >>> print(y.shape)  # (1, 64, 224, 224)
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights with Kaiming initialization
        from .core import tensor
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        weight_data = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32) * scale
        self.weight = tensor(weight_data)
        self._parameters['weight'] = self.weight
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return conv2d(x, self.weight, 
                     stride=(self.stride, self.stride),
                     padding=(self.padding, self.padding))
    
    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")


class ReLU(Module):
    """ReLU activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)
    
    def __repr__(self):
        return "Sigmoid()"


class Softmax(Module):
    """Softmax activation function."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, axis=self.dim)
    
    def __repr__(self):
        return f"Softmax(dim={self.dim})"


class Sequential(Module):
    """
    Sequential container for stacking layers.
    
    Example:
        >>> model = lumen.nn.Sequential(
        ...     lumen.nn.Linear(784, 128),
        ...     lumen.nn.ReLU(),
        ...     lumen.nn.Linear(128, 10),
        ...     lumen.nn.Softmax()
        ... )
        >>> x = lumen.randn(32, 784)
        >>> y = model(x)
    """
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        
        # Collect parameters from all layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Module):
                for name, param in layer.named_parameters():
                    self._parameters[f"layer{i}_{name}"] = param
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __repr__(self):
        layer_str = "\n  ".join([f"({i}): {layer}" for i, layer in enumerate(self.layers)])
        return f"Sequential(\n  {layer_str}\n)"


class Flatten(Module):
    """Flatten input tensor."""
    
    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim
    
    def forward(self, x: Tensor) -> Tensor:
        """Flatten all dimensions after start_dim."""
        shape = x.shape
        if self.start_dim == 1:
            new_shape = (shape[0], -1)
        else:
            raise NotImplementedError("Only start_dim=1 supported currently")
        
        # For now, return reshaped numpy array as new tensor
        from .core import tensor
        flat = x.numpy().reshape(shape[0], -1)
        return tensor(flat)
    
    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"


# ============================================================================
# Example Models
# ============================================================================

class SimpleMLP(Module):
    """
    Simple Multi-Layer Perceptron for classification.
    
    Example:
        >>> model = lumen.nn.SimpleMLP(784, 128, 10)
        >>> x = lumen.randn(32, 784)
        >>> logits = model(x)
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes)
        
        # Collect parameters
        for name, param in self.fc1.named_parameters():
            self._parameters[f"fc1_{name}"] = param
        for name, param in self.fc2.named_parameters():
            self._parameters[f"fc2_{name}"] = param
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def __repr__(self):
        return f"SimpleMLP(\n  {self.fc1}\n  {self.relu}\n  {self.fc2}\n)"


def test_nn_modules():
    """Test neural network modules."""
    print("\nTesting nn.Module classes:")
    
    from .core import randn
    
    # Test Linear
    print("\n1. Linear layer:")
    linear = Linear(10, 5)
    x = randn(2, 10)
    y = linear(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    
    # Test Conv2d
    print("\n2. Conv2d layer:")
    conv = Conv2d(3, 16, kernel_size=3, padding=1)
    x = randn(1, 3, 32, 32)
    y = conv(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    
    # Test Sequential
    print("\n3. Sequential model:")
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
        Softmax()
    )
    x = randn(4, 784)
    y = model(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Model:\n{model}")
    
    # Test SimpleMLP
    print("\n4. SimpleMLP:")
    mlp = SimpleMLP(784, 256, 10)
    x = randn(8, 784)
    y = mlp(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Model:\n{mlp}")


if __name__ == "__main__":
    test_nn_modules()