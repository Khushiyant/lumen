import onnx
from onnx import helper, TensorProto
import numpy as np
import os

def main():
    # create models dir in current directory
    os.makedirs("models", exist_ok=True)
    output_path = "./models/mnist.onnx"

    print(f"Generating {output_path} using native ONNX...")

    # 1. Define Inputs/Outputs
    # Input: [1, 1, 28, 28]
    input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 28, 28])
    output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])

    # 2. Weights (Randomized)
    # Conv: [4, 1, 3, 3]
    conv_w_data = np.random.randn(4, 1, 3, 3).astype(np.float32)
    conv_w = helper.make_tensor('conv_w', TensorProto.FLOAT, [4, 1, 3, 3], conv_w_data.flatten())

    # FC: [3136, 10]
    fc_w_data = (np.random.randn(3136, 10) / 100.0).astype(np.float32)
    fc_w = helper.make_tensor('fc_w', TensorProto.FLOAT, [3136, 10], fc_w_data.flatten())

    # 3. Nodes
    # Conv2d (padding=1 for 3x3 kernel preserves 28x28 size)
    conv_node = helper.make_node(
        'Conv', inputs=['input', 'conv_w'], outputs=['conv_out'], name='Conv1',
        pads=[1, 1, 1, 1], strides=[1, 1]
    )
    # ReLU
    relu_node = helper.make_node('Relu', inputs=['conv_out'], outputs=['relu_out'], name='Relu1')
    # Flatten
    flat_node = helper.make_node('Flatten', inputs=['relu_out'], outputs=['flat_out'], name='Flatten1')
    # MatMul (FC)
    matmul_node = helper.make_node('MatMul', inputs=['flat_out', 'fc_w'], outputs=['output'], name='Fc1')

    # 4. Graph & Model
    graph_def = helper.make_graph(
        [conv_node, relu_node, flat_node, matmul_node],
        'lumen_mnist_test',
        [input_info], [output_info],
        [conv_w, fc_w]
    )
    model_def = helper.make_model(graph_def, producer_name='lumen_native', opset_imports=[helper.make_opsetid("", 13)])
    
    onnx.save(model_def, output_path)
    print(f"✅ Success! Model saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()