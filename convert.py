import torch
import os
import onnx
import keras
import subprocess
import tensorflow as tf
import torch.onnx as torch_onnx
from onnx2pytorch import ConvertModel


def convert_pytorch_to_onnx(model_name: str, model_path: str, output_dir: str):
    """
    将 PyTorch 模型转换为 ONNX 的逻辑
    :param model_path: 上传的 PyTorch 模型路径
    :param output_dir: ONNX 模型保存目录
    :return: ONNX 模型的路径
    """
    # 加载 PyTorch 模型
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    # 生成 ONNX 模型保存路径
    onnx_model_path = os.path.join(output_dir, f"{os.path.basename(model_path).split('.')[0]}.onnx")

    # 模拟的输入数据，适用于 MNIST 模型
    dummy_input = torch.randn(1, 1, 28, 28)

    # 导出 ONNX 模型
    torch_onnx.export(model, dummy_input, onnx_model_path, export_params=True)

    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # onnx_model_path = f'D:/code_python_project/model_quantization/temp/{model_name}.onnx'
    onnx_model_path = f'./temp/{model_name}.onnx'
    abs_file_path = os.path.abspath(onnx_model_path)
    return onnx_model_path


def convert_tensorflow_to_onnx(model_name: str, model_path: str, output_dir: str = None):
    model = keras.models.load_model(model_path)
    tf.saved_model.save(model, 'saved_model_directory')

    process = subprocess.Popen(['python', '-m', 'tf2onnx.convert', '--saved-model', 'saved_model_directory',
                                '--output', f'./temp/{model_name}.onnx'], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.wait()
    # onnx_model_path = f'./temp/{model_name}.onnx'
    onnx_model_path = f'./temp/{model_name}.onnx'
    abs_file_path = os.path.abspath(onnx_model_path)
    return abs_file_path


def convert_onnx_to_pytorch(model_name: str, onnx_model_path: str, output_dir: str = None):
    # 加载 ONNX 模型
    # onnx_model_path = 'model.onnx'
    onnx_model = onnx.load(onnx_model_path)

    # 将 ONNX 模型转换为 PyTorch 模型
    pytorch_model = ConvertModel(onnx_model)

    # 保存完整的模型（包括结构和参数）
    pytorch_model_path = f'./temp/{model_name}.pt'
    torch.save(pytorch_model, pytorch_model_path)
    return pytorch_model_path


# print(convert_pytorch_to_onnx('cnn_model', 'cnn_model.pt', 'temp'))  # temp\cnn_model.onnx
# print(convert_tensorflow_to_onnx('mnist_model', 'mnist_model.h5', 'temp'))  # ./temp/mnist_model.onnx
# print(convert_onnx_to_pytorch('mnist_model', './temp/mnist_model.onnx', 'temp'))  # ./temp/mnist_model.pt
# print(convert_pytorch_to_onnx('mnist_model_quantized', './temp/mnist_model_quantized.pt', 'temp'))
