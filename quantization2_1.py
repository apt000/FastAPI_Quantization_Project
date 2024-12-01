import os

import torch
import tensorflow as tf
from pathlib import Path
import torch.quantization as quant
import keras
from torch import optim
from torchvision import datasets, transforms
from convert import convert_tensorflow_to_onnx, convert_onnx_to_pytorch

global save_path, model_structure


def quantize_model(model_path: Path, quantization_type: str, model_type: str, model_name: str, quantization_method,
                   quantization_method2=None):
    global save_path, model_structure, model
    # 根据选择的量化类型
    dtype = torch.qint8 if quantization_type == "int8" else torch.qint4

    if model_type == "tensorflow":
        # TensorFlow 转 onnx
        onnx_path = convert_tensorflow_to_onnx(model_name, str(model_path))
        # onnx_path = convert_tensorflow_to_onnx(model_name, 'tf_model.h5')
        print(onnx_path)
        model_path = convert_onnx_to_pytorch(model_name, onnx_path)
        print(model_path)
        model = torch.load(model_path)
    elif model_type == "pytorch":
        # 加载模型
        model = torch.jit.load(model_path)
    model.eval()
    # 动态量化
    if quantization_method == "dynamic":
        if quantization_method2 == "MinMax":
            # 配置 MinMax 动态量化，适合 CPU 量化，主要用于推理加速。
            model.qconfig = quant.QConfig(
                activation=quant.MinMaxObserver.with_args(dtype=dtype),
                weight=quant.MinMaxObserver.with_args(dtype=dtype)
            )
        elif quantization_method2 == "EQ":
            # 对每层进行均衡化，适合 CPU 量化，主要用于推理加速。
            for name, param in model.named_parameters():
                max_val = torch.max(torch.abs(param))
                param.data = param.data / max_val
        # 进行量化
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=dtype
        )

    # 非对称量化（静态量化）
    elif quantization_method == "asymmetric":
        # 准备量化
        # model.fuse_model()  # 对模型进行层融合，通常是 Conv + BN 等层的融合
        if quantization_method2 == "KLD":
            # 设置非对称量化的 KLD 配置，适合静态量化且需要高精度场景。
            model.qconfig = quant.QConfig(
                activation=quant.HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False),
                weight=quant.MinMaxObserver.with_args(dtype=torch.qint8)
            )
        elif quantization_method2 == "EQ":
            # 均衡化（EQ）+ 非对称量化，适合精度要求高的非对称量化场景。
            for name, param in model.named_parameters():
                scale = torch.max(torch.abs(param))
                param.data = param.data / scale  # 均衡化操作
                param.data = torch.round(param.data * 127) / 127  # 8-bit 非对称量化
        else:
            model.qconfig = quant.get_default_qconfig('fbgemm')  # 设置量化配置

        # 准备量化
        torch.quantization.prepare(model, inplace=True)

        # 定义数据转换和数据集
        transform = transforms.Compose([transforms.ToTensor()])
        calibration_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=32)

        # 校准模型
        with torch.no_grad():
            for images, labels in calibration_loader:
                model(images)

        # 将模型转换为量化模型
        torch.quantization.convert(model, inplace=True)

    elif quantization_method == "nonlinear":
        # 定义简单的非线性量化（对数量化）
        def log_quantize(tensor, scale=1.0):
            tensor_sign = torch.sign(tensor)
            tensor = torch.log1p(torch.abs(tensor)) * scale  # 对数映射
            return tensor_sign * tensor

        # 对模型中的权重进行非线性量化
        for param in model.parameters():
            param.data = log_quantize(param.data, scale=0.1)

    # 保存量化后的模型
    # save_path = f'D:/code_python_project/model_quantization/temp/{model_name}_quantized.pt'
    temp_path = f'./temp/{model_name}_quantized.pt'
    save_path = os.path.abspath(temp_path)
    if model_type == "pytorch":
        model.eval()  # 切换到评估模式
        scripted_model = torch.jit.script(model)  # 转换为 TorchScript
        # scripted_model.save(scripted_model, save_path)  # 保存模型
        scripted_model.save(f"./temp/{model_name}_quantized.pt")
    elif model_type == "tensorflow":
        torch.save(model, f"./temp/{model_name}_quantized.pt")

    # 返回模型的结构
    model_structure = str(model)

    '''elif model_type == "tensorflow":
        model = keras.models.load_model(model_path)

        # TensorFlow 模型量化 (假设基于int8/int4的量化)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if quantization_type == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        else:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int4]

        tflite_model = converter.convert()

        save_path = model_path.with_suffix(f"D:/code_python_project/model_quantization/temp/{model_name}_quantized.tflite")
        with open(save_path, "wb") as f:
            f.write(tflite_model)

        # 返回模型的结构
        model_structure = model.summary(print_fn=lambda x: x)'''

    return save_path, model_structure

# print(quantize_model(Path('cnn_model.pt'),'int8','pytorch','cnn_model_qt'))
# print(quantize_model(Path('mnist_model.h5'),'int8','tensorflow','mnist_model','dynamic'))
