from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import os
import shutil
from pathlib import Path
from quantization2_1 import quantize_model
from convert import convert_pytorch_to_onnx, convert_tensorflow_to_onnx

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)  # 创建文件夹用于存储上传的文件
ONNX_DIR = "temp"

model_name = ''
model_type = ''


# 上传模型
@app.post("/quantize/")
async def quantize_model_endpoint(
        file: UploadFile = File(...),
        quantization_type: str = Form(...),  # int4 or int8
        # model_type: str = Form(...),  # pytorch or tensorflow
        quantization_method: str = Form(...),
        quantization_method2: str = Form(...),
        select_convert: str = Form(...)
):
    print(quantization_type, quantization_method, quantization_method2)
    global model_type
    try:
        # 检查文件扩展名
        if file.filename.endswith(".pt"):
            model_type = "pytorch"
        elif file.filename.endswith(".h5"):
            model_type = "tensorflow"
        else:
            raise HTTPException(status_code=400, detail="Please upload .pt or .h5 file")

        print(model_type)

        # 将上传的文件保存到服务器
        model_path = UPLOAD_FOLDER / file.filename
        global model_name  # 全局变量，保存模型名字
        model_name = os.path.splitext(file.filename)[0]
        with open(model_path, "wb") as buffer:
            buffer.write(await file.read())

        # 量化模型
        quantized_model_path, model_structure = quantize_model(model_path, quantization_type, model_type, model_name,
                                                               quantization_method, quantization_method2)

        print(quantized_model_path)

        # 是否转换成onnx格式
        if select_convert == 'yes':
            if model_type == "pytorch":
                quantized_model_path = convert_pytorch_to_onnx(model_name, str(quantized_model_path), ONNX_DIR)
            elif model_type == "tensorflow":
                onnx_model_path = f'./temp/{model_name}.onnx'
                quantized_model_path = os.path.abspath(onnx_model_path)
        '''if model_type == "pytorch":
            onnx_model_path = convert_pytorch_to_onnx(model_name, str(model_path), ONNX_DIR)
        elif model_type == "tensorflow":
            onnx_model_path = convert_tensorflow_to_onnx(model_name, str(model_path), ONNX_DIR)'''

        return {
            "quantized_model_path": str(quantized_model_path),
            "model_structure": model_structure,
            "select_convert": select_convert
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantization failed: {e}")


# 下载量化后的模型
@app.get("/download/")
async def download_model(quantized_model_path: str, select_convert=None):
    if not os.path.exists(quantized_model_path):
        raise HTTPException(status_code=404, detail="File not found")
    filename = f"{model_name}_quantized.pt"
    if select_convert == 'yes':
        filename = f"{model_name}_quantized.onnx"

    # print(filename)

    return FileResponse(path=quantized_model_path, filename=filename)


# 转换 PyTorch 模型为 ONNX
'''@app.post("/convert/")
async def convert_to_onnx(
        file: UploadFile = File(...),
        model_type: str = Form(...)):
    # 保存上传的模型文件
    model_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(model_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # model_path = UPLOAD_FOLDER / file.filename
    global model_name, onnx_model_path  # 全局变量，保存模型名字

    model_name = os.path.splitext(file.filename)[0]
    print(model_name)

    # 调用转换函数
    if model_type == "pytorch":
        onnx_model_path = convert_pytorch_to_onnx(model_name, model_path, ONNX_DIR)
    elif model_type == "tensorflow":
        onnx_model_path = convert_tensorflow_to_onnx(model_name, model_path, ONNX_DIR)

    return JSONResponse({
        "onnx_model_path": onnx_model_path
    })

# ONNX 模型下载
@app.get("/download_onnx/")
async def download_onnx_model(onnx_model: str):
    filename = f'{model_name}.onnx'
    return FileResponse(path=onnx_model, filename=filename)'''


# 显示前端页面
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index2-1.html", {"request": request})
