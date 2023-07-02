import onnx
#import onnxruntime
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import os
import time
from imagenet_lable import imagenet_label

def get_onnx_model(path):
  model = models.resnet50(pretrained=True)
  model.eval()

  x = torch.randn(1, 3, 224, 224, requires_grad=True)

  torch.onnx.export(model,
                    x,
                    path,
                    export_params=True,#导入参数
                    opset_version=10,
                    input_names=["input"], #指定输入的名称（key）
                    output_names=['output'],
                    dynamic_axes={'input':{0:'batchsize'}, 'output':{0:'batchsize'}}
  )
  

def image_process(path):
  img = Image.open(path)
  img = img.resize((224, 224))
  img = np.array(img, dtype = np.float32)
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  mean = np.array(mean).reshape((1, 1, -1))
  std = np.array(std).reshape((1, 1, -1))
  img = (img / 255.0 - mean) / std
  img = np.transpose(img, [2, 0, 1]).astype(np.float32)
  return img

# 模型推理接口
def image_infer(path, model_path):
    data = image_process(path)
    
    session = onnxruntime.InferenceSession(model_path)

    input_data = [data]
    input_name_1 = session.get_inputs()[0].name

    outputs = session.run([],{input_name_1:input_data})
    index = outputs[0][0].argmax()
    print(imagenet_label[index])
    return imagenet_label[index]
    
    
if __name__ == "__main__":
    get_onnx_model("resnet50.onnx")
    # image_infer("cat.jpg", "resnet50.onnx")
    print("run resnet_infer.py successfully")
