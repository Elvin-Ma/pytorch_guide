# 1. 部署我们自己的聊天机器人

## 1.1 代码准备
git clone git@github.com:wenda-LLM/wenda.git 
*自己download 也可以*

## 1.2 模型准备
- [链接](https://pan.baidu.com/s/1VPRGReHfnnqe_ULKjquoaQ?pwd=oaae)
- 提取码：oaae 
- 将pretrained model 放到 wenda/model 下（建议）；

## 1.3 环境准备
```shell
- conda create -n python3.10 python=3.10（建议）
- pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
- cd */wenda/requirements/ 
- pip install -r requirements.txt # update transformers==4.27.1
- pip install requirements-glm6b-lora.txt
- pip install protobuf transformers==4.27.1 cpm_kernels
```
## 1.4 运行程序
cp example.config.yml config.yml
source run_GLM6B.sh

## 1.5 连接网络进行访问
http://172.29.240.181:17860
**注意：IP要换成自己的IP，端口号 和 config.yml 保存一致**

# 2 [参考资料]
[参考资料2](https://huggingface.co/THUDM/chatglm-6b-int4)
[参考资料](https://github.com/wenda-LLM/wenda/tree/main)