# pytorch 介绍
![pytorch](https://github.com/pytorch/pytorch/blob/main/docs/source/_static/img/pytorch-logo-dark.png)

#安装
[安装链接](https://pytorch.org/)

# 安装注意事项
1. 选Stable(选择稳定版本)
2. linux 和 windows；linux 工作时候；
3. pckage：pip install，
4. libtorch：它是pytorch的c++版本，c++ 部署的时候可能会用到；
5. language: Python
6. compute platform:
  cuda: 在Nvidia 的卡上跑模型要用到；
  cpu: 在cpu上跑模型；
  rocm: amd 显卡对应的计算平台；
7. cuda 和 driver 版本对应：
[cuda driver 对照表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

# conda 环境管理
1. (base) C:\Users\86183> ： base ： 我们现在在base 这个环境里；
2. conda env list : 查看conda 环境；
3. conda create -n python=3.9
4. conda activate base

# pytorch 的软件栈
1. pytorch 软件本身；
2. cpu 版本 直接安装，依赖比较少：pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
3. cuda：nvidia 开发的一个在 GPU 上进行编程的一种语言， 和 C语言很类似；*.cu
4. cuda 最终要的作用就是加速，在GPU上进行加速；
5. GPU 非常善于处理结构化的数据，深度学习中 这些计算 ： 矩阵运算 --> 特别适合我们GPU来处理；
6. cuda --> nvcc 编译器；装好cuda 后 nvcc 自动安装；
7. cuda 一定要有nvidia的卡支持，不一定所有的nvidia的卡都支持cuda；
8. cuda 要依赖于一个 driver()；

[驱动选择](https://www.nvidia.cn/Download/index.aspx?lang=cn)

# pytorch version
- 2.0
- 1.9.0 1.10 1.11 1.12 1.13 --> 2.0
- 2023年
- 加速 pip install pytorch -i 镜像地址

# 安装其他版本
[安装链接1](https://pytorch.org/get-started/previous-versions/)
[安装链接2](https://download.pytorch.org/whl/torch/)

# 常用的pytorch 模块
- nn
- utils
- optim
- autograd : 对用户不感知
- distributed
- profiler : 算子性能追踪
- quantization: 量化模块
- jit、onnx：模型保存相关；

# 学习pytorch 相关模块
1. [pytorch doc](https://pytorch.org/docs/stable/search.html?q=&check_keywords=yes&area=default#)
2. .conda/envs/mmcv/lib/python3.8/site-packages/torch/
3. torch/_C/_VariableFunctions.pyi : pyi c++ 程序的python 接口
4. 其它 。。。

# torchvision
