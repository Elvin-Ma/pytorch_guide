# cpp 部署
- 训练框架（pytorch 等）
- 推理框架 （tensorRT，mnn、onnxruntime、libtorch）

# libtorch ： pytorch 的c++接口
- 可以进行训练
- 也可以进行推理（torch.jit.trace）

# 注意事项：
- 需要找到torch的动态库和头文件
- 假如我们已经安装了 pytorch的话，libtorch 环境肯定已经具备了；
- .conda/envs/pytorch_env/lib/python3.8/site-packages/torch/

**推理代码展示**
```c++
#include <torch/torch.h>
#include <torch/script.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
void  mnist_train() {
  torch::DeviceType device_type = torch::kCPU;
  torch::Device device(device_type);
  auto input = torch::rand({1, 1, 28, 28}, torch::kFloat32);
  auto module = torch::jit::load("../mnist_model.pt");
  torch::Tensor output = module.forward({input}).toTensor();
  auto max_result = output.max(1, true);
  auto max_index = std::get<1>(max_result).item<float>();
  std::cout << "=========cpu max_index: " << max_index << std::endl;
}
```
# 操作流程
0. cmake/g++
1. cd mnist
2. 修改 CMakelist.txt 设置自己的 torch路径
3. make build && cd build
4. cmake ..
5. make -j16
6. ./mnist-infer

