#include <torch/torch.h> // 
#include <torch/script.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int  mnist_train() {
  torch::DeviceType device_type = torch::kCPU;
  torch::Device device(device_type);
  auto input = torch::rand({1, 1, 28, 28}, torch::kFloat32);
  auto module = torch::jit::load("../traced_model.pt");
  torch::Tensor output = module.forward({input}).toTensor();
  auto max_result = output.max(1, true);
  auto max_index = std::get<1>(max_result).item<float>();
  std::cout << "=========cpu max_index: " << max_index << std::endl;
  return max_index;
}

int main() {
  mnist_train();
  std::cout << "run mnist-infer.cpp successfully !!!" << std::endl;
  return 0;
}
