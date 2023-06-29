cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(mnist)
set(CMAKE_CXX_STANDARD 14)

# 设置自己的 torch 路径
set(Torch_DIR /home/elvin/.conda/envs/pytorch_env/lib/python3.8/site-packages/torch/share/cmake/Torch)
find_package(Torch REQUIRED)

option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(mnist-infer mnist-infer.cpp)
add_executable(mnist-train mnist-train.cpp)
target_compile_features(mnist-train PUBLIC cxx_range_for)
target_compile_features(mnist-infer PUBLIC cxx_range_for)
target_link_libraries(mnist-train ${TORCH_LIBRARIES})
target_link_libraries(mnist-infer ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET mnist
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:mnist>)
endif (MSVC)
