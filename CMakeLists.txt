cmake_minimum_required(VERSION 3.10)
project(onnx_cpp_demo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# ONNX Runtime 头文件和库路径
set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# 添加 OpenCV 库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# 生成可执行文件
add_executable(onnx_demo main.cpp)

# 链接 OpenCV 库和 ONNX Runtime 库
target_link_libraries(onnx_demo onnxruntime ${OpenCV_LIBS})
