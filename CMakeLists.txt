cmake_minimum_required(VERSION 3.5)
project(image-gd-kernel CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

# build optimizer library (includes SGD, Adam, Gauss-Newton)
add_library(optimizers STATIC src/gradient_descent.cpp src/adam_optimizer.cpp src/gauss_newton.cpp)
target_include_directories(optimizers PUBLIC include)

add_executable(train_kernel src/train_kernel.cpp)
target_link_libraries(train_kernel optimizers ${OpenCV_LIBS})
