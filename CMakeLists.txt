cmake_minimum_required(VERSION 3.18)
project(MonteCarloProCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70)

# find pybind11
find_package(pybind11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/C++/include)

# libraries
add_library(mc_cpu STATIC C++/src/option.cpp C++/src/simulator.cpp)
set_target_properties(mc_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# GPU implementation (object library)
add_library(mc_gpu_obj OBJECT C++/src/gpu_simulator.cu)
set_target_properties(mc_gpu_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Python extension via pybind11
pybind11_add_module(_backend C++/src/pybind_module.cpp $<TARGET_OBJECTS:mc_gpu_obj> C++/src/option.cpp C++/src/simulator.cpp)
target_include_directories(_backend PRIVATE ${CMAKE_SOURCE_DIR}/C++/include)
target_link_libraries(_backend PRIVATE OpenMP::OpenMP_CXX)

# install (optional)
install(TARGETS _backend DESTINATION "${CMAKE_INSTALL_PREFIX}/")
