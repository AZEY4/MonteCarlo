# MonteCarlo
High-performance Monte Carlo option pricing engine with CPU (OpenMP) and GPU (CUDA) backends, supporting European and Asian options with Python bindings via pybind11.

# Build instructions (Linux) -- summary

Requirements:
 - CMake >= 3.18
 - CUDA toolkit
 - nvcc available in PATH
 - A C++ compiler supporting C++17 (gcc/clang)
 - pybind11 (cmake findable or installed into system)
 - OpenMP

Steps:

1. Create build directory
   mkdir build && cd build

2. Configure with CMake (point to python executable if needed)
   cmake ..

3. Build
   cmake --build . --config Release -j

4. After build, the python extension module `_backend` will be available in the build directory. You can either:
   - Install the package into your python environment manually (python -c "import sys; print(sys.path)") or
   - Copy the compiled `_backend.*.so` into `Python/monte_carlo_pro/` so `import monte_carlo_pro` finds it.

Example quick test (from project root):
   mkdir -p build && cd build
   cmake ..
   cmake --build . -j
   cp _backend*.so ../Python/monte_carlo_pro/
   python3 -c "from monte_carlo_pro import MonteCarloPro; m=MonteCarloPro(backend='gpu', paths=100000); print(m.price_option(100,100,1.0,0.01,0.2))"
