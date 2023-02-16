project(gpgpu)
cmake_minimum_required(VERSION 2.8)

# Compute capability. See -arch option in nvcc --help.
set (CC 35)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

cuda_add_executable(matrix_vector_1thread 01-matrix_vector_1thread.cu)
cuda_add_executable(main_cuda_v1 main_cuda_v1.cu read_csr.c read_ellpack.c mmio.c wtime.c)
cuda_add_executable(main_cuda_v2 main_cuda_v2.cu read_csr.c read_ellpack.c mmio.c wtime.c)