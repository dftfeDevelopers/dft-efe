cmake_minimum_required(VERSION 3.17)
project(reframe_linear_algebra_test)
set(CMAKE_CXX_STANDARD 14)
include(ExternalProject)
# setting environment variables
#set (DFT_EFE_LINKER $ENV{DFT_EFE_LINKER})

set(DFTEFE_BLAS_LIBRARIES "-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl")
set(DFTEFE_SCALAPACK_LIBRARIES "-L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl")
set(CMAKE_PREFIX_PATH "/home/vikramg/DFT-FE-softwares/dealiiDevCustomized/install_gcc8.2.0_openmpi4.0.6_minimal")
set(ENABLE_CUDA OFF)

set(MAIN_PROJECT_DIR /home/iancclin/project/DFT-EFE-hackathon/dft-efe)

message("${CMAKE_CURRENT_BINARY_DIR}")
message("${CMAKE_BINARY_DIR}")

add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
include_directories(${MAIN_PROJECT_DIR}/src)

add_executable(OperationHost OperationHost.cpp)
target_link_libraries(OperationHost PUBLIC dft-efe-linalg)
