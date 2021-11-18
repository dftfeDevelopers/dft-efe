cmake_minimum_required(VERSION 3.17)
project(reframe_linear_algebra_test)
set(CMAKE_CXX_STANDARD 14)

# setting environment variables
set (DFT_EFE_LINKER $ENV{DFT_EFE_LINKER})



add_executable(OperationHost OperationHost.cpp)
target_link_libraries(test1 ${DFT_EFE_LINKER})
