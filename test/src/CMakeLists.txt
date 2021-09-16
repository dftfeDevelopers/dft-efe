cmake_minimum_required(VERSION 3.17)
project(reframe_test)
set(CMAKE_CXX_STANDARD 14)

# setting environment variables
set (DFT_EFE_LINKER $ENV{DFT_EFE_LINKER})

message("${DFT_EFE_LINKER}")
add_executable(test test.cpp)
target_link_libraries(test ${DFT_EFE_LINKER})