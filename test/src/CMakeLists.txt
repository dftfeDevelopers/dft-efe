cmake_minimum_required(VERSION 3.17)
project(reframe_test)
set(CMAKE_CXX_STANDARD 14)

# setting environment variables
set (DFT_EFE_LINKER $ENV{DFT_EFE_LINKER})

add_executable(test1 test1.cpp)
target_link_libraries(test1 ${DFT_EFE_LINKER})

add_executable(test2 test2.cpp)
target_link_libraries(test2 ${DFT_EFE_LINKER})

add_executable(test3 test3.cpp)
target_link_libraries(test3 ${DFT_EFE_LINKER})
