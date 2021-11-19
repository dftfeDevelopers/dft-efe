cmake_minimum_required(VERSION 3.17)
project(reframe_linear_algebra_test)
set(CMAKE_CXX_STANDARD 14)

set(MAIN_PROJECT_DIR ${DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
include_directories(${MAIN_PROJECT_DIR}/src)

add_executable(OperationHost OperationHost.cpp)
target_link_libraries(OperationHost PUBLIC dft-efe-linalg)
