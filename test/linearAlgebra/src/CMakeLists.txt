cmake_minimum_required(VERSION 3.17)
project(reframe_linear_algebra_test)
set(CMAKE_CXX_STANDARD 14)

set(MAIN_PROJECT_DIR ${DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
include_directories(${MAIN_PROJECT_DIR}/src)

# CPU: testing +=, -=, add, copy constructor
add_executable(TestVectorAggregate1 TestVectorAggregate1.cpp)
target_link_libraries(TestVectorAggregate1 PUBLIC dft-efe-linalg)

# CPU: testing +=, -=, add, copy constructor, a dummy test to show how to run multiple executables in one tests
add_executable(TestVectorAggregate2 TestVectorAggregate2.cpp)
target_link_libraries(TestVectorAggregate2 PUBLIC dft-efe-linalg)

# GPU: testing +=, -=, add, copy constructor
add_executable(TestVectorAggregate3 TestVectorAggregate3.cpp)
target_link_libraries(TestVectorAggregate3 PUBLIC dft-efe-linalg)
