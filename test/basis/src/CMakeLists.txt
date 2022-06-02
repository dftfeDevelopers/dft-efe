cmake_minimum_required(VERSION 3.20)
project(reframe_linear_algebra_test)
set(CMAKE_CXX_STANDARD 17)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/basis ${CMAKE_CURRENT_BINARY_DIR}/buildBasis)
include_directories(${MAIN_PROJECT_DIR}/src)
add_executable(TestTriangulationDealiiSerial TestTriangulationDealiiSerial.cpp)
target_link_libraries(TestTriangulationDealiiSerial PUBLIC dft-efe-basis dft-efe-utils)
