cmake_minimum_required(VERSION 3.17)
project(reframe_quadrature_test)
set(CMAKE_CXX_STANDARD 14)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/basis ${CMAKE_CURRENT_BINARY_DIR}/buildBasis)
add_subdirectory(${MAIN_PROJECT_DIR}/src/quadrature ${CMAKE_CURRENT_BINARY_DIR}/buildQuadrature)
include_directories(${MAIN_PROJECT_DIR}/src)

add_executable(TestAdaptiveQuadratureExpNeg100x TestAdaptiveQuadratureExpNeg100x.cpp)
target_link_libraries(TestAdaptiveQuadratureExpNeg100x PUBLIC dft-efe-quadrature dft-efe-basis dft-efe-utils)

add_executable(TestAdaptiveQuadratureExpNeg10x TestAdaptiveQuadratureExpNeg10x.cpp)
target_link_libraries(TestAdaptiveQuadratureExpNeg10x PUBLIC dft-efe-quadrature dft-efe-basis dft-efe-utils)
