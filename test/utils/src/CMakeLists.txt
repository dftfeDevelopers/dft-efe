cmake_minimum_required(VERSION 3.17)
project(reframe_quadrature_test)
set(CMAKE_CXX_STANDARD 14)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
include_directories(${MAIN_PROJECT_DIR}/src)

add_executable(TestCubicSplineAlglib TestCubicSplineAlglib.cpp)
target_link_libraries(TestCubicSplineAlglib PUBLIC dft-efe-utils)

add_executable(TestCubicSplineTK TestCubicSplineTK.cpp)
target_link_libraries(TestCubicSplineTK PUBLIC dft-efe-utils)

add_executable(TestCubicSplineAlglibConv TestCubicSplineAlglibConv.cpp)
target_link_libraries(TestCubicSplineAlglibConv PUBLIC dft-efe-utils)
