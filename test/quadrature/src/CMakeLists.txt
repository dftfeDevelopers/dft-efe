cmake_minimum_required(VERSION 3.20)
project(reframe_quadrature_test)
set(CMAKE_CXX_STANDARD 17)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

include_directories(${MAIN_PROJECT_DIR}/src)
add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/basis ${CMAKE_CURRENT_BINARY_DIR}/buildBasis)
add_subdirectory(${MAIN_PROJECT_DIR}/src/quadrature ${CMAKE_CURRENT_BINARY_DIR}/buildQuadrature)

add_executable(TestAdaptiveQuadratureExpNeg100x TestAdaptiveQuadratureExpNeg100x.cpp)
target_link_libraries(TestAdaptiveQuadratureExpNeg100x PUBLIC dft-efe-quadrature dft-efe-basis dft-efe-utils)

add_executable(TestAdaptiveQuadratureExpNeg10x TestAdaptiveQuadratureExpNeg10x.cpp)
target_link_libraries(TestAdaptiveQuadratureExpNeg10x PUBLIC dft-efe-quadrature dft-efe-basis dft-efe-utils)

add_executable(TestGaussIteratedQuadratureExpNeg100xSerial TestGaussIteratedQuadratureExpNeg100xSerial.cpp)
target_link_libraries(TestGaussIteratedQuadratureExpNeg100xSerial PUBLIC dft-efe-utils dft-efe-quadrature dft-efe-basis)

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
add_executable(TestGaussIteratedQuadratureExpNeg100x TestGaussIteratedQuadratureExpNeg100x.cpp)
target_link_libraries(TestGaussIteratedQuadratureExpNeg100x PUBLIC dft-efe-utils dft-efe-quadrature dft-efe-basis)
endif()