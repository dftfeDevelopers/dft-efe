cmake_minimum_required(VERSION 3.17)
project(reframe_quadrature_test)
set(CMAKE_CXX_STANDARD 14)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
include_directories(${MAIN_PROJECT_DIR}/src)

add_executable(TestCubicSplineTK TestCubicSplineTK.cpp)
target_link_libraries(TestCubicSplineTK PUBLIC dft-efe-utils)

###### Add all the MPI dependent tests below #######
if(ENABLE_MPI)
	if(NOT DEFINED DDFTEFE_WITH_MPI)
		ADD_DEFINITIONS(-DDFTEFE_WITH_MPI)
	endif()
	add_executable(TestMPIRequestersNBX TestMPIRequestersNBX.cpp)
	target_link_libraries(TestMPIRequestersNBX PUBLIC dft-efe-utils)
endif()
