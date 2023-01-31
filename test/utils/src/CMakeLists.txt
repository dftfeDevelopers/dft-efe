cmake_minimum_required(VERSION 3.20)
project(reframe_quadrature_test)
set(CMAKE_CXX_STANDARD 17)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

if (ENABLE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(DFTEFE_WITH_DEVICE)
    add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
endif ()

add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${MAIN_PROJECT_DIR}/test/utils/lib/buildUtils)
include_directories(${MAIN_PROJECT_DIR}/src)

add_executable(TestCubicSplineTK TestCubicSplineTK.cpp)
target_link_libraries(TestCubicSplineTK PUBLIC dft-efe-utils)
set_target_properties(TestCubicSplineTK PROPERTIES OUTPUT_NAME "TestCubicSplineTK" SUFFIX ".x")

add_executable(TestOptimizedIndexSet TestOptimizedIndexSet.cpp)
target_link_libraries(TestOptimizedIndexSet PUBLIC dft-efe-utils)
set_target_properties(TestOptimizedIndexSet PROPERTIES OUTPUT_NAME "TestOptimizedIndexSet" SUFFIX ".x")

###### Add all the MPI dependent tests below #######
if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	add_executable(TestMPIRequestersNBX TestMPIRequestersNBX.cpp)
	target_link_libraries(TestMPIRequestersNBX PUBLIC dft-efe-utils)
	set_target_properties(TestMPIRequestersNBX PROPERTIES OUTPUT_NAME "TestMPIRequestersNBX" SUFFIX ".x")
	
	add_executable(TestMPIPatternP2PGhostData TestMPIPatternP2PGhostData.cpp)
	target_link_libraries(TestMPIPatternP2PGhostData PUBLIC dft-efe-utils)
	set_target_properties(TestMPIPatternP2PGhostData PROPERTIES OUTPUT_NAME "TestMPIPatternP2PGhostData" SUFFIX ".x")
	
	add_executable(TestMPIPatternP2PTargetData TestMPIPatternP2PTargetData.cpp)
	target_link_libraries(TestMPIPatternP2PTargetData PUBLIC dft-efe-utils)
	set_target_properties(TestMPIPatternP2PTargetData PROPERTIES OUTPUT_NAME "TestMPIPatternP2PTargetData" SUFFIX ".x")

	add_executable(TestMPICommunicatorP2PUpdateGhosts TestMPICommunicatorP2PUpdateGhosts.cpp)
	target_link_libraries(TestMPICommunicatorP2PUpdateGhosts PUBLIC dft-efe-utils)
	set_target_properties(TestMPICommunicatorP2PUpdateGhosts PROPERTIES OUTPUT_NAME "TestMPICommunicatorP2PUpdateGhosts" SUFFIX ".x")

	add_executable(TestMPICommunicatorP2PAccumulateAdd TestMPICommunicatorP2PAccumulateAdd.cpp)
	target_link_libraries(TestMPICommunicatorP2PAccumulateAdd PUBLIC dft-efe-utils)
	set_target_properties(TestMPICommunicatorP2PAccumulateAdd PROPERTIES OUTPUT_NAME "TestMPICommunicatorP2PAccumulateAdd" SUFFIX ".x")

if (ENABLE_CUDA)
	add_executable(TestMPICommunicatorP2PUpdateGhostsDevice TestMPICommunicatorP2PUpdateGhostsDevice.cpp)
	target_link_libraries(TestMPICommunicatorP2PUpdateGhostsDevice PUBLIC dft-efe-utils)
	set_target_properties(TestMPICommunicatorP2PUpdateGhostsDevice PROPERTIES OUTPUT_NAME "TestMPICommunicatorP2PUpdateGhostsDevice" SUFFIX ".x")
endif()

endif()
