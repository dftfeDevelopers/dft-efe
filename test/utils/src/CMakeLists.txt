cmake_minimum_required(VERSION 3.20)
project(reframe_quadrature_test)
set(CMAKE_CXX_STANDARD 17)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

IF (ENABLE_GPU)
add_compile_definitions(DFTEFE_WITH_DEVICE)

IF ("${GPU_VENDOR}" STREQUAL "nvidia")
    add_compile_definitions(DFTEFE_WITH_DEVICE_NVIDIA)
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_CUDA)
ELSEIF ("${GPU_VENDOR}" STREQUAL "amd")
    add_compile_definitions(DFTEFE_WITH_DEVICE_AMD)
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_HIP)
ELSEIF ("${GPU_VENDOR}" STREQUAL "intel")
    add_compile_definitions(DFTEFE_WITH_DEVICE_INTEL)
    add_compile_definitions(DFTEFE_WITH_DEVICE_MKL)
    add_compile_definitions(DFTEFE_WITH_DEVICE_LANG_SYCL)
ENDIF()
ENDIF()

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
	
    add_executable(TestMPIPatternP2PGhostDataPermute TestMPIPatternP2PGhostDataPermute.cpp)
	target_link_libraries(TestMPIPatternP2PGhostDataPermute PUBLIC dft-efe-utils)
	set_target_properties(TestMPIPatternP2PGhostDataPermute PROPERTIES OUTPUT_NAME "TestMPIPatternP2PGhostDataPermute" SUFFIX ".x")
    
    add_executable(TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIds TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIds.cpp)
	target_link_libraries(TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIds PUBLIC dft-efe-utils)
	set_target_properties(TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIds PROPERTIES OUTPUT_NAME "TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIds" SUFFIX ".x")
	
    add_executable(TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIdsComposite TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIdsComposite.cpp)
	target_link_libraries(TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIdsComposite PUBLIC dft-efe-utils)
	set_target_properties(TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIdsComposite PROPERTIES OUTPUT_NAME "TestMPIPatternP2PGhostDataPermuteSomeZeroLocalIdsComposite" SUFFIX ".x")
    
    add_executable(TestMPIPatternP2PTargetDataPermuteSomeZeroLocalIdsComposite TestMPIPatternP2PTargetDataPermuteSomeZeroLocalIdsComposite.cpp)
	target_link_libraries(TestMPIPatternP2PTargetDataPermuteSomeZeroLocalIdsComposite PUBLIC dft-efe-utils)
	set_target_properties(TestMPIPatternP2PTargetDataPermuteSomeZeroLocalIdsComposite PROPERTIES OUTPUT_NAME "TestMPIPatternP2PTargetDataPermuteSomeZeroLocalIdsComposite" SUFFIX ".x")
	
    add_executable(TestMPIPatternP2PTargetData TestMPIPatternP2PTargetData.cpp)
	target_link_libraries(TestMPIPatternP2PTargetData PUBLIC dft-efe-utils)
	set_target_properties(TestMPIPatternP2PTargetData PROPERTIES OUTPUT_NAME "TestMPIPatternP2PTargetData" SUFFIX ".x")

	add_executable(TestMPICommunicatorP2PUpdateGhosts TestMPICommunicatorP2PUpdateGhosts.cpp)
	target_link_libraries(TestMPICommunicatorP2PUpdateGhosts PUBLIC dft-efe-utils)
	set_target_properties(TestMPICommunicatorP2PUpdateGhosts PROPERTIES OUTPUT_NAME "TestMPICommunicatorP2PUpdateGhosts" SUFFIX ".x")

if (ENABLE_GPU)
	if (ENABLE_MPI_DEVICE_AWARE)
		add_compile_definitions(DFTEFE_WITH_DEVICE_AWARE_MPI)
		MESSAGE(STATUS "DFTEFE with GPU Aware MPI in test")
	endif()
	add_executable(TestMPICommunicatorP2PAccumulateAdd TestMPICommunicatorP2PAccumulateAdd.cpp)
	target_link_libraries(TestMPICommunicatorP2PAccumulateAdd PUBLIC dft-efe-utils)
	set_target_properties(TestMPICommunicatorP2PAccumulateAdd PROPERTIES OUTPUT_NAME "TestMPICommunicatorP2PAccumulateAdd" SUFFIX ".x")

	add_executable(TestMPICommunicatorP2PUpdateGhostsDevice TestMPICommunicatorP2PUpdateGhostsDevice.cpp)
	target_link_libraries(TestMPICommunicatorP2PUpdateGhostsDevice PUBLIC dft-efe-utils)
	set_target_properties(TestMPICommunicatorP2PUpdateGhostsDevice PROPERTIES OUTPUT_NAME "TestMPICommunicatorP2PUpdateGhostsDevice" SUFFIX ".x")
endif()

endif()
