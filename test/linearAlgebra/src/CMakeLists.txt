cmake_minimum_required(VERSION 3.20)
project(reframe_linear_algebra_test)
set(CMAKE_CXX_STANDARD 17)

set(MAIN_PROJECT_DIR ${DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

if (ENABLE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(DFTEFE_WITH_DEVICE)
    add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
endif ()

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	find_package(MPI REQUIRED)
	include_directories(${MPI_CXX_INCLUDE_PATH})
	set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
endif()

add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${MAIN_PROJECT_DIR}/test/linearAlgebra/lib/buildLinearAlgebra)
include_directories(${MAIN_PROJECT_DIR}/src)

# CPU: testing +=, -=, add, copy constructor
#add_executable(TestSerialVectorDoubleSizeHost TestSerialVectorDoubleSizeHost.cpp)
#target_link_libraries(TestSerialVectorDoubleSizeHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleSizeHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleSizeHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorDoubleGetValuesHost TestSerialVectorDoubleGetValuesHost.cpp)
#target_link_libraries(TestSerialVectorDoubleGetValuesHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleGetValuesHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleGetValuesHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorDoubleSetValuesHost TestSerialVectorDoubleSetValuesHost.cpp)
#target_link_libraries(TestSerialVectorDoubleSetValuesHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleSetValuesHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleSetValuesHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorDoubleSetStorageHost TestSerialVectorDoubleSetStorageHost.cpp)
#target_link_libraries(TestSerialVectorDoubleSetStorageHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleSetStorageHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleSetStorageHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorDoubleNormHost TestSerialVectorDoubleNormHost.cpp)
#target_link_libraries(TestSerialVectorDoubleNormHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleNormHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleNormHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorComplexDoubleNormHost TestSerialVectorComplexDoubleNormHost.cpp)
#target_link_libraries(TestSerialVectorComplexDoubleNormHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorComplexDoubleNormHost PROPERTIES OUTPUT_NAME "TestSerialVectorComplexDoubleNormHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorDoubleAddHost TestSerialVectorDoubleAddHost.cpp)
#target_link_libraries(TestSerialVectorDoubleAddHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleAddHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleAddHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorDoubleInPlaceAddHost TestSerialVectorDoubleInPlaceAddHost.cpp)
#target_link_libraries(TestSerialVectorDoubleInPlaceAddHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleInPlaceAddHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleInPlaceAddHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorDoubleSubtractHost TestSerialVectorDoubleSubtractHost.cpp)
#target_link_libraries(TestSerialVectorDoubleSubtractHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleSubtractHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleSubtractHost" SUFFIX ".x")
#
#add_executable(TestSerialVectorDoubleInPlaceSubtractHost TestSerialVectorDoubleInPlaceSubtractHost.cpp)
#target_link_libraries(TestSerialVectorDoubleInPlaceSubtractHost PUBLIC dft-efe-linalg)
#set_target_properties(TestSerialVectorDoubleInPlaceSubtractHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleInPlaceSubtractHost" SUFFIX ".x")
#
#add_executable(TestBlasLapackDoubleGemmHost TestBlasLapackDoubleGemmHost.cpp)
#target_link_libraries(TestBlasLapackDoubleGemmHost PUBLIC dft-efe-linalg)
#set_target_properties(TestBlasLapackDoubleGemmHost PROPERTIES OUTPUT_NAME "TestBlasLapackDoubleGemmHost" SUFFIX ".x")
#
#add_executable(TestBlasLapackComplexDoubleGemmHost TestBlasLapackComplexDoubleGemmHost.cpp)
#target_link_libraries(TestBlasLapackComplexDoubleGemmHost PUBLIC dft-efe-linalg)
#set_target_properties(TestBlasLapackComplexDoubleGemmHost PROPERTIES OUTPUT_NAME "TestBlasLapackComplexDoubleGemmHost" SUFFIX ".x")
#

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})


if (ENABLE_MPI)

# add_executable(TestParallelMultiVectorCholeskyGSHostComplexDouble  TestParallelMultiVectorCholeskyGSHostComplexDouble.cpp)
# target_link_libraries(TestParallelMultiVectorCholeskyGSHostComplexDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
# set_target_properties(TestParallelMultiVectorCholeskyGSHostComplexDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

# add_executable(TestParallelMultiVectorCholeskyGSHostDouble  TestParallelMultiVectorCholeskyGSHostDouble.cpp)
# target_link_libraries(TestParallelMultiVectorCholeskyGSHostDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
# set_target_properties(TestParallelMultiVectorCholeskyGSHostDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

# add_executable(TestLanczosEigenSolverHostDouble  TestLanczosEigenSolverHostDouble.cpp)
# target_link_libraries(TestLanczosEigenSolverHostDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
# set_target_properties(TestLanczosEigenSolverHostDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

# add_executable(TestRayleighRitzHostDouble  TestRayleighRitzHostDouble.cpp)
# target_link_libraries(TestRayleighRitzHostDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
# set_target_properties(TestRayleighRitzHostDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

add_executable(TestChebyshevFilteredEigenSolveHostDouble  TestChebyshevFilteredEigenSolveHostDouble.cpp)
target_link_libraries(TestChebyshevFilteredEigenSolveHostDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
set_target_properties(TestChebyshevFilteredEigenSolveHostDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

# add_executable(TestParallelMultiVectorMultipassLowdinHostDouble  TestParallelMultiVectorMultipassLowdinHostDouble.cpp)
# target_link_libraries(TestParallelMultiVectorMultipassLowdinHostDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
# set_target_properties(TestParallelMultiVectorMultipassLowdinHostDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

# add_executable(TestParallelMultiVectorMultipassLowdinHostComplexDouble  TestParallelMultiVectorMultipassLowdinHostComplexDouble.cpp)
# target_link_libraries(TestParallelMultiVectorMultipassLowdinHostComplexDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
# set_target_properties(TestParallelMultiVectorMultipassLowdinHostComplexDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

# add_executable(TestCGLinearSolverSerialHostDouble  TestCGLinearSolverSerialHostDouble.cpp)
# target_link_libraries(TestCGLinearSolverSerialHostDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
# set_target_properties(TestCGLinearSolverSerialHostDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

#add_executable(TestCGLinearSolverSerialHostComplexDouble  TestCGLinearSolverSerialHostComplexDouble.cpp)
#target_link_libraries(TestCGLinearSolverSerialHostComplexDouble PUBLIC dft-efe-linalg mkl_intel_lp64 mkl_sequential mkl_core)
#set_target_properties(TestCGLinearSolverSerialHostComplexDouble PROPERTIES OUTPUT_NAME "" SUFFIX ".x")
endif()

if (ENABLE_CUDA)
    add_executable(TestSerialVectorDoubleAddDevice TestSerialVectorDoubleAddDevice.cpp)
    target_link_libraries(TestSerialVectorDoubleAddDevice PUBLIC dft-efe-linalg)
    set_target_properties(TestSerialVectorDoubleAddDevice PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleAddDevice" SUFFIX ".x")

    add_executable(TestSerialVectorComplexDoubleAddDevice TestSerialVectorComplexDoubleAddDevice.cpp)
    target_link_libraries(TestSerialVectorComplexDoubleAddDevice PUBLIC dft-efe-linalg)
    set_target_properties(TestSerialVectorComplexDoubleAddDevice PROPERTIES OUTPUT_NAME "TestSerialVectorComplexDoubleAddDevice" SUFFIX ".x")

    add_executable(TestBlasLapackDoubleGemmDevice TestBlasLapackDoubleGemmDevice.cpp)
    target_link_libraries(TestBlasLapackDoubleGemmDevice PUBLIC dft-efe-linalg)
    set_target_properties(TestBlasLapackDoubleGemmDevice PROPERTIES OUTPUT_NAME "TestBlasLapackDoubleGemmDevice" SUFFIX ".x")

    add_executable(TestBlasLapackComplexDoubleGemmDevice TestBlasLapackComplexDoubleGemmDevice.cpp)
    target_link_libraries(TestBlasLapackComplexDoubleGemmDevice PUBLIC dft-efe-linalg)
    set_target_properties(TestBlasLapackComplexDoubleGemmDevice PROPERTIES OUTPUT_NAME "TestBlasLapackComplexDoubleGemmDevice" SUFFIX ".x")

endif ()
