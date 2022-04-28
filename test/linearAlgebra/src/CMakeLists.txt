cmake_minimum_required(VERSION 3.17)
project(reframe_linear_algebra_test)
set(CMAKE_CXX_STANDARD 14)

set(MAIN_PROJECT_DIR ${DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

if (ENABLE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(DFTEFE_WITH_DEVICE)
    add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
endif()

add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${MAIN_PROJECT_DIR}/test/linearAlgebra/lib/buildLinearAlgebra)
include_directories(${MAIN_PROJECT_DIR}/src)

# CPU: testing +=, -=, add, copy constructor
add_executable(TestSerialVectorDoubleSizeHost TestSerialVectorDoubleSizeHost.cpp)
target_link_libraries(TestSerialVectorDoubleSizeHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleSizeHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleSizeHost" SUFFIX ".x")

add_executable(TestSerialVectorDoubleGetValuesHost TestSerialVectorDoubleGetValuesHost.cpp)
target_link_libraries(TestSerialVectorDoubleGetValuesHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleGetValuesHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleGetValuesHost" SUFFIX ".x")

add_executable(TestSerialVectorDoubleSetValuesHost TestSerialVectorDoubleSetValuesHost.cpp)
target_link_libraries(TestSerialVectorDoubleSetValuesHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleSetValuesHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleSetValuesHost" SUFFIX ".x")

add_executable(TestSerialVectorDoubleSetStorageHost TestSerialVectorDoubleSetStorageHost.cpp)
target_link_libraries(TestSerialVectorDoubleSetStorageHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleSetStorageHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleSetStorageHost" SUFFIX ".x")

add_executable(TestSerialVectorDoubleNormHost TestSerialVectorDoubleNormHost.cpp)
target_link_libraries(TestSerialVectorDoubleNormHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleNormHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleNormHost" SUFFIX ".x")

add_executable(TestSerialVectorComplexDoubleNormHost TestSerialVectorComplexDoubleNormHost.cpp)
target_link_libraries(TestSerialVectorComplexDoubleNormHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorComplexDoubleNormHost PROPERTIES OUTPUT_NAME "TestSerialVectorComplexDoubleNormHost" SUFFIX ".x")

add_executable(TestSerialVectorDoubleAddHost TestSerialVectorDoubleAddHost.cpp)
target_link_libraries(TestSerialVectorDoubleAddHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleAddHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleAddHost" SUFFIX ".x")

add_executable(TestSerialVectorDoubleInPlaceAddHost TestSerialVectorDoubleInPlaceAddHost.cpp)
target_link_libraries(TestSerialVectorDoubleInPlaceAddHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleInPlaceAddHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleInPlaceAddHost" SUFFIX ".x")

add_executable(TestSerialVectorDoubleSubtractHost TestSerialVectorDoubleSubtractHost.cpp)
target_link_libraries(TestSerialVectorDoubleSubtractHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleSubtractHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleSubtractHost" SUFFIX ".x")

add_executable(TestSerialVectorDoubleInPlaceSubtractHost TestSerialVectorDoubleInPlaceSubtractHost.cpp)
target_link_libraries(TestSerialVectorDoubleInPlaceSubtractHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleInPlaceSubtractHost PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleInPlaceSubtractHost" SUFFIX ".x")

add_executable(TestBlasLapackDoubleGemmHost TestBlasLapackDoubleGemmHost.cpp)
target_link_libraries(TestBlasLapackDoubleGemmHost PUBLIC dft-efe-linalg)
set_target_properties(TestBlasLapackDoubleGemmHost PROPERTIES OUTPUT_NAME "TestBlasLapackDoubleGemmHost" SUFFIX ".x")

add_executable(TestBlasLapackComplexDoubleGemmHost TestBlasLapackComplexDoubleGemmHost.cpp)
target_link_libraries(TestBlasLapackComplexDoubleGemmHost PUBLIC dft-efe-linalg)
set_target_properties(TestBlasLapackComplexDoubleGemmHost PROPERTIES OUTPUT_NAME "TestBlasLapackComplexDoubleGemmHost" SUFFIX ".x")

add_executable(TestSerialDenseMatrixDoubleFrobeniusNormHost TestSerialDenseMatrixDoubleFrobeniusNormHost.cpp)
target_link_libraries(TestSerialDenseMatrixDoubleFrobeniusNormHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialDenseMatrixDoubleFrobeniusNormHost PROPERTIES OUTPUT_NAME "TestSerialDenseMatrixDoubleFrobeniusNormHost" SUFFIX ".x")

add_executable(TestSerialDenseMatrixDoubleCopyHost TestSerialDenseMatrixDoubleCopyHost.cpp)
target_link_libraries(TestSerialDenseMatrixDoubleCopyHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialDenseMatrixDoubleCopyHost PROPERTIES OUTPUT_NAME "TestSerialDenseMatrixDoubleCopyHost" SUFFIX ".x")

add_executable(TestSerialDenseMatrixDoubleSizeHost TestSerialDenseMatrixDoubleSizeHost.cpp)
target_link_libraries(TestSerialDenseMatrixDoubleSizeHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialDenseMatrixDoubleSizeHost PROPERTIES OUTPUT_NAME "TestSerialDenseMatrixDoubleSizeHost" SUFFIX ".x")

add_executable(TestSerialDenseMatrixDoubleConstructorHost TestSerialDenseMatrixDoubleConstructorHost.cpp)
target_link_libraries(TestSerialDenseMatrixDoubleConstructorHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialDenseMatrixDoubleConstructorHost PROPERTIES OUTPUT_NAME "TestSerialDenseMatrixDoubleConstructorHost" SUFFIX ".x")

add_executable(TestSerialDenseMatrixDoubleSetValuesHost TestSerialDenseMatrixDoubleSetValuesHost.cpp)
target_link_libraries(TestSerialDenseMatrixDoubleSetValuesHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialDenseMatrixDoubleSetValuesHost PROPERTIES OUTPUT_NAME "TestSerialDenseMatrixDoubleSetValuesHost" SUFFIX ".x")

add_executable(TestSerialDenseMatrixDoubleGetValuesHost TestSerialDenseMatrixDoubleGetValuesHost.cpp)
target_link_libraries(TestSerialDenseMatrixDoubleGetValuesHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialDenseMatrixDoubleGetValuesHost PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

add_executable(TestSerialMultiVectorDoubleAddHost TestSerialMultiVectorDoubleAddHost.cpp)
target_link_libraries(TestSerialMultiVectorDoubleAddHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialMultiVectorDoubleAddHost PROPERTIES OUTPUT_NAME "TestSerialMultiVectorDoubleAddHost" SUFFIX ".x")

add_executable(TestSerialMultiVectorDoubleNormHost TestSerialMultiVectorDoubleNormHost.cpp)
target_link_libraries(TestSerialMultiVectorDoubleNormHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialMultiVectorDoubleNormHost PROPERTIES OUTPUT_NAME "TestSerialMultiVectorDoubleNormHost" SUFFIX ".x")

add_executable(TestSerialDenseMatrixDoubleInPlaceAddHost TestSerialDenseMatrixDoubleInPlaceAddHost.cpp)
target_link_libraries(TestSerialDenseMatrixDoubleInPlaceAddHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialDenseMatrixDoubleInPlaceAddHost PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

add_executable(TestSerialDenseMatrixDoubleInPlaceSubtractHost TestSerialDenseMatrixDoubleInPlaceSubtractHost.cpp)
target_link_libraries(TestSerialDenseMatrixDoubleInPlaceSubtractHost PUBLIC dft-efe-linalg)
set_target_properties(TestSerialDenseMatrixDoubleInPlaceSubtractHost PROPERTIES OUTPUT_NAME "" SUFFIX ".x")

if (ENABLE_CUDA)
  add_executable(TestSerialVectorDoubleAddDevice TestSerialVectorDoubleAddDevice.cpp)
  target_link_libraries(TestSerialVectorDoubleAddDevice PUBLIC dft-efe-linalg)
  set_target_properties(TestSerialVectorDoubleAddDevice PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleAddDevice" SUFFIX ".x")
endif()


## CPU: testing +=, -=, add, copy constructor
#add_executable(TestVectorAggregate1 TestVectorAggregate1.cpp)
#target_link_libraries(TestVectorAggregate1 PUBLIC dft-efe-linalg)
#set_target_properties(TestVectorAggregate1 PROPERTIES OUTPUT_NAME "TestVectorAggregate1" SUFFIX ".x")
#
## CPU: testing l2Norm, lInfNorm
#add_executable(TestVectorNorms TestVectorNorms.cpp)
#target_link_libraries(TestVectorNorms PUBLIC dft-efe-linalg)
#set_target_properties(TestVectorNorms PROPERTIES OUTPUT_NAME "TestVectorNorms" SUFFIX ".x")
#
## GPU: testing +=, -=, add, copy constructor
#if (ENABLE_CUDA)
#    add_executable(TestVectorAggregate2 TestVectorAggregate2.cpp)
#    target_link_libraries(TestVectorAggregate2 PUBLIC dft-efe-linalg)
#    set_target_properties(TestVectorAggregate2 PROPERTIES OUTPUT_NAME "TestVectorAggregate2" SUFFIX ".x")
#endif()
