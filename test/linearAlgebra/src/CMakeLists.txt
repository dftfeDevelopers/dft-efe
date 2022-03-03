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
add_executable(TestSerialVectorDoubleHostSize TestSerialVectorDoubleHostSize.cpp)
target_link_libraries(TestSerialVectorDoubleHostSize PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleHostSize PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleHostSize" SUFFIX ".x")

add_executable(TestSerialVectorDoubleHostGetValues TestSerialVectorDoubleHostGetValues.cpp)
target_link_libraries(TestSerialVectorDoubleHostGetValues PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleHostGetValues PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleHostGetValues" SUFFIX ".x")

add_executable(TestSerialVectorDoubleHostSetValues TestSerialVectorDoubleHostSetValues.cpp)
target_link_libraries(TestSerialVectorDoubleHostSetValues PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleHostSetValues PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleHostSetValues" SUFFIX ".x")

add_executable(TestSerialVectorDoubleHostSetStorage TestSerialVectorDoubleHostSetStorage.cpp)
target_link_libraries(TestSerialVectorDoubleHostSetStorage PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleHostSetStorage PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleHostSetStorage" SUFFIX ".x")

add_executable(TestSerialVectorDoubleHostSetStorage2 TestSerialVectorDoubleHostSetStorage2.cpp)
target_link_libraries(TestSerialVectorDoubleHostSetStorage2 PUBLIC dft-efe-linalg)
set_target_properties(TestSerialVectorDoubleHostSetStorage2 PROPERTIES OUTPUT_NAME "TestSerialVectorDoubleHostSetStorage2" SUFFIX ".x")

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
