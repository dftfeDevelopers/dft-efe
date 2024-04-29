cmake_minimum_required(VERSION 3.20)
project(reframe_test)
set(CMAKE_CXX_STANDARD 17)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})

include_directories(${MAIN_PROJECT_DIR}/src)
add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${MAIN_PROJECT_DIR}/test/basis/lib/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/basis ${MAIN_PROJECT_DIR}/test/basis/lib/buildBasis)
add_subdirectory(${MAIN_PROJECT_DIR}/src/atoms ${MAIN_PROJECT_DIR}/test/basis/lib/buildAtoms)
add_subdirectory(${MAIN_PROJECT_DIR}/src/quadrature ${MAIN_PROJECT_DIR}/test/basis/lib/buildQuadrature)
add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${MAIN_PROJECT_DIR}/test/linearAlgebra/lib/buildLinearAlgebra)

add_executable(TestAdaptiveQuad TestAdaptiveQuad.cpp )
target_link_libraries(TestAdaptiveQuad PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
set_target_properties(TestAdaptiveQuad PROPERTIES OUTPUT_NAME "TestAdaptiveQuad" SUFFIX ".x")

add_executable(TestAtomIdsPartitionSerial TestAtomIdsPartitionSerial.cpp )
target_link_libraries(TestAtomIdsPartitionSerial PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
set_target_properties(TestAtomIdsPartitionSerial PROPERTIES OUTPUT_NAME "TestAtomIdsPartitionSerial" SUFFIX ".x")

add_executable(TestBasisStorageSerialTria TestBasisStorageSerialTria.cpp )
target_link_libraries(TestBasisStorageSerialTria PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
set_target_properties(TestBasisStorageSerialTria PROPERTIES OUTPUT_NAME "TestBasisStorageSerialTria" SUFFIX ".x")

add_executable(TestHomogeneousConstraintMatrix TestHomogeneousConstraintMatrix.cpp )
target_link_libraries(TestHomogeneousConstraintMatrix PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
set_target_properties(TestHomogeneousConstraintMatrix PROPERTIES OUTPUT_NAME "TestHomogeneousConstraintMatrix" SUFFIX ".x")

add_executable(TestInhomogeneousConstraintMatrix TestInhomogeneousConstraintMatrix.cpp )
target_link_libraries(TestInhomogeneousConstraintMatrix PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
set_target_properties(TestInhomogeneousConstraintMatrix PROPERTIES OUTPUT_NAME "TestInhomogeneousConstraintMatrix" SUFFIX ".x")

add_executable(TestEnrichmentIdsPartitionSerial TestEnrichmentIdsPartitionSerial.cpp )
target_link_libraries(TestEnrichmentIdsPartitionSerial PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
set_target_properties(TestEnrichmentIdsPartitionSerial PROPERTIES OUTPUT_NAME "TestEnrichmentIdsPartitionSerial" SUFFIX ".x")

add_executable(TestInterpolateWithBasisGradient TestInterpolateWithBasisGradient.cpp )
target_link_libraries(TestInterpolateWithBasisGradient PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
set_target_properties(TestInterpolateWithBasisGradient PROPERTIES OUTPUT_NAME "TestInterpolateWithBasisGradient" SUFFIX ".x")

add_executable(TestTriangulationDealiiSerial TestTriangulationDealiiSerial.cpp)
target_link_libraries(TestTriangulationDealiiSerial PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
set_target_properties(TestTriangulationDealiiSerial PROPERTIES OUTPUT_NAME "TestTriangulationDealiiSerial" SUFFIX ".x")


if(ENABLE_MPI)

	add_compile_definitions(DFTEFE_WITH_MPI)

	add_executable(TestComputeFEMatrices TestComputeFEMatrices.cpp )
	target_link_libraries(TestComputeFEMatrices PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestComputeFEMatrices PROPERTIES OUTPUT_NAME "TestComputeFEMatrices" SUFFIX ".x")

	add_executable(TestAtomIdsPartitionParallel TestAtomIdsPartitionParallel.cpp )
	target_link_libraries(TestAtomIdsPartitionParallel PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestAtomIdsPartitionParallel PROPERTIES OUTPUT_NAME "TestAtomIdsPartitionParallel" SUFFIX ".x")

	add_executable(TestCFEOverlapMatrix TestCFEOverlapMatrix.cpp )
	target_link_libraries(TestCFEOverlapMatrix PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestCFEOverlapMatrix PROPERTIES OUTPUT_NAME "TestCFEOverlapMatrix" SUFFIX ".x")

	add_executable(TestClassicalEnrichmentIdsMPICommunication TestClassicalEnrichmentIdsMPICommunication.cpp)
	target_link_libraries(TestClassicalEnrichmentIdsMPICommunication PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestClassicalEnrichmentIdsMPICommunication PROPERTIES OUTPUT_NAME "TestClassicalEnrichmentIdsMPICommunication" SUFFIX ".x")

	add_executable(TestOrthoEFEOverlapMatrix TestOrthoEFEOverlapMatrix.cpp )
	target_link_libraries(TestOrthoEFEOverlapMatrix PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestOrthoEFEOverlapMatrix PROPERTIES OUTPUT_NAME "TestOrthoEFEOverlapMatrix" SUFFIX ".x")

	add_executable(TestEnrichmentBasisManager TestEnrichmentBasisManager.cpp )
	target_link_libraries(TestEnrichmentBasisManager PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestEnrichmentBasisManager PROPERTIES OUTPUT_NAME "TestEnrichmentBasisManager" SUFFIX ".x")

	add_executable(TestEnrichmentIdsPartitionParallel TestEnrichmentIdsPartitionParallel.cpp )
	target_link_libraries(TestEnrichmentIdsPartitionParallel PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestEnrichmentIdsPartitionParallel PROPERTIES OUTPUT_NAME "TestEnrichmentIdsPartitionParallel" SUFFIX ".x")

	add_executable(TestOrthoEnrichmentBasisManager TestOrthoEnrichmentBasisManager.cpp )
	target_link_libraries(TestOrthoEnrichmentBasisManager PUBLIC dft-efe-basis dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestOrthoEnrichmentBasisManager PROPERTIES OUTPUT_NAME "TestOrthoEnrichmentBasisManager" SUFFIX ".x")

endif()