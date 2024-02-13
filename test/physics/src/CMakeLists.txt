cmake_minimum_required(VERSION 3.20)
project(reframe_test)
set(CMAKE_CXX_STANDARD 17)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})


include_directories(${MAIN_PROJECT_DIR}/src)
add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${MAIN_PROJECT_DIR}/test/physics/lib/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/basis ${MAIN_PROJECT_DIR}/test/physics/lib/buildBasis)
add_subdirectory(${MAIN_PROJECT_DIR}/src/atoms ${MAIN_PROJECT_DIR}/test/physics/lib/buildAtoms)
add_subdirectory(${MAIN_PROJECT_DIR}/src/quadrature ${MAIN_PROJECT_DIR}/test/physics/lib/buildQuadrature)
add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${MAIN_PROJECT_DIR}/test/physics/lib/buildLinearAlgebra)
add_subdirectory(${MAIN_PROJECT_DIR}/src/physics ${MAIN_PROJECT_DIR}/test/physics/lib/buildPhysics)

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	# add_executable(TestLaplaceOperator TestLaplaceOperator.cpp )
	# target_link_libraries(TestLaplaceOperator PUBLIC dft-efe-basis dft-efe-physics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	# set_target_properties(TestLaplaceOperator PROPERTIES OUTPUT_NAME "TestLaplaceOperator" SUFFIX ".x")

	add_executable(TestFunctionClassical TestFunctionClassical.cpp )
	target_link_libraries(TestFunctionClassical PUBLIC dft-efe-basis dft-efe-physics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	set_target_properties(TestFunctionClassical PROPERTIES OUTPUT_NAME "TestFunctionClassical" SUFFIX ".x")

	# add_executable(TestPoissonProblemClassical TestPoissonProblemClassical.cpp )
	# target_link_libraries(TestPoissonProblemClassical PUBLIC dft-efe-basis dft-efe-physics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	# set_target_properties(TestPoissonProblemClassical PROPERTIES OUTPUT_NAME "TestPoissonProblemClassical" SUFFIX ".x")

	# add_executable(TestPoissonProblemEnrichment TestPoissonProblemEnrichment.cpp )
	# target_link_libraries(TestPoissonProblemEnrichment PUBLIC dft-efe-basis dft-efe-physics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	# set_target_properties(TestPoissonProblemEnrichment PROPERTIES OUTPUT_NAME "TestPoissonProblemEnrichment" SUFFIX ".x")

	# add_executable(TestPoissonProblemEnrichmentAdaptiveQuad TestPoissonProblemEnrichmentAdaptiveQuad.cpp )
	# target_link_libraries(TestPoissonProblemEnrichmentAdaptiveQuad PUBLIC dft-efe-basis dft-efe-physics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	# set_target_properties(TestPoissonProblemEnrichmentAdaptiveQuad PROPERTIES OUTPUT_NAME "TestPoissonProblemEnrichmentAdaptiveQuad" SUFFIX ".x")

	# add_executable(TestPoissonProblemOrthoEnrichmentAdaptiveQuad TestPoissonProblemOrthoEnrichmentAdaptiveQuad.cpp )
	# target_link_libraries(TestPoissonProblemOrthoEnrichmentAdaptiveQuad PUBLIC dft-efe-basis dft-efe-physics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	# set_target_properties(TestPoissonProblemOrthoEnrichmentAdaptiveQuad PROPERTIES OUTPUT_NAME "TestPoissonProblemOrthoEnrichmentAdaptiveQuad" SUFFIX ".x")

	# add_executable(TestTwoBodyInteraction TestTwoBodyInteraction.cpp )
	# target_link_libraries(TestTwoBodyInteraction PUBLIC dft-efe-basis dft-efe-physics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms)
	# set_target_properties(TestTwoBodyInteraction PROPERTIES OUTPUT_NAME "TestTwoBodyInteraction" SUFFIX ".x")
endif()
