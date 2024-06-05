cmake_minimum_required(VERSION 3.20)
project(reframe_test)
set(CMAKE_CXX_STANDARD 17)

set(MAIN_PROJECT_DIR $ENV{DFTEFE_PATH})
message(${MAIN_PROJECT_DIR})


include_directories(${MAIN_PROJECT_DIR}/src)
add_subdirectory(${MAIN_PROJECT_DIR}/src/utils ${MAIN_PROJECT_DIR}/test/ksdft/lib/buildUtils)
add_subdirectory(${MAIN_PROJECT_DIR}/src/basis ${MAIN_PROJECT_DIR}/test/ksdft/lib/buildBasis)
add_subdirectory(${MAIN_PROJECT_DIR}/src/atoms ${MAIN_PROJECT_DIR}/test/ksdft/lib/buildAtoms)
add_subdirectory(${MAIN_PROJECT_DIR}/src/quadrature ${MAIN_PROJECT_DIR}/test/ksdft/lib/buildQuadrature)
add_subdirectory(${MAIN_PROJECT_DIR}/src/linearAlgebra ${MAIN_PROJECT_DIR}/test/ksdft/lib/buildLinearAlgebra)
add_subdirectory(${MAIN_PROJECT_DIR}/src/electrostatics ${MAIN_PROJECT_DIR}/test/ksdft/lib/buildElectrostatics)
add_subdirectory(${MAIN_PROJECT_DIR}/src/ksdft ${MAIN_PROJECT_DIR}/test/ksdft/lib/buildKsdft)

if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)

	# add_executable(TestHamiltonianClassical TestHamiltonianClassical.cpp )
	# target_link_libraries(TestHamiltonianClassical PUBLIC dft-efe-basis dft-efe-electrostatics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms dft-efe-ksdft)
	# set_target_properties(TestHamiltonianClassical PROPERTIES OUTPUT_NAME "TestHamiltonianClassical" SUFFIX ".x")

	# add_executable(TestKSAllElectronEigenSolveClassical TestKSAllElectronEigenSolveClassical.cpp )
	# target_link_libraries(TestKSAllElectronEigenSolveClassical PUBLIC dft-efe-basis dft-efe-electrostatics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms dft-efe-ksdft)
	# set_target_properties(TestKSAllElectronEigenSolveClassical PROPERTIES OUTPUT_NAME "TestKSAllElectronEigenSolveClassical" SUFFIX ".x")

	# add_executable(TestKSAllElectronEigenSolveOrthoEFE TestKSAllElectronEigenSolveOrthoEFE.cpp )
	# target_link_libraries(TestKSAllElectronEigenSolveOrthoEFE PUBLIC dft-efe-basis dft-efe-electrostatics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms dft-efe-ksdft)
	# set_target_properties(TestKSAllElectronEigenSolveOrthoEFE PROPERTIES OUTPUT_NAME "TestKSAllElectronEigenSolveOrthoEFE" SUFFIX ".x")

	add_executable(TestKSDFTClassical TestKSDFTClassical.cpp )
	target_link_libraries(TestKSDFTClassical PUBLIC dft-efe-basis dft-efe-electrostatics dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms dft-efe-ksdft)
	set_target_properties(TestKSDFTClassical PROPERTIES OUTPUT_NAME "TestKSDFTClassical" SUFFIX ".x")	
endif()
