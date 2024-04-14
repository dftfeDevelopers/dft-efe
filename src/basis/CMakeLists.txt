if (NOT TARGET dft-efe-basis)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Add source files here
set(DFT-EFE-BASIS-SOURCES
	TriangulationCellDealii.cpp
	TriangulationDealiiSerial.cpp
	LinearCellMappingDealii.cpp
	ParentToChildCellsManagerDealii.cpp
	TriangulationDealiiParallel.cpp
	ConstraintsInternal.cpp
	ConstraintsInternal.cu
	CFEConstraintsLocalDealii.cpp
	CFEBasisDofHandlerDealii.cpp
	FEBasisManager.cpp
	CFEBasisDataStorageDealii.cpp
	FEBasisOperations.cpp
	AtomIdsPartition.cpp
	EnrichmentIdsPartition.cpp
	CFEOverlapOperatorContext.cpp
	EFEOverlapOperatorContext.cpp
	EnrichmentClassicalInterfaceSpherical.cpp
	L2ProjectionLinearSolverFunction.cpp
	Defaults.cpp
	CFEOverlapInverseOperatorContext.cpp
	OrthoEFEOverlapInverseOpContextGLL.cpp
	EFEBasisDataStorageDealii.cpp
	EFEBasisDofHandlerDealii.cpp
	EFEConstraintsLocalDealii.cpp
	)

add_library(dft-efe-basis SHARED ${DFT-EFE-BASIS-SOURCES})

include_directories(../)

find_package(deal.II 9.5.1 REQUIRED HINTS ${DEALII_PATH})
target_include_directories(dft-efe-basis PUBLIC ${DEALII_PATH}/${DEAL_II_INCLUDE_RELDIR})
target_link_libraries (dft-efe-basis PUBLIC dealii::dealii_release)
#target_link_libraries (dft-efe-basis PUBLIC ${DEALII_PATH}/lib)


if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
		target_include_directories(dft-efe-basis PUBLIC ${MPI_CXX_INCLUDE_PATH})
		set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
	endif()
endif()

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()

if (NOT TARGET dft-efe-quadrature)
 add_subdirectory(../quadrature ${CMAKE_CURRENT_BINARY_DIR}/buildQuadrature)
endif()

if (NOT TARGET dft-efe-linalg)
 add_subdirectory(../linearAlgebra ${CMAKE_CURRENT_BINARY_DIR}/buildLinearAlgebra)
endif()

if (NOT TARGET dft-efe-atoms)
  add_subdirectory(../atoms ${CMAKE_CURRENT_BINARY_DIR}/buildAtoms)
endif()

target_link_libraries(dft-efe-basis PUBLIC dft-efe-utils dft-efe-quadrature dft-efe-linalg dft-efe-atoms ${DFTEFE_MPI_LIBRARIES})
endif()
