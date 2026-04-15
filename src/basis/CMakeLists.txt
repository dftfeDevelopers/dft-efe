if (NOT TARGET dft-efe-basis)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

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

# Add source files here
set(DFT-EFE-BASIS-SOURCES
	TriangulationCellDealii.cpp
	TriangulationDealiiSerial.cpp
	LinearCellMappingDealii.cpp
	ParentToChildCellsManagerDealii.cpp
	TriangulationDealiiParallel.cpp
	ConstraintsInternal.cpp
	#ConstraintsInternal.cu
	CFEConstraintsLocalDealii.cpp
	CFEBasisDofHandlerDealii.cpp
	FEBasisManager.cpp
	CFEBasisDataStorageDealii.cpp
	CFEBDSOnTheFlyComputeDealii.cpp
	FEBasisOperations.cpp
	AtomIdsPartition.cpp
	EnrichmentIdsPartition.cpp
	CFEOverlapOperatorContext.cpp
	PristineEFEOverlapOperatorContext.cpp
	OrthoEFEOverlapOperatorContext.cpp
	L2ProjectionLinearSolverFunction.cpp
	Defaults.cpp
	CFEOverlapInverseOpContextGLL.cpp
	EnrichmentClassicalInterfaceSpherical.cpp
	OrthoEFEOverlapInverseOpContextGLL.cpp
	EFEBasisDataStorageDealii.cpp
	EFEBDSOnTheFlyComputeDealii.cpp
	EFEBasisDofHandlerDealii.cpp
	EFEConstraintsLocalDealii.cpp
	GenerateMesh.cpp
	DealiiFEEvaluationWrapper.cpp
	OEFEAtomBlockOverlapInvOpContextGLL.cpp
	AtomCenterNonLocalOpContextFE.cpp
	FECellWiseDataOperations.cpp
	EnrichmentDataEvalKernels.cpp
	)

IF (ENABLE_GPU)
IF ("${GPU_LANG}" STREQUAL "cuda")
SET(DFT-EFE-BASIS-DEVICE-SOURCES
	ConstraintsInternalDeviceKernels.cpp
	FECellWiseDataOperationsDeviceKernels.cpp
	EnrichmentDataEvalDeviceKernels.cpp
	)
ELSEIF ("${GPU_LANG}" STREQUAL "hip")
SET(DFT-EFE-BASIS-DEVICE-SOURCES
	ConstraintsInternalDeviceKernels.cpp
	FECellWiseDataOperationsDeviceKernels.cpp
	EnrichmentDataEvalDeviceKernels.cpp
	)
ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
SET(DFT-EFE-BASIS-DEVICE-SOURCES
	ConstraintsInternalDeviceKernels.cpp
	FECellWiseDataOperationsDeviceKernels.cpp
	EnrichmentDataEvalDeviceKernels.cpp
	)
ENDIF()
ENDIF()

IF (ENABLE_GPU)
	IF ("${GPU_LANG}" STREQUAL "cuda")
	set_source_files_properties(${DFT-EFE-BASIS-DEVICE-SOURCES} PROPERTIES LANGUAGE CUDA)
	ELSEIF ("${GPU_LANG}" STREQUAL "hip")
	set_source_files_properties(${DFT-EFE-BASIS-DEVICE-SOURCES} PROPERTIES LANGUAGE HIP)
	IF ("${GPU_VENDOR}" STREQUAL "amd")
		ADD_DEFINITIONS(-D__HIP_PLATFORM_AMD__)
	ELSEIF ("${GPU_VENDOR}" STREQUAL "nvidia")
		ADD_DEFINITIONS(-D__HIP_PLATFORM_NVIDIA__)
	ENDIF()
	ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
	set_source_files_properties(${DFT-EFE-BASIS-DEVICE-SOURCES} PROPERTIES COMPILE_FLAGS ${CMAKE_SYCL_FLAGS})
	ENDIF()

LIST(APPEND DFT-EFE-BASIS-SOURCES ${DFT-EFE-BASIS-DEVICE-SOURCES})
ENDIF()

add_library(dft-efe-basis SHARED ${DFT-EFE-BASIS-SOURCES})

include_directories(../)

find_package(deal.II 9.7.0 REQUIRED HINTS ${DEALII_PATH})
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
