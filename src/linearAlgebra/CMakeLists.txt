message (" DFTEFE_BLAS_LIBRARIES = ${DFTEFE_BLAS_LIBRARIES}")
message (" DFTEFE_SCALAPACK_LIBRARIES = ${DFTEFE_SCALAPACK_LIBRARIES}")


if ("${DFTEFE_BLAS_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFTEFE_BLAS libraries not specified. Please specify -DDFTEFE_BLAS_LIBRARIES in command line")
endif ()


if ("${DFTEFE_SCALAPACK_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFTEFE_SCALAPACK libraries not specified. Please specify -DDFTEFE_SCALAPACK_LIBRARIES in command line")
endif ()

# Add source files here
set(DFT-EFE-LINALG-SOURCES
    Vector.cpp)

IF (ENABLE_CUDA)
    set_source_files_properties(${DFT-EFE-LINALG-SOURCES} PROPERTIES LANGUAGE CUDA)
ENDIF()

add_library(dft-efe-linalg SHARED ${DFT-EFE-LINALG-SOURCES} )
target_include_directories(dft-efe-linalg INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

IF (ENABLE_CUDA)
   find_package(CUDAToolkit REQUIRED)
   target_link_libraries(dft-efe-linalg PRIVATE CUDA::cudart CUDA::cublas)
ENDIF()

target_link_libraries(dft-efe-linalg PRIVATE dft-efe-utils ${DFTEFE_MPI_LIBRARIES} ${DFTEFE_BLAS_LIBRARIES} ${DFTEFE_SCALAPACK_LIBRARIES})
