
# Add source files here
set(DFT-EFE-UTILS-SOURCES
    MemoryManager.cpp
    CUDAUtils.cpp)

IF (ENABLE_CUDA)
    set_source_files_properties(${DFT-EFE-UTILS-SOURCES} PROPERTIES LANGUAGE CUDA)
ENDIF()

add_library(dft-efe-utils SHARED  ${DFT-EFE-UTILS-SOURCES} )
target_include_directories(dft-efe-utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(dft-efe-utils PRIVATE ${DFTEFE_MPI_LIBRARIES})

IF (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(dft-efe-utils PRIVATE CUDA::cudart)
ENDIF()
