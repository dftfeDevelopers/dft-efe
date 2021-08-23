cmake_minimum_required(VERSION 3.17)
project(dft-efe VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

message (" DFT_EFE_BLAS_DIR = ${DFT_EFE_BLAS_DIR}")
message (" DFT_EFE_BLAS_LIBRARIES = ${DFT_EFE_BLAS_LIBRARIES}")

if (NOT EXISTS ${DFT_EFE_BLAS_DIR})
	message (FATAL_ERROR  "DFT_EFE_Blas directory not specified. Please specify -DDFT_EFE_BLAS_DIR in command line")
endif ()
	
if ("${DFT_EFE_BLAS_LIBRARIES}" STREQUAL "")
	message (FATAL_ERROR  "DFT_EFE_BLAS libraries not specified. Please specify -DDFT_EFE_BLAS_LIBRARIES in command line")
endif ()	

link_directories(${DFT_EFE_BLAS_DIR})


include_directories(${MPI_CXX_INCLUDE_PATH})
set(DFT_EFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})

set(DFT_EFE_LIBRARIES ${DFT_EFE_MPI_LIBRARIES} ${DFT_EFE_BLAS_LIBRARIES})


add_executable(dft-efe src/main.cpp)
target_link_libraries(dft-efe ${DFT_EFE_LIBRARIES})		
