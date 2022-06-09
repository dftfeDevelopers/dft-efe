if (NOT TARGET dft-efe-linalg)
    if (${CMAKE_CXX_STANDARD})
        set(CMAKE_CXX_STANDARD 14)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif ()

    # this part should be removed in the future and move the SLATE_DIR definition to be configured by install.py automatically
    if (ENABLE_CUDA)
        set(SLATE_DIR "/home/vikramg/DFT-FE-softwares/slate/gpu/gcc8.2.0_openmpi4.0.6_nvcc11.4.0")
    else ()
        set(SLATE_DIR "/home/vikramg/DFT-FE-softwares/slate/cpu/gcc8.2.0_openmpi4.0.6")
    endif ()

    # Add source files here
    set(DFT-EFE-LINALG-SOURCES
            BlasLapack.cpp
            BlasLapackKernels.cpp
            BlasLapackKernels.cu
            DistributedVector.cpp
            Matrix.cpp
            MatrixOperations.cpp
            QueueManager.cpp
            SerialDenseMatrix.cpp
            SerialVector.cpp
            VectorAttributes.cpp)

    add_library(dft-efe-linalg SHARED ${DFT-EFE-LINALG-SOURCES})

    include_directories(../)

    if (ENABLE_MPI)
        add_compile_definitions(DFTEFE_WITH_MPI)
	message("DFTEFE_WITH_MPI in linAlg")
        if (NOT MPI_FOUND)
            find_package(MPI REQUIRED)
            target_include_directories(dft-efe-linalg PUBLIC ${MPI_CXX_INCLUDE_PATH})
            set(DFTEFE_LINEARALGEBRA_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
        endif ()
    endif ()

    target_include_directories(dft-efe-linalg PUBLIC ${SLATE_DIR}/include)
    target_link_directories(dft-efe-linalg PUBLIC ${SLATE_DIR}/lib64)

    if (ENABLE_CUDA)
        enable_language(CUDA)
        add_compile_definitions(DFTEFE_WITH_DEVICE)
        add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
        set_target_properties(dft-efe-linalg PROPERTIES
                CUDA_STANDARD 14
                CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(dft-efe-linalg PUBLIC CUDA::cublas)
    endif ()

    if (NOT TARGET dft-efe-utils)
        add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
    endif ()


    target_link_libraries(dft-efe-linalg PUBLIC blaspp dft-efe-utils ${DFTEFE_LINEARALGEBRA_MPI_LIBRARIES} ${DFTEFE_BLAS_LIBRARIES})
endif ()
