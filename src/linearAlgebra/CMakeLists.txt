if (NOT TARGET dft-efe-linalg)
    if (${CMAKE_CXX_STANDARD})
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif ()

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
    set(DFT-EFE-LINALG-SOURCES
	    Defaults.cpp
	    LinearAlgebraTypes.cpp
	    LinearAlgebraProfiler.cpp
        BlasAPIWrapperHost.cpp
        LapackAPIWrapperHost.cpp
        BlasLapack.cpp
        BlasLapackKernels.cpp
        LinAlgOpContext.cpp
        VectorAttributes.cpp
	    Vector.cpp
	    MultiVector.cpp
	    LinearSolver.cpp
	    CGLinearSolver.cpp
        PreconditionerJacobi.cpp
        PreconditionerNone.cpp
        OrthonormalizationFunctions.cpp
        IdentityOperatorContext.cpp
        LanczosExtremeEigenSolver.cpp
        ChebyshevFilter.cpp
        RayleighRitzEigenSolver.cpp
        ChebyshevFilteredEigenSolver.cpp
        NewtonRaphsonSolver.cpp
        ProcessGrid.cpp
        ScalapackWrapper.cpp
        ElpaScalapackOperations.cpp
        ElpaScalapackManager.cpp)

    IF (ENABLE_GPU)
    IF ("${GPU_LANG}" STREQUAL "cuda")
    SET(DFT-EFE-LINALG-DEVICE-SOURCES
        BlasAPIWrapperDevice.cpp
        BlasLapackKernelsDevice.cpp
        )
    ELSEIF ("${GPU_LANG}" STREQUAL "hip")
    SET(DFT-EFE-LINALG-DEVICE-SOURCES
        BlasAPIWrapperDevice.cpp
        BlasLapackKernelsDevice.cpp
        )
    ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
    SET(DFT-EFE-LINALG-DEVICE-SOURCES
        BlasAPIWrapperDevice.cpp
        BlasLapackKernelsDevice.cpp
        )
    ENDIF()
    ENDIF()

    IF (ENABLE_GPU)
        IF ("${GPU_LANG}" STREQUAL "cuda")
        set_source_files_properties(${DFT-EFE-LINALG-DEVICE-SOURCES} PROPERTIES LANGUAGE CUDA)
        ELSEIF ("${GPU_LANG}" STREQUAL "hip")
        set_source_files_properties(${DFT-EFE-LINALG-DEVICE-SOURCES} PROPERTIES LANGUAGE HIP)
        IF ("${GPU_VENDOR}" STREQUAL "amd")
            ADD_DEFINITIONS(-D__HIP_PLATFORM_AMD__)
        ELSEIF ("${GPU_VENDOR}" STREQUAL "nvidia")
            ADD_DEFINITIONS(-D__HIP_PLATFORM_NVIDIA__)
        ENDIF()
        ELSEIF ("${GPU_LANG}" STREQUAL "sycl")
        set_source_files_properties(${DFT-EFE-LINALG-DEVICE-SOURCES} PROPERTIES COMPILE_FLAGS ${CMAKE_SYCL_FLAGS})
        ENDIF()

    LIST(APPEND DFT-EFE-LINALG-SOURCES ${DFT-EFE-LINALG-DEVICE-SOURCES})
    ENDIF()

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

    # if (ENABLE_CUDA)
    #     enable_language(CUDA)
    #     add_compile_definitions(DFTEFE_WITH_DEVICE)
    #     add_compile_definitions(DFTEFE_WITH_DEVICE_CUDA)
    #     set_target_properties(dft-efe-linalg PROPERTIES
    #             CUDA_STANDARD 14
    #             CUDA_STANDARD_REQUIRED ON)
    #     find_package(CUDAToolkit REQUIRED)
    #     target_link_libraries(dft-efe-linalg PUBLIC CUDA::cublas)
    # endif ()

    if (NOT TARGET dft-efe-utils)
        add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
    endif ()

    #NOTE : DFTFE doesn't link to scalapack directly. Check this.
    target_link_libraries(dft-efe-linalg PUBLIC ${SCALAPACK_LIBRARIES})

    find_package(ELPA 2023 REQUIRED)
    target_link_libraries(dft-efe-linalg PUBLIC ELPA::ELPA dft-efe-utils ${DFTEFE_LINEARALGEBRA_MPI_LIBRARIES})
endif ()
