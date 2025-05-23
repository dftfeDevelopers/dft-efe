if (NOT TARGET dft-efe-linalg)
    if (${CMAKE_CXX_STANDARD})
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif ()

    # Add source files here
    set(DFT-EFE-LINALG-SOURCES
	    Defaults.cpp
	    LinearAlgebraTypes.cpp
	    LinearAlgebraProfiler.cpp
            BlasLapack.cpp
            BlasLapackKernels.cpp
            BlasLapackKernels.cu
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
        NewtonRaphsonSolver.cpp)

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

    target_include_directories(dft-efe-linalg PUBLIC ${SLATE_DIR}/include)
    target_link_directories(dft-efe-linalg PUBLIC ${SLATE_DIR}/lib64)

    target_link_libraries(dft-efe-linalg PUBLIC blaspp lapackpp slate dft-efe-utils ${DFTEFE_LINEARALGEBRA_MPI_LIBRARIES})
endif ()
