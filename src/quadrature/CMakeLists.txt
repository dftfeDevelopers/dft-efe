if (NOT TARGET dft-efe-quadrature)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


if(ENABLE_MPI)
	add_compile_definitions(DFTEFE_WITH_MPI)
	if (NOT MPI_FOUND)
		find_package(MPI REQUIRED)
		target_include_directories(dft-efe-quadrature PUBLIC ${MPI_CXX_INCLUDE_PATH})
		set(DFTEFE_MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
	endif()
endif()

# Add source files here
set(DFT-EFE-QUADRATURE-SOURCES
	QuadratureAttributes.cpp
	QuadratureRule.cpp
	QuadratureRuleContainer.cpp
        QuadratureRuleGauss.cpp
	QuadratureRuleGLL.cpp
	QuadratureRuleAdaptive.cpp
	QuadratureRuleGaussIterated.cpp
	QuadratureValuesContainer.cpp
	Integrate.cpp
	Defaults.cpp)

add_library(dft-efe-quadrature SHARED ${DFT-EFE-QUADRATURE-SOURCES})

find_package(deal.II 9.5.1 REQUIRED HINTS ${DEALII_PATH})
target_include_directories(dft-efe-quadrature PUBLIC ${DEALII_PATH}/${DEAL_II_INCLUDE_RELDIR})
target_link_libraries(dft-efe-quadrature PUBLIC dealii::dealii_release)
#target_link_libraries (dft-efe-quadrature PUBLIC ${DEALII_PATH}/${DEAL_II_LIBRARY_RELDIR})

if (NOT TARGET dft-efe-utils)
  add_subdirectory(../utils ${CMAKE_CURRENT_BINARY_DIR}/buildUtils)
endif()

target_include_directories(dft-efe-quadrature PUBLIC ${SLATE_DIR}/include)
target_link_directories(dft-efe-quadrature PUBLIC ${SLATE_DIR}/lib64)

include_directories(../)
target_link_libraries(dft-efe-quadrature PUBLIC blaspp lapackpp slate dft-efe-utils ${DFTEFE_MPI_LIBRARIES})
endif()
