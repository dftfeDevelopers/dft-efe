/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Avirup Sircar
 */

#ifndef dftefeEnrichmentClassicalInterfaceSpherical_h
#define dftefeEnrichmentClassicalInterfaceSpherical_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <vector>
#include <string>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <memory>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <basis/FEBasisManager.h>
#include <basis/FEBasisHandler.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/FEBasisOperations.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
#include <basis/TriangulationBase.h>
#  include <linearAlgebra/LinAlgOpContext.h>

namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Class to get the interface between Classical and Enrichment basis. It takes as the classical basis as input. 
     * The main functionalities of the class are:
     * 1. Partition the enrichment degrees of freedom based on a.) periodic/non-periodic BCs b.) Orthogonalized or pristine enrichment dofs needed.
     * 2. Acts as input to the EFEBasisManager the basic class for all EFE operations.
     * 3. For Orthogonalized EFE it carries an extra set of d_i's according to Md = f obtained from classical dofs.
     * 4. For periodic BC, there will be contributions to enrichment set from periodic images.
     */
    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    class EnrichmentClassicalInterfaceSpherical
    {
    public:
      /**
       * @brief This Constructor for orthogonalized EFE which takes as input
       * @param[in] atomCoordinates Vector of Coordinates of the atoms
       * @param[in] tolerance set the tolerance for partitioning the atomids
       * @param[in] comm MPI_Comm object if defined with MPI
       */
      EnrichmentClassicalInterfaceSpherical(
      std::shared_ptr<const FEBasisDataStorage<ValueTypeBasisData, memorySpace>> cfeBasisDataStorage,
      std::shared_ptr<const FEBasisHandler<ValueTypeBasisData, memorySpace, dim>> cfeBasisHandler,
      const quadrature::QuadratureRuleAttributes l2ProjQuadAttr,
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const double                     atomPartitionTolerance,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      std::string                      basisInterfaceCoeffConstraint,
      std::shared_ptr< const linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContext,
      const utils::mpi::MPIComm &      comm);

      EnrichmentClassicalInterfaceSpherical(
      std::shared_ptr<const TriangulationBase> triangulation,
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const double                     atomPartitionTolerance,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      const utils::mpi::MPIComm &      comm);

      /**
       * @brief Destructor for the class
       */
      ~EnrichmentClassicalInterfaceSpherical() = default;

      /**
       * @brief Function to return AtomSphericalDataContainerObject
       */
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
      getAtomSphericalDataContainer() const;

      std::shared_ptr<const EnrichmentIdsPartition<dim>>
      getEnrichmentIdsPartition() const;

      std::shared_ptr<const AtomIdsPartition<dim>>
      getAtomIdsPartition() const;

      std::shared_ptr<const FEBasisHandler<ValueTypeBasisData, memorySpace, dim>>
      getCFEBasisHandler() const;

      std::string
      getBasisInterfaceCoeffConstraint() const;

      linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>
      getBasisInterfaceCoeff() const;

      bool
      isOrthgonalized() const;

    private:
      std::shared_ptr<const EnrichmentIdsPartition<dim>> 
                                d_enrichmentIdsPartition;
      std::shared_ptr<const AtomIdsPartition<dim>> 
                                d_atomIdsPartition;
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                d_atomSphericalDataContainer;
      linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace> d_basisInterfaceCoeff;
      const bool d_isOrthogonalized;
      std::shared_ptr<const FEBasisHandler<ValueTypeBasisData, memorySpace, dim>> d_cfeBasisHandler;
      const std::string d_basisInterfaceCoeffConstraint;

    }; // end of class
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/EnrichmentClassicalInterfaceSpherical.t.cpp>
#endif // dftefeEnrichmentClassicalInterfaceSpherical_h
