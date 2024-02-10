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
#include <basis/FEBasisDofHandler.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/CFEOverlapOperatorContext.h>
#include <basis/FEBasisOperations.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
#include <basis/TriangulationBase.h>
#include <linearAlgebra/LinAlgOpContext.h>

namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Class to get the interface between Classical and Enrichment basis. It takes as the classical basis as input.
     * The main functionalities of the class are:
     * 1. Partition the enrichment degrees of freedom based on a.)
     * periodic/non-periodic BCs b.) Orthogonalized or pristine enrichment dofs
     * needed.
     * 2. Acts as input to the EFEBasisDofHandler the basic class for all EFE
     * operations.
     * 3. For Orthogonalized EFE it carries an extra set of c's from Mc = d
     * obtained from classical dofs.
     * 4. For periodic BC, there will be contributions to enrichment set from
     * periodic images.
     */
    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class EnrichmentClassicalInterfaceSpherical
    {
    public:
      /**
       * @brief This Constructor for augmenting the orthogonalized EFE basis with classical FE basis.
       * and then 1. Partition the enrichment dofs and 2. Form the
       * orthogonalized basis by L2 projecting the pristine enrichment functions
       * on the classicla FE basis. One can write any field \f$u^h_{\alpha}(x) =
       * \sum_{i=1}^{n_h}N_i^Cu^C_{\alpha,i}
       * +\sum_{I=1}^{N_e}N_I^{O,u}u_{\alpha,I}\f$ where \f$\alpha\f$ represents
       * the number of discretized fields, C is the classical part and O is the
       * orthogonalized enriched basis part,\f$N_I\f$ represents the number of
       * enrichment dofs and \f$n_h\f$ represents the number of classical finite
       * element dofs. The orthogonalized basis functions can be written as
       * \f$N^{O,u}_I(x) = N^{A,u}_I(x) − N^{B,u}_I(x)\f$, where
       * \f$N^{A,u}_I(x)\f$ represents the pristine enrichment functions and
       * \f$N^{B,u}_I(x)\f is the component of enrichment function along to the
       * classical basis and can be written in terms of classical basis
       * functions as \f$N^{B,u}_I(x) = \sum^{n_h}_{l=1}c_{I,l}N^{C}_{l}(x)\f.
       * Since the \f$N^{O,u}_I(x)\f$'s are orthogonal to classical basis, one
       * can solve the equation \f$M^{cc}.c = d\f$ where M is the discretized
       * overlap matrix in classical FE basis and \f$M^{cc}_{j,l} =
       * \integral_{\omega}N^{C}_{l}(x)N^{C}_{j}(x)dx\f$ and d is the RHS
       * written as \f$d_{I,j,k} =
       * \integral_{\omega}N^{A,u}_{i,j}(x)N^{C}_{k}(x)dx\f$ and \f$c\f$ is the
       * \p basisInterfaceCoefficient written as a multivector.
       * @param[in] atomCoordinates Vector of Coordinates of the atoms
       * @param[in] atomPartitionTolerance set the tolerance for partitioning
       * the atomids
       * @param[in] comm MPI_Comm object if defined with MPI
       * @param[in] cfeBasisDataStorageRhs FEBasisDataStorage object for RHS
       * \f$d_{I}\f$
       * @param[in] cfeBasisDataStorageOverlapMatrix FEBasisDataStorage object
       * for the OverlapMatrix \f$M^{cc}\f$
       * @param[in] linAlgOpContext The linearAlgebraOperator context for
       * solving the linear equation
       * @param[in] fieldName The fieldname of the enrichment function
       */
      EnrichmentClassicalInterfaceSpherical(
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          cfeBasisDataStorageOverlapMatrix,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          cfeBasisDataStorageRhs,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const double                     atomPartitionTolerance,
        const std::vector<std::string> & atomSymbolVec,
        const std::vector<utils::Point> &atomCoordinatesVec,
        const std::string                fieldName,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm);

      /**
       * @brief This Constructor for augmenting the EFE basis with classical FE basis.
       * @param[in] atomCoordinates Vector of Coordinates of the atoms
       * @param[in] atomPartitionTolerance set the tolerance for partitioning
       * the atomids
       * @param[in] comm MPI_Comm object if defined with MPI
       * @param[in] fieldName The fieldname of the enrichment function
       */
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

      /**
       * @brief Function to return EnrichmentIDsObject
       */
      std::shared_ptr<const EnrichmentIdsPartition<dim>>
      getEnrichmentIdsPartition() const;

      std::shared_ptr<const AtomIdsPartition<dim>>
      getAtomIdsPartition() const;

      std::shared_ptr<const BasisManager<ValueTypeBasisData, memorySpace>>
      getCFEBasisManager() const;

      std::shared_ptr<const BasisDofHandler>
      getCFEBasisDofHandler() const;

      const linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace> &
      getBasisInterfaceCoeff() const;

      bool
      isOrthgonalized() const;

      std::vector<std::string>
      getAtomSymbolVec() const;

      std::vector<utils::Point>
      getAtomCoordinatesVec() const;

      std::string
      getFieldName() const;

      std::shared_ptr<const TriangulationBase>
      getTriangulation() const;


      /**
       * @brief The localid is determined by the storage pattern of the components of
       * basisInterfaceCoeff multivector. The multivector has a storage pattern
       * of [LocallyownedEnrichmentIds, GhostEnrichemntIds]
       */

      global_size_type
      getEnrichmentId(size_type cellId, size_type enrichmentCellLocalId) const;

      size_type
      getEnrichmentLocalId(global_size_type enrichmentId) const;

      size_type
      getEnrichmentLocalId(size_type cellId,
                           size_type enrichmentCellLocalId) const;

    private:
      std::shared_ptr<const EnrichmentIdsPartition<dim>>
                                                   d_enrichmentIdsPartition;
      std::shared_ptr<const AtomIdsPartition<dim>> d_atomIdsPartition;
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                               d_atomSphericalDataContainer;
      std::shared_ptr<const TriangulationBase> d_triangulation;
      std::shared_ptr<
        linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>>
           d_basisInterfaceCoeff;
      bool d_isOrthogonalized;
      std::shared_ptr<
        const FEBasisDofHandler<ValueTypeBasisData, memorySpace, dim>>
        d_cfeBasisDofHandler;
      std::shared_ptr<const FEBasisManager<ValueTypeBasisData,
                                           ValueTypeBasisData,
                                           memorySpace,
                                           dim>>
                                      d_cfeBasisManager;
      const std::vector<std::string>  d_atomSymbolVec;
      const std::vector<utils::Point> d_atomCoordinatesVec;
      const std::string               d_fieldName;
      std::vector<std::vector<global_size_type>>
        d_overlappingEnrichmentIdsInCells;

    }; // end of class
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/EnrichmentClassicalInterfaceSpherical.t.cpp>
#endif // dftefeEnrichmentClassicalInterfaceSpherical_h
