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

#ifndef dftefeEFEBasisManagerDealii_h
#  define dftefeEFEBasisManagerDealii_h

#  include <utils/TypeConfig.h>
#  include <utils/Point.h>
#  include <basis/EFEBasisManager.h>
#  include <memory>
#  include <deal.II/fe/fe_q.h>
#  include <basis/AtomIdsPartition.h>
#  include <atoms/AtomSphericalDataContainer.h>
#  include <utils/Exceptions.h>
#  include <utils/MPITypes.h>
#  include <utils/MPIWrapper.h>
#  include <linearAlgebra/LinAlgOpContext.h>

/// dealii includes
#  include <deal.II/dofs/dof_handler.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * A derived class of FEBasisManager to handle the FE basis evaluations
     * through dealii
     */
    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    class EFEBasisManagerDealii : public EFEBasisManager<ValueTypeBasisData, memorySpace, dim>
    {
    public:

      EFEBasisManagerDealii(
        std::shared_ptr<const EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace, dim>> 
          enrichmentClassicalInterface,
        const size_type                  feOrder);

      double
      getBasisFunctionValue(const size_type     basisId,
                            const utils::Point &point) const override;
      std::vector<double>
      getBasisFunctionDerivative(
        const size_type     basisId,
        const utils::Point &point,
        const size_type     derivativeOrder = 1) const override;

      ////// FE specific  member functions /////
      void
      reinit(std::shared_ptr<const EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace, dim>> 
        enrichmentClassicalInterface, const size_type  feOrder);

      std::shared_ptr<const TriangulationBase>
      getTriangulation() const override;

      size_type
      nLocalCells() const override;
      size_type
      nLocallyOwnedCells() const override;

      size_type
      nGlobalCells() const override;
      size_type
      getFEOrder(size_type cellId) const override;

      size_type
      nCellDofs(size_type cellId) const override;

      bool
      isVariableDofsPerCell() const override;

      size_type
      nLocalNodes() const override;

      std::vector<std::pair<global_size_type, global_size_type>>
      getLocallyOwnedRanges() const override;

      std::vector<std::pair<global_size_type, global_size_type>>
      getGlobalRanges() const override;

      std::map<BasisIdAttribute, size_type>
      getBasisAttributeToRangeIdMap() const override;

      global_size_type
      nGlobalNodes() const override;

      std::vector<size_type>
      getLocalNodeIds(size_type cellId) const override;

      std::vector<size_type>
      getGlobalNodeIds() const override;

      void
      getCellDofsGlobalIds(
        size_type                      cellId,
        std::vector<global_size_type> &vecGlobalNodeId) const override;

      std::vector<size_type>
      getBoundaryIds() const override;

      FEBasisManager::FECellIterator
      beginLocallyOwnedCells() override;

      FEBasisManager::FECellIterator
      endLocallyOwnedCells() override;

      FEBasisManager::const_FECellIterator
      beginLocallyOwnedCells() const override;

      FEBasisManager::const_FECellIterator
      endLocallyOwnedCells() const override;

      FEBasisManager::FECellIterator
      beginLocalCells() override;
      FEBasisManager::FECellIterator
      endLocalCells() override;
      FEBasisManager::const_FECellIterator
      beginLocalCells() const override;
      FEBasisManager::const_FECellIterator
      endLocalCells() const override;
      unsigned int
      getDim() const override;

      size_type
      nCumulativeLocallyOwnedCellDofs() const override;

      size_type
      nCumulativeLocalCellDofs() const override;

      // This assumes a linear cell mapping
      void
      getBasisCenters(
        std::map<global_size_type, utils::Point> &dofCoords) const override;

      //
      // dealii specific functions
      //
      std::shared_ptr<const dealii::DoFHandler<dim>>
      getDoFHandler() const;

      const dealii::FiniteElement<dim> &
      getReferenceFE(const size_type cellId) const;

      // Enrichment functions with dealii mesh. The enrichedid is the cell local
      // id.
      double
      getEnrichmentValue(const size_type             cellId,
                         const size_type             cellLocalEnrichmentId,
                         const dftefe::utils::Point &point) const override;

      std::vector<double>
      getEnrichmentDerivative(const size_type             cellId,
                              const size_type             cellLocalEnrichmentId,
                              const dftefe::utils::Point &point) const override;

      std::vector<double>
      getEnrichmentHessian(const size_type             cellId,
                           const size_type             cellLocalEnrichmentId,
                           const dftefe::utils::Point &point) const override;

      std::vector<global_size_type>
      getGhostEnrichmentGlobalIds() const override;

      global_size_type
      nGlobalEnrichmentNodes() const override;

      std::shared_ptr<const EnrichmentIdsPartition<dim>>
      getEnrichmentIdsPartition() const override;

      std::shared_ptr<const EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace, dim>>
      getEnrichmentClassicalInterface() const override;

      bool
      isOrthogonalized() const override;

      size_type
      totalRanges() const override;

    private:
      std::shared_ptr<dealii::DoFHandler<dim>> d_dofHandler;
      bool                                     d_isVariableDofsPerCell;
      std::vector<std::shared_ptr<FECellBase>> d_localCells;
      std::vector<std::shared_ptr<FECellBase>> d_locallyOwnedCells;
      size_type d_numCumulativeLocallyOwnedCellDofs;
      size_type d_numCumulativeLocalCellDofs;
      std::shared_ptr<const EnrichmentIdsPartition<dim>>
        d_enrichmentIdsPartition;
      std::vector<std::vector<global_size_type>>
                      d_overlappingEnrichmentIdsInCells;
      size_type d_totalRanges;
      std::vector<std::pair<global_size_type, global_size_type>>
        d_locallyOwnedRanges;
      std::vector<std::pair<global_size_type, global_size_type>> 
        d_globalRanges;
      std::vector<global_size_type> d_ghostEnrichmentGlobalIds;
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                d_atomSphericalDataContainer;
      std::vector<std::string>  d_atomSymbolVec;
      std::vector<utils::Point> d_atomCoordinatesVec;
      std::string               d_fieldName;
      bool                      d_isOrthogonalized;
      std::shared_ptr<const EnrichmentClassicalInterfaceSpherical
        <ValueTypeBasisData, memorySpace, dim>> d_enrichClassIntfce;

    }; // end of EFEBasisManagerDealii
  }    // end of namespace basis
} // end of namespace dftefe
#  include "EFEBasisManagerDealii.t.cpp"
#endif // dftefeEFEBasisManagerDealii_h
//
