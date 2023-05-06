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

#ifndef dftefeFEBasisManagerDealii_h
#define dftefeFEBasisManagerDealii_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <basis/FEBasisManager.h>
#include <memory>
#include <deal.II/fe/fe_q.h>

/// dealii includes
#include <deal.II/dofs/dof_handler.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * A derived class of FEBasisManager to handle the FE basis evaluations
     * through dealii
     */
    template <size_type dim>
    class EFEBasisManagerDealii : public EFEBasisManager
    {
    public:

      // add attribute to the classical and enriched ids for accessing locallyowned ranges
      enum  basisIdAttribute{
        classical,
        enriched
      };

      EFEBasisManagerDealii(
        std::shared_ptr<const TriangulationBase>     triangulation,
        std::shared_ptr<const atoms::AtomSphericalDataContainer> atomSphericalDataContainer,
        const size_type                              feOrder,
        const double                                 atomPartitionTolerance,
        const std::vector<std::string> &             atomSymbol,
        const std::vector<utils::Point> &            atomCoordinates,
        const std::string                            fieldName);

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
      reinit(std::shared_ptr<const TriangulationBase> triangulation,
             const size_type                          feOrder) override;

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
      isHPRefined() const override;

      size_type
      nLocalNodes() const override;

      std::pair<global_size_type, global_size_type>
      getLocallyOwnedRange() const override;

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

      void
      getCellDofsLocalIds(
        size_type                      cellId,
        std::vector<global_size_type> &vecGlobalNodeId) const override;

      std::vector<size_type>
      getBoundaryIds() const override;

      FEBasisManager::FECellIterator
      beginLocallyOwnedCells() override;;

      FEBasisManager::FECellIterator
      endLocallyOwnedCells() override;;

      FEBasisManager::const_FECellIterator
      beginLocallyOwnedCells() const override;;

      FEBasisManager::const_FECellIterator
      endLocallyOwnedCells() const override;;

      FEBasisManager::FECellIterator
      beginLocalCells() override;;
      FEBasisManager::FECellIterator
      endLocalCells() override;;
      FEBasisManager::const_FECellIterator
      beginLocalCells() const override;;
      FEBasisManager::const_FECellIterator
      endLocalCells() const override;;
      unsigned int
      getDim() const override;;

      virtual size_type
      nCumulativeLocallyOwnedCellDofs() const override;

      virtual size_type
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

      // Enrichment functions with dealii mesh. The argument type is the processor local enriched ids.
      double
      getEnrichmentValue(size_type enrichmentId) const override;

      virtual std::vector<double>
      getEnrichmentDerivative(size_type enrichmentId) const override;

      virtual std::vector<double>
      getEnrichmentHessian(size_type enrichmentd) const override;

    private:

      std::shared_ptr<const TriangulationBase> d_triangulation;
      std::shared_ptr<dealii::DoFHandler<dim>> d_dofHandler;
      bool                                     d_isHPRefined;
      std::vector<std::shared_ptr<FECellBase>> d_localCells;
      std::vector<std::shared_ptr<FECellBase>> d_locallyOwnedCells;
      size_type d_numCumulativeLocallyOwnedCellDofs;
      size_type d_numCumulativeLocalCellDofs;
      std::shared_ptr<const EnrichmentIdsPartition> d_enrichmentIdsPartition;
      std::shared_ptr<const AtomIdsPartition> d_atomIdsPartition;

    }; // end of FEBasisManagerDealii
  }    // end of namespace basis
} // end of namespace dftefe
#include "EFEBasisManagerDealii.t.cpp"
#endif // dftefeFEBasisManagerDealii_h
//
