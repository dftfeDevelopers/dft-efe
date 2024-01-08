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
 * @author Bikash Kanungo, Vishal Subramanian
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
    class FEBasisManagerDealii : public FEBasisManager
    {
    public:
      using FECellIterator       = FEBasisManager::FECellIterator;
      using const_FECellIterator = FEBasisManager::const_FECellIterator;

      FEBasisManagerDealii(
        std::shared_ptr<const TriangulationBase> triangulation,
        const size_type                          feOrder);
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
             const size_type                          feOrder);

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

      FECellIterator
      beginLocallyOwnedCells() override;

      FECellIterator
      endLocallyOwnedCells() override;

      const_FECellIterator
      beginLocallyOwnedCells() const override;

      const_FECellIterator
      endLocallyOwnedCells() const override;

      FECellIterator
      beginLocalCells() override;
      FECellIterator
      endLocalCells() override;
      const_FECellIterator
      beginLocalCells() const override;
      const_FECellIterator
      endLocalCells() const override;
      unsigned int
      getDim() const override;

      size_type
      nCumulativeLocallyOwnedCellDofs() const override;

      size_type
      nCumulativeLocalCellDofs() const override;

      size_type
      totalRanges() const override;

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

      std::vector<std::vector<global_size_type>>
      getBoundaryGlobalNodeIds() const;

    private:
      std::shared_ptr<const TriangulationBase> d_triangulation;
      std::shared_ptr<dealii::DoFHandler<dim>> d_dofHandler;
      bool                                     d_isVariableDofsPerCell;
      std::vector<std::shared_ptr<FECellBase>> d_localCells;
      std::vector<std::shared_ptr<FECellBase>> d_locallyOwnedCells;
      size_type d_numCumulativeLocallyOwnedCellDofs;
      size_type d_numCumulativeLocalCellDofs;
      size_type d_totalRanges;

    }; // end of FEBasisManagerDealii
  }    // end of namespace basis
} // end of namespace dftefe
#include "FEBasisManagerDealii.t.cpp"
#endif // dftefeFEBasisManagerDealii_h
