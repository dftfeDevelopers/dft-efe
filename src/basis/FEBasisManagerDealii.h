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
      using FECellIterator       = FEBasisManager<dim>::FECellIterator;
      using const_FECellIterator = FEBasisManager<dim>::const_FECellIterator;

      FEBasisManagerDealii(TriangulationBase &tria);
      ~FEBasisManagerDealii();
      double
      getBasisFunctionValue(const size_type     basisId,
                            const utils::Point &point) const;
      std::vector<double>
      getBasisFunctionDerivative(const size_type     basisId,
                                 const utils::Point &point,
                                 const size_type     derivativeOrder = 1) const;

      ////// FE specific  member functions /////
      void
      reinit(const TriangulationBase &triangulation, const size_type feOrder);
      size_type
      nLocallyActiveCells() const = 0;
      size_type
      nOwnedCells() const = 0;
      size_type
      nGloballyActiveCells() const = 0;
      size_type
      getFEOrder(size_type cellId) const;
      size_type
      nCellDofs(size_type cellId) const;
      bool
      isHPRefined() const;
      size_type
      nLocalNodes() const;
      global_size_type
      nGlobalNodes() const;
      std::vector<size_type>
      getLocalNodeIds(size_type cellId) const;
      std::vector<size_type>
      getGlobalNodeIds() const;
      std::vector<size_type>
      getCellDofsLocalIds(size_type cellId) const;
      std::vector<size_type>
      getBoundaryIds() const;
      FECellIterator
      beginLocallyOwnedCells();
      FECellIterator
      endLocallyOwnedCells();
      const_FECellIterator
      beginLocallyOwnedCells() const;
      const_FECellIterator
      endLocallyOwnedCells() const;
      FECellIterator
      beginLocallyActiveCells();
      FECellIterator
      endLocallyActiveCells();
      const_FECellIterator
      beginLocallyActiveCells() const;
      const_FECellIterator
      endLocallyActiveCells() const;
      unsigned int
      getDim() const;

      //
      // dealii specific functions
      //
      std::shared_ptr<const dealii::DoFHandler<dim>>
      getDoFHandler();

      const dealii::FiniteElement<dim> &
      getReferenceFE(const size_type cellId) const;

    private:
      std::shared_ptr<const TriangulationBase> d_triangulation;
      std::shared_ptr<dealii::DoFHandler<dim>> d_dofHandler;
      bool                                     d_isHPRefined;
      std::vector<std::shared_ptr<FECellBase>> d_locallyActiveCells;
      std::vector<std::shared_ptr<FECellBase>> d_locallyOwnedCells;

    }; // end of FEBasisManagerDealii
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeFEBasisManagerDealii_h
