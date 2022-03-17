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
      using cellIterator       = FEBasisManager<dim>::cellIterator;
      using const_cellIterator = FEBasisManager<dim>::const_cellIterator;

      FEBasisManagerDealii();
      ~FEBasisManagerDealii();
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
      reinit(const TriangulationBase &triangulation,
             const size_type          feOrder) override;
      size_type
      nLocalCells() const override;
      size_type
      nGlobalCells() const override;
      size_type
      getFEOrder(size_type cellId) const override;
      size_type
      nCellDofs(size_type cellId) const override;
      size_type
      nLocalNodes() const override;
      global_size_type
      nGlobalNodes() const override;
      std::vector<size_type>
      getLocalNodeIds(size_type cellId) const override;
      std::vector<size_type>
      getGlobalNodeIds() const override;
      virtual std::vector<size_type>
      getCellDofsLocalIds(size_type cellId) const override;
      std::vector<size_type>
      getBoundaryIds() const override;
      cellIterator
      beginLocal() override;
      cellIterator
      endLocal() override;
      const_cellIterator
      beginLocal() const override;
      const_cellIterator
      endLocal() const override;
      unsigned int
      getDim() const override;

      //
      // dealii specific functions
      //
      std::shared_ptr<const dealii::DoFHandler<dim>>
      getDoFHandler();

    private:
      std::shared_ptr<const TriangulationBase> d_triangulation;
      std::shared_ptr<dealii::DoFHandler<dim>> d_dofHandler;


    }; // end of FEBasisManagerDealii
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeFEBasisManagerDealii_h
