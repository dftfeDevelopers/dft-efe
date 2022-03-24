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

#ifndef dftefeFECellDealii_h
#define dftefeFECellDealii_h

#include <utils/TypeConfig.h>
#include <deal.II/dofs/dof_accessor.h>
namespace dftefe 
{
  namespace basis 
  {

    template <size_type dim>
      class FECellDealii: public FECellBase 
      {
      using DealiiFECellIterator =
        typename dealii::DoFHandler<dim>::active_cell_iterator;

    public:
      FECellDealii(DealiiFECellIterator dealiiFECellIter);
      ~FECellDealii();
      
      std::vector<std::shared_ptr<Point>>
      getVertices() const override ;

      std::shared_ptr<Point>
      getVertex(size_type i) const override ;

      std::vector<std::shared_ptr<Point>>
      getNodalPoints() const override ;

      size_type
      getId() const override ;

      bool
      isPointInside(std::shared_ptr<const Point> point) const override ;

      bool
      isAtBoundary(const unsigned int i) const override ;

      bool
      isAtBoundary() const override ;

      void
      setRefineFlag() override ;

      void
      clearRefineFlag() override ;

      void
      setCoarsenFlag() override ;

      void
      clearCoarsenFlag() override ;

      bool
      isActive() const override ;

      bool
      isLocallyOwned() const override ;

       bool
      isGhost() const override ;

       bool
      isArtificial() const override ;

       int
      getDim() const override ;

       std::shared_ptr<Point>
      getParametricPoint(std::shared_ptr<const Point> realPoint,
                         const CellMappingBase &cellMapping) const override ;

       std::shared_ptr<Point>
      getRealPoint(std::shared_ptr<const Point> parametricPoint,
                   const CellMappingBase &      cellMapping) const override ;
       
      global_size_type
      getLocalToGlobalDoFId(size_type i) const override ;
       size_type
      getFEOrder() const override;

      DealiiFECellIterator &
	getDealiiFECellIter();

    private:
	DealiiFECellIterator d_dealiiFECellIter;


      };// end of class FECellDealii
  }

}

#endif // dftefeFECellDealii_h
