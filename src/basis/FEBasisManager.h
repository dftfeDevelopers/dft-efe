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

#ifndef dftefeFEBasisManager_h
#define dftefeFEBasisManager_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <basis/BasisManager.h>
#include <basis/TriangulationBase.h>
#include <basis/FECellBase.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * An abstract class to handle finite element (FE) basis related operations,
     * such as evaluating the value and gradients of any basis function at a
     * point, getting cell and nodal information, etc.
     *
     */
    class FEBasisManager : public BasisManager
    {
      //
      // Typedefs
      //
    public:
      typedef std::vector<std::shared_ptr<FECellBase>>::iterator FECellIterator;
      typedef std::vector<std::shared_ptr<FECellBase>>::const_iterator
        const_FECellIterator;

      virtual double
      getBasisFunctionValue(const size_type     basisId,
                            const utils::Point &point) const = 0;
      virtual std::vector<double>
      getBasisFunctionDerivative(const size_type     basisId,
                                 const utils::Point &point,
                                 const size_type derivativeOrder = 1) const = 0;

      ////// FE specific virtual member functions /////
      virtual void
      reinit(std::shared_ptr<const TriangulationBase> triangulation,
             const size_type                          feOrder) = 0;
      virtual size_type
      nLocalCells() const = 0;
      virtual size_type
      nLocallyOwnedCells() const = 0;
      virtual size_type
      nGlobalCells() const = 0;
      virtual size_type
      getFEOrder(size_type cellId) const = 0;
      virtual size_type
      nCellDofs(size_type cellId) const = 0;
      virtual bool
      isHPRefined() const = 0;
      
      std::pair<global_size_type, global_size_type>
	getLocallyOwnedRange() const = 0;
      
      virtual size_type
      nLocalNodes() const = 0;
      virtual global_size_type
      nGlobalNodes() const = 0;
      virtual std::vector<size_type>
      getLocalNodeIds(size_type cellId) const = 0;
      virtual std::vector<size_type>
      getGlobalNodeIds() const = 0;
      virtual void
      getCellDofsGlobalIds(
        size_type                      cellId,
        std::vector<global_size_type> &vecGlobalNodeId) const = 0;
      virtual std::vector<size_type>
      getBoundaryIds() const = 0;
      virtual FECellIterator
      beginLocallyOwnedCells() = 0;
      virtual FECellIterator
      endLocallyOwnedCells() = 0;
      virtual const_FECellIterator
      beginLocallyOwnedCells() const = 0;
      virtual const_FECellIterator
      endLocallyOwnedCells() const = 0;
      virtual FECellIterator
      beginLocalCells() = 0;
      virtual FECellIterator
      endLocalCells() = 0;
      virtual const_FECellIterator
      beginLocalCells() const = 0;
      virtual const_FECellIterator
      endLocalCells() const = 0;
      virtual unsigned int
      getDim() const = 0;
    }; // end of FEBasisManager
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeFEBasisManager_h
