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

#ifndef dftefeCFEOverlapOperatorContext_h
#define dftefeCFEOverlapOperatorContext_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <basis/FEBasisManager.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <quadrature/QuadratureAttributes.h>
#include <memory>

namespace dftefe
{
  namespace basis
  {
    /**
     *@brief A derived class of linearAlgebra::OperatorContext to encapsulate
     * the action of a discrete operator on vectors, matrices, etc.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator.
     * Ususlly ValueTypeBasisData in FE basis
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act. Ususlly ValueTypeBasisCoeff in FE basis
     * @tparam memorySpace The memory space (HOST, DEVICE, HOST_PINNED, etc.) in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class CFEOverlapOperatorContext
      : public linearAlgebra::
          OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
    {
    public:
      /**
       * @brief define ValueType as the superior (bigger set) of the
       * ValueTypeOperator and ValueTypeOperand
       * (e.g., between double and complex<double>, complex<double>
       * is the bigger set)
       */
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

      using Storage =
        dftefe::utils::MemoryStorage<ValueTypeOperator, memorySpace>;

    public:
      /**
       * @brief Constructor
       */
      CFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             feBasisDataStorage,
        const size_type maxCellBlock,
        const size_type maxFieldBlock,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);

      CFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &feBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);

      ~CFEOverlapOperatorContext() = default;

      // void
      // apply(const linearAlgebra::Vector<ValueTypeOperand, memorySpace> &x,
      //       linearAlgberba::Vector<ValueType, memorySpace> &y) const
      //       override;

      void
      apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
            linearAlgebra::MultiVector<ValueType, memorySpace> &       Y,
            bool updateGhostX = false,
            bool updateGhostY = false) const override;

      // get overlap of two basis functions in a cell
      Storage
      getBasisOverlap(const size_type cellId,
                      const size_type basisId1,
                      const size_type basisId2) const;

      // get overlap of all the basis functions in a cell
      Storage
      getBasisOverlapInCell(const size_type cellId) const;

      // get overlap of all the basis functions in all cells
      const Storage &
      getBasisOverlapInAllCells() const;

    private:
      const FEBasisManager<ValueTypeOperand,
                           ValueTypeOperator,
                           memorySpace,
                           dim> *d_feBasisManager;
      std::shared_ptr<Storage>   d_basisOverlap;
      std::vector<size_type>     d_cellStartIdsBasisOverlap;
      std::vector<size_type>     d_dofsInCell;
      const size_type            d_maxCellBlock;
      const size_type            d_maxFieldBlock;
      bool                       d_isMassLumping;
      std::shared_ptr<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>
        d_diagonal;

    }; // end of class CFEOverlapOperatorContext
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/CFEOverlapOperatorContext.t.cpp>
#endif // dftefeCFEOverlapOperatorContext_h
