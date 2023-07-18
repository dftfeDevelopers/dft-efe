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
 * @author Bikash Kanungo
 */

#ifndef dftefeLaplaceOperatorContextFE_h
#define dftefeLaplaceOperatorContextFE_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <basis/FEBasisHandlerDealii.h>
#include <basis/EFEBasisHandlerDealii.h>
#include <basis/FEBasisDataStorageDealii.h>
#include <quadrature/QuadratureAttributes.h>
#include <memory>

namespace dftefe
{
  namespace physics
  {
    /**
     *@brief A derived class of linearAlgebra::OperatorContext to encapsulate
     * the action of a discrete operator on vectors, matrices, etc.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNED, etc.) in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class LaplaceOperatorContextFE
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

    public:
      /**
       * @brief Constructor
       */
      LaplaceOperatorContextFE(
        const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>
          & feBasisHandler,
        const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
          & feBasisDataStorage,
        const std::string               constraintsX,
        const std::string               constraintsY,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 maxCellTimesNumVecs);

      // void
      // apply(const linearAlgebra::Vector<ValueTypeOperand, memorySpace> &x,
      //       linearAlgberba::Vector<ValueType, memorySpace> &y) const override;

      void
      apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
            linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const override;

    private:
      const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>
                                      *d_feBasisHandler;
      const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
                                      *d_feBasisDataStorage;
      const std::string               d_constraintsX;
      const std::string               d_constraintsY;
      const quadrature::QuadratureRuleAttributes &d_quadratureRuleAttributes;
      const size_type                 d_maxCellTimesNumVecs;
    }; // end of class LaplaceOperatorContextFE
  }    // end of namespace physics
} // end of namespace dftefe
#include <physics/LaplaceOperatorContextFE.t.cpp>
#endif // dftefeLaplaceOperatorContextFE_h