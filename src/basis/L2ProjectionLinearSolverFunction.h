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

#ifndef dftefeL2ProjectionLinearSolverFunction_h
#define dftefeL2ProjectionLinearSolverFunction_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/LinearSolverFunction.h>
#include <linearAlgebra/OperatorContext.h>
#include <basis/BasisOverlapOperatorContext.h>
#include <linearAlgebra/PreconditionerJacobi.h>
#include <linearAlgebra/PreconditionerNone.h>
#include <basis/FEBasisHandler.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEBasisDataStorage.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <vector>
#include <memory>

namespace dftefe
{
  namespace basis
  {
    /**
     *@brief A derived class of linearAlgebra::LinearSolverFunction
     * to encapsulate the L2 Projecton partial differential equation
     * (PDE) discretized in a finite element (FE) basis.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNES, etc.) in which the data of the operator
     * and its operands reside
     * @tparam dim Dimension of the Poisson problem
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class L2ProjectionLinearSolverFunction
      : public linearAlgebra::
          LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
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
       * @brief This constructor creates an instance of a base LinearSolverFunction called PoissonLinearSolverFE
       */
      L2ProjectionLinearSolverFunction(
        std::shared_ptr<
          const FEBasisHandler<ValueTypeOperator, memorySpace, dim>>
                                             cfeBasisHandler,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          cfeBasisDataStorage,
        FEBasisOperations<ValueTypeBasisOperand, ValurTypeBasisOperator,memorySpace,
         dim> cfeBasisOperations,
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpace> & inp,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const std::string basisInterfaceCoeffConstraint,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellTimesNumVecs);

      const linearAlgebra::
        OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
        getAxContext() const override;

      const linearAlgebra::
        OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
        getPCContext() const override;

      void
      setSolution(
        const linearAlgebra::MultiVector<ValueType, memorySpace> &x) override;

      void
      getSolution(
        linearAlgebra::MultiVector<ValueType, memorySpace> &solution) override;

      const linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &
      getRhs() const override;

      const linearAlgebra::MultiVector<ValueType, memorySpace> &
      getInitialGuess() const override;

      const utils::mpi::MPIComm &
      getMPIComm() const override;

    private:
      std::shared_ptr<
        const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>>
        d_feBasisHandler;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
        d_feBasisDataStorage;
      std::shared_ptr<const linearAlgebra::OperatorContext<ValueTypeOperator,
                                                           ValueTypeOperand,
                                                           memorySpace>>
        d_AxContext;
      std::shared_ptr<const linearAlgebra::OperatorContext<ValueTypeOperator,
                                                           ValueTypeOperand,
                                                           memorySpace>>
        d_PCContext;
      linearAlgebra::MultiVector<ValueType, memorySpace> d_x;
      linearAlgebra::MultiVector<ValueType, memorySpace> d_b;
      linearAlgebra::PreconditionerType d_pcType;
      const linearAlgebra::MultiVector<ValueType, memorySpace> d_initial;
      std::string  d_basisInterfaceCoeffConstraint;
    }; // end of class L2ProjectionLinearSolverFunction
  }    // namespace basis
} // end of namespace dftefe
#include <basis/L2ProjectionLinearSolverFunction.t.cpp>
#endif // dftefeL2ProjectionLinearSolverFunction_h
