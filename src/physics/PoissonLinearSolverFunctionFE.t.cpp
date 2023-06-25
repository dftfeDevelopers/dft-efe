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

#include <utils/Defaults.h>
namespace dftefe
{
  namespace physics
  {
    //
    // Constructor
    //
    // template <typename ValueTypeOperator,
    //           typename ValueTypeOperand,
    //           utils::MemorySpace memorySpace,
    //           size_type          dim>
    // PoissonLinearSolverFunctionFE<ValueTypeOperator,
    //                               ValueTypeOperand,
    //                               memorySpace,
    //                               dim>::
    //   PoissonLinearSolverFunctionFE(
    //     const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>
    //       &feBasisHandler,
    //     const utils::FEBasisDataStorage<ValueTypeOperator, memorySpace>
    //       &                                                  feBasisDataStorage,
    //     const linearAlgebra::Vector<ValueType, memorySpace> &b,
    //     const std::string                                    constraintsName,
    //     const linearAlgebra::PreconditionerType              pcType)
    //   : d_feBasisHandler(&feBasisHandler)
    //   , d_feBasisDataStorage(&feBasisDataStorage)
    //   , d_b(b)
    //   , d_constraintsName(constraintsName)
    //   , d_pcType(pcType)
    //   , d_x(b.getMPIPatternP2P(),
    //         b.getLinAlgOpContext(),
    //         utils::Types<ValueType>::zero)
    // {}

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::PoissonLinearSolverFunctionFE(
      std::shared_ptr<const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>>
        feBasisHandler,
      const basis::FEBasisOperations<ValueTypeOperator,ValueTypeOperand,memorySpace,dim>
        & feBasisOperations,
      std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
        feBasisDataStorage,
      const quadrature::QuadratureValuesContainer<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
        ValueTypeOperand>, memorySpace> 
        & inp,
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
      const std::string                                    constraintsName,
      const linearAlgebra::PreconditionerType              pcType,
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                    mpiPatternP2P,
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContext,
      const size_type                 maxCellTimesNumVecs)
      : d_feBasisHandler(feBasisHandler)
      , d_feBasisDataStorage(feBasisDataStorage)
      , d_b(mpiPatternP2P,linAlgOpContext,inp.getNumberComponents())
      , d_constraintsName(constraintsName)
      , d_pcType(pcType)
      , d_x(mpiPatternP2P,linAlgOpContext,inp.getNumberComponents())
    {
      feBasisOperations.integrateWithBasisValues(
        inp,
        quadratureRuleAttributes,
        *d_feBasisHandler,
        constraintsName,
        d_b);

        d_mpiPatternP2P = mpiPatternP2P;

      d_AxContext = std::make_shared<physics::LaplaceOperatorContextFE
        <ValueTypeOperator, ValueTypeOperand, memorySpace, dim>>( *d_feBasisHandler,
        *d_feBasisDataStorage,
        constraintsName,
        quadratureRuleAttributes,
        maxCellTimesNumVecs);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(d_mpiPatternP2P,
             linAlgOpContext,
             (ValueTypeOperator)1.0);   // TODO

      if (d_pcType == linearAlgebra::PreconditionerType::JACOBI)
        d_PCContext = std::make_shared<linearAlgebra::PreconditionerJacobi
          <ValueTypeOperator, ValueTypeOperand, memorySpace>>(diagonal);
      else if (d_pcType == linearAlgebra::PreconditionerType::NONE)
        d_PCContext = std::make_shared<linearAlgebra::PreconditionerJacobi
          <ValueTypeOperator, ValueTypeOperand, memorySpace>>(diagonal);
      else
      utils::throwException(
        false,
        "Unknown PreConditionerType");
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getAxContext() const
    {
      return *d_AxContext;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getPCContext() const
    {
      return *d_PCContext;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::setSolution(const linearAlgebra::MultiVector<linearAlgebra::blasLapack::scalar_type
                                    <ValueTypeOperator,ValueTypeOperand>, memorySpace>
                                   &x)
    {
      d_x = x;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    linearAlgebra::MultiVector<linearAlgebra::blasLapack::scalar_type
                                    <ValueTypeOperator,ValueTypeOperand>, memorySpace> &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getSolution() 
    {
      return d_x;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getRhs() const
    {
      return d_b;
    }
                            
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::MultiVector<linearAlgebra::blasLapack::scalar_type
                                    <ValueTypeOperator,ValueTypeOperand>, memorySpace> &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getInitialGuess() const
    {
      return d_x;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const utils::mpi::MPIComm &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getMPIComm() const
    {
      return d_mpiPatternP2P->mpiCommunicator();
    }

  } // end of namespace physics
} // end of namespace dftefe