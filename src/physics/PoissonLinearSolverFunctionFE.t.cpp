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
      const std::string                                    constraintsNameRhs,
      const std::string                                    constraintsNameLhs,
      const linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> & rhsAddOnVector,
      const linearAlgebra::PreconditionerType              pcType,
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContext,
      const size_type                 maxCellTimesNumVecs)
      : d_feBasisHandler(feBasisHandler)
      , d_feBasisDataStorage(feBasisDataStorage)
      , d_b(feBasisHandler->getMPIPatternP2P(constraintsNameRhs),linAlgOpContext,inp.getNumberComponents())
      , d_pcType(pcType)
      , d_x(feBasisHandler->getMPIPatternP2P(constraintsNameLhs),linAlgOpContext,inp.getNumberComponents())
      , d_initial(feBasisHandler->getMPIPatternP2P(constraintsNameLhs),linAlgOpContext,inp.getNumberComponents())
    {
      d_mpiPatternP2PLhs = feBasisHandler->getMPIPatternP2P(constraintsNameLhs);

      d_AxContext = std::make_shared<physics::LaplaceOperatorContextFE
        <ValueTypeOperator, ValueTypeOperand, memorySpace, dim>>( *d_feBasisHandler,
        *d_feBasisDataStorage,
        constraintsNameLhs,
        quadratureRuleAttributes,
        maxCellTimesNumVecs);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(d_mpiPatternP2PLhs,
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

      // Compute RHS

      std::vector<ValueTypeOperand> ones(0);
      ones.resize(inp.getNumberComponents(), (ValueTypeOperand)1.0);

      d_b.setValue(0.0);
      linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> b(d_b, 0.0);

      feBasisOperations.integrateWithBasisValues(
        inp,
        quadratureRuleAttributes,
        *d_feBasisHandler,
        constraintsNameRhs,
        b);

      linearAlgebra::add(ones, b, ones, rhsAddOnVector, d_b);

//  for (unsigned int i = 0 ; i < d_b.locallyOwnedSize() ; i++)
//   {
//     std::cout << "d_b[" <<i<<"] : "<< *(d_b.data()+i) << "," << "b[" <<i<<"] : "<< *(b.data()+i)<<"\n";
//   }

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
		   return d_initial;
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
      return d_mpiPatternP2PLhs->mpiCommunicator();
    }

  } // end of namespace physics
} // end of namespace dftefe