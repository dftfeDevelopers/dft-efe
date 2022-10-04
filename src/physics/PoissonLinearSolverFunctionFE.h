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

#ifndef dftefePoissonLinearSolverFunctionFE_h
#define dftefePoissonLinearSolverFunctionFE_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
namespace dftefe
{
  namespace physics
  {
    /**
     *@brief A derived class of linearAlgebra::LinearSolverFunction
     * to encapsulate the Poisson partial differential equation
     * (PDE) discretized in a finite element (FE) basis.
     * The Possion PDE is given as:
     * \f$\nabla^2 v(\textbf{r}) = -4 \pi \rho(\textbf{r})$\f
     * with the boundary condition on
     * \f$v(\textbf{r})|_{\partial \Omega}=g(\textbf{r})$\f
     * (\f$\\partial Omega$\f denoting the boundary of a domain \f$\Omega$\f).
     * Here \f$v$\f has the physical notion of a potential (e.g.,
     * Hartree potential, nuclear potential, etc.) arising due to a charge
     * distributin \f$\rho$\f.
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
    class PoissonLinearSolverFunctionFE
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
      PoissonLinearSolverFunctionFE(
        const basis::FEBasisHandler<ValueTypeOperator,
        memorySpace,
        dim > &feBasisHandler,
        const utils::FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &                                                  feBasisDataStorage,
        const linearAlgebra::Vector<ValueType, memorySpace> &b,
        const std::string                                    constraintsName,
	const linearAlgebra::PreconditionerType pcType);

      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
      getAxContext() const override;

      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
      getPCContext() const override;

      void
      setSolution(const Vector<ValueTypeOperand, memorySpace> &x) override;

      virtual Vector<ValueType, memorySpace>
      getRhs() const override;

      virtual Vector<ValueTypeOperand, memorySpace>
      getInitialGuess() const override;

      const utils::mpi::MPIComm &
      getMPIComm() const override;

     private:
      const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim> * d_feBasisHandler;
      const utils::FEBasisDataStorage<ValueTypeOperator, memorySpace> * d_feBasisDataStorage;
      linearAlgebra::Vector<ValueType, memorySpace> d_b;
      linearAlgebra::Vector<ValueTypeOperand, memorySpace> d_x;
      std::string d_constraintsName;
      linearAlgebra::PreconditionerType d_pcType;
    }; // end of class PoissonLinearSolverFunctionFE
  } // namespace physics
} // end of namespace dftefe
#include <physics/PoissonLinearSolverFunctionFE.t.cpp>
#endif // dftefePoissonLinearSolverFunctionFE_h
