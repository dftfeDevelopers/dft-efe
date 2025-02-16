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

#ifndef dftefePoissonSolverDealiiMatrixFreeFE_h
#define dftefePoissonSolverDealiiMatrixFreeFE_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <basis/FEBasisManager.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEBasisDataStorage.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <basis/DealiiFEEvaluationWrapper.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <basis/CFEConstraintsLocalDealii.h>
#include <basis/CFEBDSOnTheFlyComputeDealii.h>
#include <vector>
#include <memory>
#include <utils/Profiler.h>

namespace dftefe
{
  namespace electrostatics
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
    class PoissonSolverDealiiMatrixFreeFE
    {
    public:
      /**
       * @brief define ValueType as the superior (bigger set) of the
       * ValueTypeOperator and ValueTypeOperand
       * (e.g., between double and complex<double>, complex<double>
       * is the bigger set)
       */

      template <typename T>
      using distributedCPUVec =
        basis::FEEvaluationWrapperBase::distributedCPUVec<T>;

      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

    public:
      /**
       * @brief This constructor creates an instance of a base LinearSolverFunction called PoissonSolverDealiiMatrixFreeFE
       */
      PoissonSolverDealiiMatrixFreeFE(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          feBasisDataStorageStiffnessMatrix,
        const std::map<
          std::string,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>>
          &feBasisDataStorageRhs,
        const std::map<
          std::string,
          const quadrature::QuadratureValuesContainer<ValueType, memorySpace> &>
          &                                     inpRhs,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);

      /**
       * @brief This constructor creates an instance of a base LinearSolverFunction called PoissonSolverDealiiMatrixFreeFE
       */
      PoissonSolverDealiiMatrixFreeFE(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          feBasisDataStorageStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          feBasisDataStorageRhs,
        const quadrature::QuadratureValuesContainer<ValueType, memorySpace>
          &                                     inpRhs,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);

      void
      reinit(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        const std::map<
          std::string,
          const quadrature::QuadratureValuesContainer<ValueType, memorySpace> &>
          &inpRhs);

      void
      reinit(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        const quadrature::QuadratureValuesContainer<ValueType, memorySpace>
          &inpRhs);

      ~PoissonSolverDealiiMatrixFreeFE() = default;

      void
      solve(const double absTolerance, const unsigned int maxNumberIterations);

      void
      getSolution(linearAlgebra::MultiVector<ValueType, memorySpace> &solution);

      const utils::mpi::MPIComm &
      getMPIComm() const;

    private:
      const distributedCPUVec<ValueTypeOperand> &
      getRhs() const;

      const distributedCPUVec<ValueType> &
      getInitialGuess() const;

      void
      setSolution(const distributedCPUVec<ValueType> &x);

      void
      computeRhs(distributedCPUVec<double> &rhs,
                 const std::map<
                   std::string,
                   const quadrature::QuadratureValuesContainer<
                     linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                            ValueTypeOperand>,
                     memorySpace> &> &inpRhs);

      void
      vmult(distributedCPUVec<double> &Ax, distributedCPUVec<double> &x);

      void
      precondition_Jacobi(distributedCPUVec<double> &      dst,
                          const distributedCPUVec<double> &src) const;

      void
      computeDiagonalA();

      void
      AX(const dealii::MatrixFree<dim, double> &      matrixFreeData,
         distributedCPUVec<double> &                  dst,
         const distributedCPUVec<double> &            src,
         const std::pair<unsigned int, unsigned int> &cell_range) const;


      size_type d_numComponents;
      std::shared_ptr<
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>>
                                        d_feBasisManagerField;
      linearAlgebra::PreconditionerType d_pcType;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
                                d_feBasisDataStorageStiffnessMatrix;
      utils::Profiler           d_p;
      utils::ConditionalOStream d_rootCout;


      distributedCPUVec<ValueTypeOperator> d_x, d_rhs, d_initial, d_diagonalA;
      std::shared_ptr<dealii::MatrixFree<dim, ValueTypeOperator>>
                                                     d_dealiiMatrixFree;
      std::shared_ptr<const dealii::DoFHandler<dim>> d_dealiiDofHandler;
      dealii::AffineConstraints<ValueTypeOperand>
        *d_dealiiAffineConstraintMatrix;
      dealii::AffineConstraints<ValueTypeOperand> *d_constraintsInfo;
      unsigned int                        d_num1DQuadPointsStiffnessMatrix;
      std::map<std::string, unsigned int> d_num1DQuadPointsRhs;
      size_type                           d_feOrder;
      unsigned int                        d_dofHandlerIndex;

      const std::map<
        std::string,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>>
        d_feBasisDataStorageRhs;

      unsigned int d_matrixFreeQuadCompStiffnessMatrix;

    }; // end of class PoissonSolverDealiiMatrixFreeFE
  }    // namespace electrostatics
} // end of namespace dftefe
#include <electrostatics/PoissonSolverDealiiMatrixFreeFE.t.cpp>
#endif // dftefePoissonSolverDealiiMatrixFreeFE_h
