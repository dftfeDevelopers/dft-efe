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
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/Vector.h>
#include <vector>
#include <memory>
#include <utils/Profiler.h>
#ifdef DFTEFE_WITH_DEVICE
#  include "MatrixFreeWrapper.h"
#  include "PoissonSolverDealiiMatrixFreeFEDeviceKernels.h"
#endif

namespace dftefe
{
  namespace electrostatics
  {
    static constexpr utils::MemorySpace memorySpaceHost =
      utils::MemorySpace::HOST;
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
                                                    memorySpaceHost,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>
          feBasisDataStorageStiffnessMatrix,
        const std::map<
          std::string,
          std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeOperator,
                                                          memorySpaceHost>>>
          &feBasisDataStorageRhs,
        const std::map<
          std::string,
          const quadrature::QuadratureValuesContainer<ValueType,
                                                      memorySpaceHost> &>
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
                                                    memorySpaceHost,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>
          feBasisDataStorageStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>
          feBasisDataStorageRhs,
        const quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>
          &                                     inpRhs,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);

      void
      reinit(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpaceHost,
                                                    dim>> feBasisManagerField,
        const std::map<
          std::string,
          const quadrature::QuadratureValuesContainer<ValueType,
                                                      memorySpaceHost> &>
          &inpRhs);

      void
      reinit(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpaceHost,
                                                    dim>> feBasisManagerField,
        const quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>
          &inpRhs);

      ~PoissonSolverDealiiMatrixFreeFE() = default;

      void
      solve(const double absTolerance, const size_type maxNumberIterations);

      void
      getSolution(
        linearAlgebra::MultiVector<ValueType, memorySpaceHost> &solution);

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
                     memorySpaceHost> &> &inpRhs);

      void
      vmult(distributedCPUVec<double> &Ax, distributedCPUVec<double> &x);

      void
      precondition_Jacobi(distributedCPUVec<double> &      dst,
                          const distributedCPUVec<double> &src) const;

      void
      computeDiagonalA();

      void
      AX(const dealii::MatrixFree<dim, double> &matrixFreeData,
         distributedCPUVec<double> &            dst,
         const distributedCPUVec<double> &      src,
         const std::pair<size_type, size_type> &cell_range) const;

      void
      CGsolve(const double    absTolerance,
              const size_type maxNumberIterations,
              bool            distributeFlag);

      const linearAlgebra::Vector<ValueTypeOperator, utils::MemorySpace::DEVICE>
        &
        getRhsDevice() const;

      const linearAlgebra::Vector<ValueTypeOperator, utils::MemorySpace::DEVICE>
        &
        getInitialGuessDevice() const;

      // Device-specific methods: only called when memorySpace == DEVICE.
      // Requires DFTEFE_WITH_DEVICE for the GPU matrix-free AX kernel.
      void
      CGsolveDevice(const double    absTolerance,
                    const size_type maxNumberIterations,
                    bool            distributeFlag);

      void
      computeAXDevice(linearAlgebra::Vector<ValueTypeOperator,
                                            utils::MemorySpace::DEVICE> &Ax,
                      linearAlgebra::Vector<ValueTypeOperator,
                                            utils::MemorySpace::DEVICE> &x);

      size_type d_numComponents;
      std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                  ValueTypeOperator,
                                                  memorySpaceHost,
                                                  dim>>
                                        d_feBasisManagerField;
      linearAlgebra::PreconditionerType d_pcType;
      utils::Profiler<memorySpace>      d_p;


      std::shared_ptr<basis::FEBasisManager<ValueTypeOperand,
                                            ValueTypeOperator,
                                            memorySpaceHost,
                                            dim>>
        d_feBasisManagerHomo;

      distributedCPUVec<ValueTypeOperator> d_x, d_rhs, d_initial, d_diagonalA;
      std::shared_ptr<dealii::MatrixFree<dim, ValueTypeOperator>>
                                                     d_dealiiMatrixFree;
      std::shared_ptr<const dealii::DoFHandler<dim>> d_dealiiDofHandler;
      const dealii::AffineConstraints<ValueTypeOperand>
        *d_dealiiAffineConstraintMatrix;
      const dealii::AffineConstraints<ValueTypeOperand> *d_constraintsInfo;
      size_type                        d_num1DQuadPointsStiffnessMatrix;
      std::map<std::string, size_type> d_num1DQuadPointsRhs;
      size_type                        d_feOrder;
      size_type                        d_dofHandlerIndex;

      std::map<
        std::string,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>>
        d_feBasisDataStorageRhs;

      std::vector<distributedCPUVec<ValueType>> d_nonTensorSructuredQuadeRhs;

      size_type                           d_matrixFreeQuadCompStiffnessMatrix;
      std::map<dealii::CellId, size_type> d_cellIdToCellIndexMap;

      std::vector<dealii::Quadrature<dim>> d_dealiiQuadratureRuleVec;
      dealii::MappingQ1<dim, dim>          d_mappingDealii;
      // dealii::IndexSet d_ghostIndexSet, d_locallyOwnedIndexSet;
      utils::ConditionalOStream pcout;

      linearAlgebra::MultiVector<
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>,
        memorySpaceHost>
        d_scratchMultiVecHost;

      // Stored LinAlgOpContext needed for device vector creation and BLAS.
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;

      // Device-resident copies of the solution, RHS, and Jacobi diagonal.
      // Constructed lazily in the constructor body when memorySpace == DEVICE.
      // Use unique_ptr because MPIPatternP2P (needed for construction) is not
      // available at member-initializer-list time.
      std::unique_ptr<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>
        d_diagonalADevice;
      std::unique_ptr<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>
        d_rhsDevice;
      std::unique_ptr<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>
        d_initialDevice;
      /// define some temporary vectors for cgsolver device
      linearAlgebra::Vector<ValueTypeOperator, memorySpace> d_qvec, d_rvec,
        d_dvec;

#ifdef DFTEFE_WITH_DEVICE
      size_type d_xLocalDof;
      double *  d_devSumPtr;
      dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
        d_devSum;

      std::shared_ptr<
        const utils::mpi::MPIPatternP2P<dftefe::utils::MemorySpace::DEVICE>>
        d_mpiPatternP2PDevice;
      // Device-side matrix-free Laplace operator (from dftfe).
      std::unique_ptr<
        dftefe::MatrixFreeWrapperClass<ValueTypeOperator,
                                       dftefe::operatorList::Laplace,
                                       dftefe::utils::MemorySpace::DEVICE,
                                       false>>
        d_matrixFreeWrapperDevice;

      /**
       * @brief Combines precondition and dot product
       *
       */
      double
      applyPreconditionAndComputeDotProduct(const double *jacobi);

      /**
       * @brief Combines precondition, sadd and dot product
       *
       */
      double
      applyPreconditionComputeDotProductAndSadd(const double *jacobi);

      /**
       * @brief Combines scaling and norm
       *
       */
      double
      scaleXRandComputeNorm(double *x, const double &alpha);

      void
      dotDevice(const size_type                              size,
                double *                                     x,
                double *                                     y,
                double &                                     alpha,
                linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext);
#endif // DFTEFE_WITH_DEVICE

    }; // end of class PoissonSolverDealiiMatrixFreeFE
  }    // namespace electrostatics
} // end of namespace dftefe
#include <electrostatics/PoissonSolverDealiiMatrixFreeFE.t.cpp>
#endif // dftefePoissonSolverDealiiMatrixFreeFE_h
