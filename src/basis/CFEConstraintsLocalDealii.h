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
 * @author Vishal Subramanian, Avirup Sircar,
 */

#ifndef dftefeCFEConstraintsLocalDealii_h
#define dftefeCFEConstraintsLocalDealii_h

#include <basis/ConstraintsLocal.h>
#include <utils/TypeConfig.h>
#include <deal.II/lac/affine_constraints.h>
#include <utils/MemoryStorage.h>
#include <utils/ScalarSpatialFunction.h>
#include <unordered_map>
#include <unordered_set>

#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <utils/MPITypes.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class CFEConstraintsLocalDealii
      : public ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>
    {
    public:
      // constraint coefficients, inhomogeneities and the dealii
      // constraint matrix are geometric data, hence real
      using RealTypeBasisCoeff =
        linearAlgebra::blasLapack::real_type<ValueTypeBasisCoeff>;
      using GlobalSizeTypeVector =
        utils::MemoryStorage<global_size_type, memorySpace>;
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;

      CFEConstraintsLocalDealii(const dealii::IndexSet &locally_owned_dofs,
                                const dealii::IndexSet &locally_relevant_dofs);

      CFEConstraintsLocalDealii(
        dealii::AffineConstraints<RealTypeBasisCoeff>
          &dealiiAffineConstraintMatrix,
        std::vector<std::pair<global_size_type, global_size_type>>
          &                            locallyOwnedRanges,
        std::vector<global_size_type> &ghostIndices,
        std::unordered_map<global_size_type, size_type>
          &globalToLocalMapLocalDofs);

      ~CFEConstraintsLocalDealii() = default;

      //
      // Copy function - note one has to call close after calling copyFrom
      //
      void
      copyFrom(const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>
                 &constraintsLocalIn) override;

      void
      clear() override;
      bool
      isConstrained(global_size_type basisId) const override;
      void
      close() override;
      bool
      isClosed() const override;

      void
      setInhomogeneity(global_size_type    basisId,
                       ValueTypeBasisCoeff constraintValue) override;

      const std::vector<std::pair<global_size_type, RealTypeBasisCoeff>> *
      getConstraintEntries(const global_size_type lineDof) const override;

      bool
      isInhomogeneouslyConstrained(const global_size_type index) const override;

      ValueTypeBasisCoeff
      getInhomogeneity(const global_size_type lineDof) const override;

      void
      distributeChildToParent(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const override;
      void
      distributeParentToChild(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const override;

      void
      setConstrainedNodesToZero(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const override;

      void
      setConstrainedNodes(linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                     memorySpace> &vectorData,
                          size_type                                blockSize,
                          ValueTypeBasisCoeff alpha) const override;


      /**
       * @brief Builds the mean-value constraint that pins the null space of the
       * Poisson operator under full periodic boundary conditions, by enforcing
       * \f$\int_\Omega \phi \, d\Omega = \sum_i w_i \phi_i = 0\f$ with
       * \f$w_i = \int_\Omega N_i \, d\Omega\f$. One pinned dof \f$o\f$ is
       * isolated so that \f$\phi_o = \sum_{i \neq o} a_i \phi_i\f$ with
       * \f$a_i = -w_i / w_o\f$, which is a standard slave-from-masters
       * constraint, but one whose masters are every other dof rather than the
       * handful a hanging node or a periodic slave has. Eliminating it into
       * the dealii AffineConstraints object the way those are would contribute
       * a dense outer product a a^T to the operator, coupling every dof to
       * every other one, so it is kept out of that object and folded into the
       * four distribute methods instead, as a rank-1 operation costing O(N)
       * work and only the coefficient vector in storage. Matches dftfe's
       * treatment (poissonSolverProblem.cc:500-631).
       *
       * Once built, every distribute call on this object applies it, so the
       * dftefe-native Poisson path needs no changes of its own.
       */
      void
      applyMeanValueConstraintDistributeP2C(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const override;

      void
      applyMeanValueConstraintDistributeC2P(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const override;

      void
      setMeanValueConstraint(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                        basisIntegrals,
        const utils::mpi::MPIComm &mpiComm) override;

      bool
      hasMeanValueConstraint() const override;

      const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &
      getMeanValueConstraintVec() const override;

      global_size_type
      getMeanValueConstraintNodeIdGlobal() const override;

      size_type
      getMeanValueConstraintProcId() const override;

      //
      // dealii function
      //
      const dealii::AffineConstraints<RealTypeBasisCoeff> &
      getAffineConstraints() const;

      //
      // private functions
      //
    private:
      void
      addEntries(
        const global_size_type constrainedDofIndex,
        const std::vector<std::pair<global_size_type, RealTypeBasisCoeff>>
          &colWeightPairs);

      void
      copyConstraintsDataFromDealiiToDealii(
        const CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>
          &constraintsDataIn);

      void
      copyConstraintsDataFromDealiiToDftefe();

      void
      addLine(const global_size_type lineDof);

      bool
      isGhostEntry(const global_size_type globalId) const;

      bool
      inLocallyOwnedRanges(const global_size_type globalId) const;

      size_type
      globalToLocal(const global_size_type globalId) const;


      dealii::AffineConstraints<RealTypeBasisCoeff>
           d_dealiiAffineConstraintMatrix;
      bool d_isCleared;
      bool d_isClosed;

      GlobalSizeTypeVector d_rowConstraintsIdsGlobal;
      SizeTypeVector       d_rowConstraintsIdsLocal;
      SizeTypeVector       d_columnConstraintsIdsLocal;
      SizeTypeVector       d_constraintRowSizesAccumulated;
      GlobalSizeTypeVector d_columnConstraintsIdsGlobal;

      utils::MemoryStorage<double, memorySpace> d_columnConstraintsValues;
      utils::MemoryStorage<RealTypeBasisCoeff, memorySpace>
        d_constraintsInhomogenities;

      SizeTypeVector d_rowConstraintsSizes;

      std::vector<std::pair<global_size_type, global_size_type>>
                                                      d_locallyOwnedRanges;
      std::vector<global_size_type>                   d_ghostIndices;
      std::unordered_set<global_size_type>            d_ghostIndicesSet;
      std::unordered_map<global_size_type, size_type> d_globalToLocalMap;

      // Mean-value constraint state; inactive unless computeMeanValueConstraint
      // has been called, in which case every branch guarded on
      // d_isMeanValueConstraintActive below is dead and behaviour is unchanged.
      bool d_isMeanValueConstraintActive;
      linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
                          d_meanValueConstraintVec;
      size_type           d_meanValueConstraintNodeIdLocal;
      size_type           d_meanValueConstraintProcId;
      global_size_type    d_meanValueConstraintNodeIdGlobal;
      utils::mpi::MPIComm d_meanValueMpiComm;
      int                 d_meanValueMyRank;
    };

  } // namespace basis
} // namespace dftefe
#include <basis/CFEConstraintsLocalDealii.t.cpp>
#endif // dftefeCFEConstraintsLocalDealii_h
