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
 * @author Vishal Subramanian
 */

#ifndef dftefeConstraintsLocal_h
#define dftefeConstraintsLocal_h

#include <utils/TypeConfig.h>
#include <utils/MPIPatternP2P.h>

#include <linearAlgebra/MultiVector.h>
#include <utils/ScalarSpatialFunction.h>
#include <utils/MPITypes.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * An abstract class to handle the constraints related to a basis
     */
    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    class ConstraintsLocal
    {
    public:
      // constraint coefficients, inhomogeneities and the dealii
      // constraint matrix are geometric data, hence real
      using RealTypeBasisCoeff =
        linearAlgebra::blasLapack::real_type<ValueTypeBasisCoeff>;

      virtual ~ConstraintsLocal() = default;

      //
      // Copy function - note one has to call close after calling copyFrom
      //
      virtual void
      copyFrom(const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>
                 &constraintsLocalIn) = 0;

      virtual void
      clear() = 0;
      virtual void
      setInhomogeneity(global_size_type    basisId,
                       ValueTypeBasisCoeff constraintValue) = 0;
      virtual void
      close() = 0;
      virtual bool
      isClosed() const = 0;
      virtual bool
      isConstrained(global_size_type basisId) const = 0;

      virtual const std::vector<
        std::pair<global_size_type, RealTypeBasisCoeff>> *
      getConstraintEntries(const global_size_type lineDof) const = 0;

      virtual bool
      isInhomogeneouslyConstrained(const global_size_type index) const = 0;

      virtual ValueTypeBasisCoeff
      getInhomogeneity(const global_size_type lineDof) const = 0;

      virtual void
      distributeChildToParent(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const = 0;
      virtual void
      distributeParentToChild(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const = 0;

      virtual void
      setConstrainedNodesToZero(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const = 0;

      virtual void
      setConstrainedNodes(linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                     memorySpace> &vectorData,
                          size_type                                blockSize,
                          ValueTypeBasisCoeff alpha) const = 0;

      /**
       * @brief Installs the mean-value constraint that pins the null space of
       * the Poisson operator under full periodic boundary conditions, from the
       * pre-assembled basis integrals
       * \f$w_i = \int_\Omega N_i \, d\Omega\f$. The caller assembles
       * \p basisIntegrals because that needs the basis-data type; everything
       * from here on needs the constraint matrix and so lives on this side.
       *
       * Unlike a hanging node or a periodic slave, which have a handful of
       * masters each, this constraint has every other dof as a master. Storing
       * it as a sparse row and eliminating it the way the others are would
       * therefore couple every dof to every other one, since the elimination
       * contributes a dense outer product a a^T to the operator. It is
       * instead kept out of the sparse storage and applied by the four
       * distribute methods above, as a rank-1 operation costing O(N) work and
       * only the coefficient vector in storage. That also means the operator
       * never has to be assembled for it, which is what lets the matrix-free
       * Poisson path pick it up without any changes of its own.
       *
       * Mirrors dftfe's poissonSolverProblem.cc:500-631.
       */
      /**
       * @brief Sets the pinned dof from its masters, vec[o] = dot(a, vec).
       * No-op unless a mean-value constraint has been installed.
       *
       * Deliberately not folded into distributeParentToChild: the caller has
       * to be able to place it relative to the halo exchange, which the
       * transpose below depends on.
       */
      virtual void
      applyMeanValueConstraintDistributeP2C(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const = 0;

      /**
       * @brief Transpose of the above: redistributes what has accumulated on
       * the pinned dof onto its masters as vec += vec[o] * a, then zeroes the
       * pinned entry. No-op unless a mean-value constraint has been installed.
       *
       * Must be called only after the ghost contributions have been summed
       * into their owners, i.e. after accumulateAddLocallyOwned. Unlike a
       * hanging or periodic slave, whose masters are all locally relevant and
       * which can therefore be condensed before the exchange, the pinned dof's
       * masters span the whole domain: applying this beforehand would both
       * broadcast an incomplete vec[o] and double count every master once the
       * exchange adds the ghost copies back in. That is why it is not folded
       * into distributeChildToParent.
       */
      virtual void
      applyMeanValueConstraintDistributeC2P(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const = 0;

      virtual void
      setMeanValueConstraint(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                        basisIntegrals,
        const utils::mpi::MPIComm &mpiComm) = 0;

      virtual bool
      hasMeanValueConstraint() const = 0;

      virtual const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
        &
        getMeanValueConstraintVec() const = 0;

      virtual global_size_type
      getMeanValueConstraintNodeIdGlobal() const = 0;

      virtual size_type
      getMeanValueConstraintProcId() const = 0;
    };

  } // namespace basis
} // namespace dftefe

#endif // dftefeConstraints_h
