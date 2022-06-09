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

#ifndef dftefeFEConstraintsDealii_h
#define dftefeFEConstraintsDealii_h

#include <basis/FEConstraintsBase.h>
#include <basis/FEBasisManager.h>
#include <utils/TypeConfig.h>
#include <deal.II/lac/affine_constraints.h>
#include "FEBasisManagerDealii.h"
#include <utils/MemoryStorage.h>
#include <utils/MPIPatternP2P.h>

#include <linearAlgebra/Vector.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    class FEConstraintsDealii : public FEConstraintsBase<ValueType, memorySpace>
    {
    public:
      using GlobalSizeTypeVector =
        utils::MemoryStorage<global_size_type, memorySpace>;
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;
      FEConstraintsDealii();
      ~FEConstraintsDealii() = default;
      void
      clear() override;
      void
      setInhomogeneity(global_size_type basisId,
                       ValueType        constraintValue) override;
      bool
      isConstrained(global_size_type basisId) const override;
      void
      close() override;
      bool
      isClosed() const override;
      void
      setHomogeneousDirichletBC() override;

      std::vector<std::pair<global_size_type, ValueType>> *
      getConstraintEntries(const global_size_type lineDof) const override;

      bool
      isInhomogeneouslyConstrained(const global_size_type index) const override;

      ValueType
      getInhomogeneity(const global_size_type lineDof) const override;

      void
      copyConstraintsData(
        const Constraints<ValueType, memorySpace> &constraintsDataIn,
        const utils::MPIPatternP2P<memorySpace> &  mpiPattern) override;
      void
      populateConstraintsData(
        const utils::MPIPatternP2P<memorySpace> &mpiPattern) override;


      void
      distributeChildToParent(
        linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
        size_type blockSize = 1) const override;
      void
      distributeParentToChild(
        linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
        size_type blockSize = 1) const override;
      void
      setConstrainedNodesToZero(
        linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
        size_type blockSize = 1) const override;

      //
      // FE related functions
      //
      void
      makeHangingNodeConstraint(
        std::shared_ptr<FEBasisManager> feBasis) override;

      //
      // dealii specific fucntions
      //
      const dealii::AffineConstraints<ValueType> &
      getAffineConstraints() const;

    private:
      void
      addEntries(const global_size_type constrainedDofIndex,
                 const std::vector<std::pair<global_size_type, ValueType>>
                   &colWeightPairs);

      void
      addLine(const global_size_type lineDof);

      dealii::AffineConstraints<ValueType>             d_constraintMatrix;
      std::shared_ptr<const FEBasisManagerDealii<dim>> d_feBasisManager;
      bool                                             d_isCleared;
      bool                                             d_isClosed;

      GlobalSizeTypeVector d_rowConstraintsIdsGlobal;
      SizeTypeVector       d_rowConstraintsIdsLocal;
      SizeTypeVector       d_columnConstraintsIdsLocal;
      GlobalSizeTypeVector d_columnConstraintsIdsGlobal;

      utils::MemoryStorage<double, memorySpace>    d_columnConstraintsValues;
      utils::MemoryStorage<ValueType, memorySpace> d_constraintsInhomogenities;

      SizeTypeVector d_rowConstraintsSizes;
    };

  } // namespace basis
} // namespace dftefe
#include "FEConstraintsDealii.t.cpp"
#endif // dftefeFEConstraintsDealii_h
