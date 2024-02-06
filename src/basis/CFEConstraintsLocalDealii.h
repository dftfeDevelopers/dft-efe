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
 * @author Vishal Subramanian, Avirup Sircar
 */

#ifndef dftefeCFEConstraintsLocalDealii_h
#define dftefeCFEConstraintsLocalDealii_h

#include <basis/ConstraintsLocal.h>
#include <utils/TypeConfig.h>
#include <deal.II/lac/affine_constraints.h>
#include <utils/MemoryStorage.h>
#include <utils/ScalarSpatialFunction.h>
#include <unordered_map>

#include <linearAlgebra/Vector.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    class CFEConstraintsLocalDealii
      : public ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>
    {
    public:
      using GlobalSizeTypeVector =
        utils::MemoryStorage<global_size_type, memorySpace>;
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;

      CFEConstraintsLocalDealii();

      CFEConstraintsLocalDealii(
        dealii::AffineConstraints<ValueTypeBasisCoeff>
         & dealiiAffineConstraintMatrix,
        std::vector<std::pair<global_size_type, global_size_type>>
                                      & locallyOwnedRanges,
        std::vector<global_size_type> & ghostIndices,
        std::unordered_map<global_size_type, size_type>
          & globalToLocalMapLocalDofs);

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

      const std::vector<std::pair<global_size_type, ValueTypeBasisCoeff>> *
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

      //
      // dealii function
      //
      const dealii::AffineConstraints<ValueTypeBasisCoeff> &
      getAffineConstraints() const;

      //
      // private functions
      //
    private:
      void
      addEntries(
        const global_size_type constrainedDofIndex,
        const std::vector<std::pair<global_size_type, ValueTypeBasisCoeff>>
          &colWeightPairs);

      void
      copyConstraintsDataFromDealiiToDealii(
        const CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>
          &constraintsDataIn);

        void copyConstraintsDataFromDealiiToDftefe();

      void
      addLine(const global_size_type lineDof);

      bool
      isGhostEntry(const global_size_type globalId) const;

      bool
      inLocallyOwnedRanges(const global_size_type globalId) const;

      size_type
      globalToLocal(const global_size_type globalId) const;


      dealii::AffineConstraints<ValueTypeBasisCoeff> d_dealiiAffineConstraintMatrix;
      bool                                           d_isCleared;
      bool                                           d_isClosed;

      GlobalSizeTypeVector d_rowConstraintsIdsGlobal;
      SizeTypeVector       d_rowConstraintsIdsLocal;
      SizeTypeVector       d_columnConstraintsIdsLocal;
      SizeTypeVector       d_constraintRowSizesAccumulated;
      GlobalSizeTypeVector d_columnConstraintsIdsGlobal;

      utils::MemoryStorage<double, memorySpace> d_columnConstraintsValues;
      utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
        d_constraintsInhomogenities;

      SizeTypeVector d_rowConstraintsSizes;

      std::vector<std::pair<global_size_type, global_size_type>>
                                                      d_locallyOwnedRanges;
      std::vector<global_size_type>                   d_ghostIndices;
      std::unordered_map<global_size_type, size_type> d_globalToLocalMap;
    };

  } // namespace basis
} // namespace dftefe
#include <basis/CFEConstraintsLocalDealii.t.cpp>
#endif // dftefeCFEConstraintsLocalDealii_h
