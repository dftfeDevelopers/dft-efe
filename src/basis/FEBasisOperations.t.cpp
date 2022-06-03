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
 * @author Bikash Kanungo, Vishal Subramanian
 */
#include <utils/Exceptions.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <basis/FEBasisOperationsInternal.h>
namespace dftefe
{
  namespace basis
  {
    //
    // Constructor
    //
    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    FEBasisOperations<ValueType, memorySpace, dim>::FEBasisOperations(
      std::shared_ptr<const BasisDataStorage<ValueType, memorySpace>>
                      basisDataStorage,
      const size_type maxCellTimesFieldBlock)
      : d_maxCellTimesFieldBlock(maxCellTimesFieldBlock)
    {
      d_feBasisDataStorage = std::dynamic_pointer_cast<
        const FEBasisDataStorage<ValueType, memorySpace>>(
        basisDataStorage);
      utils::throwException(
        d_feBasisDataStorage != nullptr,
        "Could not cast BasisDataStorage to FEBasisDataStorage in the constructor of FEBasisOperations");
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    void
    FEBasisOperations<ValueType, memorySpace, dim>::interpolate(
      const Field<ValueType, memorySpace> &       field,
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        &quadValuesContainer) const
    {
      const BasisHandler<ValueType,memorySpace> &basisHandler = field.getBasisHandler();
      const FEBasisHandler<ValueType, memorySpace, dim> &feBasisHandler =
        dynamic_cast<const FEBasisHandler<ValueType, memorySpace, dim> &>(
          basisHandler);
      utils::throwException(
        &feBasisHandler != nullptr,
        "Could not cast BasisHandler to FEBasisHandler in FEBasisOperations.interpolate()");
      const BasisManager &basisManagerField = basisHandler.getBasisManager();
      const BasisManager &basisManagerDataStorage =
        d_feBasisDataStorage->getBasisManager();
      utils::throwException(
        &basisManagerField == &basisManagerDataStorage,
        "Mismatch in BasisManager used in Field and BasisDataStorage.");
      const FEBasisManager &feBasisManager =
        dynamic_cast<const FEBasisManager &>(basisManagerField);
      utils::throwException(
        &feBasisManager != nullptr,
        "Could not cast BasisManager to FEBasisManager in FEBasisOperations.interpolate()");

      const size_type   numComponents   = 1;
      const std::string constraintsName = field.getConstraintsName();
      const size_type   numLocallyOwnedCells =
        feBasisHandler.nLocallyOwnedCells();
      const size_type numCumulativeLocallyOwnedCellDofs =
        feBasisHandler.nCumulativeLocallyOwnedCellDofs();
      auto itCellLocalIdsBegin =
        feBasisHandler.locallyOwnedCellLocalDofIdsBegin(constraintsName);
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = feBasisHandler.nLocallyOwnedCellDofs(iCell);


      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();
      bool sameQuadRuleInAllCells = false;
      if (quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
          quadratureFamily == quadrature::QuadratureFamily::GLL)
        sameQuadRuleInAllCells = true;
      bool hpRefined = feBasisManager.isHPRefined();

      // Perform
      // Ce = Ae*Be, where Ce_ij = interpolated value of the i-th component at
      // j-th quad point in e-th cell Ae_ik = i-th field components at k-th
      // basis function of e-th cell Be_kj = k-th basis function value at j-th
      // quad point in e-th cell
      //

      //
      // For better performance, we evaluate Ce for multiple cells at a time
      //

      //
      // @note: The Be matrix is stored with the quad point as the fastest
      // index. That is Be_kj (k-th basis function value at j-th quad point in
      // e-th cell) is stored in a row-major format. Instead of copying it to a
      // column major format, we use the transpose of Be matrix. That is, we
      // perform Ce = Ae*(Be)^T, with Be stored in row major format
      //
      const bool zeroStrideB = sameQuadRuleInAllCells && (!hpRefined);
      linearAlgebra::blasLapack::Layout layout =
        linearAlgebra::blasLapack::Layout::ColMajor;
      size_type       cellLocalIdsOffset = 0;
      size_type       BStartOffset       = 0;
      size_type       CStartOffset       = 0;
      const size_type cellBlockSize = d_maxCellTimesFieldBlock / numComponents;
      for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
           cellStartId += cellBlockSize)
        {
          const size_type cellEndId =
            std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
          const size_type        numCellsInBlock = cellEndId - cellStartId;
          std::vector<size_type> numCellsInBlockDofs(numCellsInBlock, 0);
          std::copy(numCellDofs.begin() + cellStartId,
                    numCellDofs.begin() + cellEndId,
                    numCellsInBlockDofs.begin());

          const size_type numCumulativeDofsCellsInBlock =
            std::accumulate(numCellsInBlockDofs.begin(),
                            numCellsInBlockDofs.end(),
                            0);
          utils::MemoryStorage<ValueType, memorySpace> fieldCellValues(
            numCumulativeDofsCellsInBlock);
          FEBasisOperationsInternal<ValueType, memorySpace>::
            copyFieldToCellWiseData(field.begin(),
                                    numComponents,
                                    itCellLocalIdsBegin + cellLocalIdsOffset,
                                    numCellsInBlockDofs,
                                    fieldCellValues);

          std::vector<linearAlgebra::blasLapack::Op> transA(
            numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
          std::vector<linearAlgebra::blasLapack::Op> transB(
            numCellsInBlock, linearAlgebra::blasLapack::Op::Trans);
          std::vector<size_type> mSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> nSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> kSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> ldaSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> ldbSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> ldcSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> strideATmp(numCellsInBlock, 0);
          std::vector<size_type> strideBTmp(numCellsInBlock, 0);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              const size_type cellId = cellStartId + iCell;
              mSizesTmp[iCell]       = numComponents;
              nSizesTmp[iCell] =
                quadValuesContainer.nCellQuadraturePoints(cellId);
              kSizesTmp[iCell]   = numCellsInBlockDofs[iCell];
              ldaSizesTmp[iCell] = mSizesTmp[iCell];
              ldbSizesTmp[iCell] = nSizesTmp[iCell];
              ldcSizesTmp[iCell] = mSizesTmp[iCell];
              if (iCell > 0)
                strideATmp[iCell] = strideATmp[iCell - 1] +
                                    mSizesTmp[iCell - 1] * kSizesTmp[iCell - 1];
              if (!zeroStrideB && iCell > 0)
                strideBTmp[iCell] = strideBTmp[iCell - 1] +
                                    kSizesTmp[iCell - 1] * nSizesTmp[iCell - 1];
            }

          utils::MemoryStorage<size_type, memorySpace> mSizes(numCellsInBlock);
          utils::MemoryStorage<size_type, memorySpace> nSizes(numCellsInBlock);
          utils::MemoryStorage<size_type, memorySpace> kSizes(numCellsInBlock);
          utils::MemoryStorage<size_type, memorySpace> ldaSizes(
            numCellsInBlock);
          utils::MemoryStorage<size_type, memorySpace> ldbSizes(
            numCellsInBlock);
          utils::MemoryStorage<size_type, memorySpace> ldcSizes(
            numCellsInBlock);
          utils::MemoryStorage<size_type, memorySpace> strideA(numCellsInBlock);
          utils::MemoryStorage<size_type, memorySpace> strideB(numCellsInBlock);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransfer;
          memoryTransfer.copy(numCellsInBlock, mSizes.data(), mSizesTmp.data());
          memoryTransfer.copy(numCellsInBlock, nSizes.data(), nSizesTmp.data());
          memoryTransfer.copy(numCellsInBlock, kSizes.data(), kSizesTmp.data());
          memoryTransfer.copy(numCellsInBlock,
                              ldaSizes.data(),
                              ldaSizesTmp.data());
          memoryTransfer.copy(numCellsInBlock,
                              ldbSizes.data(),
                              ldbSizesTmp.data());
          memoryTransfer.copy(numCellsInBlock,
                              ldcSizes.data(),
                              ldcSizesTmp.data());
          memoryTransfer.copy(numCellsInBlock,
                              strideA.data(),
                              strideATmp.data());
          memoryTransfer.copy(numCellsInBlock,
                              strideB.data(),
                              strideBTmp.data());

          ValueType                           alpha = 1.0;
          ValueType                           beta  = 0.0;
          const linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
            field.getLinAlgOpContext();

          const ValueType *B =
            (d_feBasisDataStorage->getBasisDataInAllCells(quadratureRuleAttributes))
              .data() +
            BStartOffset;
          ValueType *C = quadValuesContainer.begin() + CStartOffset;
          linearAlgebra::blasLapack::gemmStridedVarBatched<ValueType,memorySpace>(layout,
                                               numLocallyOwnedCells,
                                               transA.data(),
                                               transB.data(),
                                               strideA.data(),
                                               strideB.data(),
                                               mSizes.data(),
                                               nSizes.data(),
                                               kSizes.data(),
                                               alpha,
                                               fieldCellValues.data(),
                                               ldaSizes.data(),
                                               B,
                                               ldbSizes.data(),
                                               beta,
                                               C,
                                               ldcSizes.data(),
                                               linAlgOpContext.getBlasQueue());

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              BStartOffset += kSizesTmp[iCell] * nSizesTmp[iCell];
              CStartOffset += mSizesTmp[iCell] * nSizesTmp[iCell];
              cellLocalIdsOffset += numCellDofs[cellStartId + iCell];
            }
        }
    }

  } // namespace basis
} // namespace dftefe
