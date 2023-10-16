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
#include <basis/FECellWiseDataOperations.h>
namespace dftefe
{
  namespace basis
  {
    //
    // Constructor
    //
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::
      FEBasisOperations(
        std::shared_ptr<const BasisDataStorage<ValueTypeBasisData, memorySpace>>
                        basisDataStorage,
        const size_type maxCellTimesFieldBlock)
      : d_maxCellTimesFieldBlock(maxCellTimesFieldBlock)
    {
      d_feBasisDataStorage = std::dynamic_pointer_cast<
        const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>(
        basisDataStorage);
      utils::throwException(
        d_feBasisDataStorage != nullptr,
        "Could not cast BasisDataStorage to FEBasisDataStorage in the constructor of FEBasisOperations");
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::
      interpolate(
        const Field<ValueTypeBasisCoeff, memorySpace> &field,
        quadrature::QuadratureValuesContainer<ValueTypeUnion, memorySpace>
          &quadValuesContainer) const
    {
      const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
        &               vectorData      = field.getVector();
      const std::string constraintsName = field.getConstraintsName();
      const BasisHandler<ValueTypeBasisCoeff, memorySpace> &basisHandler =
        field.getBasisHandler();

      interpolate(vectorData,
                  constraintsName,
                  basisHandler,
                  quadValuesContainer);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::
      interpolate(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                                   vectorData,
        const std::string &                                   constraintsName,
        const BasisHandler<ValueTypeBasisCoeff, memorySpace> &basisHandler,
        quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                 ValueTypeBasisData>,
          memorySpace> &quadValuesContainer) const

    {
      quadrature::QuadratureRuleAttributes quadratureRuleAttributes = 
        d_feBasisDataStorage->getQuadratureRuleContainer()->getQuadratureRuleAttributes();
      const FEBasisHandler<ValueTypeBasisCoeff, memorySpace, dim>
        &feBasisHandler = dynamic_cast<
          const FEBasisHandler<ValueTypeBasisCoeff, memorySpace, dim> &>(
          basisHandler);
      utils::throwException(
        &feBasisHandler != nullptr,
        "Could not cast BasisHandler of the input vector to FEBasisHandler in "
        "FEBasisOperations.interpolate()");

      const BasisManager &basisManager = basisHandler.getBasisManager();

      const FEBasisManager &feBasisManager =
        dynamic_cast<const FEBasisManager &>(basisManager);
      utils::throwException(
        &feBasisManager != nullptr,
        "Could not cast BasisManager of the input vector to FEBasisManager "
        "in FEBasisOperations.interpolate()");

      const size_type numComponents = vectorData.getNumberComponents();
      const size_type numLocallyOwnedCells =
        feBasisHandler.nLocallyOwnedCells();
      const size_type numCumulativeLocallyOwnedCellDofs =
        feBasisHandler.nCumulativeLocallyOwnedCellDofs();
      auto itCellLocalIdsBegin =
        feBasisHandler.locallyOwnedCellLocalDofIdsBegin(constraintsName);
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = feBasisHandler.nLocallyOwnedCellDofs(iCell);

      //
      // reinit the quadValuesContainer
      //
      utils::throwException(
        quadratureRuleAttributes == quadValuesContainer.getQuadratureRuleContainer()->getQuadratureRuleAttributes(),
        "The quadRuleAttributes do not match with that in the quadValuesContainer");

      utils::throwException(
        numComponents == quadValuesContainer.getNumberComponents(),
        "The number of components of input vector do not match with that in the quadValuesContainer");

      std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainer =
        d_feBasisDataStorage->getQuadratureRuleContainer();
      quadValuesContainer.reinit(quadRuleContainer,
                                 numComponents,
                                 ValueTypeUnion());

      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();
      bool sameQuadRuleInAllCells = false;
      if (quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
          quadratureFamily == quadrature::QuadratureFamily::GLL)
        sameQuadRuleInAllCells = true;
      bool variableDofsPerCell = feBasisManager.isVariableDofsPerCell();

      // Perform
      // Ce = Ae*Be, where Ce_ij = interpolated value of the i-th component at
      // j-th quad point in e-th cell Ae_ik = i-th vector components at k-th
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
      // column major format (which is assumed native format for Blas/Lapack
      // data), we use the transpose of Be matrix. That is, we perform Ce =
      // Ae*(Be)^T, with Be stored in row major format
      //
      const bool zeroStrideB = sameQuadRuleInAllCells && (!variableDofsPerCell);
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
          utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
            fieldCellValues(numCumulativeDofsCellsInBlock * numComponents);

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransfer;

          utils::MemoryStorage<size_type, memorySpace>
            numCellsInBlockDofsMemSpace(numCellsInBlock);
          memoryTransfer.copy(numCellsInBlock,
                              numCellsInBlockDofsMemSpace.data(),
                              numCellsInBlockDofs.data());
          FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
            copyFieldToCellWiseData(vectorData.begin(),
                                    numComponents,
                                    itCellLocalIdsBegin + cellLocalIdsOffset,
                                    numCellsInBlockDofsMemSpace,
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
          std::vector<size_type> strideCTmp(numCellsInBlock, 0);

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
              strideATmp[iCell]  = mSizesTmp[iCell] * kSizesTmp[iCell];
              strideCTmp[iCell]  = mSizesTmp[iCell] * nSizesTmp[iCell];
              if (!zeroStrideB)
                strideBTmp[iCell] = kSizesTmp[iCell] * nSizesTmp[iCell];
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
          utils::MemoryStorage<size_type, memorySpace> strideC(numCellsInBlock);
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
          memoryTransfer.copy(numCellsInBlock,
                              strideC.data(),
                              strideCTmp.data());

          ValueTypeUnion                               alpha = 1.0;
          ValueTypeUnion                               beta  = 0.0;
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
            *(vectorData.getLinAlgOpContext().get());

          const ValueTypeBasisData *B =
            (d_feBasisDataStorage->getBasisDataInAllCells())
              .data() +
            BStartOffset;

          ValueTypeUnion *C = quadValuesContainer.begin() + CStartOffset;
          linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace>(
            layout,
            numCellsInBlock,
            transA.data(),
            transB.data(),
            strideA.data(),
            strideB.data(),
            strideC.data(),
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
            linAlgOpContext);


          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              if (!zeroStrideB)
                {
                  BStartOffset += kSizesTmp[iCell] * nSizesTmp[iCell];
                }
              CStartOffset += mSizesTmp[iCell] * nSizesTmp[iCell];
              cellLocalIdsOffset += numCellDofs[cellStartId + iCell];
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::
      interpolateWithBasisGradient(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
                                                              & vectorData,
        const std::string &                                   constraintsName,
        const BasisHandler<ValueTypeBasisCoeff, memorySpace> &basisHandler,
        quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                 ValueTypeBasisData>,
          memorySpace> &quadValuesContainer) const

    {
      quadrature::QuadratureRuleAttributes quadratureRuleAttributes = 
        d_feBasisDataStorage->getQuadratureRuleContainer()->getQuadratureRuleAttributes();
      const FEBasisHandler<ValueTypeBasisCoeff, memorySpace, dim>
        &feBasisHandler = dynamic_cast<
          const FEBasisHandler<ValueTypeBasisCoeff, memorySpace, dim> &>(
          basisHandler);
      utils::throwException(
        &feBasisHandler != nullptr,
        "Could not cast BasisHandler of the input vector to FEBasisHandler in "
        "FEBasisOperations.interpolate()");

      const BasisManager &basisManager = basisHandler.getBasisManager();

      const FEBasisManager &feBasisManager =
        dynamic_cast<const FEBasisManager &>(basisManager);
      utils::throwException(
        &feBasisManager != nullptr,
        "Could not cast BasisManager of the input vector to FEBasisManager "
        "in FEBasisOperations.interpolate()");

      const size_type numComponents = vectorData.getNumberComponents();
      const size_type numLocallyOwnedCells =
        feBasisHandler.nLocallyOwnedCells();
      const size_type numCumulativeLocallyOwnedCellDofs =
        feBasisHandler.nCumulativeLocallyOwnedCellDofs();
      auto itCellLocalIdsBegin =
        feBasisHandler.locallyOwnedCellLocalDofIdsBegin(constraintsName);
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = feBasisHandler.nLocallyOwnedCellDofs(iCell);

      //
      // reinit the quadValuesContainer
      //
      utils::throwException(
        quadratureRuleAttributes == quadValuesContainer.getQuadratureRuleContainer()->getQuadratureRuleAttributes(),
        "The quadRuleAttributes do not match with that in the quadValuesContainer");

      utils::throwException(
        numComponents*dim == quadValuesContainer.getNumberComponents(),
        "The number of components of input vector do not match with that in the quadValuesContainer*dim");

      std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainer =
        d_feBasisDataStorage->getQuadratureRuleContainer();
      quadValuesContainer.reinit(quadRuleContainer,
                                 numComponents*dim,
                                 ValueTypeUnion());

      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();
      bool sameQuadRuleInAllCells = false;
      if (quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
          quadratureFamily == quadrature::QuadratureFamily::GLL)
        sameQuadRuleInAllCells = true;
      bool variableDofsPerCell = feBasisManager.isVariableDofsPerCell();

      // Perform
      // Ce = Ae*Be, where Ce_ij = interpolated value of the i-th component at
      // j-th quad point in e-th cell Ae_ik = i-th vector components at k-th
      // basis function of e-th cell Be_kj = k-th basis function gradient value at j-th
      // quad point in e-th cell
      //

      //
      // For better performance, we evaluate Ce for multiple cells at a time
      //

      //
      // @note: The Be matrix is stored with the quad point as the fastest
      // index. That is Be_kj (k-th basis function value at j-th quad point in
      // e-th cell) is stored in a row-major format. Instead of copying it to a
      // column major format (which is assumed native format for Blas/Lapack
      // data), we use the transpose of Be matrix. That is, we perform Ce =
      // Ae*(Be)^T, with Be stored in row major format
      //
      const bool zeroStrideB = sameQuadRuleInAllCells && (!variableDofsPerCell);
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
          utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
            fieldCellValues(numCumulativeDofsCellsInBlock * numComponents);

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransfer;

          utils::MemoryStorage<size_type, memorySpace>
            numCellsInBlockDofsMemSpace(numCellsInBlock);
          memoryTransfer.copy(numCellsInBlock,
                              numCellsInBlockDofsMemSpace.data(),
                              numCellsInBlockDofs.data());
          FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
            copyFieldToCellWiseData(vectorData.begin(),
                                    numComponents,
                                    itCellLocalIdsBegin + cellLocalIdsOffset,
                                    numCellsInBlockDofsMemSpace,
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
          std::vector<size_type> strideCTmp(numCellsInBlock, 0);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              const size_type cellId = cellStartId + iCell;
              mSizesTmp[iCell]       = numComponents;
              nSizesTmp[iCell] =
                quadValuesContainer.nCellQuadraturePoints(cellId)*dim;
              kSizesTmp[iCell]   = numCellsInBlockDofs[iCell];
              ldaSizesTmp[iCell] = mSizesTmp[iCell];
              ldbSizesTmp[iCell] = nSizesTmp[iCell];
              ldcSizesTmp[iCell] = mSizesTmp[iCell];
              strideATmp[iCell]  = mSizesTmp[iCell] * kSizesTmp[iCell];
              strideCTmp[iCell]  = mSizesTmp[iCell] * nSizesTmp[iCell];
              if (!zeroStrideB)
                strideBTmp[iCell] = kSizesTmp[iCell] * nSizesTmp[iCell];
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
          utils::MemoryStorage<size_type, memorySpace> strideC(numCellsInBlock);
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
          memoryTransfer.copy(numCellsInBlock,
                              strideC.data(),
                              strideCTmp.data());

          ValueTypeUnion                               alpha = 1.0;
          ValueTypeUnion                               beta  = 0.0;
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
            *(vectorData.getLinAlgOpContext().get());

          const ValueTypeBasisData *B =
            (d_feBasisDataStorage->getBasisGradientDataInAllCells())
              .data() +
            BStartOffset;

          ValueTypeUnion *C = quadValuesContainer.begin() + CStartOffset;
          linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace>(
            layout,
            numCellsInBlock,
            transA.data(),
            transB.data(),
            strideA.data(),
            strideB.data(),
            strideC.data(),
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
            linAlgOpContext);


          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              if (!zeroStrideB)
                {
                  BStartOffset += kSizesTmp[iCell] * nSizesTmp[iCell];
                }
              CStartOffset += mSizesTmp[iCell] * nSizesTmp[iCell];
              cellLocalIdsOffset += numCellDofs[cellStartId + iCell];
            }
        }
    }

    // Assess i,j,k element by C[i*numvec*dim + j*numvec + k (numvec is the fastest)] cell->dim->numVec (i,j,k)

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::
      integrateWithBasisValues(
        const quadrature::QuadratureValuesContainer<ValueTypeUnion, memorySpace>
          &                                         inp,
        Field<ValueTypeBasisCoeff, memorySpace> &   f) const
    {
      linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &vectorData =
        f.getVector();
      const BasisHandler<ValueTypeBasisCoeff, memorySpace> &basisHandler =
        f.getBasisHandler();
      const std::string constraintsName = f.getConstraintsName();

      integrateWithBasisValues(inp,
                               basisHandler,
                               constraintsName,
                               vectorData);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::
      integrateWithBasisValues(
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                 ValueTypeBasisData>,
          memorySpace> &                            inp,
        const BasisHandler<ValueTypeBasisCoeff, memorySpace> &basisHandler,
        const std::string &                                   constraintsName,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &vectorData) const

    {
      quadrature::QuadratureRuleAttributes quadratureRuleAttributes = 
        d_feBasisDataStorage->getQuadratureRuleContainer()->getQuadratureRuleAttributes();
      std::shared_ptr<const quadrature::QuadratureRuleContainer> quadRuleContainer =
        inp.getQuadratureRuleContainer();
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributesInp =
        quadRuleContainer->getQuadratureRuleAttributes();
      utils::throwException(
        quadratureRuleAttributes == quadratureRuleAttributesInp,
        "Mismatch in the underlying QuadratureRuleAttributes of the "
        "input QuadratureValuesContainer and the one passed to the "
        " FEBasisOperations::integrateWithBasisValues function");

      const FEBasisHandler<ValueTypeBasisCoeff, memorySpace, dim>
        &feBasisHandler = dynamic_cast<
          const FEBasisHandler<ValueTypeBasisCoeff, memorySpace, dim> &>(
          basisHandler);
      utils::throwException(
        &feBasisHandler != nullptr,
        "Could not cast BasisHandler of the input Field to FEBasisHandler in "
        "FEBasisOperations integrateWithBasisValues()");

      const BasisManager &basisManagerField = basisHandler.getBasisManager();
      const BasisManager &basisManagerDataStorage =
        d_feBasisDataStorage->getBasisManager();
      utils::throwException(
        &basisManagerField == &basisManagerDataStorage,
        "Mismatch in BasisManager used in the Field and the BasisDataStorage "
        "in FEBasisOperations integrateWithBasisValues().");
      const FEBasisManager &feBasisManager =
        dynamic_cast<const FEBasisManager &>(basisManagerField);
      utils::throwException(
        &feBasisManager != nullptr,
        "Could not cast BasisManager of the input Field to FEBasisManager "
        "in FEBasisOperations integrateWithBasisValues()");


      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
        *(vectorData.getLinAlgOpContext().get());
      auto jxwStorage =
        d_feBasisDataStorage->getJxWInAllCells();

      const size_type numComponents = inp.getNumberComponents();
      utils::throwException(
        vectorData.getNumberComponents() == numComponents,
        "Mismatch in number of components in input and output "
        "in FEBasisOperations integrateWithBasisValues().");
      const size_type numLocallyOwnedCells =
        feBasisHandler.nLocallyOwnedCells();
      const size_type numCumulativeLocallyOwnedCellDofs =
        feBasisHandler.nCumulativeLocallyOwnedCellDofs();
      auto itCellLocalIdsBegin =
        feBasisHandler.locallyOwnedCellLocalDofIdsBegin(constraintsName);
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = feBasisHandler.nLocallyOwnedCellDofs(iCell);

      std::vector<size_type> numCellQuad(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellQuad[iCell] = quadRuleContainer->nCellQuadraturePoints(iCell);


      bool                               sameQuadRuleInAllCells = false;
      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();
      if (quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
          quadratureFamily == quadrature::QuadratureFamily::GLL)
        sameQuadRuleInAllCells = true;

      bool       variableDofsPerCell = feBasisManager.isVariableDofsPerCell();
      const bool zeroStrideB = sameQuadRuleInAllCells && (!variableDofsPerCell);
      linearAlgebra::blasLapack::Layout layout =
        linearAlgebra::blasLapack::Layout::ColMajor;
      size_type cellLocalIdsOffset = 0;
      size_type BStartOffset       = 0;
      size_type CStartOffset       = 0;

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

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransfer;

          utils::MemoryStorage<size_type, memorySpace>
            numCellsInBlockDofsMemSpace(numCellsInBlock);
          memoryTransfer.copy(numCellsInBlock,
                              numCellsInBlockDofsMemSpace.data(),
                              numCellsInBlockDofs.data());
          std::vector<size_type> numCellsInBlockQuad(numCellsInBlock, 0);
          std::copy(numCellQuad.begin() + cellStartId,
                    numCellQuad.begin() + cellEndId,
                    numCellsInBlockQuad.begin());

          // std::vector<size_type> numCellsInBlockDofsQuad(numCellsInBlock, 0);

          // for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
          //   {
          //     numCellsInBlockDofsQuad[iCell] =
          //       numCellsInBlockQuad[iCell] * numCellsInBlockDofs[iCell];
          //   }

          const size_type numCumulativeQuadCellsInBlock =
            std::accumulate(numCellsInBlockQuad.begin(),
                            numCellsInBlockQuad.end(),
                            0);


          utils::MemoryStorage<ValueTypeUnion, memorySpace> inpJxW(
            numComponents * numCumulativeQuadCellsInBlock, ValueTypeUnion());

          // std::cout << "numCumulativeDofsCellsInBlock = "
          //           << numCumulativeDofsCellsInBlock
          //           << " numComponents  = " << numComponents <<
          //           "locallyownedcells =" << numLocallyOwnedCells << "\n";

          utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
            outputFieldCellValues(numCumulativeDofsCellsInBlock * numComponents,
                                  ValueTypeUnion());


          // KhatriRao product for inp and JxW
          linearAlgebra::blasLapack::khatriRaoProduct(
            layout,
            1,
            numComponents,
            numCumulativeQuadCellsInBlock,
            jxwStorage.data() +
              quadRuleContainer->getCellQuadStartId(cellStartId),
            inp.begin(cellStartId),
            inpJxW.data(),
            linAlgOpContext);

          // // Hadamard product for inp and JxW
          // linearAlgebra::blasLapack::blockedHadamardProduct(
          //   numCumulativeQuadCellsInBlock,
          //   numComponents,
          //   inp.begin(cellStartId),
          //   jxwStorage.data() +
          //     quadRuleContainer->getCellQuadStartId(cellStartId),
          //   inpJxW.data(),
          //   linAlgOpContext);

          // TODO check if these are right ?? Why is the B Transposed
          std::vector<linearAlgebra::blasLapack::Op> transA(
            numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
          std::vector<linearAlgebra::blasLapack::Op> transB(
            numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
          std::vector<size_type> mSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> nSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> kSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> ldaSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> ldbSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> ldcSizesTmp(numCellsInBlock, 0);
          std::vector<size_type> strideATmp(numCellsInBlock, 0);
          std::vector<size_type> strideBTmp(numCellsInBlock, 0);
          std::vector<size_type> strideCTmp(numCellsInBlock, 0);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              const size_type cellId = cellStartId + iCell;
              mSizesTmp[iCell]       = numComponents;
              nSizesTmp[iCell]       = numCellsInBlockDofs[iCell];
              kSizesTmp[iCell]       = numCellsInBlockQuad[iCell];
              ldaSizesTmp[iCell]     = mSizesTmp[iCell];
              ldbSizesTmp[iCell]     = kSizesTmp[iCell];
              ldcSizesTmp[iCell]     = mSizesTmp[iCell];
              strideATmp[iCell]      = mSizesTmp[iCell] * kSizesTmp[iCell];
              strideCTmp[iCell]      = mSizesTmp[iCell] * nSizesTmp[iCell];
              if (!zeroStrideB)
                strideBTmp[iCell] = kSizesTmp[iCell] * nSizesTmp[iCell];
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
          utils::MemoryStorage<size_type, memorySpace> strideC(numCellsInBlock);
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
          memoryTransfer.copy(numCellsInBlock,
                              strideC.data(),
                              strideCTmp.data());

          ValueTypeUnion alpha = 1.0;
          ValueTypeUnion beta  = 0.0;

          const ValueTypeBasisData *B =
            (d_feBasisDataStorage->getBasisDataInAllCells())
              .data() +
            BStartOffset;

          ValueTypeUnion *C = outputFieldCellValues.begin();
          linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace>(
            layout,
            numCellsInBlock,
            transA.data(),
            transB.data(),
            strideA.data(),
            strideB.data(),
            strideC.data(),
            mSizes.data(),
            nSizes.data(),
            kSizes.data(),
            alpha,
            inpJxW.data(),
            ldaSizes.data(),
            B,
            ldbSizes.data(),
            beta,
            C,
            ldcSizes.data(),
            linAlgOpContext);


          FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
            addCellWiseDataToFieldData(outputFieldCellValues,
                                       numComponents,
                                       itCellLocalIdsBegin + cellLocalIdsOffset,
                                       numCellsInBlockDofsMemSpace,
                                       vectorData.begin());

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              if (!zeroStrideB)
                {
                  BStartOffset += kSizesTmp[iCell] * nSizesTmp[iCell];
                }
              CStartOffset += mSizesTmp[iCell] * nSizesTmp[iCell];
              cellLocalIdsOffset += numCellDofs[cellStartId + iCell];
            }
        }

      const Constraints<ValueTypeBasisCoeff, memorySpace> &constraints =
        feBasisHandler.getConstraints(constraintsName);
      constraints.distributeChildToParent(vectorData,
                                          vectorData.getNumberComponents());

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      vectorData.accumulateAddLocallyOwned();
      vectorData.updateGhostValues();
    }


  } // namespace basis
} // namespace dftefe
