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
 * @author Bikash Kanungo, Vishal Subramanian, Avirup Sircar
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
    FEBasisOperations<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::interpolate(const Field<ValueTypeBasisCoeff, memorySpace> &field,
                        quadrature::QuadratureValuesContainer<ValueTypeUnion,
                                                              memorySpace>
                          &quadValuesContainer) const
    {
      const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
        &vectorData = field.getVector();
      const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager =
        field.getBasisManager();

      interpolate(vectorData, basisManager, quadValuesContainer);
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
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                 ValueTypeBasisData>,
          memorySpace> &quadValuesContainer) const

    {
      quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        d_feBasisDataStorage->getQuadratureRuleContainer()
          ->getQuadratureRuleAttributes();
      const FEBasisManager<ValueTypeBasisCoeff,
                           ValueTypeBasisData,
                           memorySpace,
                           dim> &feBasisManager =
        dynamic_cast<const FEBasisManager<ValueTypeBasisCoeff,
                                          ValueTypeBasisData,
                                          memorySpace,
                                          dim> &>(basisManager);
      utils::throwException(
        &feBasisManager != nullptr,
        "Could not cast BasisManager of the input vector to FEBasisManager in "
        "FEBasisOperations.interpolate()");

      const BasisDofHandler &basisDofHandler =
        basisManager.getBasisDofHandler();

      const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>
        &feBasisDofHandler = dynamic_cast<
          const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim> &>(
          basisDofHandler);
      utils::throwException(
        &feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input vector to FEBasisDofHandler "
        "in FEBasisOperations.interpolate()");

      const size_type numComponents = vectorData.getNumberComponents();
      const size_type numLocallyOwnedCells =
        feBasisManager.nLocallyOwnedCells();
      const size_type numCumulativeLocallyOwnedCellDofs =
        feBasisManager.nCumulativeLocallyOwnedCellDofs();
      auto itCellLocalIdsBegin =
        feBasisManager.locallyOwnedCellLocalDofIdsBegin();
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = feBasisManager.nLocallyOwnedCellDofs(iCell);

      //
      // reinit the quadValuesContainer
      //
      utils::throwException(
        quadratureRuleAttributes ==
          quadValuesContainer.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes(),
        "The quadRuleAttributes do not match with that in the quadValuesContainer");

      utils::throwException(
        numComponents == quadValuesContainer.getNumberComponents(),
        "The number of components of input vector do not match with that in the quadValuesContainer");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBasisDataStorage->getQuadratureRuleContainer();
      quadValuesContainer.reinit(quadRuleContainer,
                                 numComponents,
                                 ValueTypeUnion());

      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();
      bool sameQuadRuleInAllCells = false;
      if (quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
          quadratureFamily == quadrature::QuadratureFamily::GLL ||
          quadratureFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED)
        sameQuadRuleInAllCells = true;
      bool variableDofsPerCell = feBasisDofHandler.isVariableDofsPerCell();

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
      const bool zeroStrideA = sameQuadRuleInAllCells && (!variableDofsPerCell);
      linearAlgebra::blasLapack::Layout layout =
        linearAlgebra::blasLapack::Layout::ColMajor;
      size_type       cellLocalIdsOffset = 0;
      size_type       AStartOffset       = 0;
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
              mSizesTmp[iCell] =
                quadValuesContainer.nCellQuadraturePoints(cellId);
              nSizesTmp[iCell]   = numComponents;
              kSizesTmp[iCell]   = numCellsInBlockDofs[iCell];
              ldaSizesTmp[iCell] = mSizesTmp[iCell];
              ldbSizesTmp[iCell] = nSizesTmp[iCell];
              ldcSizesTmp[iCell] = mSizesTmp[iCell];
              strideBTmp[iCell]  = kSizesTmp[iCell] * nSizesTmp[iCell];
              strideCTmp[iCell]  = mSizesTmp[iCell] * nSizesTmp[iCell];
              if (!zeroStrideA)
                strideATmp[iCell] = mSizesTmp[iCell] * kSizesTmp[iCell];
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

          const ValueTypeBasisData *A =
            (d_feBasisDataStorage->getBasisDataInAllCells()).data() +
            AStartOffset;

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
            A,
            ldaSizes.data(),
            fieldCellValues.data(),
            ldbSizes.data(),
            beta,
            C,
            ldcSizes.data(),
            linAlgOpContext);


          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              if (!zeroStrideA)
                {
                  AStartOffset += mSizesTmp[iCell] * kSizesTmp[iCell];
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
          &                                                   vectorData,
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                 ValueTypeBasisData>,
          memorySpace> &quadValuesContainer) const

    {
      quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        d_feBasisDataStorage->getQuadratureRuleContainer()
          ->getQuadratureRuleAttributes();
      const FEBasisManager<ValueTypeBasisCoeff,
                           ValueTypeBasisData,
                           memorySpace,
                           dim> &feBasisManager =
        dynamic_cast<const FEBasisManager<ValueTypeBasisCoeff,
                                          ValueTypeBasisData,
                                          memorySpace,
                                          dim> &>(basisManager);
      utils::throwException(
        &feBasisManager != nullptr,
        "Could not cast BasisManager of the input vector to FEBasisManager in "
        "FEBasisOperations.interpolate()");

      const BasisDofHandler &basisDofHandler =
        basisManager.getBasisDofHandler();

      const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>
        &feBasisDofHandler = dynamic_cast<
          const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim> &>(
          basisDofHandler);
      utils::throwException(
        &feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input vector to FEBasisDofHandler"
        "in FEBasisOperations.interpolate()");

      const size_type numComponents = vectorData.getNumberComponents();
      const size_type numLocallyOwnedCells =
        feBasisManager.nLocallyOwnedCells();
      const size_type numCumulativeLocallyOwnedCellDofs =
        feBasisManager.nCumulativeLocallyOwnedCellDofs();
      auto itCellLocalIdsBegin =
        feBasisManager.locallyOwnedCellLocalDofIdsBegin();

      //
      // reinit the quadValuesContainer
      //
      utils::throwException(
        quadratureRuleAttributes ==
          quadValuesContainer.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes(),
        "The quadRuleAttributes do not match with that in the quadValuesContainer");

      utils::throwException(
        numComponents * dim == quadValuesContainer.getNumberComponents(),
        "The number of components of input vector do not match with that in the quadValuesContainer*dim");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBasisDataStorage->getQuadratureRuleContainer();
      quadValuesContainer.reinit(quadRuleContainer,
                                 numComponents * dim,
                                 ValueTypeUnion());

      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();

      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      std::vector<size_type> numCellQuad(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        {
          numCellDofs[iCell] = feBasisManager.nLocallyOwnedCellDofs(iCell);
          numCellQuad[iCell] = quadValuesContainer.nCellQuadraturePoints(iCell);
        }

      // Perform
      // Ce = Ae*Be, where Ce_ij = interpolated value of the i-th component at
      // j-th quad point in e-th cell Be_ik = i-th vector components at k-th
      // basis function of e-th cell Ae_kj = k-th basis function gradient value
      // at j-th quad point in e-th cell
      //

      //
      // For better performance, we evaluate Ce for multiple cells at a time
      //

      //
      // @note: The Ae matrix is stored with the quad point as the fastest
      // index. That is Ae_kj (k-th basis function value at j-th quad point in
      // e-th cell) is stored in a row-major format. That is, we perform Ce =
      // (Ae)*(Be)^T, with Be stored in row major format
      //

      linearAlgebra::blasLapack::Layout layout =
        linearAlgebra::blasLapack::Layout::ColMajor;
      size_type       cellLocalIdsOffset = 0;
      size_type       AStartOffset       = 0;
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

          size_type numCellsInBlockDim = numCellsInBlock * dim;
          std::vector<linearAlgebra::blasLapack::Op> transA(
            numCellsInBlockDim, linearAlgebra::blasLapack::Op::NoTrans);
          std::vector<linearAlgebra::blasLapack::Op> transB(
            numCellsInBlockDim, linearAlgebra::blasLapack::Op::Trans);
          std::vector<size_type> mSizesTmp(numCellsInBlockDim, 0);
          std::vector<size_type> nSizesTmp(numCellsInBlockDim, 0);
          std::vector<size_type> kSizesTmp(numCellsInBlockDim, 0);
          std::vector<size_type> ldaSizesTmp(numCellsInBlockDim, 0);
          std::vector<size_type> ldbSizesTmp(numCellsInBlockDim, 0);
          std::vector<size_type> ldcSizesTmp(numCellsInBlockDim, 0);
          std::vector<size_type> strideATmp(numCellsInBlockDim, 0);
          std::vector<size_type> strideBTmp(numCellsInBlockDim, 0);
          std::vector<size_type> strideCTmp(numCellsInBlockDim, 0);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              const size_type cellId = cellStartId + iCell;
              for (size_type iDim = 0; iDim < dim; iDim++)
                {
                  size_type index = iCell * dim + iDim;
                  mSizesTmp[index] =
                    quadValuesContainer.nCellQuadraturePoints(cellId);
                  nSizesTmp[index]   = numComponents;
                  kSizesTmp[index]   = numCellsInBlockDofs[iCell];
                  ldaSizesTmp[index] = mSizesTmp[index];
                  ldbSizesTmp[index] = nSizesTmp[index];
                  ldcSizesTmp[index] = mSizesTmp[index];
                  strideATmp[index]  = mSizesTmp[index] * kSizesTmp[index];
                  strideCTmp[index]  = mSizesTmp[index] * nSizesTmp[index];
                  if (iDim == dim - 1)
                    strideBTmp[index] = kSizesTmp[index] * nSizesTmp[index];
                  else
                    strideBTmp[index] = 0;
                }
            }

          utils::MemoryStorage<size_type, memorySpace> mSizes(
            numCellsInBlockDim);
          utils::MemoryStorage<size_type, memorySpace> nSizes(
            numCellsInBlockDim);
          utils::MemoryStorage<size_type, memorySpace> kSizes(
            numCellsInBlockDim);
          utils::MemoryStorage<size_type, memorySpace> ldaSizes(
            numCellsInBlockDim);
          utils::MemoryStorage<size_type, memorySpace> ldbSizes(
            numCellsInBlockDim);
          utils::MemoryStorage<size_type, memorySpace> ldcSizes(
            numCellsInBlockDim);
          utils::MemoryStorage<size_type, memorySpace> strideA(
            numCellsInBlockDim);
          utils::MemoryStorage<size_type, memorySpace> strideB(
            numCellsInBlockDim);
          utils::MemoryStorage<size_type, memorySpace> strideC(
            numCellsInBlockDim);
          memoryTransfer.copy(numCellsInBlockDim,
                              mSizes.data(),
                              mSizesTmp.data());
          memoryTransfer.copy(numCellsInBlockDim,
                              nSizes.data(),
                              nSizesTmp.data());
          memoryTransfer.copy(numCellsInBlockDim,
                              kSizes.data(),
                              kSizesTmp.data());
          memoryTransfer.copy(numCellsInBlockDim,
                              ldaSizes.data(),
                              ldaSizesTmp.data());
          memoryTransfer.copy(numCellsInBlockDim,
                              ldbSizes.data(),
                              ldbSizesTmp.data());
          memoryTransfer.copy(numCellsInBlockDim,
                              ldcSizes.data(),
                              ldcSizesTmp.data());
          memoryTransfer.copy(numCellsInBlockDim,
                              strideA.data(),
                              strideATmp.data());
          memoryTransfer.copy(numCellsInBlockDim,
                              strideB.data(),
                              strideBTmp.data());
          memoryTransfer.copy(numCellsInBlockDim,
                              strideC.data(),
                              strideCTmp.data());

          ValueTypeUnion                               alpha = 1.0;
          ValueTypeUnion                               beta  = 0.0;
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
            *(vectorData.getLinAlgOpContext().get());

          const ValueTypeBasisData *A =
            (d_feBasisDataStorage->getBasisGradientDataInAllCells()).data() +
            AStartOffset;

          ValueTypeUnion *C = quadValuesContainer.begin() + CStartOffset;
          linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace>(
            layout,
            numCellsInBlockDim,
            transA.data(),
            transB.data(),
            strideA.data(),
            strideB.data(),
            strideC.data(),
            mSizes.data(),
            nSizes.data(),
            kSizes.data(),
            alpha,
            A,
            ldaSizes.data(),
            fieldCellValues.data(),
            ldbSizes.data(),
            beta,
            C,
            ldcSizes.data(),
            linAlgOpContext);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              for (size_type iDim = 0; iDim < dim; iDim++)
                {
                  size_type index = iCell * dim + iDim;
                  AStartOffset += mSizesTmp[index] * kSizesTmp[index];
                  CStartOffset += mSizesTmp[index] * nSizesTmp[index];
                }
              cellLocalIdsOffset += numCellDofs[cellStartId + iCell];
            }
        }
    }

    /**
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
              &                                                   vectorData,
            const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
            std::vector<quadrature::QuadratureValuesContainer<
              linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData>,
              memorySpace>> &quadValuesContainerVec) const

        {
          quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
            d_feBasisDataStorage->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes();
          const FEBasisManager<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim> &feBasisManager =
            dynamic_cast<const FEBasisManager<ValueTypeBasisCoeff,
                                              ValueTypeBasisData,
                                              memorySpace,
                                              dim> &>(basisManager);
          utils::throwException(
            &feBasisManager != nullptr,
            "Could not cast BasisManager of the input vector to FEBasisManager
    in " "FEBasisOperations.interpolate()");

          const BasisDofHandler &basisDofHandler =
            basisManager.getBasisDofHandler();

          const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>
            &feBasisDofHandler = dynamic_cast<
              const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim> &>(
              basisDofHandler);
          utils::throwException(
            &feBasisDofHandler != nullptr,
            "Could not cast BasisDofHandler of the input vector to
    FEBasisDofHandler " "in FEBasisOperations.interpolate()");

          const size_type numComponents = vectorData.getNumberComponents();
          const size_type numLocallyOwnedCells =
            feBasisManager.nLocallyOwnedCells();
          const size_type numCumulativeLocallyOwnedCellDofs =
            feBasisManager.nCumulativeLocallyOwnedCellDofs();
          auto itCellLocalIdsBegin =
            feBasisManager.locallyOwnedCellLocalDofIdsBegin();

          //
          // reinit the quadValuesContainerVec
          //

          utils::throwException(
            dim == quadValuesContainerVec.size(),
            "The dim do not match with that of size of quadValuesContainerVec");
          for (auto &i : quadValuesContainerVec)
            {
              utils::throwException(
                quadratureRuleAttributes ==
                  i.getQuadratureRuleContainer()->getQuadratureRuleAttributes(),
                "The quadRuleAttributes do not match with that in the input
    quadValuesContainer"); utils::throwException( numComponents ==
    i.getNumberComponents(), "The number of components of input vector do not
    match with that in the input quadValuesContainer");
            }

          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainer =
    d_feBasisDataStorage->getQuadratureRuleContainer(); for (auto &i :
    quadValuesContainerVec)
            {
              i.reinit(quadRuleContainer, numComponents, ValueTypeUnion());
            }

          const quadrature::QuadratureFamily quadratureFamily =
            quadratureRuleAttributes.getQuadratureFamily();

          std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
          std::vector<size_type> numCellQuad(numLocallyOwnedCells, 0);
          for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
            {
              numCellDofs[iCell] = feBasisManager.nLocallyOwnedCellDofs(iCell);
              numCellQuad[iCell] =
    quadRuleContainer->nCellQuadraturePoints(iCell);
            }

          // Perform
          // Ce = Ae*Be, where Ce_ij = interpolated value of the i-th component
    at
          // j-th quad point in e-th cell Ae_ik = i-th vector components at k-th
          // basis function of e-th cell Be_kj = k-th basis function gradient
    value
          // at j-th quad point in e-th cell
          //

          //
          // For better performance, we evaluate Ce for multiple cells at a time
          //

          //
          // @note: The Be matrix is stored with the quad point as the fastest
          // index. That is Be_kj (k-th basis function value at j-th quad point
    in
          // e-th cell) is stored in a row-major format. Instead of copying it
    to a
          // column major format (which is assumed native format for Blas/Lapack
          // data), we use the transpose of Be matrix. That is, we perform Ce =
          // Ae*(Be)^T, with Be stored in row major format
          //

          linearAlgebra::blasLapack::Layout layout =
            linearAlgebra::blasLapack::Layout::ColMajor;
          size_type              cellLocalIdsOffset = 0;
          std::vector<size_type> BStartOffset(dim, 0);
          for (size_type iDim = 0; iDim < dim; iDim++)
            BStartOffset[iDim] =
              numCellDofs[0] * quadRuleContainer->nCellQuadraturePoints(0) *
    iDim; size_type       quadValueContainerStartOffset = 0; const size_type
    cellBlockSize = d_maxCellTimesFieldBlock / numComponents; for (size_type
    cellStartId = 0; cellStartId < numLocallyOwnedCells; cellStartId +=
    cellBlockSize)
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
                                        itCellLocalIdsBegin +
    cellLocalIdsOffset, numCellsInBlockDofsMemSpace, fieldCellValues);

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
                    quadRuleContainer->nCellQuadraturePoints(cellId);
                  kSizesTmp[iCell]   = numCellsInBlockDofs[iCell];
                  ldaSizesTmp[iCell] = mSizesTmp[iCell];
                  ldbSizesTmp[iCell] = nSizesTmp[iCell];
                  ldcSizesTmp[iCell] = mSizesTmp[iCell];
                  strideATmp[iCell]  = mSizesTmp[iCell] * kSizesTmp[iCell];
                  strideCTmp[iCell]  = mSizesTmp[iCell] * nSizesTmp[iCell];
                  strideBTmp[iCell]  = kSizesTmp[iCell] * nSizesTmp[iCell] *
    dim;
                }

              utils::MemoryStorage<size_type, memorySpace>
    mSizes(numCellsInBlock); utils::MemoryStorage<size_type, memorySpace>
    nSizes(numCellsInBlock); utils::MemoryStorage<size_type, memorySpace>
    kSizes(numCellsInBlock); utils::MemoryStorage<size_type, memorySpace>
    ldaSizes( numCellsInBlock); utils::MemoryStorage<size_type, memorySpace>
    ldbSizes( numCellsInBlock); utils::MemoryStorage<size_type, memorySpace>
    ldcSizes( numCellsInBlock); utils::MemoryStorage<size_type, memorySpace>
    strideA(numCellsInBlock); utils::MemoryStorage<size_type, memorySpace>
    strideB(numCellsInBlock); utils::MemoryStorage<size_type, memorySpace>
    strideC(numCellsInBlock); memoryTransfer.copy(numCellsInBlock,
    mSizes.data(), mSizesTmp.data()); memoryTransfer.copy(numCellsInBlock,
    nSizes.data(), nSizesTmp.data()); memoryTransfer.copy(numCellsInBlock,
    kSizes.data(), kSizesTmp.data()); memoryTransfer.copy(numCellsInBlock,
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

              std::vector<size_type> numCellsInBlockQuad(numCellsInBlock, 0);
              std::copy(numCellQuad.begin() + cellStartId,
                        numCellQuad.begin() + cellEndId,
                        numCellsInBlockQuad.begin());

              const size_type numCumulativeQuadCellsInBlock =
                std::accumulate(numCellsInBlockQuad.begin(),
                                numCellsInBlockQuad.end(),
                                0);

              for (size_type iDim = 0; iDim < dim; iDim++)
                {
                  const ValueTypeBasisData *B =
                    (d_feBasisDataStorage->getBasisGradientDataInAllCells())
                      .begin() +
                    BStartOffset[iDim];

                  ValueTypeUnion *C = quadValuesContainerVec[iDim].begin() +
                                      quadValueContainerStartOffset;
                  linearAlgebra::blasLapack::gemmStridedVarBatched<
                    ValueTypeBasisCoeff,
                    ValueTypeBasisData,
                    memorySpace>(layout,
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
                                 fieldCellValues.begin(),
                                 ldaSizes.data(),
                                 B,
                                 ldbSizes.data(),
                                 beta,
                                 C,
                                 ldcSizes.data(),
                                 linAlgOpContext);
                }

              for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                {
                  for (size_type iDim = 0; iDim < dim; iDim++)
                    BStartOffset[iDim] += kSizesTmp[iCell] * nSizesTmp[iCell] *
    dim; cellLocalIdsOffset += numCellDofs[cellStartId + iCell];
                }
              quadValueContainerStartOffset +=
                numComponents * numCumulativeQuadCellsInBlock;
            }
        }

        // Assess i,j,k element by C[i*numvec*dim + j*numvec + k (numvec is the
        // fastest)] cell->dim->numVec (i,j,k)
    **/
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
          &                                      inp,
        Field<ValueTypeBasisCoeff, memorySpace> &f) const
    {
      linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &vectorData =
        f.getVector();
      const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager =
        f.getBasisManager();

      integrateWithBasisValues(inp, basisManager, vectorData);
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
          memorySpace> &                                      inp,
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &vectorData) const

    {
      quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        d_feBasisDataStorage->getQuadratureRuleContainer()
          ->getQuadratureRuleAttributes();
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = inp.getQuadratureRuleContainer();
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributesInp =
        quadRuleContainer->getQuadratureRuleAttributes();
      utils::throwException(
        quadratureRuleAttributes == quadratureRuleAttributesInp,
        "Mismatch in the underlying QuadratureRuleAttributes of the "
        "input QuadratureValuesContainer and the one passed to the "
        " FEBasisOperations::integrateWithBasisValues function");

      const FEBasisManager<ValueTypeBasisCoeff,
                           ValueTypeBasisData,
                           memorySpace,
                           dim> &feBasisManager =
        dynamic_cast<const FEBasisManager<ValueTypeBasisCoeff,
                                          ValueTypeBasisData,
                                          memorySpace,
                                          dim> &>(basisManager);
      utils::throwException(
        &feBasisManager != nullptr,
        "Could not cast BasisManager of the input Field to FEBasisManager in "
        "FEBasisOperations integrateWithBasisValues()");

      const BasisDofHandler &basisDofHandlerField =
        basisManager.getBasisDofHandler();
      std::shared_ptr<const BasisDofHandler> basisDofHandlerDataStorage =
        d_feBasisDataStorage->getBasisDofHandler();
      utils::throwException(
        &basisDofHandlerField == basisDofHandlerDataStorage.get(),
        "Mismatch in BasisDofHandler used in the Field and the BasisDataStorage "
        "in FEBasisOperations integrateWithBasisValues().");
      const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>
        &feBasisDofHandler = dynamic_cast<
          const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim> &>(
          basisDofHandlerField);
      utils::throwException(
        &feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input Field to FEBasisDofHandler "
        "in FEBasisOperations integrateWithBasisValues()");


      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
        *(vectorData.getLinAlgOpContext().get());
      auto jxwStorage = d_feBasisDataStorage->getJxWInAllCells();

      const size_type numComponents = inp.getNumberComponents();
      utils::throwException(
        vectorData.getNumberComponents() == numComponents,
        "Mismatch in number of components in input and output "
        "in FEBasisOperations integrateWithBasisValues().");
      const size_type numLocallyOwnedCells =
        feBasisManager.nLocallyOwnedCells();
      const size_type numCumulativeLocallyOwnedCellDofs =
        feBasisManager.nCumulativeLocallyOwnedCellDofs();
      auto itCellLocalIdsBegin =
        feBasisManager.locallyOwnedCellLocalDofIdsBegin();
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = feBasisManager.nLocallyOwnedCellDofs(iCell);

      std::vector<size_type> numCellQuad(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellQuad[iCell] = quadRuleContainer->nCellQuadraturePoints(iCell);


      bool                               sameQuadRuleInAllCells = false;
      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();
      if (quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
          quadratureFamily == quadrature::QuadratureFamily::GLL ||
          quadratureFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED)
        sameQuadRuleInAllCells = true;

      bool variableDofsPerCell = feBasisDofHandler.isVariableDofsPerCell();
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

          const size_type numCumulativeQuadCellsInBlock =
            std::accumulate(numCellsInBlockQuad.begin(),
                            numCellsInBlockQuad.end(),
                            0);


          utils::MemoryStorage<ValueTypeUnion, memorySpace> inpJxW(
            numComponents * numCumulativeQuadCellsInBlock, ValueTypeUnion());

          utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
            outputFieldCellValues(numCumulativeDofsCellsInBlock * numComponents,
                                  ValueTypeUnion());

          // TransposedKhatriRao product for inp and JxW
          size_type cumulativeA = 0, cumulativeB = 0, cumulativeC = 0;
          for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
            {
              linearAlgebra::blasLapack::transposedKhatriRaoProduct(
                layout,
                1,
                numComponents,
                numCellsInBlockQuad[iCell],
                jxwStorage.data() +
                  quadRuleContainer->getCellQuadStartId(cellStartId) +
                  cumulativeA,
                inp.begin(cellStartId) + cumulativeB,
                inpJxW.data() + cumulativeC,
                linAlgOpContext);
              cumulativeA += numCellsInBlockQuad[iCell];
              cumulativeB += numCellsInBlockQuad[iCell] * numComponents;
              cumulativeC += numCellsInBlockQuad[iCell] * numComponents;
            }

          /**
          // Blocked Hadamard product for inp and JxW
          linearAlgebra::blasLapack::blockedHadamardProduct(
            numCumulativeQuadCellsInBlock,
            numComponents,
            inp.begin(cellStartId),
            jxwStorage.data() +
              quadRuleContainer->getCellQuadStartId(cellStartId),
            inpJxW.data(),
            linAlgOpContext);

          **/

          // TODO check if these are right ?? Why is the B Transposed
          std::vector<linearAlgebra::blasLapack::Op> transA(
            numCellsInBlock, linearAlgebra::blasLapack::Op::Trans);
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
              ldaSizesTmp[iCell]     = kSizesTmp[iCell];
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
            (d_feBasisDataStorage->getBasisDataInAllCells()).data() +
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

      const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace> &constraints =
        feBasisManager.getConstraints();
      constraints.distributeChildToParent(vectorData,
                                          vectorData.getNumberComponents());

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      vectorData.accumulateAddLocallyOwned();
      vectorData.updateGhostValues();
    }


  } // namespace basis
} // namespace dftefe
