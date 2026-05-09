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
      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      FEBasisOperations<ValueTypeBasisCoeff,
                        ValueTypeBasisData,
                        memorySpace,
                        dim>::
      BasisWeakFormKernelWithField(
        realspace::LinearLocalOp L1,
        realspace::VectorMathOp  Op1,
        realspace::VectorMathOp  Op2,
        realspace::LinearLocalOp L2,
        const quadrature::QuadratureValuesContainer<ValueTypeUnion, memorySpace> &f,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                                             feBasisDataStorage,
        const size_type                                      cellBlockSize,
        StorageUnion &cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &        linAlgOpContext) const
      {
        if (L1 == realspace::LinearLocalOp::IDENTITY &&
            Op1 == realspace::VectorMathOp::MULT &&
            Op2 == realspace::VectorMathOp::MULT &&
            L2 == realspace::LinearLocalOp::IDENTITY)
          {
            /*Check the quadAttributes of f and basisDataStorage*/
            quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
              feBasisDataStorage->getQuadratureRuleContainer()
                ->getQuadratureRuleAttributes();
            std::shared_ptr<const quadrature::QuadratureRuleContainer>
              quadRuleContainer = f.getQuadratureRuleContainer();
            const quadrature::QuadratureRuleAttributes
              &quadratureRuleAttributesf =
                f.getQuadratureRuleContainer()->getQuadratureRuleAttributes();
            utils::throwException(
              quadratureRuleAttributes == quadratureRuleAttributesf,
              "Mismatch in the underlying QuadratureRuleAttributes of the "
              "input QuadratureValuesContainer and the one passed to the "
              " computeFEMatrices()");

            size_type basisOverlapSize = 0;

            for (size_type iCell = 0; iCell < d_numLocallyOwnedCells; iCell++)
              {
                basisOverlapSize += d_numCellDofs[iCell] * d_numCellDofs[iCell];
              }

            if(cellWiseFEData.size() != basisOverlapSize)
              cellWiseFEData.resize(basisOverlapSize, (ValueTypeBasisData)0);
            auto jxwStorage = feBasisDataStorage->getJxWInAllCells();

            const bool zeroStrideBasisVal =
              d_sameQuadRuleInAllCells && (!d_variableDofsPerCell);
            linearAlgebra::blasLapack::Layout layout =
              linearAlgebra::blasLapack::Layout::ColMajor;
            size_type NifNjStartOffset = 0, quadCellsInBlockOffSet = 0;

            /** --- Storages --------- **/
            const size_type numCumulativeQuadCells =
              std::accumulate(d_numCellQuad.begin(), d_numCellQuad.end(), 0);

            StorageUnion fxJxW(1 /*numComponents of f*/ *
                                 numCumulativeQuadCells,
                               ValueTypeUnion());

            if(d_fxJxWxNBlock.size() != d_maxQuadInCell * cellBlockSize * d_maxDofInCell)
              d_fxJxWxNBlock.resize(d_maxQuadInCell * cellBlockSize *  d_maxDofInCell, ValueTypeUnion());

            d_basisDataCellRangeCached = false;
            if(d_basisDataInCellRange.size() < d_maxQuadInCell * cellBlockSize * d_maxDofInCell)
              d_basisDataInCellRange.resize(d_maxQuadInCell * cellBlockSize * d_maxDofInCell,
                                        ValueTypeBasisData());
            /** --- Storages --------- **/

            if (f.getNumberComponents() == 1)
              {
                linearAlgebra::blasLapack::hadamardProduct(jxwStorage.size(),
                                                           jxwStorage.data(),
                                                           f.begin(),
                                                           fxJxW.data(),
                                                           linAlgOpContext);
              }
            else
              {
                utils::throwException(
                  false,
                  "quadValuesContainer f in BasisWeakFormKernelWithField"
                  " can be only a scalar field in real space with 1 component.");
              }

            for (size_type cellStartId = 0; cellStartId < d_numLocallyOwnedCells;
                 cellStartId += cellBlockSize)
              {
                const size_type cellEndId =
                  std::min(cellStartId + cellBlockSize, d_numLocallyOwnedCells);
                const size_type numCellsInBlock = cellEndId - cellStartId;
                std::vector<size_type> numCellsInBlockDofs(numCellsInBlock, 0);
                std::copy(d_numCellDofs.begin() + cellStartId,
                          d_numCellDofs.begin() + cellEndId,
                          numCellsInBlockDofs.begin());

                std::vector<size_type> numCellsInBlockQuad(numCellsInBlock, 0);
                std::copy(d_numCellQuad.begin() + cellStartId,
                          d_numCellQuad.begin() + cellEndId,
                          numCellsInBlockQuad.begin());

                const size_type numCumulativeQuadCellsInBlock =
                  std::accumulate(numCellsInBlockQuad.begin(),
                                  numCellsInBlockQuad.end(),
                                  0);

                // size_type numCumulativeQuadxDofsCellsInBlock = 0;
                size_type numCumulativeDofsxDofsCellsInBlock = 0;
                for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
                  {
                    // numCumulativeQuadxDofsCellsInBlock +=
                    //   numCellsInBlockQuad[iCell] *
                    //   numCellsInBlockDofs[iCell];
                    numCumulativeDofsxDofsCellsInBlock +=
                      numCellsInBlockDofs[iCell] * numCellsInBlockDofs[iCell];
                  }


                utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
                  memoryTransfer;

                linearAlgebra::blasLapack::ScalarOp scalarOpA =
                  linearAlgebra::blasLapack::ScalarOp::Identity;
                linearAlgebra::blasLapack::ScalarOp scalarOpB =
                  linearAlgebra::blasLapack::ScalarOp::Identity;
                std::vector<size_type> m(numCellsInBlock, 0);
                std::vector<size_type> n(numCellsInBlock, 0);
                std::vector<size_type> k(numCellsInBlock, 0);
                std::vector<size_type> stA(numCellsInBlock, 0);
                std::vector<size_type> stB(numCellsInBlock, 0);
                std::vector<size_type> stC(numCellsInBlock, 0);

                for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
                  {
                    m[iCell]   = 1; // only for fxJxW numComponents = 1
                    n[iCell]   = numCellsInBlockDofs[iCell];
                    k[iCell]   = numCellsInBlockQuad[iCell];
                    stA[iCell] = m[iCell] * k[iCell];
                    if (!zeroStrideBasisVal)
                      stB[iCell] = n[iCell] * k[iCell];
                    stC[iCell] = m[iCell] * n[iCell] * k[iCell];
                  }

                if (!zeroStrideBasisVal || cellStartId == 0)
                  feBasisDataStorage->getBasisDataInCellRange(
                    std::make_pair(cellStartId, cellEndId), d_basisDataInCellRange);

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeBasisCoeff,
                  ValueTypeBasisData,
                  memorySpace>(numCellsInBlock,
                               layout,
                               scalarOpA,
                               scalarOpB,
                               stA.data(),
                               stB.data(),
                               stC.data(),
                               m.data(),
                               n.data(),
                               k.data(),
                               fxJxW.data() + quadCellsInBlockOffSet,
                               d_basisDataInCellRange.data(),
                               d_fxJxWxNBlock.data(),
                               linAlgOpContext);

                /*--------- Do the integration -----------------*/
                std::vector<char>      transA(numCellsInBlock, 'N');
                std::vector<char>      transB(numCellsInBlock, 'C');
                std::vector<size_type> mSizes(numCellsInBlock, 0);
                std::vector<size_type> nSizes(numCellsInBlock, 0);
                std::vector<size_type> kSizes(numCellsInBlock, 0);
                std::vector<size_type> ldaSizes(numCellsInBlock, 0);
                std::vector<size_type> ldbSizes(numCellsInBlock, 0);
                std::vector<size_type> ldcSizes(numCellsInBlock, 0);
                std::vector<size_type> strideA(numCellsInBlock, 0);
                std::vector<size_type> strideB(numCellsInBlock, 0);
                std::vector<size_type> strideC(numCellsInBlock, 0);

                for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                  {
                    mSizes[iCell]   = numCellsInBlockDofs[iCell];
                    nSizes[iCell]   = numCellsInBlockDofs[iCell];
                    kSizes[iCell]   = numCellsInBlockQuad[iCell];
                    ldaSizes[iCell] = mSizes[iCell];
                    ldbSizes[iCell] = nSizes[iCell];
                    ldcSizes[iCell] = mSizes[iCell];
                    if (!zeroStrideBasisVal)
                      strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                    strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                    strideC[iCell] = mSizes[iCell] * nSizes[iCell];
                  }

                ValueTypeUnion alpha = 1.0;
                ValueTypeUnion beta  = 0.0;

                ValueTypeUnion *C = cellWiseFEData.begin() + NifNjStartOffset;
                linearAlgebra::blasLapack::gemmStridedVarBatched<
                  ValueTypeBasisData,
                  ValueTypeUnion,
                  memorySpace>(numCellsInBlock,
                               transA.data(),
                               transB.data(),
                               strideA.data(),
                               strideB.data(),
                               strideC.data(),
                               mSizes.data(),
                               nSizes.data(),
                               kSizes.data(),
                               alpha,
                               d_fxJxWxNBlock.data(),
                               ldaSizes.data(),
                               d_basisDataInCellRange.data(),
                               ldbSizes.data(),
                               beta,
                               C,
                               ldcSizes.data(),
                               linAlgOpContext);

                NifNjStartOffset += numCumulativeDofsxDofsCellsInBlock;
                quadCellsInBlockOffSet += numCumulativeQuadCellsInBlock;
              }
          }
        else
          {
            utils::throwException(
              false,
              "computeFEMatrices for the given choices of L1, Op1, Op2, L2"
              "is not defined in FEBasisOperations.");
          }
          deleteScratch();
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
      BasisWeakFormKernel(
        realspace::LinearLocalOp L1,
        realspace::VectorMathOp  Op1,
        realspace::LinearLocalOp L2,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                                             feBasisDataStorage,
        const size_type                                      cellBlockSize,
        StorageBasis &cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &        linAlgOpContext) const
      {
        if (L1 == realspace::LinearLocalOp::GRAD &&
            Op1 == realspace::VectorMathOp::DOT &&
            L2 == realspace::LinearLocalOp::GRAD)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
              memoryTransfer;

            size_type basisStiffnessSize = 0;

            for (size_type iCell = 0; iCell < d_numLocallyOwnedCells; iCell++)
              {
                basisStiffnessSize += d_numCellDofs[iCell] * d_numCellDofs[iCell];
              }

            if(cellWiseFEData.size() != basisStiffnessSize)
              cellWiseFEData.resize(basisStiffnessSize, (ValueTypeBasisData)0);
            auto jxwStorage = feBasisDataStorage->getJxWInAllCells();

            linearAlgebra::blasLapack::Layout layout =
              linearAlgebra::blasLapack::Layout::ColMajor;
            // size_type GradNStartOffset        = 0;
            size_type gradNigradNjStartOffset = 0;

            /** --- Storages --------- **/
            if(d_JxWxGradNBlock.size() != d_maxQuadInCell * cellBlockSize * d_maxDofInCell * dim)
              d_JxWxGradNBlock.resize(d_maxQuadInCell * cellBlockSize * d_maxDofInCell * dim, ValueTypeUnion());

            if(d_basisGradientDataInCellRange.size() != d_maxQuadInCell * cellBlockSize * d_maxDofInCell * dim)
              d_basisGradientDataInCellRange.resize(d_maxQuadInCell * cellBlockSize * d_maxDofInCell * dim,
                                        ValueTypeBasisData());                                                    
            /** --- Storages --------- **/

            for (size_type cellStartId = 0; cellStartId < d_numLocallyOwnedCells;
                 cellStartId += cellBlockSize)
              {
                const size_type cellEndId =
                  std::min(cellStartId + cellBlockSize, d_numLocallyOwnedCells);
                const size_type numCellsInBlock = cellEndId - cellStartId;
                std::vector<size_type> numCellsInBlockDofs(numCellsInBlock, 0);
                std::copy(d_numCellDofs.begin() + cellStartId,
                          d_numCellDofs.begin() + cellEndId,
                          numCellsInBlockDofs.begin());

                std::vector<size_type> numCellsInBlockQuad(numCellsInBlock, 0);
                std::copy(d_numCellQuad.begin() + cellStartId,
                          d_numCellQuad.begin() + cellEndId,
                          numCellsInBlockQuad.begin());

                size_type numCumulativeQuadxDofsCellsInBlock = 0;
                size_type numCumulativeDofsxDofsCellsInBlock = 0;
                for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
                  {
                    numCumulativeQuadxDofsCellsInBlock +=
                      numCellsInBlockQuad[iCell] * numCellsInBlockDofs[iCell];
                    numCumulativeDofsxDofsCellsInBlock +=
                      numCellsInBlockDofs[iCell] * numCellsInBlockDofs[iCell];
                  }

                feBasisDataStorage->getBasisGradientDataInCellRange(
                  std::make_pair(cellStartId, cellEndId), d_basisGradientDataInCellRange);

                linearAlgebra::blasLapack::ScalarOp scalarOpA =
                  linearAlgebra::blasLapack::ScalarOp::Identity;
                linearAlgebra::blasLapack::ScalarOp scalarOpB =
                  linearAlgebra::blasLapack::ScalarOp::Identity;
                std::vector<size_type> m(numCellsInBlock, 0);
                std::vector<size_type> n(numCellsInBlock, 0);
                std::vector<size_type> k(numCellsInBlock, 0);
                std::vector<size_type> stA(numCellsInBlock, 0);
                std::vector<size_type> stB(numCellsInBlock, 0);
                std::vector<size_type> stC(numCellsInBlock, 0);

                for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
                  {
                    m[iCell]   = 1;
                    n[iCell]   = numCellsInBlockDofs[iCell] * dim;
                    k[iCell]   = numCellsInBlockQuad[iCell];
                    stA[iCell] = m[iCell] * k[iCell];
                    stB[iCell] = n[iCell] * k[iCell];
                    stC[iCell] = m[iCell] * n[iCell] * k[iCell];
                  }

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeBasisData,
                  ValueTypeBasisData,
                  memorySpace>(
                  numCellsInBlock,
                  layout,
                  scalarOpA,
                  scalarOpB,
                  stA.data(),
                  stB.data(),
                  stC.data(),
                  m.data(),
                  n.data(),
                  k.data(),
                  jxwStorage.data() +
                    d_quadratureRuleContainer->getCellQuadStartId(cellStartId),
                  d_basisGradientDataInCellRange.data(),
                  d_JxWxGradNBlock.data(),
                  linAlgOpContext);

                std::vector<char>      transA(numCellsInBlock, 'N');
                std::vector<char>      transB(numCellsInBlock, 'C');
                std::vector<size_type> mSizes(numCellsInBlock, 0);
                std::vector<size_type> nSizes(numCellsInBlock, 0);
                std::vector<size_type> kSizes(numCellsInBlock, 0);
                std::vector<size_type> ldaSizes(numCellsInBlock, 0);
                std::vector<size_type> ldbSizes(numCellsInBlock, 0);
                std::vector<size_type> ldcSizes(numCellsInBlock, 0);
                std::vector<size_type> strideA(numCellsInBlock, 0);
                std::vector<size_type> strideB(numCellsInBlock, 0);
                std::vector<size_type> strideC(numCellsInBlock, 0);

                for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
                  {
                    size_type index = iCell;
                    mSizes[index]   = numCellsInBlockDofs[iCell];
                    nSizes[index]   = numCellsInBlockDofs[iCell];
                    kSizes[index]   = numCellsInBlockQuad[iCell] * dim;
                    ldaSizes[index] = mSizes[index];
                    ldbSizes[index] = nSizes[index];
                    ldcSizes[index] = mSizes[index];
                    strideA[index]  = mSizes[index] * kSizes[index];
                    strideB[index]  = kSizes[index] * nSizes[index];
                    strideC[index]  = mSizes[index] * nSizes[index];
                  }

                ValueTypeBasisData alpha = 1.0;
                ValueTypeBasisData beta  = 0.0;


                const ValueTypeBasisData *GradN = d_basisGradientDataInCellRange.data();

                ValueTypeBasisData *C =
                  cellWiseFEData.begin() + gradNigradNjStartOffset;

                linearAlgebra::blasLapack::gemmStridedVarBatched<
                  ValueTypeBasisData,
                  ValueTypeBasisData,
                  memorySpace>(numCellsInBlock,
                               transA.data(),
                               transB.data(),
                               strideA.data(),
                               strideB.data(),
                               strideC.data(),
                               mSizes.data(),
                               nSizes.data(),
                               kSizes.data(),
                               alpha,
                               GradN,
                               ldaSizes.data(),
                               d_JxWxGradNBlock.data(),
                               ldbSizes.data(),
                               beta,
                               C,
                               ldcSizes.data(),
                               linAlgOpContext);
                gradNigradNjStartOffset += numCumulativeDofsxDofsCellsInBlock;
              }
          }
        else
          {
            utils::throwException(
              false,
              "computeFEMatrices for the given choices of L1, Op1, L2"
              "is not defined in FEBasisOperations.");
          }
          deleteScratch();
      }

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
        const size_type maxCellBlock,
        const size_type maxFieldBlock)
      : d_maxCellBlock(0)
      , d_maxFieldBlock(0)
      , d_fieldCellValues(0)
      , d_fxJxWxNBlock(0)
      , d_JxWxGradNBlock(0)
      , d_JxWxNBlock(0)
      , d_basisDataInCellRange(0)
      , d_basisGradientDataInCellRange(0)
      , d_basisDataCellRange(0, 0)
      , d_basisDataCellRangeCached(false)
    {
      d_feBasisDataStorage = std::dynamic_pointer_cast<
        const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>(
        basisDataStorage);
      utils::throwException(
        d_feBasisDataStorage != nullptr,
        "Could not cast BasisDataStorage to FEBasisDataStorage in the constructor of FEBasisOperations");
      
      std::shared_ptr<const BasisDofHandler> basisDofHandlerDataStorage =
        d_feBasisDataStorage->getBasisDofHandler();

      d_feBasisDofHandler =
          dynamic_cast<const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>*>(
              basisDofHandlerDataStorage.get());

      utils::throwException(
        d_feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input basisDataStorage "
        "in FEBasisOperations constructor()");

      d_numLocallyOwnedCells = d_feBasisDofHandler->nLocallyOwnedCells();
      d_quadratureRuleContainer = d_feBasisDataStorage->getQuadratureRuleContainer();
      
      d_numCellDofs.resize(d_numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < d_numLocallyOwnedCells; ++iCell)
        d_numCellDofs[iCell] = d_feBasisDofHandler->nCellDofs(iCell);

      d_numCellQuad.resize(d_numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < d_numLocallyOwnedCells; ++iCell)
        d_numCellQuad[iCell] = d_quadratureRuleContainer->nCellQuadraturePoints(iCell);

      d_maxDofInCell =
        *std::max_element(d_numCellDofs.begin(), d_numCellDofs.end());
      d_maxQuadInCell =
        *std::max_element(d_numCellQuad.begin(), d_numCellQuad.end());

      quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
              d_feBasisDataStorage->getQuadratureRuleContainer()->getQuadratureRuleAttributes();
      d_sameQuadRuleInAllCells = false;
      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();
      if (quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
          quadratureFamily == quadrature::QuadratureFamily::GLL ||
          quadratureFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED)
        d_sameQuadRuleInAllCells = true;

      d_variableDofsPerCell = d_feBasisDofHandler->isVariableDofsPerCell();

      reinit(maxCellBlock, maxFieldBlock);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::reinit(const size_type maxCellBlock,
                                   const size_type maxFieldBlock)
    {
      d_maxCellBlock  = maxCellBlock;
      d_maxFieldBlock = maxFieldBlock;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::deleteScratch() const
    {
      d_fieldCellValues.resize(0);
      d_fxJxWxNBlock.resize(0);
      d_JxWxGradNBlock.resize(0);
      d_JxWxNBlock.resize(0);
      d_basisDataInCellRange.resize(0);
      d_basisGradientDataInCellRange.resize(0);
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

      utils::throwException(
        &feBasisDofHandler == d_feBasisDofHandler,
        "Mismatch in BasisDofHandler used in the BasisManager and the BasisDataStorage "
        "in FEBasisOperations inpterpolateWithBasisGrad.");

      const size_type numComponents = vectorData.getNumberComponents();

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
        numComponents == quadValuesContainer.getNumberComponents(),
        "The number of components of input vector do not match with that in the quadValuesContainer");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBasisDataStorage->getQuadratureRuleContainer();

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
      const bool zeroStrideB = d_sameQuadRuleInAllCells && (!d_variableDofsPerCell);
      size_type  cellLocalIdsOffset = 0;
      size_type       CStartOffset  = 0;
      const size_type cellBlockSize = d_maxCellBlock * ((d_maxFieldBlock != 0) ? d_maxFieldBlock/numComponents : 1);

      /** --- Storages --------- **/
      if(d_fieldCellValues.size() != cellBlockSize * d_maxDofInCell * numComponents)
        d_fieldCellValues.resize(cellBlockSize * d_maxDofInCell * numComponents);

      d_basisDataCellRangeCached = false;
      if(d_basisDataInCellRange.size() < d_maxQuadInCell * cellBlockSize * d_maxDofInCell)
        d_basisDataInCellRange.resize(d_maxQuadInCell * cellBlockSize * d_maxDofInCell,
                                  ValueTypeBasisData());
      /** --- Storages --------- **/

      for (size_type cellStartId = 0; cellStartId < d_numLocallyOwnedCells;
           cellStartId += cellBlockSize)
        {
          const size_type cellEndId =
            std::min(cellStartId + cellBlockSize, d_numLocallyOwnedCells);
          const size_type        numCellsInBlock = cellEndId - cellStartId;
          std::vector<size_type> numCellsInBlockDofs(numCellsInBlock, 0);
          std::copy(d_numCellDofs.begin() + cellStartId,
                    d_numCellDofs.begin() + cellEndId,
                    numCellsInBlockDofs.begin());

          const size_type numCumulativeDofsCellsInBlock =
            std::accumulate(numCellsInBlockDofs.begin(),
                            numCellsInBlockDofs.end(),
                            0);

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransfer;

          FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
            copyFieldToCellWiseData(vectorData.begin(),
                                    numComponents,
                                    itCellLocalIdsBegin + cellLocalIdsOffset,
                                    numCumulativeDofsCellsInBlock,
                                    d_fieldCellValues);

          std::vector<char>      transA(numCellsInBlock, 'N');
          std::vector<char>      transB(numCellsInBlock, 'N');
          std::vector<size_type> mSizes(numCellsInBlock, 0);
          std::vector<size_type> nSizes(numCellsInBlock, 0);
          std::vector<size_type> kSizes(numCellsInBlock, 0);
          std::vector<size_type> ldaSizes(numCellsInBlock, 0);
          std::vector<size_type> ldbSizes(numCellsInBlock, 0);
          std::vector<size_type> ldcSizes(numCellsInBlock, 0);
          std::vector<size_type> strideA(numCellsInBlock, 0);
          std::vector<size_type> strideB(numCellsInBlock, 0);
          std::vector<size_type> strideC(numCellsInBlock, 0);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              const size_type cellId = cellStartId + iCell;
              mSizes[iCell]          = numComponents;
              nSizes[iCell] = quadValuesContainer.nCellQuadraturePoints(cellId);
              kSizes[iCell] = numCellsInBlockDofs[iCell];
              ldaSizes[iCell] = mSizes[iCell];
              ldbSizes[iCell] = kSizes[iCell];
              ldcSizes[iCell] = mSizes[iCell];
              if (!zeroStrideB)
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
              strideC[iCell] = mSizes[iCell] * nSizes[iCell];
              strideA[iCell] = mSizes[iCell] * kSizes[iCell];
            }

          ValueTypeUnion                               alpha = 1.0;
          ValueTypeUnion                               beta  = 0.0;
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
            *(vectorData.getLinAlgOpContext().get());

          if (!zeroStrideB || cellStartId == 0)
            d_feBasisDataStorage->getBasisDataInCellRange(
              std::make_pair(cellStartId, cellEndId), d_basisDataInCellRange);

          ValueTypeUnion *C = quadValuesContainer.begin() + CStartOffset;
          linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace>(
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
            d_fieldCellValues.data(),
            ldaSizes.data(),
            d_basisDataInCellRange.data(),
            ldbSizes.data(),
            beta,
            C,
            ldcSizes.data(),
            linAlgOpContext);


          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              CStartOffset += mSizes[iCell] * nSizes[iCell];
              cellLocalIdsOffset += d_numCellDofs[cellStartId + iCell];
            }
        }
        deleteScratch();
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
        const std::pair<size_type, size_type>                 cellRange,
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                               ValueTypeBasisData>
          *quadValuesInCellRangePtr) const
    {
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
        &feBasisDofHandler == d_feBasisDofHandler,
        "Mismatch in BasisDofHandler used in the BasisManager and the BasisDataStorage "
        "in FEBasisOperations interpolate(cellRange).");

      const size_type numComponents = vectorData.getNumberComponents();
      auto itCellLocalIdsBegin =
        feBasisManager.locallyOwnedCellLocalDofIdsBegin();

      const size_type cellStartId     = cellRange.first;
      const size_type cellEndId       = cellRange.second;
      const size_type numCellsInRange = cellEndId - cellStartId;

      std::vector<size_type> numCellsInRangeDofs(numCellsInRange, 0);
      std::copy(d_numCellDofs.begin() + cellStartId,
                d_numCellDofs.begin() + cellEndId,
                numCellsInRangeDofs.begin());

      const size_type numCumulativeDofsCellsInRange =
        std::accumulate(numCellsInRangeDofs.begin(),
                        numCellsInRangeDofs.end(),
                        0);

      size_type cellLocalIdsOffset = 0;
      for (size_type iCell = 0; iCell < cellStartId; ++iCell)
        cellLocalIdsOffset += d_numCellDofs[iCell];

      const bool zeroStrideB = d_sameQuadRuleInAllCells && (!d_variableDofsPerCell);

      /** --- Storages --------- **/
      if(d_fieldCellValues.size() < numCellsInRange * d_maxDofInCell * numComponents)
        d_fieldCellValues.resize(numCellsInRange * d_maxDofInCell * numComponents);

      const size_type requiredBasisSize =
        d_maxQuadInCell * numCellsInRange * d_maxDofInCell;
      if (d_basisDataInCellRange.size() < requiredBasisSize)
        {
          d_basisDataInCellRange.resize(requiredBasisSize, ValueTypeBasisData());
          d_basisDataCellRangeCached = false;
        }
      /** --- Storages --------- **/

      if (!d_basisDataCellRangeCached || d_basisDataCellRange != cellRange)
        {
          d_feBasisDataStorage->getBasisDataInCellRange(cellRange,
                                                        d_basisDataInCellRange);
          d_basisDataCellRange       = cellRange;
          d_basisDataCellRangeCached = true;
        }

      FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
        copyFieldToCellWiseData(vectorData.begin(),
                                numComponents,
                                itCellLocalIdsBegin + cellLocalIdsOffset,
                                numCumulativeDofsCellsInRange,
                                d_fieldCellValues);

      std::vector<char>      transA(numCellsInRange, 'N');
      std::vector<char>      transB(numCellsInRange, 'N');
      std::vector<size_type> mSizes(numCellsInRange, 0);
      std::vector<size_type> nSizes(numCellsInRange, 0);
      std::vector<size_type> kSizes(numCellsInRange, 0);
      std::vector<size_type> ldaSizes(numCellsInRange, 0);
      std::vector<size_type> ldbSizes(numCellsInRange, 0);
      std::vector<size_type> ldcSizes(numCellsInRange, 0);
      std::vector<size_type> strideA(numCellsInRange, 0);
      std::vector<size_type> strideB(numCellsInRange, 0);
      std::vector<size_type> strideC(numCellsInRange, 0);

      for (size_type iCell = 0; iCell < numCellsInRange; ++iCell)
        {
          const size_type cellId = cellStartId + iCell;
          mSizes[iCell]          = numComponents;
          nSizes[iCell]          = d_numCellQuad[cellId];
          kSizes[iCell]          = numCellsInRangeDofs[iCell];
          ldaSizes[iCell]        = mSizes[iCell];
          ldbSizes[iCell]        = kSizes[iCell];
          ldcSizes[iCell]        = mSizes[iCell];
          if (!zeroStrideB)
            strideB[iCell] = kSizes[iCell] * nSizes[iCell];
          strideC[iCell] = mSizes[iCell] * nSizes[iCell];
          strideA[iCell] = mSizes[iCell] * kSizes[iCell];
        }

      ValueTypeUnion alpha = 1.0;
      ValueTypeUnion beta  = 0.0;
      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
        *(vectorData.getLinAlgOpContext().get());

      linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace>(
        numCellsInRange,
        transA.data(),
        transB.data(),
        strideA.data(),
        strideB.data(),
        strideC.data(),
        mSizes.data(),
        nSizes.data(),
        kSizes.data(),
        alpha,
        d_fieldCellValues.data(),
        ldaSizes.data(),
        d_basisDataInCellRange.data(),
        ldbSizes.data(),
        beta,
        quadValuesInCellRangePtr,
        ldcSizes.data(),
        linAlgOpContext);
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

      utils::throwException(
        &feBasisDofHandler == d_feBasisDofHandler,
        "Mismatch in BasisDofHandler used in the BasisManager and the BasisDataStorage "
        "in FEBasisOperations inpterpolateWithBasisGrad.");

      const size_type numComponents = vectorData.getNumberComponents();

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
      // quadValuesContainer.reinit(quadRuleContainer,
      //                            numComponents * dim,
      //                            ValueTypeUnion());

      const quadrature::QuadratureFamily quadratureFamily =
        quadratureRuleAttributes.getQuadratureFamily();

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

      size_type cellLocalIdsOffset = 0;
      size_type       CStartOffset  = 0;
      const size_type cellBlockSize = d_maxCellBlock * ((d_maxFieldBlock != 0) ? d_maxFieldBlock/numComponents : 1);
      /** --- Storages --------- **/
      if(d_fieldCellValues.size() != cellBlockSize * d_maxDofInCell * numComponents)
        d_fieldCellValues.resize(cellBlockSize * d_maxDofInCell * numComponents);

      if(d_basisGradientDataInCellRange.size() != d_maxQuadInCell * cellBlockSize * d_maxDofInCell * dim)
        d_basisGradientDataInCellRange.resize(d_maxQuadInCell * cellBlockSize * d_maxDofInCell * dim,
                                  ValueTypeBasisData());
      /** --- Storages --------- **/

      for (size_type cellStartId = 0; cellStartId < d_numLocallyOwnedCells;
           cellStartId += cellBlockSize)
        {
          const size_type cellEndId =
            std::min(cellStartId + cellBlockSize, d_numLocallyOwnedCells);
          const size_type        numCellsInBlock = cellEndId - cellStartId;
          std::vector<size_type> numCellsInBlockDofs(numCellsInBlock, 0);
          std::copy(d_numCellDofs.begin() + cellStartId,
                    d_numCellDofs.begin() + cellEndId,
                    numCellsInBlockDofs.begin());

          const size_type numCumulativeDofsCellsInBlock =
            std::accumulate(numCellsInBlockDofs.begin(),
                            numCellsInBlockDofs.end(),
                            0);

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransfer;

          FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
            copyFieldToCellWiseData(vectorData.begin(),
                                    numComponents,
                                    itCellLocalIdsBegin + cellLocalIdsOffset,
                                    numCumulativeDofsCellsInBlock,
                                    d_fieldCellValues);

          std::vector<char>      transA(numCellsInBlock, 'N');
          std::vector<char>      transB(numCellsInBlock, 'N');
          std::vector<size_type> mSizes(numCellsInBlock, 0);
          std::vector<size_type> nSizes(numCellsInBlock, 0);
          std::vector<size_type> kSizes(numCellsInBlock, 0);
          std::vector<size_type> ldaSizes(numCellsInBlock, 0);
          std::vector<size_type> ldbSizes(numCellsInBlock, 0);
          std::vector<size_type> ldcSizes(numCellsInBlock, 0);
          std::vector<size_type> strideA(numCellsInBlock, 0);
          std::vector<size_type> strideB(numCellsInBlock, 0);
          std::vector<size_type> strideC(numCellsInBlock, 0);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              const size_type cellId = cellStartId + iCell;
              size_type       index  = iCell;
              mSizes[index]          = numComponents;
              nSizes[index] =
                quadValuesContainer.nCellQuadraturePoints(cellId) * dim;
              kSizes[index]   = numCellsInBlockDofs[iCell];
              ldaSizes[index] = mSizes[index];
              ldbSizes[index] = kSizes[index];
              ldcSizes[index] = mSizes[index];
              strideA[index]  = mSizes[index] * kSizes[index];
              strideC[index]  = mSizes[index] * nSizes[index];
              strideB[index]  = kSizes[index] * nSizes[index];
            }

          ValueTypeUnion                               alpha = 1.0;
          ValueTypeUnion                               beta  = 0.0;
          linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
            *(vectorData.getLinAlgOpContext().get());

          d_feBasisDataStorage->getBasisGradientDataInCellRange(
            std::make_pair(cellStartId, cellEndId), d_basisGradientDataInCellRange);

          ValueTypeUnion *C = quadValuesContainer.begin() + CStartOffset;
          linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace>(
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
            d_fieldCellValues.data(),
            ldaSizes.data(),
            d_basisGradientDataInCellRange.data(),
            ldbSizes.data(),
            beta,
            C,
            ldcSizes.data(),
            linAlgOpContext);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              size_type index = iCell;
              CStartOffset += mSizes[index] * nSizes[index];
              cellLocalIdsOffset += d_numCellDofs[cellStartId + iCell];
            }
        }
        deleteScratch();
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
          memorySpace> &                                      f,
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &vectorData) const

    {
      quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        d_feBasisDataStorage->getQuadratureRuleContainer()
          ->getQuadratureRuleAttributes();
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = f.getQuadratureRuleContainer();
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
      utils::throwException(
        &basisDofHandlerField == d_feBasisDofHandler,
        "Mismatch in BasisDofHandler used in the Field and the BasisDataStorage "
        "in FEBasisOperations integrateWithBasisValues().");


      linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
        *(vectorData.getLinAlgOpContext().get());
      auto jxwStorage = d_feBasisDataStorage->getJxWInAllCells();

      const size_type numComponents = f.getNumberComponents();
      utils::throwException(
        vectorData.getNumberComponents() == numComponents,
        "Mismatch in number of components in input and output "
        "in FEBasisOperations integrateWithBasisValues().");

      auto itCellLocalIdsBegin =
        feBasisManager.locallyOwnedCellLocalDofIdsBegin();

      const bool zeroStrideB = d_sameQuadRuleInAllCells && (!d_variableDofsPerCell);
      linearAlgebra::blasLapack::Layout layout =
        linearAlgebra::blasLapack::Layout::ColMajor;
      size_type cellLocalIdsOffset = 0;
      size_type CStartOffset = 0;

      vectorData.setValue((ValueTypeBasisCoeff)0);
      const size_type cellBlockSize = d_maxCellBlock * ((d_maxFieldBlock != 0) ? d_maxFieldBlock/numComponents : 1);

      /** --- Storages --------- **/
      if(d_JxWxNBlock.size() != d_maxQuadInCell * cellBlockSize * d_maxDofInCell)
        d_JxWxNBlock.resize(d_maxQuadInCell * cellBlockSize * d_maxDofInCell, ValueTypeBasisData());

      if(d_fieldCellValues.size() != cellBlockSize * d_maxDofInCell * numComponents)
        d_fieldCellValues.resize(cellBlockSize * d_maxDofInCell * numComponents,
                              ValueTypeUnion());

      d_basisDataCellRangeCached = false;
      if(d_basisDataInCellRange.size() < d_maxQuadInCell * cellBlockSize * d_maxDofInCell)
        d_basisDataInCellRange.resize(d_maxQuadInCell * cellBlockSize * d_maxDofInCell,
                                  ValueTypeBasisData());
      /** --- Storages --------- **/

      for (size_type cellStartId = 0; cellStartId < d_numLocallyOwnedCells;
           cellStartId += cellBlockSize)
        {
          const size_type cellEndId =
            std::min(cellStartId + cellBlockSize, d_numLocallyOwnedCells);
          const size_type        numCellsInBlock = cellEndId - cellStartId;
          std::vector<size_type> numCellsInBlockDofs(numCellsInBlock, 0);
          std::copy(d_numCellDofs.begin() + cellStartId,
                    d_numCellDofs.begin() + cellEndId,
                    numCellsInBlockDofs.begin());

          const size_type numCumulativeDofsCellsInBlock =
            std::accumulate(numCellsInBlockDofs.begin(),
                            numCellsInBlockDofs.end(),
                            0);

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
            memoryTransfer;

          std::vector<size_type> numCellsInBlockQuad(numCellsInBlock, 0);
          std::copy(d_numCellQuad.begin() + cellStartId,
                    d_numCellQuad.begin() + cellEndId,
                    numCellsInBlockQuad.begin());

          const size_type numCumulativeQuadCellsInBlock =
            std::accumulate(numCellsInBlockQuad.begin(),
                            numCellsInBlockQuad.end(),
                            0);

          linearAlgebra::blasLapack::ScalarOp scalarOpA =
            linearAlgebra::blasLapack::ScalarOp::Identity;
          linearAlgebra::blasLapack::ScalarOp scalarOpB =
            linearAlgebra::blasLapack::ScalarOp::Identity;
          std::vector<size_type> m(numCellsInBlock, 0);
          std::vector<size_type> n(numCellsInBlock, 0);
          std::vector<size_type> k(numCellsInBlock, 0);
          std::vector<size_type> stA(numCellsInBlock, 0);
          std::vector<size_type> stB(numCellsInBlock, 0);
          std::vector<size_type> stC(numCellsInBlock, 0);

          if (!zeroStrideB || cellStartId == 0)
            d_feBasisDataStorage->getBasisDataInCellRange(
              std::make_pair(cellStartId, cellEndId), d_basisDataInCellRange);

          for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
            {
              m[iCell]   = 1;
              n[iCell]   = numCellsInBlockDofs[iCell];
              k[iCell]   = numCellsInBlockQuad[iCell];
              stA[iCell] = m[iCell] * k[iCell];
              stB[iCell] = zeroStrideB ? 0 : n[iCell] * k[iCell];
              stC[iCell] = m[iCell] * n[iCell] * k[iCell];
            }

          linearAlgebra::blasLapack::scaleStridedVarBatched<
            ValueTypeBasisData,
            ValueTypeBasisData,
            memorySpace>(
            numCellsInBlock,
            layout,
            scalarOpA,
            scalarOpB,
            stA.data(),
            stB.data(),
            stC.data(),
            m.data(),
            n.data(),
            k.data(),
            jxwStorage.data() +
              quadRuleContainer->getCellQuadStartId(cellStartId),
            d_basisDataInCellRange.data(),
            d_JxWxNBlock.data(),
            linAlgOpContext);

          // TODO check if these are right ?? Why is the B Transposed
          std::vector<char>      transA(numCellsInBlock, 'N');
          std::vector<char>      transB(numCellsInBlock, 'T');
          std::vector<size_type> mSizes(numCellsInBlock, 0);
          std::vector<size_type> nSizes(numCellsInBlock, 0);
          std::vector<size_type> kSizes(numCellsInBlock, 0);
          std::vector<size_type> ldaSizes(numCellsInBlock, 0);
          std::vector<size_type> ldbSizes(numCellsInBlock, 0);
          std::vector<size_type> ldcSizes(numCellsInBlock, 0);
          std::vector<size_type> strideA(numCellsInBlock, 0);
          std::vector<size_type> strideB(numCellsInBlock, 0);
          std::vector<size_type> strideC(numCellsInBlock, 0);

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              const size_type cellId = cellStartId + iCell;
              mSizes[iCell]          = numComponents;
              nSizes[iCell]          = numCellsInBlockDofs[iCell];
              kSizes[iCell]          = numCellsInBlockQuad[iCell];
              ldaSizes[iCell]        = mSizes[iCell];
              ldbSizes[iCell]        = nSizes[iCell];
              ldcSizes[iCell]        = mSizes[iCell];
              strideA[iCell]         = mSizes[iCell] * kSizes[iCell];
              strideC[iCell]         = mSizes[iCell] * nSizes[iCell];
              strideB[iCell] = kSizes[iCell] * nSizes[iCell];
            }

          ValueTypeUnion alpha = 1.0;
          ValueTypeUnion beta  = 0.0;

          const ValueTypeBasisData *B = d_JxWxNBlock.data();

          ValueTypeUnion *C = d_fieldCellValues.begin();
          linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace>(
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
            f.begin(cellStartId),
            ldaSizes.data(),
            B,
            ldbSizes.data(),
            beta,
            C,
            ldcSizes.data(),
            linAlgOpContext);


          FECellWiseDataOperations<ValueTypeBasisCoeff, memorySpace>::
            addCellWiseDataToFieldData(d_fieldCellValues,
                                       numComponents,
                                       itCellLocalIdsBegin + cellLocalIdsOffset,
                                       numCumulativeDofsCellsInBlock,
                                       vectorData.begin());

          for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
            {
              CStartOffset += mSizes[iCell] * nSizes[iCell];
              cellLocalIdsOffset += d_numCellDofs[cellStartId + iCell];
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
      deleteScratch();
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
      computeFEMatrices(
        realspace::LinearLocalOp L1,
        realspace::VectorMathOp  Op1,
        realspace::VectorMathOp  Op2,
        realspace::LinearLocalOp L2,
        const quadrature::QuadratureValuesContainer<ValueTypeUnion, memorySpace>
          &                                          f,
        StorageUnion &                               cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext) const
    {
      utils::throwException(
        f.getNumberComponents() == 1 /*|| f.getNumberComponents() == dim*/,
        "quadValuesContainer f in computeFEMatrices"
        " can be only a scalar field in real space with 1 component.");

      const size_type cellBlockSize = d_maxCellBlock;
      BasisWeakFormKernelWithField(L1,
             Op1,
             Op2,
             L2,
             f,
             d_feBasisDataStorage,
             cellBlockSize,
             cellWiseFEData,
             linAlgOpContext);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>::computeFEMatrices(realspace::LinearLocalOp L1,
                                              realspace::VectorMathOp  Op1,
                                              realspace::LinearLocalOp L2,
                                              StorageBasis &cellWiseFEData,
                                              linearAlgebra::LinAlgOpContext<
                                                memorySpace> &linAlgOpContext) const
    {
      BasisWeakFormKernel(L1,
                          Op1,
                          L2,
                          d_feBasisDataStorage,
                          d_maxCellBlock,
                          cellWiseFEData,
                          linAlgOpContext);
    }

  } // namespace basis
} // namespace dftefe
