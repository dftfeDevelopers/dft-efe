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
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <basis/FECellWiseDataOperations.h>
#include <utils/OptimizedIndexSet.h>
#include <unordered_map>
#include <vector>
#include <basis/EnrichmentClassicalInterfaceSpherical.h>

namespace dftefe
{
  namespace basis
  {
    namespace OrthoEFEOverlapOperatorContextInternal
    {
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeBasisOverlapMatrix(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &cfeBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &efeBasisDataStorage,
        std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
          &                     basisOverlap,
        std::vector<size_type> &cellStartIdsBasisOverlap,
        std::vector<size_type> &dofsInCellVec,
        bool                    calculateWings = true)
      {
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          cfeBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            cfeBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          cfeBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in OrthoEFEOverlapOperatorContext");

        std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                                 ValueTypeOperator,
                                                 memorySpace,
                                                 dim>>
          efeBDH = std::dynamic_pointer_cast<
            const EFEBasisDofHandler<ValueTypeOperand,
                                     ValueTypeOperator,
                                     memorySpace,
                                     dim>>(
            efeBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          efeBDH != nullptr,
          "Could not cast BasisDofHandler to EFEBasisDofHandler "
          "in OrthoEFEOverlapOperatorContext");

        // NOTE: cellId 0 passed as we assume only H refined in this function

        utils::throwException(
          cfeBDH->getTriangulation() == efeBDH->getTriangulation() &&
            cfeBDH->getFEOrder(0) == efeBDH->getFEOrder(0),
          "The EFEBasisDataStorage and and Classical FEBasisDataStorage have different triangulation or FEOrder. ");

        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        dofsInCellVec.resize(numLocallyOwnedCells, 0);
        cellStartIdsBasisOverlap.resize(numLocallyOwnedCells, 0);
        size_type cumulativeBasisOverlapId = 0;

        size_type       basisOverlapSize = 0;
        size_type       cellId           = 0;
        const size_type feOrder          = efeBDH->getFEOrder(cellId);

        size_type       dofsPerCell;
        const size_type dofsPerCellCFE = cfeBDH->nCellDofs(cellId);

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = efeBDH->nCellDofs(cellId);
            basisOverlapSize += dofsInCellVec[cellId] * dofsInCellVec[cellId];
            cellId++;
          }

        std::vector<ValueTypeOperator> basisOverlapTmp(0);

        basisOverlap = std::make_shared<
          utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          basisOverlapSize);
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeOperator(0));

        auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex           = 0;

        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            const utils::MemoryStorage<ValueTypeOperator, memorySpace>
              &basisDataInCellCFE =
                cfeBasisDataStorage.getBasisDataInCell(cellIndex);
            const utils::MemoryStorage<ValueTypeOperator, memorySpace>
              &basisDataInCellEFE =
                efeBasisDataStorage.getBasisDataInCell(cellIndex);

            dofsPerCell = dofsInCellVec[cellIndex];
            size_type nQuadPointInCellCFE =
              cfeBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesCFE =
              cfeBasisDataStorage.getQuadratureRuleContainer()->getCellJxW(
                cellIndex);

            size_type nQuadPointInCellEFE =
              efeBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEFE =
              efeBasisDataStorage.getQuadratureRuleContainer()->getCellJxW(
                cellIndex);

            const ValueTypeOperator *cumulativeCFEDofQuadPoints =
              basisDataInCellCFE.data();

            const ValueTypeOperator *cumulativeEFEDofQuadPoints =
              basisDataInCellEFE.data();

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisOverlapTmpIter = 0.0;
                    if (iNode < dofsPerCellCFE && jNode < dofsPerCellCFE)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellCFE;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeCFEDofQuadPoints +
                                dofsPerCellCFE * qPoint + iNode
                                /*nQuadPointInCellCFE * iNode + qPoint*/) *
                              *(cumulativeCFEDofQuadPoints +
                                dofsPerCellCFE * qPoint + jNode
                                /*nQuadPointInCellCFE * jNode + qPoint*/) *
                              cellJxWValuesCFE[qPoint];
                          }
                      }
                    else if (((iNode >= dofsPerCellCFE &&
                                 jNode < dofsPerCellCFE ||
                               iNode < dofsPerCellCFE &&
                                 jNode >= dofsPerCellCFE) &&
                              calculateWings) ||
                             iNode >= dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEFE;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeEFEDofQuadPoints +
                                dofsPerCell * qPoint + iNode
                                /*nQuadPointInCellEFE * iNode + qPoint*/) *
                              *(cumulativeEFEDofQuadPoints +
                                dofsPerCell * qPoint + jNode
                                /*nQuadPointInCellEFE * jNode + qPoint*/) *
                              cellJxWValuesEFE[qPoint];
                          }
                      }
                    basisOverlapTmpIter++;
                  }
              }

            cellStartIdsBasisOverlap[cellIndex] = cumulativeBasisOverlapId;
            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(), basisOverlap->data(), basisOverlapTmp.data());
      }

      // Use this for data storage of orthogonalized EFE only
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeBasisOverlapMatrix(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
          &                                          basisOverlap,
        std::vector<size_type> &                     cellStartIdsBasisOverlap,
        std::vector<size_type> &                     dofsInCellVec,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        bool                                         calculateWings = true)
      {
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          ccfeBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            classicalBlockBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          ccfeBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in OrthoEFEOverlapOperatorContext for the Classical data storage of classical dof block.");

        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          ecfeBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            enrichmentBlockClassicalBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          ecfeBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in OrthoEFEOverlapOperatorContext for the Classical data storage of enrichment dof blocks.");

        std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                                 ValueTypeOperator,
                                                 memorySpace,
                                                 dim>>
          eefeBDH = std::dynamic_pointer_cast<
            const EFEBasisDofHandler<ValueTypeOperand,
                                     ValueTypeOperator,
                                     memorySpace,
                                     dim>>(
            enrichmentBlockEnrichmentBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          eefeBDH != nullptr,
          "Could not cast BasisDofHandler to EFEBasisDofHandler "
          "in OrthoEFEOverlapOperatorContext for the Enrichment data storage of enrichment dof blocks.");

        utils::throwException(
          ccfeBDH->getTriangulation() == ecfeBDH->getTriangulation() &&
            ccfeBDH->getFEOrder(0) == ecfeBDH->getFEOrder(0) &&
            ccfeBDH->getTriangulation() == eefeBDH->getTriangulation() &&
            ccfeBDH->getFEOrder(0) == eefeBDH->getFEOrder(0),
          "The EFEBasisDataStorage and and Classical FEBasisDataStorage have different triangulation or FEOrder"
          "in OrthoEFEOverlapOperatorContext.");

        utils::throwException(
          eefeBDH->isOrthogonalized(),
          "The Enrcihment data storage of enrichment dof blocks should have isOrthogonalized as true in OrthoEFEOverlapOperatorContext.");

        std::shared_ptr<
          const EnrichmentClassicalInterfaceSpherical<ValueTypeOperator,
                                                      memorySpace,
                                                      dim>>
          eci = eefeBDH->getEnrichmentClassicalInterface();

        size_type nTotalEnrichmentIds =
          eci->getEnrichmentIdsPartition()->nTotalEnrichmentIds();

        // interpolate the ci 's to the enrichment quadRuleAttr quadpoints

        const EFEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorageEFE = dynamic_cast<
            const EFEBasisDataStorage<ValueTypeOperator, memorySpace> &>(
            enrichmentBlockEnrichmentBasisDataStorage);
        utils::throwException(
          &enrichmentBlockEnrichmentBasisDataStorageEFE != nullptr,
          "Could not cast FEBasisDataStorage to EFEBasisDataStorage "
          "in OrthoEFEOverlapOperatorContext for enrichmentBlockEnrichmentBasisDataStorage.");

        // Set up the overlap matrix quadrature storages.

        const size_type numLocallyOwnedCells = eefeBDH->nLocallyOwnedCells();
        dofsInCellVec.resize(numLocallyOwnedCells, 0);
        cellStartIdsBasisOverlap.resize(numLocallyOwnedCells, 0);
        size_type cumulativeBasisOverlapId = 0;

        size_type       basisOverlapSize = 0;
        size_type       cellId           = 0;
        const size_type feOrder          = eefeBDH->getFEOrder(cellId);

        size_type       dofsPerCell;
        const size_type dofsPerCellCFE = ccfeBDH->nCellDofs(cellId);

        auto locallyOwnedCellIter = eefeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != eefeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = eefeBDH->nCellDofs(cellId);
            basisOverlapSize += dofsInCellVec[cellId] * dofsInCellVec[cellId];
            cellId++;
          }

        std::vector<ValueTypeOperator> basisOverlapTmp(0);

        basisOverlap = std::make_shared<
          utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          basisOverlapSize);
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeOperator(0));

        auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex           = 0;

        //
        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap =
            &eci->getClassicalComponentLocalIdsMap();
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeOperator>>
          *enrichmentIdToInterfaceCoeffMap =
            &eci->getClassicalComponentCoeffMap();
        std::shared_ptr<const FEBasisManager<ValueTypeOperator,
                                             ValueTypeOperator,
                                             memorySpace,
                                             dim>>
          cfeBasisManager =
            std::dynamic_pointer_cast<const FEBasisManager<ValueTypeOperator,
                                                           ValueTypeOperator,
                                                           memorySpace,
                                                           dim>>(
              eci->getCFEBasisManager());

        locallyOwnedCellIter = eefeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != eefeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            const utils::MemoryStorage<ValueTypeOperator, memorySpace>
              &basisDataInCellClassicalBlock =
                classicalBlockBasisDataStorage.getBasisDataInCell(cellIndex);
            const utils::MemoryStorage<ValueTypeOperator, memorySpace>
              &basisDataInCellEnrichmentBlockClassical =
                enrichmentBlockClassicalBasisDataStorage.getBasisDataInCell(
                  cellIndex);
            const utils::MemoryStorage<ValueTypeOperator, memorySpace>
              &basisDataInCellEnrichmentBlockEnrichment =
                enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInCell(
                  cellIndex);

            dofsPerCell = dofsInCellVec[cellIndex];
            size_type nQuadPointInCellClassicalBlock =
              classicalBlockBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesClassicalBlock =
              classicalBlockBasisDataStorage.getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);

            size_type nQuadPointInCellEnrichmentBlockClassical =
              enrichmentBlockClassicalBasisDataStorage
                .getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEnrichmentBlockClassical =
              enrichmentBlockClassicalBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);

            size_type nQuadPointInCellEnrichmentBlockEnrichment =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEnrichmentBlockEnrichment =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);

            const ValueTypeOperator *cumulativeClassicalBlockDofQuadPoints =
              basisDataInCellClassicalBlock.data();

            const ValueTypeOperator
              *cumulativeEnrichmentBlockClassicalDofQuadPoints =
                basisDataInCellEnrichmentBlockClassical.data();

            const ValueTypeOperator
              *cumulativeEnrichmentBlockEnrichmentDofQuadPoints =
                basisDataInCellEnrichmentBlockEnrichment.data();

            std::vector<utils::Point> quadRealPointsVec =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellRealPoints(cellIndex);

            std::vector<size_type> vecClassicalLocalNodeId(0);

            size_type numEnrichmentIdsInCell = dofsPerCell - dofsPerCellCFE;

            std::vector<ValueTypeOperator> classicalComponentInQuadValuesEC(0);

            classicalComponentInQuadValuesEC.resize(
              nQuadPointInCellEnrichmentBlockClassical * numEnrichmentIdsInCell,
              (ValueTypeOperator)0);


            std::vector<ValueTypeOperator> classicalComponentInQuadValuesEE(0);

            classicalComponentInQuadValuesEE.resize(
              nQuadPointInCellEnrichmentBlockEnrichment *
                numEnrichmentIdsInCell,
              (ValueTypeOperator)0);


            std::vector<ValueTypeOperator> enrichmentValuesVec(
              numEnrichmentIdsInCell *
                nQuadPointInCellEnrichmentBlockEnrichment,
              0);

            if (numEnrichmentIdsInCell > 0)
              {
                cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                                     vecClassicalLocalNodeId);

                std::vector<ValueTypeOperator> coeffsInCell(
                  dofsPerCellCFE * numEnrichmentIdsInCell, 0);

                for (size_type cellEnrichId = 0;
                     cellEnrichId < numEnrichmentIdsInCell;
                     cellEnrichId++)
                  {
                    // get the enrichmentIds
                    global_size_type enrichmentId =
                      eci->getEnrichmentId(cellIndex, cellEnrichId);

                    // get the vectors of non-zero localIds and coeffs
                    auto iter =
                      enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
                    auto it =
                      enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
                    if (iter != enrichmentIdToInterfaceCoeffMap->end() &&
                        it != enrichmentIdToClassicalLocalIdMap->end())
                      {
                        const std::vector<ValueTypeOperator>
                          &coeffsInLocalIdsMap = iter->second;

                        for (size_type i = 0; i < dofsPerCellCFE; i++)
                          {
                            size_type pos   = 0;
                            bool      found = false;
                            it->second.getPosition(vecClassicalLocalNodeId[i],
                                                   pos,
                                                   found);
                            if (found)
                              {
                                coeffsInCell[numEnrichmentIdsInCell * i +
                                             cellEnrichId] =
                                  coeffsInLocalIdsMap[pos];
                              }
                          }
                      }
                  }

                dftefe::utils::MemoryStorage<ValueTypeOperator, memorySpace>
                  basisValInCellEC =
                    enrichmentBlockClassicalBasisDataStorage.getBasisDataInCell(
                      cellIndex);

                ValueTypeOperator *B = basisValInCellEC.data();
                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockClassical,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  B,
                  dofsPerCellCFE,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEC.data(),
                  numEnrichmentIdsInCell,
                  *eci->getLinAlgOpContext());


                dftefe::utils::MemoryStorage<ValueTypeOperator, memorySpace>
                  basisValInCellEE = enrichmentBlockEnrichmentBasisDataStorage
                                       .getBasisDataInCell(cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector
                B = basisValInCellEE.data();

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockEnrichment,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  basisValInCellEE.data(),
                  dofsPerCell,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEE.data(),
                  numEnrichmentIdsInCell,
                  *eci->getLinAlgOpContext());

                const std::vector<double> &enrichValAtQuadPts =
                  eefeBDH->getEnrichmentValue(cellIndex, quadRealPointsVec);
                for (size_type i = 0; i < numEnrichmentIdsInCell; i++)
                  {
                    for (unsigned int qPoint = 0;
                         qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                         qPoint++)
                      {
                        *(enrichmentValuesVec.data() + numEnrichmentIdsInCell * qPoint
                           + i
                          /*nQuadPointInCellEnrichmentBlockEnrichment * i +
                          qPoint*/) =
                            *(enrichValAtQuadPts.data() + nQuadPointInCellEnrichmentBlockEnrichment * i + qPoint);
                      }
                  }
              }

            std::vector<ValueTypeOperator> basisOverlapClassicalBlock(
              dofsPerCellCFE * dofsPerCellCFE);
            std::vector<ValueTypeOperator> JxWxNCell(
              dofsPerCellCFE * nQuadPointInCellClassicalBlock, 0);

            size_type stride = 0;
            size_type m = 1, n = dofsPerCellCFE,
                      k = nQuadPointInCellClassicalBlock;

            linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                              ValueTypeOperator,
                                                              memorySpace>(
              1,
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              &stride,
              &stride,
              &stride,
              &m,
              &n,
              &k,
              cellJxWValuesClassicalBlock.data(),
              cumulativeClassicalBlockDofQuadPoints,
              JxWxNCell.data(),
              linAlgOpContext);

            linearAlgebra::blasLapack::
              gemm<ValueTypeOperand, ValueTypeOperand, memorySpace>(
                linearAlgebra::blasLapack::Layout::ColMajor,
                linearAlgebra::blasLapack::Op::NoTrans,
                linearAlgebra::blasLapack::Op::ConjTrans,
                n,
                n,
                k,
                (ValueTypeOperand)1.0,
                JxWxNCell.data(),
                n,
                cumulativeClassicalBlockDofQuadPoints,
                n,
                (ValueTypeOperand)0.0,
                basisOverlapClassicalBlock.data(),
                n,
                linAlgOpContext);

            std::vector<ValueTypeOperator> basisOverlapECBlockEnrich(
              dofsPerCell * numEnrichmentIdsInCell);

            std::vector<ValueTypeOperator> basisOverlapECBlockClass(
              dofsPerCellCFE * numEnrichmentIdsInCell);

            if (numEnrichmentIdsInCell > 0)
              {
                JxWxNCell.resize(dofsPerCell *
                                   nQuadPointInCellEnrichmentBlockEnrichment,
                                 0);

                m = 1, n = dofsPerCell,
                k = nQuadPointInCellEnrichmentBlockEnrichment;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  memorySpace>(1,
                               linearAlgebra::blasLapack::Layout::ColMajor,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               &stride,
                               &stride,
                               &stride,
                               &m,
                               &n,
                               &k,
                               cellJxWValuesEnrichmentBlockEnrichment.data(),
                               cumulativeEnrichmentBlockEnrichmentDofQuadPoints,
                               JxWxNCell.data(),
                               linAlgOpContext);

                linearAlgebra::blasLapack::
                  gemm<ValueTypeOperand, ValueTypeOperand, memorySpace>(
                    linearAlgebra::blasLapack::Layout::ColMajor,
                    linearAlgebra::blasLapack::Op::NoTrans,
                    linearAlgebra::blasLapack::Op::ConjTrans,
                    dofsPerCell,
                    numEnrichmentIdsInCell,
                    k,
                    (ValueTypeOperand)1.0,
                    JxWxNCell.data(),
                    n,
                    enrichmentValuesVec.data(),
                    numEnrichmentIdsInCell,
                    (ValueTypeOperand)0.0,
                    basisOverlapECBlockEnrich.data(),
                    dofsPerCell,
                    linAlgOpContext);

                JxWxNCell.resize(dofsPerCellCFE *
                                   nQuadPointInCellEnrichmentBlockClassical,
                                 0);

                m = 1, n = dofsPerCellCFE,
                k = nQuadPointInCellEnrichmentBlockClassical;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  memorySpace>(1,
                               linearAlgebra::blasLapack::Layout::ColMajor,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               &stride,
                               &stride,
                               &stride,
                               &m,
                               &n,
                               &k,
                               cellJxWValuesEnrichmentBlockClassical.data(),
                               cumulativeEnrichmentBlockClassicalDofQuadPoints,
                               JxWxNCell.data(),
                               linAlgOpContext);

                linearAlgebra::blasLapack::
                  gemm<ValueTypeOperand, ValueTypeOperator, memorySpace>(
                    linearAlgebra::blasLapack::Layout::ColMajor,
                    linearAlgebra::blasLapack::Op::NoTrans,
                    linearAlgebra::blasLapack::Op::ConjTrans,
                    n,
                    numEnrichmentIdsInCell,
                    k,
                    (ValueTypeOperand)1.0,
                    JxWxNCell.data(),
                    n,
                    classicalComponentInQuadValuesEC.data(),
                    numEnrichmentIdsInCell,
                    (ValueTypeOperator)0.0,
                    basisOverlapECBlockClass.data(),
                    n,
                    linAlgOpContext);
              }

            std::vector<ValueTypeOperator> basisOverlapEEBlock1(
              numEnrichmentIdsInCell * numEnrichmentIdsInCell, 0),
              basisOverlapEEBlock2(numEnrichmentIdsInCell *
                                     numEnrichmentIdsInCell,
                                   0),
              basisOverlapEEBlock3(numEnrichmentIdsInCell *
                                     numEnrichmentIdsInCell,
                                   0);

            if (numEnrichmentIdsInCell > 0)
              {
                // Ni_pristine*Ni_pristine at quadpoints
                JxWxNCell.resize(numEnrichmentIdsInCell *
                                   nQuadPointInCellEnrichmentBlockEnrichment,
                                 0);

                m = 1, n = numEnrichmentIdsInCell,
                k = nQuadPointInCellEnrichmentBlockEnrichment;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  memorySpace>(1,
                               linearAlgebra::blasLapack::Layout::ColMajor,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               &stride,
                               &stride,
                               &stride,
                               &m,
                               &n,
                               &k,
                               cellJxWValuesEnrichmentBlockEnrichment.data(),
                               enrichmentValuesVec.data(),
                               JxWxNCell.data(),
                               linAlgOpContext);

                linearAlgebra::blasLapack::
                  gemm<ValueTypeOperand, ValueTypeOperand, memorySpace>(
                    linearAlgebra::blasLapack::Layout::ColMajor,
                    linearAlgebra::blasLapack::Op::NoTrans,
                    linearAlgebra::blasLapack::Op::ConjTrans,
                    n,
                    n,
                    k,
                    (ValueTypeOperand)1.0,
                    JxWxNCell.data(),
                    n,
                    enrichmentValuesVec.data(),
                    n,
                    (ValueTypeOperand)0.0,
                    basisOverlapEEBlock1.data(),
                    n,
                    linAlgOpContext);

                // interpolated ci's in Ni_classicalQuadrature of Mc = d
                // * interpolated ci's in Ni_classicalQuadrature of Mc =
                // d
                JxWxNCell.resize(numEnrichmentIdsInCell *
                                   nQuadPointInCellEnrichmentBlockClassical,
                                 0);

                m = 1, n = numEnrichmentIdsInCell,
                k = nQuadPointInCellEnrichmentBlockClassical;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  memorySpace>(1,
                               linearAlgebra::blasLapack::Layout::ColMajor,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               &stride,
                               &stride,
                               &stride,
                               &m,
                               &n,
                               &k,
                               cellJxWValuesEnrichmentBlockClassical.data(),
                               classicalComponentInQuadValuesEC.data(),
                               JxWxNCell.data(),
                               linAlgOpContext);

                linearAlgebra::blasLapack::
                  gemm<ValueTypeOperand, ValueTypeOperator, memorySpace>(
                    linearAlgebra::blasLapack::Layout::ColMajor,
                    linearAlgebra::blasLapack::Op::NoTrans,
                    linearAlgebra::blasLapack::Op::ConjTrans,
                    n,
                    n,
                    k,
                    (ValueTypeOperand)1.0,
                    JxWxNCell.data(),
                    n,
                    classicalComponentInQuadValuesEC.data(),
                    n,
                    (ValueTypeOperator)0.0,
                    basisOverlapEEBlock2.data(),
                    n,
                    linAlgOpContext);

                // Ni_pristine* interpolated ci's in
                // Ni_classicalQuadratureOfPristine at quadpoints

                JxWxNCell.resize(numEnrichmentIdsInCell *
                                   nQuadPointInCellEnrichmentBlockEnrichment,
                                 0);

                m = 1, n = numEnrichmentIdsInCell,
                k = nQuadPointInCellEnrichmentBlockEnrichment;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  memorySpace>(1,
                               linearAlgebra::blasLapack::Layout::ColMajor,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               linearAlgebra::blasLapack::ScalarOp::Identity,
                               &stride,
                               &stride,
                               &stride,
                               &m,
                               &n,
                               &k,
                               cellJxWValuesEnrichmentBlockEnrichment.data(),
                               enrichmentValuesVec.data(),
                               JxWxNCell.data(),
                               linAlgOpContext);

                linearAlgebra::blasLapack::
                  gemm<ValueTypeOperand, ValueTypeOperator, memorySpace>(
                    linearAlgebra::blasLapack::Layout::ColMajor,
                    linearAlgebra::blasLapack::Op::NoTrans,
                    linearAlgebra::blasLapack::Op::ConjTrans,
                    n,
                    n,
                    k,
                    (ValueTypeOperand)1.0,
                    JxWxNCell.data(),
                    n,
                    classicalComponentInQuadValuesEE.data(),
                    n,
                    (ValueTypeOperator)0.0,
                    basisOverlapEEBlock3.data(),
                    n,
                    linAlgOpContext);

                // linearAlgebra::blasLapack::
                //   gemm<ValueTypeOperand, ValueTypeOperator, memorySpace>(
                //     linearAlgebra::blasLapack::Layout::ColMajor,
                //     linearAlgebra::blasLapack::Op::Trans,
                //     linearAlgebra::blasLapack::Op::Trans,
                //     n,
                //     n,
                //     k,
                //     (ValueTypeOperand)1.0,
                //     classicalComponentInQuadValuesEC.data(),
                //     k,
                //     JxWxNCell.data(),
                //     n,
                //     (ValueTypeOperator)0.0,
                //     basisOverlapEEBlock4.data(),
                //     n,
                //     linAlgOpContext);
              }

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisOverlapTmpIter = 0.0;
                    // Ni_classical* Ni_classical of the classicalBlockBasisData
                    if (iNode < dofsPerCellCFE && jNode < dofsPerCellCFE)
                      {
                        *basisOverlapTmpIter =
                          *(basisOverlapClassicalBlock.data() +
                            iNode * dofsPerCellCFE + jNode);
                        // for (unsigned int qPoint = 0;
                        //      qPoint < nQuadPointInCellClassicalBlock;
                        //      qPoint++)
                        //   {
                        //     *basisOverlapTmpIter +=
                        //       *(cumulativeClassicalBlockDofQuadPoints +
                        //         dofsPerCellCFE * qPoint + iNode
                        //         /*nQuadPointInCellClassicalBlock * iNode +
                        //         qPoint*/) *
                        //       *(cumulativeClassicalBlockDofQuadPoints +
                        //         dofsPerCellCFE * qPoint + jNode
                        //         /*nQuadPointInCellClassicalBlock * jNode +
                        //         qPoint*/) *
                        //       cellJxWValuesClassicalBlock[qPoint];
                        //   }
                      }
                    else if (iNode >= dofsPerCellCFE &&
                             jNode < dofsPerCellCFE && calculateWings)
                      {
                        // ValueTypeOperator NpiNcj   = (ValueTypeOperator)0,
                        //                   ciNciNcj = (ValueTypeOperator)0;
                        // // Ni_pristine*Ni_classical at quadpoints
                        // for (unsigned int qPoint = 0;
                        //      qPoint <
                        //      nQuadPointInCellEnrichmentBlockEnrichment;
                        //      qPoint++)
                        //   {
                        //     NpiNcj +=
                        //       *(enrichmentValuesVec.data() +
                        //         (iNode - dofsPerCellCFE) *
                        //           nQuadPointInCellEnrichmentBlockEnrichment +
                        //         qPoint) *
                        //       *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints
                        //       +
                        //         dofsPerCell * qPoint + jNode
                        //         /*nQuadPointInCellEnrichmentBlockEnrichment *
                        //           jNode +
                        //         qPoint*/) *
                        //       cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                        //   }

                        // // Ni_classical using Mc = d quadrature *
                        // //interpolated
                        // // ci's in Ni_classicalQuadrature of Mc = d
                        // for (unsigned int qPoint = 0;
                        //      qPoint <
                        //      nQuadPointInCellEnrichmentBlockClassical;
                        //      qPoint++)
                        //   {
                        //     ciNciNcj +=
                        //       classicalComponentInQuadValuesEC
                        //         [numEnrichmentIdsInCell * qPoint +
                        //          (iNode - dofsPerCellCFE)] *
                        //       *(cumulativeEnrichmentBlockClassicalDofQuadPoints
                        //       +
                        //         dofsPerCellCFE * qPoint + jNode
                        //         /*nQuadPointInCellEnrichmentBlockClassical *
                        //           jNode + qPoint*/) *
                        //       cellJxWValuesEnrichmentBlockClassical[qPoint];
                        //   }
                        // *basisOverlapTmpIter += NpiNcj - ciNciNcj;

                        *basisOverlapTmpIter =
                          *(basisOverlapECBlockEnrich.data() +
                            (iNode - dofsPerCellCFE) * dofsPerCell + jNode) -
                          *(basisOverlapECBlockClass.data() +
                            (iNode - dofsPerCellCFE) * dofsPerCellCFE + jNode);
                      }

                    else if (iNode < dofsPerCellCFE &&
                             jNode >= dofsPerCellCFE && calculateWings)
                      {
                        // ValueTypeOperator NciNpj   = (ValueTypeOperator)0,
                        //                   NcicjNcj = (ValueTypeOperator)0;
                        // // Ni_pristine*Ni_classical at quadpoints
                        // for (unsigned int qPoint = 0;
                        //      qPoint <
                        //      nQuadPointInCellEnrichmentBlockEnrichment;
                        //      qPoint++)
                        //   {
                        //     NciNpj +=
                        //       *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints
                        //       +
                        //         dofsPerCell * qPoint + iNode
                        //         /*nQuadPointInCellEnrichmentBlockEnrichment *
                        //           iNode +
                        //         qPoint*/) *
                        //       *(enrichmentValuesVec.data() +
                        //         (jNode - dofsPerCellCFE) *
                        //           nQuadPointInCellEnrichmentBlockEnrichment +
                        //         qPoint) *
                        //       cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                        //   }

                        // // Ni_classical using Mc = d quadrature *
                        // //interpolated
                        // // ci's in Ni_classicalQuadrature of Mc = d
                        // for (unsigned int qPoint = 0;
                        //      qPoint <
                        //      nQuadPointInCellEnrichmentBlockClassical;
                        //      qPoint++)
                        //   {
                        //     NcicjNcj +=
                        //       *(cumulativeEnrichmentBlockClassicalDofQuadPoints
                        //       +
                        //         dofsPerCellCFE * qPoint + iNode
                        //         /*nQuadPointInCellEnrichmentBlockClassical *
                        //           iNode + qPoint*/) *
                        //       classicalComponentInQuadValuesEC
                        //         [numEnrichmentIdsInCell * qPoint +
                        //          (jNode - dofsPerCellCFE)] *
                        //       cellJxWValuesEnrichmentBlockClassical[qPoint];
                        //   }
                        // *basisOverlapTmpIter += NciNpj - NcicjNcj;

                        *basisOverlapTmpIter =
                          *(basisOverlapECBlockEnrich.data() +
                            (jNode - dofsPerCellCFE) * dofsPerCell + iNode) -
                          *(basisOverlapECBlockClass.data() +
                            (jNode - dofsPerCellCFE) * dofsPerCellCFE + iNode);
                      }

                    else if (iNode >= dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
                        /**
                        // Ni_pristine*Ni_pristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints
                        + nQuadPointInCellEnrichmentBlockEnrichment * iNode +
                                qPoint) *
                              *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints
                        + nQuadPointInCellEnrichmentBlockEnrichment * jNode +
                                qPoint) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        **/

                        /**
                        ValueTypeOperator NpiNpj     = (ValueTypeOperator)0,
                                          ciNciNpj   = (ValueTypeOperator)0,
                                          NpicjNcj   = (ValueTypeOperator)0,
                                          ciNcicjNcj = (ValueTypeOperator)0;
                        // Ni_pristine*Ni_pristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NpiNpj +=
                              *(enrichmentValuesVec.data() +
                                (iNode - dofsPerCellCFE) *
                                  nQuadPointInCellEnrichmentBlockEnrichment +
                                qPoint) *
                              *(enrichmentValuesVec.data() +
                                (jNode - dofsPerCellCFE) *
                                  nQuadPointInCellEnrichmentBlockEnrichment +
                                qPoint) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // Ni_pristine* interpolated ci's in
                        // Ni_classicalQuadratureOfPristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            ciNciNpj +=
                              classicalComponentInQuadValuesEE
                                [numEnrichmentIdsInCell * qPoint +
                                 (iNode - dofsPerCellCFE)] *
                              *(enrichmentValuesVec.data() +
                                (jNode - dofsPerCellCFE) *
                                  nQuadPointInCellEnrichmentBlockEnrichment +
                                qPoint) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // Ni_pristine* interpolated ci's in
                        // Ni_classicalQuadratureOfPristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NpicjNcj +=
                              *(enrichmentValuesVec.data() +
                                (iNode - dofsPerCellCFE) *
                                  nQuadPointInCellEnrichmentBlockEnrichment +
                                qPoint) *
                              classicalComponentInQuadValuesEE
                                [numEnrichmentIdsInCell * qPoint +
                                 (jNode - dofsPerCellCFE)] *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // interpolated ci's in Ni_classicalQuadrature of Mc = d
                        // * interpolated ci's in Ni_classicalQuadrature of Mc =
                        // d
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockClassical;
                             qPoint++)
                          {
                            ciNcicjNcj +=
                              classicalComponentInQuadValuesEC
                                [numEnrichmentIdsInCell * qPoint +
                                 (iNode - dofsPerCellCFE)] *
                              classicalComponentInQuadValuesEC
                                [numEnrichmentIdsInCell * qPoint +
                                 (jNode - dofsPerCellCFE)] *
                              cellJxWValuesEnrichmentBlockClassical[qPoint];
                          }
                        // *basisOverlapTmpIter +=
                        double b = NpiNpj - NpicjNcj - ciNciNpj + ciNcicjNcj;
                        **/

                        *basisOverlapTmpIter =
                          *(basisOverlapEEBlock1.data() +
                            (iNode - dofsPerCellCFE) * numEnrichmentIdsInCell +
                            (jNode - dofsPerCellCFE)) +
                          *(basisOverlapEEBlock2.data() +
                            (iNode - dofsPerCellCFE) * numEnrichmentIdsInCell +
                            (jNode - dofsPerCellCFE)) -
                          *(basisOverlapEEBlock3.data() +
                            (iNode - dofsPerCellCFE) * numEnrichmentIdsInCell +
                            (jNode - dofsPerCellCFE)) -
                          *(basisOverlapEEBlock3.data() +
                            (jNode - dofsPerCellCFE) * numEnrichmentIdsInCell +
                            (iNode - dofsPerCellCFE));
                      }
                    basisOverlapTmpIter++;
                  }
              }

            cellStartIdsBasisOverlap[cellIndex] = cumulativeBasisOverlapId;
            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(), basisOverlap->data(), basisOverlapTmp.data());
      }


      template <utils::MemorySpace memorySpace>
      void
      storeSizes(utils::MemoryStorage<size_type, memorySpace> &mSizes,
                 utils::MemoryStorage<size_type, memorySpace> &nSizes,
                 utils::MemoryStorage<size_type, memorySpace> &kSizes,
                 utils::MemoryStorage<size_type, memorySpace> &ldaSizes,
                 utils::MemoryStorage<size_type, memorySpace> &ldbSizes,
                 utils::MemoryStorage<size_type, memorySpace> &ldcSizes,
                 utils::MemoryStorage<size_type, memorySpace> &strideA,
                 utils::MemoryStorage<size_type, memorySpace> &strideB,
                 utils::MemoryStorage<size_type, memorySpace> &strideC,
                 const std::vector<size_type> &cellsInBlockNumDoFs,
                 const size_type               numVecs)
      {
        const size_type        numCellsInBlock = cellsInBlockNumDoFs.size();
        std::vector<size_type> mSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> nSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> kSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> ldaSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> ldbSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> ldcSizesSTL(numCellsInBlock, 0);
        std::vector<size_type> strideASTL(numCellsInBlock, 0);
        std::vector<size_type> strideBSTL(numCellsInBlock, 0);
        std::vector<size_type> strideCSTL(numCellsInBlock, 0);

        for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
          {
            mSizesSTL[iCell]   = numVecs;
            nSizesSTL[iCell]   = cellsInBlockNumDoFs[iCell];
            kSizesSTL[iCell]   = cellsInBlockNumDoFs[iCell];
            ldaSizesSTL[iCell] = mSizesSTL[iCell];
            ldbSizesSTL[iCell] = kSizesSTL[iCell];
            ldcSizesSTL[iCell] = mSizesSTL[iCell];
            strideASTL[iCell]  = mSizesSTL[iCell] * kSizesSTL[iCell];
            strideBSTL[iCell]  = kSizesSTL[iCell] * nSizesSTL[iCell];
            strideCSTL[iCell]  = mSizesSTL[iCell] * nSizesSTL[iCell];
          }

        mSizes.copyFrom(mSizesSTL);
        nSizes.copyFrom(nSizesSTL);
        kSizes.copyFrom(kSizesSTL);
        ldaSizes.copyFrom(ldaSizesSTL);
        ldbSizes.copyFrom(ldbSizesSTL);
        ldcSizes.copyFrom(ldcSizesSTL);
        strideA.copyFrom(strideASTL);
        strideB.copyFrom(strideBSTL);
        strideC.copyFrom(strideCSTL);
      }

      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace>
      void
      computeAxCellWiseLocal(
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &                     basisOverlapInAllCells,
        const ValueTypeOperand *x,
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand> *y,
        const size_type                                           numVecs,
        const size_type                              numLocallyOwnedCells,
        const std::vector<size_type> &               numCellDofs,
        const size_type *                            cellLocalIdsStartPtrX,
        const size_type *                            cellLocalIdsStartPtrY,
        const size_type                              cellBlockSize,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      {
        //
        // Perform ye = Ae * xe, where
        // Ae is the discrete Overlap operator for the e-th cell.
        // That is, \f$Ae_ij=\int_{\Omega_e}  N_i \cdot N_j
        // d\textbf{r} $\f,
        // (\f$Ae_ij$\f is the integral of the dot product of the
        // i-th and j-th basis function in the e-th cell.
        //
        // xe, ye are the part of the input (x) and output vector (y),
        // respectively, belonging to e-th cell.
        //

        //
        // For better performance, we evaluate ye for multiple cells at a time
        //

        linearAlgebra::blasLapack::Layout layout =
          linearAlgebra::blasLapack::Layout::ColMajor;

        size_type BStartOffset       = 0;
        size_type cellLocalIdsOffset = 0;
        for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
             cellStartId += cellBlockSize)
          {
            const size_type cellEndId =
              std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
            const size_type        numCellsInBlock = cellEndId - cellStartId;
            std::vector<size_type> cellsInBlockNumDoFsSTL(numCellsInBlock, 0);
            std::copy(numCellDofs.begin() + cellStartId,
                      numCellDofs.begin() + cellEndId,
                      cellsInBlockNumDoFsSTL.begin());

            const size_type cellsInBlockNumCumulativeDoFs =
              std::accumulate(cellsInBlockNumDoFsSTL.begin(),
                              cellsInBlockNumDoFsSTL.end(),
                              0);

            utils::MemoryStorage<size_type, memorySpace> cellsInBlockNumDoFs(
              numCellsInBlock);
            cellsInBlockNumDoFs.copyFrom(cellsInBlockNumDoFsSTL);

            // allocate memory for cell-wise data for x
            utils::MemoryStorage<ValueTypeOperand, memorySpace> xCellValues(
              cellsInBlockNumCumulativeDoFs * numVecs,
              utils::Types<linearAlgebra::blasLapack::scalar_type<
                ValueTypeOperator,
                ValueTypeOperand>>::zero);

            // copy x to cell-wise data
            basis::FECellWiseDataOperations<ValueTypeOperand, memorySpace>::
              copyFieldToCellWiseData(x,
                                      numVecs,
                                      cellLocalIdsStartPtrX +
                                        cellLocalIdsOffset,
                                      cellsInBlockNumDoFs,
                                      xCellValues);

            std::vector<linearAlgebra::blasLapack::Op> transA(
              numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
            std::vector<linearAlgebra::blasLapack::Op> transB(
              numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);

            utils::MemoryStorage<size_type, memorySpace> mSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> nSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> kSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> ldaSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> ldbSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> ldcSizes(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> strideA(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> strideB(
              numCellsInBlock);
            utils::MemoryStorage<size_type, memorySpace> strideC(
              numCellsInBlock);

            OrthoEFEOverlapOperatorContextInternal::storeSizes(
              mSizes,
              nSizes,
              kSizes,
              ldaSizes,
              ldbSizes,
              ldcSizes,
              strideA,
              strideB,
              strideC,
              cellsInBlockNumDoFsSTL,
              numVecs);

            // allocate memory for cell-wise data for y
            utils::MemoryStorage<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                     ValueTypeOperand>,
              memorySpace>
              yCellValues(cellsInBlockNumCumulativeDoFs * numVecs,
                          utils::Types<linearAlgebra::blasLapack::scalar_type<
                            ValueTypeOperator,
                            ValueTypeOperand>>::zero);

            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand>
              alpha = 1.0;
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand>
              beta = 0.0;

            const ValueTypeOperator *B =
              basisOverlapInAllCells.data() + BStartOffset;
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand> *C =
              yCellValues.begin();
            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperator,
                                                             ValueTypeOperand,
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
              xCellValues.data(),
              ldaSizes.data(),
              B,
              ldbSizes.data(),
              beta,
              C,
              ldcSizes.data(),
              linAlgOpContext);

            basis::FECellWiseDataOperations<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                     ValueTypeOperand>,
              memorySpace>::addCellWiseDataToFieldData(yCellValues,
                                                       numVecs,
                                                       cellLocalIdsStartPtrY +
                                                         cellLocalIdsOffset,
                                                       cellsInBlockNumDoFs,
                                                       y);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                BStartOffset +=
                  cellsInBlockNumDoFsSTL[iCell] * cellsInBlockNumDoFsSTL[iCell];
                cellLocalIdsOffset += cellsInBlockNumDoFsSTL[iCell];
              }
          }
      }

    } // end of namespace OrthoEFEOverlapOperatorContextInternal

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                   ValueTypeOperand,
                                   memorySpace,
                                   dim>::
      OrthoEFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             enrichmentBlockBasisDataStorage,
        const size_type maxCellBlock,
        const size_type maxFieldBlock,
        const bool      calculateWings)
      : d_feBasisManager(&feBasisManager)
      , d_maxCellBlock(maxCellBlock)
      , d_maxFieldBlock(maxFieldBlock)
      , d_cellStartIdsBasisOverlap(0)
      , d_isMassLumping(false)
      , d_isEnrichAtomBlockDiagonalApprox(false)
    {
      OrthoEFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockBasisDataStorage,
             enrichmentBlockBasisDataStorage,
             d_basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell,
             calculateWings);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                   ValueTypeOperand,
                                   memorySpace,
                                   dim>::
      OrthoEFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             enrichmentBlockClassicalBasisDataStorage,
        const size_type maxCellBlock,
        const size_type maxFieldBlock,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                   linAlgOpContext,
        const bool calculateWings)
      : d_feBasisManager(&feBasisManager)
      , d_maxCellBlock(maxCellBlock)
      , d_maxFieldBlock(maxFieldBlock)
      , d_cellStartIdsBasisOverlap(0)
      , d_isMassLumping(false)
      , d_isEnrichAtomBlockDiagonalApprox(false)
    {
      OrthoEFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockBasisDataStorage,
             enrichmentBlockEnrichmentBasisDataStorage,
             enrichmentBlockClassicalBasisDataStorage,
             d_basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell,
             *linAlgOpContext,
             calculateWings);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                   ValueTypeOperand,
                                   memorySpace,
                                   dim>::
      OrthoEFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        const bool isEnrichAtomBlockDiagonalApprox)
      : d_feBasisManager(&feBasisManager)
      , d_maxCellBlock(0)
      , d_maxFieldBlock(0)
      , d_cellStartIdsBasisOverlap(0)
      , d_isMassLumping(true)
      , d_isEnrichAtomBlockDiagonalApprox(isEnrichAtomBlockDiagonalApprox)
    {
      const size_type numLocallyOwnedCells =
        d_feBasisManager->nLocallyOwnedCells();

      const BasisDofHandler &basisDofHandler =
        feBasisManager.getBasisDofHandler();

      const EFEBasisDofHandler<ValueTypeOperand,
                               ValueTypeOperator,
                               memorySpace,
                               dim> &efebasisDofHandler =
        dynamic_cast<const EFEBasisDofHandler<ValueTypeOperand,
                                              ValueTypeOperator,
                                              memorySpace,
                                              dim> &>(basisDofHandler);
      utils::throwException(
        &efebasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input to EFEBasisDofHandler.");

      d_efebasisDofHandler = &efebasisDofHandler;

      utils::throwException(
        efebasisDofHandler.isOrthogonalized(),
        "The Enrichment functions have to be orthogonalized for this class to do the application of overlap.");

      utils::throwException(
        classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily() == quadrature::QuadratureFamily::GLL,
        "The quadrature rule for integration of Classical FE dofs has to be GLL."
        "Contact developers if extra options are needed.");

      std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                               ValueTypeOperator,
                                               memorySpace,
                                               dim>>
        efeBDH =
          std::dynamic_pointer_cast<const EFEBasisDofHandler<ValueTypeOperand,
                                                             ValueTypeOperator,
                                                             memorySpace,
                                                             dim>>(
            enrichmentBlockBasisDataStorage.getBasisDofHandler());
      utils::throwException(
        efeBDH != nullptr,
        "Could not cast BasisDofHandler to EFEBasisDofHandler "
        "in OrthoEFEOverlapOperatorContext");

      utils::throwException(
        &efebasisDofHandler == efeBDH.get(),
        "In OrthoEFEOverlapOperatorContext the feBasisManager and enrichmentBlockBasisDataStorage should"
        "come from same basisDofHandler.");


      const size_type numCellClassicalDofs = utils::mathFunctions::sizeTypePow(
        (efebasisDofHandler.getFEOrder(0) + 1), dim);
      d_nglobalEnrichmentIds = efebasisDofHandler.nGlobalEnrichmentNodes();

      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBegin =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      OrthoEFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockGLLBasisDataStorage,
             enrichmentBlockBasisDataStorage,
             d_basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell,
             false);

      std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                         0);
      std::copy(numCellDofs.begin(),
                numCellDofs.begin() + numLocallyOwnedCells,
                locallyOwnedCellsNumDoFsSTL.begin());

      utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
        numLocallyOwnedCells);
      locallyOwnedCellsNumDoFs.template copyFrom(locallyOwnedCellsNumDoFsSTL);

      d_diagonal =
        std::make_shared<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>(
          d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      // Create the diagonal of the classical block matrix which is diagonal for
      // GLL with spectral quadrature
      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(d_basisOverlap->data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           d_diagonal->data());

      d_feBasisManager->getConstraints().distributeChildToParent(*d_diagonal,
                                                                 1);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      d_diagonal->accumulateAddLocallyOwned();

      d_diagonal->updateGhostValues();

      d_feBasisManager->getConstraints().setConstrainedNodesToZero(*d_diagonal,
                                                                   1);

      // Now form the enrichment block matrix.
      if(d_isEnrichAtomBlockDiagonalApprox)
      {
        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          basisOverlapEnrichmentBlockExact(d_nglobalEnrichmentIds * 
            d_nglobalEnrichmentIds);

        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          basisOverlapEnrichmentBlock(d_nglobalEnrichmentIds *
                                          d_nglobalEnrichmentIds,
                                        0), 
          basisOverlapEnrichmentAtomBlock(d_nglobalEnrichmentIds *
                                          d_nglobalEnrichmentIds,
                                        0);

        size_type cellId                     = 0;
        size_type cumulativeBasisDataInCells = 0;
        for (auto enrichmentVecInCell :
            efebasisDofHandler.getEnrichmentIdsPartition()
              ->overlappingEnrichmentIdsInCells())
        {
          size_type nCellEnrichmentDofs = enrichmentVecInCell.size();
          for (unsigned int j = 0; j < nCellEnrichmentDofs; j++)
            {
              for (unsigned int k = 0; k < nCellEnrichmentDofs; k++)
                {
                  *(basisOverlapEnrichmentBlockExact.data() +
                    enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                    enrichmentVecInCell[k]) +=
                    *(d_basisOverlap->data() + cumulativeBasisDataInCells +
                      (numCellClassicalDofs + nCellEnrichmentDofs) *
                        (numCellClassicalDofs + j) +
                      numCellClassicalDofs + k);

                  basis::EnrichmentIdAttribute eIdAttrj =
                    efeBDH->getEnrichmentIdsPartition()
                      ->getEnrichmentIdAttribute(enrichmentVecInCell[j]);

                  basis::EnrichmentIdAttribute eIdAttrk =
                    efeBDH->getEnrichmentIdsPartition()
                      ->getEnrichmentIdAttribute(enrichmentVecInCell[k]);

                  if (eIdAttrj.atomId == eIdAttrk.atomId)
                    {
                      *(basisOverlapEnrichmentBlock.data() +
                        enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                        enrichmentVecInCell[k]) +=
                        *(d_basisOverlap->data() + cumulativeBasisDataInCells +
                          (numCellClassicalDofs + nCellEnrichmentDofs) *
                            (numCellClassicalDofs + j) +
                          numCellClassicalDofs + k);
                    }
                }
            }
          cumulativeBasisDataInCells += utils::mathFunctions::sizeTypePow(
            (nCellEnrichmentDofs + numCellClassicalDofs), 2);
          cellId += 1;
        }

      int err = utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        basisOverlapEnrichmentBlockExact.data(),
        basisOverlapEnrichmentBlockExact.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      err = utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        basisOverlapEnrichmentBlock.data(),
        basisOverlapEnrichmentBlock.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);


      basisOverlapEnrichmentAtomBlock = basisOverlapEnrichmentBlock;
      utils::MemoryStorage<double, memorySpace> eigenValuesMemSpace(
        d_nglobalEnrichmentIds);

      linearAlgebra::blasLapack::heevd<ValueType, memorySpace>(
        linearAlgebra::blasLapack::Job::NoVec,
        linearAlgebra::blasLapack::Uplo::Lower,
        d_nglobalEnrichmentIds,
        basisOverlapEnrichmentAtomBlock.data(),
        d_nglobalEnrichmentIds,
        eigenValuesMemSpace.data(),
        *d_diagonal->getLinAlgOpContext());

      auto minAbsValueOfEigenVecAtomBlock = *(eigenValuesMemSpace.data());

      global_size_type globalEnrichmentStartId =
        efeBDH->getGlobalRanges()[1].first;

      std::pair<global_size_type, global_size_type> locOwnEidPair{
        efeBDH->getLocallyOwnedRanges()[1].first - globalEnrichmentStartId,
        efeBDH->getLocallyOwnedRanges()[1].second - globalEnrichmentStartId};

      global_size_type nlocallyOwnedEnrichmentIds =
        locOwnEidPair.second - locOwnEidPair.first;

      d_atomBlockEnrichmentOverlap.resize(nlocallyOwnedEnrichmentIds *
                                             nlocallyOwnedEnrichmentIds);

      for (global_size_type i = 0; i < nlocallyOwnedEnrichmentIds; i++)
        {
          for (global_size_type j = 0; j < nlocallyOwnedEnrichmentIds; j++)
            {
              *(d_atomBlockEnrichmentOverlap.data() +
                i * nlocallyOwnedEnrichmentIds + j) =
                *(basisOverlapEnrichmentBlock.data() +
                  (i + locOwnEidPair.first) * d_nglobalEnrichmentIds +
                  (j + locOwnEidPair.first));
            }
        }

      int rank;
      utils::mpi::MPICommRank(
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator(), &rank);

      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      ValueTypeOperator normMexact = 0;
      for (int i = 0; i < basisOverlapEnrichmentBlock.size(); i++)
        {
          *(basisOverlapEnrichmentBlock.data() + i) =
            *(basisOverlapEnrichmentBlockExact.data() + i) -
            *(basisOverlapEnrichmentBlock.data() + i);

          normMexact += *(basisOverlapEnrichmentBlockExact.data()
          + i) * *(basisOverlapEnrichmentBlockExact.data() + i);
        }
      normMexact = std::sqrt(normMexact);

      linearAlgebra::blasLapack::heevd<ValueType, memorySpace>(
        linearAlgebra::blasLapack::Job::Vec,
        linearAlgebra::blasLapack::Uplo::Lower,
        d_nglobalEnrichmentIds,
        basisOverlapEnrichmentBlock.data(),
        d_nglobalEnrichmentIds,
        eigenValuesMemSpace.data(),
        *d_diagonal->getLinAlgOpContext());

      ValueType tolerance = 1e-10;

      ValueType eigValShift = std::min(*eigenValuesMemSpace.data(), 0.) - tolerance; 
      utils::throwException(minAbsValueOfEigenVecAtomBlock + eigValShift > 0, 
        "The min eigenvalue of AtomBlockDiagonal is less than shift with the residual. values: " + 
          std::to_string(minAbsValueOfEigenVecAtomBlock) + " " + std::to_string(eigValShift));

      for(int i = 0; i < eigenValuesMemSpace.size() ;i++)
        *(eigenValuesMemSpace.data() + i) -= eigValShift;

      size_type  i = d_nglobalEnrichmentIds-1;
      while(i > 0)
      {
        ValueTypeOperand sumEigVal = 0;
        for(size_type j = i ; j < d_nglobalEnrichmentIds ; j++)
        {
          sumEigVal += *(eigenValuesMemSpace.data() + j) *
            *(eigenValuesMemSpace.data() + j);
        }
        if(std::sqrt(sumEigVal)/normMexact < 1e-6)
          break;
        i--;
        d_rank++;
      }

      rootCout << "Rank of Residual M Enrichment block matrix: " <<
      d_rank << "\n";

      d_residualEnrichOverlapEigenVec.resize(nlocallyOwnedEnrichmentIds *
                                                  d_rank,
                                                0);
      d_residualEnrichOverlapEigenVal.resize(d_rank, 0);

      for (int i = 0; i < d_rank; i++)
        {
          basisOverlapEnrichmentBlock.template copyTo<memorySpace>(
            d_residualEnrichOverlapEigenVec.begin(),
            nlocallyOwnedEnrichmentIds,
            d_nglobalEnrichmentIds * (i + d_nglobalEnrichmentIds - d_rank) +
              locOwnEidPair.first, // srcoffset
            nlocallyOwnedEnrichmentIds *
              i); // dstoffset ; col - d_rank, row , N_locowned

          *(d_residualEnrichOverlapEigenVal.data() + i) =
            *(eigenValuesMemSpace.data() + (i + d_nglobalEnrichmentIds - d_rank)) - eigValShift;
        }
      }
      else
      {
      d_basisOverlapEnrichmentBlock =
        std::make_shared<utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          d_nglobalEnrichmentIds * d_nglobalEnrichmentIds);

      std::vector<ValueTypeOperator> basisOverlapEnrichmentBlockSTL(
        d_nglobalEnrichmentIds * d_nglobalEnrichmentIds, 0),
        basisOverlapEnrichmentBlockSTLTmp(d_nglobalEnrichmentIds *
                                            d_nglobalEnrichmentIds,
                                          0);

      size_type cellId                     = 0;
      size_type cumulativeBasisDataInCells = 0;
      for (auto enrichmentVecInCell :
           efebasisDofHandler.getEnrichmentIdsPartition()
             ->overlappingEnrichmentIdsInCells())
        {
          size_type nCellEnrichmentDofs = enrichmentVecInCell.size();
          for (unsigned int j = 0; j < nCellEnrichmentDofs; j++)
            {
              for (unsigned int k = 0; k < nCellEnrichmentDofs; k++)
                {
                  *(basisOverlapEnrichmentBlockSTLTmp.data() +
                    enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                    enrichmentVecInCell[k]) +=
                    *(d_basisOverlap->data() + cumulativeBasisDataInCells +
                      (numCellClassicalDofs + nCellEnrichmentDofs) *
                        (numCellClassicalDofs + j) +
                      numCellClassicalDofs + k);
                }
            }
          cumulativeBasisDataInCells += utils::mathFunctions::sizeTypePow(
            (nCellEnrichmentDofs + numCellClassicalDofs), 2);
          cellId += 1;
        }

      int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        basisOverlapEnrichmentBlockSTLTmp.data(),
        basisOverlapEnrichmentBlockSTL.data(),
        basisOverlapEnrichmentBlockSTLTmp.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      d_basisOverlapEnrichmentBlock
        ->template copyFrom<utils::MemorySpace::HOST>(
          basisOverlapEnrichmentBlockSTL.data());
    }
    }


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                   ValueTypeOperand,
                                   memorySpace,
                                   dim>::
      OrthoEFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        const bool isEnrichAtomBlockDiagonalApprox)
      : d_feBasisManager(&feBasisManager)
      , d_maxCellBlock(0)
      , d_maxFieldBlock(0)
      , d_cellStartIdsBasisOverlap(0)
      , d_isMassLumping(true)
      , d_isEnrichAtomBlockDiagonalApprox(isEnrichAtomBlockDiagonalApprox)
    {
      const size_type numLocallyOwnedCells =
        d_feBasisManager->nLocallyOwnedCells();

      const BasisDofHandler &basisDofHandler =
        feBasisManager.getBasisDofHandler();

      const EFEBasisDofHandler<ValueTypeOperand,
                               ValueTypeOperator,
                               memorySpace,
                               dim> &efebasisDofHandler =
        dynamic_cast<const EFEBasisDofHandler<ValueTypeOperand,
                                              ValueTypeOperator,
                                              memorySpace,
                                              dim> &>(basisDofHandler);
      utils::throwException(
        &efebasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input to EFEBasisDofHandler.");

      d_efebasisDofHandler = &efebasisDofHandler;

      utils::throwException(
        classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily() == quadrature::QuadratureFamily::GLL,
        "The quadrature rule for integration of Classical FE dofs has to be GLL."
        "Contact developers if extra options are needed.");

      std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                               ValueTypeOperator,
                                               memorySpace,
                                               dim>>
        efeBDH =
          std::dynamic_pointer_cast<const EFEBasisDofHandler<ValueTypeOperand,
                                                             ValueTypeOperator,
                                                             memorySpace,
                                                             dim>>(
            enrichmentBlockEnrichmentBasisDataStorage.getBasisDofHandler());
      utils::throwException(
        efeBDH != nullptr,
        "Could not cast BasisDofHandler to EFEBasisDofHandler "
        "in OrthoEFEOverlapOperatorContext");

      utils::throwException(
        &efebasisDofHandler == efeBDH.get(),
        "In OrthoEFEOverlapOperatorContext the feBasisManager and enrichmentBlockEnrichmentBasisDataStorage should"
        "come from same basisDofHandler.");


      const size_type numCellClassicalDofs = utils::mathFunctions::sizeTypePow(
        (efebasisDofHandler.getFEOrder(0) + 1), dim);
      d_nglobalEnrichmentIds = efebasisDofHandler.nGlobalEnrichmentNodes();

      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBegin =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      OrthoEFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockGLLBasisDataStorage,
             enrichmentBlockEnrichmentBasisDataStorage,
             enrichmentBlockClassicalBasisDataStorage,
             d_basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell,
             *linAlgOpContext,
             false);

      std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                         0);
      std::copy(numCellDofs.begin(),
                numCellDofs.begin() + numLocallyOwnedCells,
                locallyOwnedCellsNumDoFsSTL.begin());

      utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
        numLocallyOwnedCells);
      locallyOwnedCellsNumDoFs.template copyFrom(locallyOwnedCellsNumDoFsSTL);

      d_diagonal =
        std::make_shared<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>(
          d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      // Create the diagonal of the classical block matrix which is diagonal for
      // GLL with spectral quadrature
      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(d_basisOverlap->data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           d_diagonal->data());

      d_feBasisManager->getConstraints().distributeChildToParent(*d_diagonal,
                                                                 1);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      d_diagonal->accumulateAddLocallyOwned();

      d_diagonal->updateGhostValues();

      d_feBasisManager->getConstraints().setConstrainedNodesToZero(*d_diagonal,
                                                                   1);

      // Now form the enrichment block matrix.

      int rank;
      utils::mpi::MPICommRank(
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator(), &rank);
      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      if(d_isEnrichAtomBlockDiagonalApprox)
      {
        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          basisOverlapEnrichmentBlockExact(d_nglobalEnrichmentIds * 
            d_nglobalEnrichmentIds);

        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          basisOverlapEnrichmentBlock(d_nglobalEnrichmentIds *
                                          d_nglobalEnrichmentIds,
                                        0), 
          basisOverlapEnrichmentAtomBlock(d_nglobalEnrichmentIds *
                                          d_nglobalEnrichmentIds,
                                        0);

        size_type cellId                     = 0;
        size_type cumulativeBasisDataInCells = 0;
        for (auto enrichmentVecInCell :
            efebasisDofHandler.getEnrichmentIdsPartition()
              ->overlappingEnrichmentIdsInCells())
        {
          size_type nCellEnrichmentDofs = enrichmentVecInCell.size();
          for (unsigned int j = 0; j < nCellEnrichmentDofs; j++)
            {
              for (unsigned int k = 0; k < nCellEnrichmentDofs; k++)
                {
                  *(basisOverlapEnrichmentBlockExact.data() +
                    enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                    enrichmentVecInCell[k]) +=
                    *(d_basisOverlap->data() + cumulativeBasisDataInCells +
                    (numCellClassicalDofs + nCellEnrichmentDofs) *
                      (numCellClassicalDofs + j) +
                    numCellClassicalDofs + k);

                  basis::EnrichmentIdAttribute eIdAttrj =
                    efeBDH->getEnrichmentIdsPartition()
                      ->getEnrichmentIdAttribute(enrichmentVecInCell[j]);

                  basis::EnrichmentIdAttribute eIdAttrk =
                    efeBDH->getEnrichmentIdsPartition()
                      ->getEnrichmentIdAttribute(enrichmentVecInCell[k]);

                  if (eIdAttrj.atomId == eIdAttrk.atomId)
                    {
                      *(basisOverlapEnrichmentBlock.data() +
                        enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                        enrichmentVecInCell[k]) +=
                        *(d_basisOverlap->data() + cumulativeBasisDataInCells +
                          (numCellClassicalDofs + nCellEnrichmentDofs) *
                            (numCellClassicalDofs + j) +
                          numCellClassicalDofs + k);
                    }
                }
            }
          cumulativeBasisDataInCells += utils::mathFunctions::sizeTypePow(
            (nCellEnrichmentDofs + numCellClassicalDofs), 2);
          cellId += 1;
        }

      int err = utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        basisOverlapEnrichmentBlockExact.data(),
        basisOverlapEnrichmentBlockExact.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);


      // rootCout << "Enrichment Block Matrix: " << std::endl;
      // for (size_type i = 0; i < d_nglobalEnrichmentIds; i++)
      //   {
      //     rootCout << "[";
      //     for (size_type j = 0; j < d_nglobalEnrichmentIds; j++)
      //       {
      //         rootCout << *(basisOverlapEnrichmentBlockExact.data() +
      //                       i * d_nglobalEnrichmentIds + j)
      //                  << "\t";
      //       }
      //     rootCout << "]" << std::endl;
      //   }
      //   rootCout << "\n\n\n\n\n";

      err = utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        basisOverlapEnrichmentBlock.data(),
        basisOverlapEnrichmentBlock.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);


      // rootCout << "Enrichment Atom Block Diagonal Matrix: " << std::endl;
      // for (size_type i = 0; i < d_nglobalEnrichmentIds; i++)
      //   {
      //     rootCout << "[";
      //     for (size_type j = 0; j < d_nglobalEnrichmentIds; j++)
      //       {
      //         rootCout << *(basisOverlapEnrichmentBlock.data() +
      //                       i * d_nglobalEnrichmentIds + j)
      //                  << "\t";
      //       }
      //     rootCout << "]" << std::endl;
      //   }
      //   rootCout << std::endl;

      basisOverlapEnrichmentAtomBlock = basisOverlapEnrichmentBlockExact;
      utils::MemoryStorage<double, memorySpace> eigenValuesMemSpace(
        d_nglobalEnrichmentIds);

      linearAlgebra::blasLapack::heevd<ValueType, memorySpace>(
        linearAlgebra::blasLapack::Job::NoVec,
        linearAlgebra::blasLapack::Uplo::Lower,
        d_nglobalEnrichmentIds,
        basisOverlapEnrichmentAtomBlock.data(),
        d_nglobalEnrichmentIds,
        eigenValuesMemSpace.data(),
        *d_diagonal->getLinAlgOpContext());

      rootCout << "EigenValues of Atom Block Overlap: ";
      for(int i = 0; i < eigenValuesMemSpace.size() ;i++)
        rootCout << *(eigenValuesMemSpace.data() + i) << "\t";
      rootCout << std::endl;

      auto minAbsValueOfEigenVecAtomBlock = *(eigenValuesMemSpace.data());

      global_size_type globalEnrichmentStartId =
        efeBDH->getGlobalRanges()[1].first;

      std::pair<global_size_type, global_size_type> locOwnEidPair{
        efeBDH->getLocallyOwnedRanges()[1].first - globalEnrichmentStartId,
        efeBDH->getLocallyOwnedRanges()[1].second - globalEnrichmentStartId};

      global_size_type nlocallyOwnedEnrichmentIds =
        locOwnEidPair.second - locOwnEidPair.first;

      d_atomBlockEnrichmentOverlap.resize(nlocallyOwnedEnrichmentIds *
                                             nlocallyOwnedEnrichmentIds);

      for (global_size_type i = 0; i < nlocallyOwnedEnrichmentIds; i++)
        {
          for (global_size_type j = 0; j < nlocallyOwnedEnrichmentIds; j++)
            {
              *(d_atomBlockEnrichmentOverlap.data() +
                i * nlocallyOwnedEnrichmentIds + j) =
                *(basisOverlapEnrichmentBlock.data() +
                  (i + locOwnEidPair.first) * d_nglobalEnrichmentIds +
                  (j + locOwnEidPair.first));
            }
        }

      ValueTypeOperator normMexact = 0;
      for (int i = 0; i < basisOverlapEnrichmentBlock.size(); i++)
        {
          *(basisOverlapEnrichmentBlock.data() + i) =
            *(basisOverlapEnrichmentBlockExact.data() + i) -
            *(basisOverlapEnrichmentBlock.data() + i);

          normMexact += *(basisOverlapEnrichmentBlockExact.data()
          + i) * *(basisOverlapEnrichmentBlockExact.data() + i);
        }
      normMexact = std::sqrt(normMexact);

      linearAlgebra::blasLapack::heevd<ValueType, memorySpace>(
        linearAlgebra::blasLapack::Job::Vec,
        linearAlgebra::blasLapack::Uplo::Lower,
        d_nglobalEnrichmentIds,
        basisOverlapEnrichmentBlock.data(),
        d_nglobalEnrichmentIds,
        eigenValuesMemSpace.data(),
        *d_diagonal->getLinAlgOpContext());

      ValueType tolerance = 1e-10;

      ValueType eigValShift = std::min(*eigenValuesMemSpace.data(), 0.) - tolerance; 
      utils::throwException(minAbsValueOfEigenVecAtomBlock + eigValShift > 0, 
        "The min eigenvalue of AtomBlockDiagonal is less than shift with the residual. values: " + 
          std::to_string(minAbsValueOfEigenVecAtomBlock) + " " + std::to_string(eigValShift));

      for(int i = 0; i < eigenValuesMemSpace.size() ;i++)
        *(eigenValuesMemSpace.data() + i) -= eigValShift;

      size_type  i = d_nglobalEnrichmentIds-1;
      while(i > 0)
      {
        ValueTypeOperand sumEigVal = 0;
        for(size_type j = i ; j < d_nglobalEnrichmentIds ; j++)
        {
          sumEigVal += *(eigenValuesMemSpace.data() + j) *
            *(eigenValuesMemSpace.data() + j);
        }
        if(std::sqrt(sumEigVal)/normMexact < 1e-6)
          break;
        i--;
        d_rank++;
      }

      rootCout << "Rank of Residual M Enrichment block matrix: " <<
      d_rank << "\n";

      d_residualEnrichOverlapEigenVec.resize(nlocallyOwnedEnrichmentIds *
                                                  d_rank,
                                                0);
      d_residualEnrichOverlapEigenVal.resize(d_rank, 0);

      for (int i = 0; i < d_rank; i++)
        {
          basisOverlapEnrichmentBlock.template copyTo<memorySpace>(
            d_residualEnrichOverlapEigenVec.begin(),
            nlocallyOwnedEnrichmentIds,
            d_nglobalEnrichmentIds * (i + d_nglobalEnrichmentIds - d_rank) +
              locOwnEidPair.first, // srcoffset
            nlocallyOwnedEnrichmentIds *
              i); // dstoffset ; col - d_rank, row , N_locowned

          *(d_residualEnrichOverlapEigenVal.data() + i) =
            *(eigenValuesMemSpace.data() + (i + d_nglobalEnrichmentIds - d_rank)) - eigValShift;
        }
      }
      else
      {
      d_basisOverlapEnrichmentBlock =
        std::make_shared<utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          d_nglobalEnrichmentIds * d_nglobalEnrichmentIds);

      std::vector<ValueTypeOperator> basisOverlapEnrichmentBlockSTL(
        d_nglobalEnrichmentIds * d_nglobalEnrichmentIds, 0),
        basisOverlapEnrichmentBlockSTLTmp(d_nglobalEnrichmentIds *
                                            d_nglobalEnrichmentIds,
                                          0);

      size_type cellId                     = 0;
      size_type cumulativeBasisDataInCells = 0;
      for (auto enrichmentVecInCell :
           efebasisDofHandler.getEnrichmentIdsPartition()
             ->overlappingEnrichmentIdsInCells())
        {
          size_type nCellEnrichmentDofs = enrichmentVecInCell.size();
          for (unsigned int j = 0; j < nCellEnrichmentDofs; j++)
            {
              for (unsigned int k = 0; k < nCellEnrichmentDofs; k++)
                {
                  *(basisOverlapEnrichmentBlockSTLTmp.data() +
                    enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                    enrichmentVecInCell[k]) +=
                    *(d_basisOverlap->data() + cumulativeBasisDataInCells +
                      (numCellClassicalDofs + nCellEnrichmentDofs) *
                        (numCellClassicalDofs + j) +
                      numCellClassicalDofs + k);
                }
            }
          cumulativeBasisDataInCells += utils::mathFunctions::sizeTypePow(
            (nCellEnrichmentDofs + numCellClassicalDofs), 2);
          cellId += 1;
        }

      int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        basisOverlapEnrichmentBlockSTLTmp.data(),
        basisOverlapEnrichmentBlockSTL.data(),
        basisOverlapEnrichmentBlockSTLTmp.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      d_basisOverlapEnrichmentBlock
        ->template copyFrom<utils::MemorySpace::HOST>(
          basisOverlapEnrichmentBlockSTL.data());
      }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    OrthoEFEOverlapOperatorContext<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<
                    linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                           ValueTypeOperand>,
                    memorySpace> &Y,
                  bool            updateGhostX,
                  bool            updateGhostY) const
    {
      if (d_isMassLumping)
        {
          updateGhostX                  = false;
          updateGhostY                  = false;
          const size_type numComponents = X.getNumberComponents();
          const size_type nlocallyOwnedEnrichmentIds =
            d_feBasisManager->getLocallyOwnedRanges()[1].second -
            d_feBasisManager->getLocallyOwnedRanges()[1].first;
          const size_type nlocallyOwnedClassicalIds =
            d_feBasisManager->getLocallyOwnedRanges()[0].second -
            d_feBasisManager->getLocallyOwnedRanges()[0].first;

          if (updateGhostX)
            X.updateGhostValues();
          // update the child nodes based on the parent nodes
          d_feBasisManager->getConstraints().distributeParentToChild(
            X, X.getNumberComponents());

          Y.setValue(0.0);

          linearAlgebra::blasLapack::khatriRaoProduct(
            linearAlgebra::blasLapack::Layout::ColMajor,
            1,
            numComponents,
            d_diagonal->localSize(),
            d_diagonal->data(),
            X.begin(),
            Y.begin(),
            *(d_diagonal->getLinAlgOpContext()));

          if(!d_isEnrichAtomBlockDiagonalApprox)
          {
          utils::MemoryStorage<ValueTypeOperand, memorySpace>
            XenrichedGlobalVec(d_nglobalEnrichmentIds * numComponents),
            XenrichedGlobalVecTmp(d_nglobalEnrichmentIds * numComponents),
            YenrichedGlobalVec(d_nglobalEnrichmentIds * numComponents);

          XenrichedGlobalVecTmp.template copyFrom<memorySpace>(
            X.begin(),
            nlocallyOwnedEnrichmentIds * numComponents,
            nlocallyOwnedClassicalIds * numComponents,
            ((d_feBasisManager->getLocallyOwnedRanges()[1].first) -
             (d_efebasisDofHandler->getGlobalRanges()[0].second)) *
              numComponents);

          int err = utils::mpi::MPIAllreduce<memorySpace>(
            XenrichedGlobalVecTmp.data(),
            XenrichedGlobalVec.data(),
            XenrichedGlobalVecTmp.size(),
            utils::mpi::MPIDouble,
            utils::mpi::MPISum,
            d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
          std::pair<bool, std::string> mpiIsSuccessAndMsg =
            utils::mpi::MPIErrIsSuccessAndMsg(err);
          utils::throwException(mpiIsSuccessAndMsg.first,
                                "MPI Error:" + mpiIsSuccessAndMsg.second);

          // Do dgemm

          ValueType alpha = 1.0;
          ValueType beta  = 0.0;

          linearAlgebra::blasLapack::
            gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::Op::NoTrans,
              linearAlgebra::blasLapack::Op::Trans,
              numComponents,
              d_nglobalEnrichmentIds,
              d_nglobalEnrichmentIds,
              alpha,
              XenrichedGlobalVec.data(),
              numComponents,
              d_basisOverlapEnrichmentBlock->data(),
              d_nglobalEnrichmentIds,
              beta,
              YenrichedGlobalVec.begin(),
              numComponents,
              *(X.getLinAlgOpContext()));

          YenrichedGlobalVec.template copyTo<memorySpace>(
            Y.begin(),
            nlocallyOwnedEnrichmentIds * numComponents,
            ((d_feBasisManager->getLocallyOwnedRanges()[1].first) -
             (d_efebasisDofHandler->getGlobalRanges()[0].second)) *
              numComponents,
            nlocallyOwnedClassicalIds * numComponents);
          }
          else
          {
            utils::MemoryStorage<ValueTypeOperand, memorySpace> XenrichedLocalVec(
              nlocallyOwnedEnrichmentIds * numComponents),
              YenrichedLocalVec(nlocallyOwnedEnrichmentIds * numComponents),
              dotProds(d_rank * numComponents);
      
            XenrichedLocalVec.template copyFrom<memorySpace>(
              X.begin(),
              nlocallyOwnedEnrichmentIds * numComponents,
              nlocallyOwnedClassicalIds * numComponents,
              0);
      
      
            ValueType alpha = 1.0;
            ValueType beta  = 0.0;
      
            if (nlocallyOwnedEnrichmentIds > 0)
              linearAlgebra::blasLapack::
                gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  numComponents,
                  nlocallyOwnedEnrichmentIds,
                  nlocallyOwnedEnrichmentIds,
                  alpha,
                  XenrichedLocalVec.data(),
                  numComponents,
                  d_atomBlockEnrichmentOverlap.data(),
                  nlocallyOwnedEnrichmentIds,
                  beta,
                  YenrichedLocalVec.begin(),
                  numComponents,
                  *(X.getLinAlgOpContext()));
        
            if (nlocallyOwnedEnrichmentIds > 0)
              linearAlgebra::blasLapack::
                gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  numComponents,
                  d_rank,
                  nlocallyOwnedEnrichmentIds,
                  alpha,
                  XenrichedLocalVec.data(),
                  numComponents,
                  d_residualEnrichOverlapEigenVec.data(),
                  nlocallyOwnedEnrichmentIds,
                  beta,
                  dotProds.begin(),
                  numComponents,
                  *(X.getLinAlgOpContext()));

            utils::mpi::MPIDatatype mpiDatatype =
              utils::mpi::Types<ValueType>::getMPIDatatype();
            utils::mpi::MPIAllreduce<memorySpace>(
              utils::mpi::MPIInPlace,
              dotProds.data(),
              dotProds.size(),
              utils::mpi::MPIDouble,
              utils::mpi::MPISum,
              (X.getMPIPatternP2P())->mpiCommunicator());

            size_type m = 1, n = numComponents, k = d_rank;
            size_type stride = 0;

            linearAlgebra::blasLapack::
              scaleStridedVarBatched<ValueType, ValueType, memorySpace>(
                1,
                linearAlgebra::blasLapack::Layout::ColMajor,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                &stride,
                &stride,
                &stride,
                &m,
                &n,
                &k,
                d_residualEnrichOverlapEigenVal.data(),
                dotProds.data(),
                dotProds.data(),
                *(X.getLinAlgOpContext()));

            beta = 1.0;

            if (nlocallyOwnedEnrichmentIds > 0)
              linearAlgebra::blasLapack::
                gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::Trans,
                  numComponents,
                  nlocallyOwnedEnrichmentIds,
                  d_rank,
                  alpha,
                  dotProds.data(),
                  numComponents,
                  d_residualEnrichOverlapEigenVec.data(),
                  nlocallyOwnedEnrichmentIds,
                  beta,
                  YenrichedLocalVec.begin(),
                  numComponents,
                  *(X.getLinAlgOpContext()));
          }

          Y.updateGhostValues();

          // function to do a static condensation to send the constraint nodes
          // to its parent nodes
          d_feBasisManager->getConstraints().distributeChildToParent(
            Y, Y.getNumberComponents());

          // Function to update the ghost values of the Y
          if (updateGhostY)
            Y.updateGhostValues();
        }
      else
        {
          const size_type numLocallyOwnedCells =
            d_feBasisManager->nLocallyOwnedCells();
          std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
          for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
            numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);

          auto itCellLocalIdsBegin =
            d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

          const size_type numVecs = X.getNumberComponents();

          // get handle to constraints
          const basis::ConstraintsLocal<
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand>,
            memorySpace> &constraints = d_feBasisManager->getConstraints();

          if (updateGhostX)
            X.updateGhostValues();
          // update the child nodes based on the parent nodes
          constraints.distributeParentToChild(X, numVecs);

          // access cell-wise discrete Overlap operator
          const utils::MemoryStorage<ValueTypeOperator, memorySpace>
            &basisOverlapInAllCells = *d_basisOverlap;

          const size_type cellBlockSize =
            (d_maxCellBlock * d_maxFieldBlock) / numVecs;
          Y.setValue(0.0);

          //
          // perform Ax on the local part of A and x
          // (A = discrete Overlap operator)
          //
          OrthoEFEOverlapOperatorContextInternal::computeAxCellWiseLocal(
            basisOverlapInAllCells,
            X.begin(),
            Y.begin(),
            numVecs,
            numLocallyOwnedCells,
            numCellDofs,
            itCellLocalIdsBegin,
            itCellLocalIdsBegin,
            cellBlockSize,
            *(X.getLinAlgOpContext()));

          // function to do a static condensation to send the constraint nodes
          // to its parent nodes
          constraints.distributeChildToParent(Y, numVecs);

          // Function to add the values to the local node from its corresponding
          // ghost nodes from other processors.
          Y.accumulateAddLocallyOwned();
          if (updateGhostY)
            Y.updateGhostValues();
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const utils::MemoryStorage<ValueTypeOperator, memorySpace> &
    OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                   ValueTypeOperand,
                                   memorySpace,
                                   dim>::getBasisOverlapInAllCells() const
    {
      return *(d_basisOverlap);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    utils::MemoryStorage<ValueTypeOperator, memorySpace>
    OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                   ValueTypeOperand,
                                   memorySpace,
                                   dim>::getBasisOverlapInCell(const size_type
                                                                 cellId) const
    {
      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
                      basisOverlapStorage = d_basisOverlap;
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      utils::MemoryStorage<ValueTypeOperator, memorySpace> returnValue(
        sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId]);
      return returnValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    utils::MemoryStorage<ValueTypeOperator, memorySpace>
    OrthoEFEOverlapOperatorContext<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::getBasisOverlap(const size_type cellId,
                            const size_type basisId1,
                            const size_type basisId2) const
    {
      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
        basisOverlapStorage = d_basisOverlap;
      utils::MemoryStorage<ValueTypeOperator, memorySpace> returnValue(1);
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId] +
          basisId1 * d_dofsInCell[cellId] + basisId2);
      return returnValue;
    }

  } // namespace basis
} // end of namespace dftefe
