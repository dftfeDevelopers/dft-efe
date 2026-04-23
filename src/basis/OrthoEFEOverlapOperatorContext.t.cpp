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
#include <linearAlgebra/Defaults.h>
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

        size_type       basisOverlapSize                    = 0;
        size_type       cellId                              = 0;
        size_type       numCumulativeDofsxQuadEFEInAllCells = 0;
        const size_type feOrder = efeBDH->getFEOrder(cellId);

        size_type       dofsPerCell;
        const size_type dofsPerCellCFE = cfeBDH->nCellDofs(cellId);

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = efeBDH->nCellDofs(cellId);
            numCumulativeDofsxQuadEFEInAllCells +=
              dofsInCellVec[cellId] *
              efeBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellId);
            basisOverlapSize += dofsInCellVec[cellId] * dofsInCellVec[cellId];
            cellId++;
          }

        size_type cumulativeDofQuadPointsOffsetCFE = 0,
                  cumulativeDofQuadPointsOffsetEFE = 0;

        bool isConstantDofsAndQuadPointsInCellCFE = false;
        quadrature::QuadratureFamily quadFamily =
          cfeBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        if ((quadFamily == quadrature::QuadratureFamily::GAUSS ||
             quadFamily == quadrature::QuadratureFamily::GLL ||
             quadFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED) &&
            !cfeBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCellCFE = true;

        std::vector<ValueTypeOperator> basisOverlapTmp(0);

        basisOverlap = std::make_shared<
          utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          basisOverlapSize);
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeOperator(0));

        const utils::MemoryStorage<ValueTypeOperator, memorySpace> &
          basisDataInAllCellsCFE = cfeBasisDataStorage.getBasisDataInAllCells();

        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          basisDataInAllCellsEFE(numCumulativeDofsxQuadEFEInAllCells);
        std::pair<size_type, size_type> cellPair(0, numLocallyOwnedCells);
        efeBasisDataStorage.getBasisDataInCellRange(cellPair,
                                                    basisDataInAllCellsEFE);

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsCFEHost(basisDataInAllCellsCFE.size());

        basisDataInAllCellsCFEHost.copyFrom(basisDataInAllCellsCFE);

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsEFEHost(basisDataInAllCellsCFE.size());

        basisDataInAllCellsEFEHost.copyFrom(basisDataInAllCellsEFE);

        auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex           = 0;

        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
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
              basisDataInAllCellsCFEHost.data() +
              cumulativeDofQuadPointsOffsetCFE;

            const ValueTypeOperator *cumulativeEFEDofQuadPoints =
              basisDataInAllCellsEFEHost.data() +
              cumulativeDofQuadPointsOffsetEFE;

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
            if (!isConstantDofsAndQuadPointsInCellCFE)
              cumulativeDofQuadPointsOffsetCFE +=
                nQuadPointInCellCFE * dofsPerCellCFE;
            cumulativeDofQuadPointsOffsetEFE +=
              nQuadPointInCellEFE * dofsPerCell;
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
          &                     basisOverlap,
        std::vector<size_type> &cellStartIdsBasisOverlap,
        std::vector<size_type> &dofsInCellVec,
        bool                    calculateWings = true)
      {
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>
          &linAlgOpContext =
            *linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST;

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

        size_type       basisOverlapSize                    = 0;
        size_type       cellId                              = 0;
        size_type       numCumulativeDofsxQuadEFEInAllCells = 0;
        size_type numCumulativeEnrichDofsxQuadEFEInAllCells = 0;
        const size_type feOrder = eefeBDH->getFEOrder(cellId);

        size_type       dofsPerCell;
        const size_type dofsPerCellCFE = ccfeBDH->nCellDofs(cellId);

        auto locallyOwnedCellIter = eefeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != eefeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = eefeBDH->nCellDofs(cellId);
            numCumulativeDofsxQuadEFEInAllCells +=
              dofsInCellVec[cellId] * enrichmentBlockEnrichmentBasisDataStorage
                                        .getQuadratureRuleContainer()
                                        ->nCellQuadraturePoints(cellId);
              numCumulativeEnrichDofsxQuadEFEInAllCells +=    
              (dofsInCellVec[cellId] - dofsPerCellCFE) * enrichmentBlockEnrichmentBasisDataStorage
                                        .getQuadratureRuleContainer()
                                        ->nCellQuadraturePoints(cellId);                                        
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

        size_type cumulativeDofQuadPointsOffsetCFE            = 0,
                  cumulativeDofQuadPointsOffsetEnrichBlockCFE = 0,
                  cumulativeDofQuadPointsOffsetEnrichBlockEFE = 0;

        bool isConstantDofsAndQuadPointsInCellCFE = false;
        quadrature::QuadratureFamily quadFamily =
          classicalBlockBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        if ((quadFamily == quadrature::QuadratureFamily::GAUSS ||
             quadFamily == quadrature::QuadratureFamily::GLL ||
             quadFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED) &&
            !ccfeBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCellCFE = true;

        bool isConstantDofsAndQuadPointsInCellEnrichBlockCFE = false;
        quadrature::QuadratureFamily quadFamily1 =
          enrichmentBlockClassicalBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        if ((quadFamily1 == quadrature::QuadratureFamily::GAUSS ||
             quadFamily1 == quadrature::QuadratureFamily::GLL ||
             quadFamily1 == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED) &&
            !ecfeBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCellEnrichBlockCFE = true;

        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsClassicalBlock =
            classicalBlockBasisDataStorage.getBasisDataInAllCells();
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsEnrichmentBlockClassical =
            enrichmentBlockClassicalBasisDataStorage.getBasisDataInAllCells();

        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          basisDataInAllCellsEnrichmentBlockEnrichment(
            numCumulativeDofsxQuadEFEInAllCells);
        std::pair<size_type, size_type> cellPair(0, numLocallyOwnedCells);
        enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInCellRange(
          cellPair, basisDataInAllCellsEnrichmentBlockEnrichment);

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsClassicalBlockHost(
            basisDataInAllCellsClassicalBlock.size());

        basisDataInAllCellsClassicalBlockHost.copyFrom(
          basisDataInAllCellsClassicalBlock);

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsEnrichmentBlockClassicalHost(
            basisDataInAllCellsEnrichmentBlockClassical.size());

        basisDataInAllCellsEnrichmentBlockClassicalHost.copyFrom(
          basisDataInAllCellsEnrichmentBlockClassical);

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsEnrichmentBlockEnrichmentHost(
            basisDataInAllCellsEnrichmentBlockEnrichment.size());

        basisDataInAllCellsEnrichmentBlockEnrichmentHost.copyFrom(
          basisDataInAllCellsEnrichmentBlockEnrichment);

      std::vector<double> quadValuesInAllCellsEnrichment(numCumulativeEnrichDofsxQuadEFEInAllCells), quadGradientsInAllCellsEnrichment;
          eefeBDH->getEnrichmentClassicalInterface()->getEnrichmentDataInAllCellsAtQuadPts(
            true,
            false,
            *enrichmentBlockEnrichmentBasisDataStorage.getQuadratureRuleContainer(),
            quadValuesInAllCellsEnrichment.data(),
            quadGradientsInAllCellsEnrichment.data(),
            *eefeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext());

        size_type cumulativeQuadEnrichBlockEnrichxenrichInCell = 0;

        locallyOwnedCellIter = eefeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != eefeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
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
              basisDataInAllCellsClassicalBlockHost.data() +
              cumulativeDofQuadPointsOffsetCFE;

            const ValueTypeOperator
              *cumulativeEnrichmentBlockClassicalDofQuadPoints =
                basisDataInAllCellsEnrichmentBlockClassicalHost.data() +
                cumulativeDofQuadPointsOffsetEnrichBlockCFE;

            const ValueTypeOperator
              *cumulativeEnrichmentBlockEnrichmentDofQuadPoints =
                basisDataInAllCellsEnrichmentBlockEnrichmentHost.data() +
                cumulativeDofQuadPointsOffsetEnrichBlockEFE;

            // std::vector<utils::Point> quadRealPointsVec =
            //   enrichmentBlockEnrichmentBasisDataStorage
            //     .getQuadratureRuleContainer()
            //     ->getCellRealPoints(cellIndex);

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

            if (numEnrichmentIdsInCell > 0)
              {

                std::vector<ValueTypeOperator> coeffsInCell(
                  dofsPerCellCFE * numEnrichmentIdsInCell, 0);

                coeffsInCell = eefeBDH->getEnrichmentClassicalInterface()->getClassicalComponentCoeffsInCellOEFE(cellIndex);

                ValueTypeOperator *B =
                  basisDataInAllCellsEnrichmentBlockClassicalHost.data() +
                  cumulativeDofQuadPointsOffsetEnrichBlockCFE;
                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'N',
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
                  linAlgOpContext);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector
                B = basisDataInAllCellsEnrichmentBlockEnrichmentHost.data() +
                    cumulativeDofQuadPointsOffsetEnrichBlockEFE;

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'N',
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockEnrichment,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  B,
                  dofsPerCell,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEE.data(),
                  numEnrichmentIdsInCell,
                  linAlgOpContext);
              }

            std::vector<ValueTypeOperator> basisOverlapClassicalBlock(
              dofsPerCellCFE * dofsPerCellCFE);
            std::vector<ValueTypeOperator> JxWxNCell(
              dofsPerCellCFE * nQuadPointInCellClassicalBlock, 0);

            size_type stride = 0;
            size_type m = 1, n = dofsPerCellCFE,
                      k = nQuadPointInCellClassicalBlock;

            linearAlgebra::blasLapack::scaleStridedVarBatched<
              ValueTypeOperator,
              ValueTypeOperator,
              utils::MemorySpace::HOST>(
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

            linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                            ValueTypeOperand,
                                            utils::MemorySpace::HOST>(
              'N',
              'C',
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
                  utils::MemorySpace::HOST>(
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
                  cellJxWValuesEnrichmentBlockEnrichment.data(),
                  cumulativeEnrichmentBlockEnrichmentDofQuadPoints,
                  JxWxNCell.data(),
                  linAlgOpContext);

                linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                                ValueTypeOperand,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'C',
                  n,
                  numEnrichmentIdsInCell,
                  k,
                  (ValueTypeOperand)1.0,
                  JxWxNCell.data(),
                  n,
                  quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
                  numEnrichmentIdsInCell,
                  (ValueTypeOperand)0.0,
                  basisOverlapECBlockEnrich.data(),
                  n,
                  linAlgOpContext);

                JxWxNCell.resize(dofsPerCellCFE *
                                   nQuadPointInCellEnrichmentBlockClassical,
                                 0);

                m = 1, n = dofsPerCellCFE,
                k = nQuadPointInCellEnrichmentBlockClassical;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  utils::MemorySpace::HOST>(
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
                  cellJxWValuesEnrichmentBlockClassical.data(),
                  cumulativeEnrichmentBlockClassicalDofQuadPoints,
                  JxWxNCell.data(),
                  linAlgOpContext);

                linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'C',
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
                  utils::MemorySpace::HOST>(
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
                  cellJxWValuesEnrichmentBlockEnrichment.data(),
                  quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
                  JxWxNCell.data(),
                  linAlgOpContext);

                linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                                ValueTypeOperand,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'C',
                  n,
                  n,
                  k,
                  (ValueTypeOperand)1.0,
                  JxWxNCell.data(),
                  n,
                  quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
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
                  utils::MemorySpace::HOST>(
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
                  cellJxWValuesEnrichmentBlockClassical.data(),
                  classicalComponentInQuadValuesEC.data(),
                  JxWxNCell.data(),
                  linAlgOpContext);

                linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'C',
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
                  utils::MemorySpace::HOST>(
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
                  cellJxWValuesEnrichmentBlockEnrichment.data(),
                  quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
                  JxWxNCell.data(),
                  linAlgOpContext);

                linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'C',
                  n,
                  n,
                  k,
                  (ValueTypeOperand)1.0,
                  classicalComponentInQuadValuesEE.data(),
                  n,
                  JxWxNCell.data(),
                  n,
                  (ValueTypeOperator)0.0,
                  basisOverlapEEBlock3.data(),
                  n,
                  linAlgOpContext);

                // linearAlgebra::blasLapack::
                //   gemm<ValueTypeOperand, ValueTypeOperator, memorySpace>(
                //     'T',
                //     'T',
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
                      }
                    else if (iNode >= dofsPerCellCFE &&
                             jNode < dofsPerCellCFE && calculateWings)
                      {
                        *basisOverlapTmpIter =
                          *(basisOverlapECBlockEnrich.data() +
                            (iNode - dofsPerCellCFE) * dofsPerCell + jNode) -
                          *(basisOverlapECBlockClass.data() +
                            (iNode - dofsPerCellCFE) * dofsPerCellCFE + jNode);
                      }
                    else if (iNode < dofsPerCellCFE &&
                             jNode >= dofsPerCellCFE && calculateWings)
                      {
                        *basisOverlapTmpIter =
                          *(basisOverlapECBlockEnrich.data() +
                            (jNode - dofsPerCellCFE) * dofsPerCell + iNode) -
                          *(basisOverlapECBlockClass.data() +
                            (jNode - dofsPerCellCFE) * dofsPerCellCFE + iNode);
                      }
                    else if (iNode >= dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
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
            if (!isConstantDofsAndQuadPointsInCellCFE)
              cumulativeDofQuadPointsOffsetCFE +=
                nQuadPointInCellClassicalBlock * dofsPerCellCFE;
            if (!isConstantDofsAndQuadPointsInCellEnrichBlockCFE)
              cumulativeDofQuadPointsOffsetEnrichBlockCFE +=
                nQuadPointInCellEnrichmentBlockClassical * dofsPerCellCFE;
            cumulativeDofQuadPointsOffsetEnrichBlockEFE +=
              nQuadPointInCellEnrichmentBlockEnrichment * dofsPerCell;
            cumulativeQuadEnrichBlockEnrichxenrichInCell += 
              numEnrichmentIdsInCell * nQuadPointInCellEnrichmentBlockEnrichment;
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
      computeBasisOverlapMatrixBlocked(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
          &                     basisOverlap,
        std::vector<size_type> &cellStartIdsBasisOverlap,
        std::vector<size_type> &dofsInCellVec,
        const size_type         cellBlockSize,   
        bool                    calculateWings = true)
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

        linearAlgebra::LinAlgOpContext<memorySpace>
          &linAlgOpContext = *eefeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext();

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
        size_type numCumulativeEnrichDofsxQuadEFEInAllCells = 0;

        dofsInCellVec.resize(numLocallyOwnedCells, 0);

        size_type       basisOverlapSize                    = 0;
        size_type       cellId                              = 0;

        const size_type dofsPerCellCFE = ccfeBDH->nCellDofs(cellId);

        auto locallyOwnedCellIter = eefeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != eefeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = eefeBDH->nCellDofs(cellId);
            basisOverlapSize += dofsInCellVec[cellId] * dofsInCellVec[cellId];
            numCumulativeEnrichDofsxQuadEFEInAllCells +=    
            (dofsInCellVec[cellId] - dofsPerCellCFE) * enrichmentBlockEnrichmentBasisDataStorage
                                      .getQuadratureRuleContainer()
                                      ->nCellQuadraturePoints(cellId);                                 
            cellId++;
          }

        basisOverlap = std::make_shared<
          utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          basisOverlapSize, (ValueTypeOperator)0);

        cellStartIdsBasisOverlap.resize(numLocallyOwnedCells, 0);

        size_type cellIndex   = 0;

      // et the enrichment values
      std::vector<double> quadValuesInAllCellsEnrichmentHost(numCumulativeEnrichDofsxQuadEFEInAllCells), quadGradientsInAllCellsEnrichmentHost;
          eefeBDH->getEnrichmentClassicalInterface()->getEnrichmentDataInAllCellsAtQuadPts(
            true,
            false,
            *enrichmentBlockEnrichmentBasisDataStorage.getQuadratureRuleContainer(),
            quadValuesInAllCellsEnrichmentHost.data(),
            quadGradientsInAllCellsEnrichmentHost.data(),
            *eefeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext());
        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          quadValuesInAllCellsEnrichment(quadValuesInAllCellsEnrichmentHost.size());
        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          quadValuesInAllCellsEnrichment.size(), quadValuesInAllCellsEnrichment.data(), quadValuesInAllCellsEnrichmentHost.data());

        auto coeffsInAllCellsHost = eefeBDH->getEnrichmentClassicalInterface()->getClassicalComponentCoeffsInAllCellsOEFE();
        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          coeffsInAllCells(coeffsInAllCellsHost.size());      
        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          coeffsInAllCellsHost.size(), coeffsInAllCells.data(), coeffsInAllCellsHost.data());

        const double* cellJxWValuesClassicalBlockPtr =
          classicalBlockBasisDataStorage.getQuadratureRuleContainer()
            ->template getJxWPtr<memorySpace>();

        const double* cellJxWValuesEnrichmentBlockClassicalPtr =
          enrichmentBlockClassicalBasisDataStorage
            .getQuadratureRuleContainer()
            ->template getJxWPtr<memorySpace>();

        const double* cellJxWValuesEnrichmentBlockEnrichmentPtr =
          enrichmentBlockEnrichmentBasisDataStorage
            .getQuadratureRuleContainer()
            ->template getJxWPtr<memorySpace>();

        size_type numCumulativeDofsxDofsCellsInBlock = 0;
        size_type cumulativeQuadEnrichBlockEnrichxenrichInCell = 0;
        size_type cumulativeCoeffsInCellRange = 0;

        for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
              cellStartId += cellBlockSize)
          {
            ValueTypeOperator* basisOverlapStartPtrInCellBlock = basisOverlap->data() + numCumulativeDofsxDofsCellsInBlock;

            size_type cellQuadStartIdsClassicalBlock = classicalBlockBasisDataStorage
              .getQuadratureRuleContainer()->getCellQuadStartId(cellStartId);
            size_type cellQuadStartIdsEnrichmentBlockClassical = enrichmentBlockClassicalBasisDataStorage
              .getQuadratureRuleContainer()->getCellQuadStartId(cellStartId);
            size_type cellQuadStartIdsEnrichmentBlockEnrichment = enrichmentBlockEnrichmentBasisDataStorage
              .getQuadratureRuleContainer()->getCellQuadStartId(cellStartId);

            const size_type cellEndId =
              std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
            const size_type numCellsInBlock = cellEndId - cellStartId;

            std::vector<size_type> dofsPerCellInCellBlock(numCellsInBlock, 0),
              numEnrichmentIdsInCellBlock(numCellsInBlock, 0),
              nQuadPointInCellBlockClassicalBlock(numCellsInBlock, 0),
              nQuadPointInCellBlockEnrichmentBlockClassical(numCellsInBlock, 0),
              nQuadPointInCellBlockEnrichmentBlockEnrichment(numCellsInBlock, 0);

            std::vector<size_type> mSizes(numCellsInBlock, 0);
            std::vector<size_type> nSizes(numCellsInBlock, 0);
            std::vector<size_type> kSizes(numCellsInBlock, 0);
            std::vector<size_type> ldaSizes(numCellsInBlock, 0);
            std::vector<size_type> ldbSizes(numCellsInBlock, 0);
            std::vector<size_type> ldcSizes(numCellsInBlock, 0);
            std::vector<size_type> strideA(numCellsInBlock, 0);
            std::vector<size_type> strideB(numCellsInBlock, 0);
            std::vector<size_type> strideC(numCellsInBlock, 0);

            std::copy(dofsInCellVec.begin() + cellStartId,
                      dofsInCellVec.begin() + cellEndId,
                      dofsPerCellInCellBlock.begin());

            size_type cumulativeDofsCFExQuadClassicalBlock = 0;            
            size_type cumulativeDofsCFExQuadEnrichmentBlockClassical = 0;
            size_type cumulativeDofsxQuadEnrichmentBlockEnrichment = 0;
            size_type cumulativeEnrichxQuadEnrichmentBlockClassical = 0;
            size_type cumulativeEnrichxQuadEnrichmentBlockEnrichment = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
              {
                cellIndex = iCell + cellStartId;

                numEnrichmentIdsInCellBlock[iCell] = dofsPerCellInCellBlock[iCell] - dofsPerCellCFE;                       
                                            
                nQuadPointInCellBlockClassicalBlock[iCell] = 
                  classicalBlockBasisDataStorage.getQuadratureRuleContainer()
                    ->nCellQuadraturePoints(cellIndex);   
                nQuadPointInCellBlockEnrichmentBlockClassical[iCell] = 
                  enrichmentBlockClassicalBasisDataStorage
                    .getQuadratureRuleContainer()
                    ->nCellQuadraturePoints(cellIndex);                
                nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell] =  
                  enrichmentBlockEnrichmentBasisDataStorage
                    .getQuadratureRuleContainer()
                    ->nCellQuadraturePoints(cellIndex);     
                    
                cumulativeDofsCFExQuadClassicalBlock +=
                  dofsPerCellCFE * nQuadPointInCellBlockClassicalBlock[iCell];
                cumulativeDofsCFExQuadEnrichmentBlockClassical +=     
                  dofsPerCellCFE * nQuadPointInCellBlockEnrichmentBlockClassical[iCell];                
                cumulativeDofsxQuadEnrichmentBlockEnrichment  += 
                  dofsPerCellInCellBlock[iCell] * nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];

                cumulativeEnrichxQuadEnrichmentBlockClassical += numEnrichmentIdsInCellBlock[iCell] * 
                  nQuadPointInCellBlockEnrichmentBlockClassical[iCell];  
                cumulativeEnrichxQuadEnrichmentBlockEnrichment += numEnrichmentIdsInCellBlock[iCell] * 
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];                     
              }

            //------- basis data in cell range ------
            std::pair<size_type, size_type> cellPair(cellStartId, cellEndId);

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              basisDataInCellRangeClassicalBlock(
                cumulativeDofsCFExQuadClassicalBlock);
            classicalBlockBasisDataStorage.getBasisDataInCellRange(
                  cellPair, basisDataInCellRangeClassicalBlock);

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              basisDataInCellRangeEnrichmentBlockClassical(
                cumulativeDofsCFExQuadEnrichmentBlockClassical);
            enrichmentBlockClassicalBasisDataStorage.getBasisDataInCellRange(
                  cellPair, basisDataInCellRangeEnrichmentBlockClassical);

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              basisDataInCellRangeEnrichmentBlockEnrichment(
                cumulativeDofsxQuadEnrichmentBlockEnrichment);
            enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInCellRange(
              cellPair, basisDataInCellRangeEnrichmentBlockEnrichment);
            //------ basis data in cell range -------

            utils::MemoryStorage<ValueTypeOperator, memorySpace> 
              classicalComponentInQuadValuesEC(
                cumulativeEnrichxQuadEnrichmentBlockClassical);

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              classicalComponentInQuadValuesEE(
                cumulativeEnrichxQuadEnrichmentBlockEnrichment);

            // Do a gemm (\Sigma c_i N_i^classical)
            std::vector<char>      transA(numCellsInBlock, 'N');
            std::vector<char>      transB(numCellsInBlock, 'N');
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                kSizes[iCell]   = dofsPerCellCFE;
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = kSizes[iCell];
                ldcSizes[iCell] = mSizes[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell];
              }
                  
                linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperator,
                                                ValueTypeOperator,
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
                  (ValueTypeOperator)1.0,
                  coeffsInAllCells.data() + cumulativeCoeffsInCellRange,
                  ldaSizes.data(),
                  basisDataInCellRangeEnrichmentBlockClassical.data(),
                  ldbSizes.data(),
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEC.data(),
                  ldcSizes.data(),
                  linAlgOpContext);

              for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                {
                  mSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                  nSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                  kSizes[iCell]   = dofsPerCellCFE;
                  ldaSizes[iCell] = mSizes[iCell];
                  ldbSizes[iCell] = dofsPerCellInCellBlock[iCell];
                  ldcSizes[iCell] = mSizes[iCell];
                  strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                  strideB[iCell]  = dofsPerCellInCellBlock[iCell] * nSizes[iCell];
                  strideC[iCell]  = mSizes[iCell] * nSizes[iCell];
                }

                // Do a gemm (\Sigma c_i N_i^classical) for enrichment quad rule

                linearAlgebra::blasLapack::gemmStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
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
                  (ValueTypeOperator)1.0,
                  coeffsInAllCells.data() + cumulativeCoeffsInCellRange,
                  ldaSizes.data(),
                  basisDataInCellRangeEnrichmentBlockEnrichment.data(),
                  ldbSizes.data(),
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEE.data(),
                  ldcSizes.data(),
                  linAlgOpContext);

            // strideC for sub-blocks written directly into basisOverlap:
            //   strideC_full[i]       = dofsPerCell[i]^2  (CC, EC)
            //   strideC_colOffset[i]  = dofsPerCell[i]^2 +
            //     dofsPerCFE*(dofsPerCell[i+1]-dofsPerCell[i])  (CE, EE)                  
            std::vector<size_type> strideC_full(numCellsInBlock, 0);
            std::vector<size_type> strideC_colOffset(numCellsInBlock, 0);
           for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                strideC_full[iCell] =
                  dofsPerCellInCellBlock[iCell] * dofsPerCellInCellBlock[iCell];
                strideC_colOffset[iCell] =
                  strideC_full[iCell] +
                  ((iCell + 1 < numCellsInBlock)
                     ? dofsPerCellCFE * (dofsPerCellInCellBlock[iCell + 1] -
                                         dofsPerCellInCellBlock[iCell])
                     : 0);
              }
              
            // ------------------- Classical - Classical Block --------------------
            size_type JxWxNCellSize = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = 1;
                nSizes[iCell]   = dofsPerCellCFE;
                kSizes[iCell]   = nQuadPointInCellBlockClassicalBlock[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
              }
            utils::MemoryStorage<ValueTypeOperator, memorySpace> JxWxNCell(
              JxWxNCellSize);

            linearAlgebra::blasLapack::scaleStridedVarBatched<
              ValueTypeOperator,
              ValueTypeOperator,
              memorySpace>(
              numCellsInBlock,
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              cellJxWValuesClassicalBlockPtr + cellQuadStartIdsClassicalBlock,
              basisDataInCellRangeClassicalBlock.data(),
              JxWxNCell.data(),
              linAlgOpContext);

            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'C');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = dofsPerCellCFE;
                nSizes[iCell]   = dofsPerCellCFE;
                kSizes[iCell]   = nQuadPointInCellBlockClassicalBlock[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = nSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell]; 
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_full[iCell]; 
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                            ValueTypeOperand,
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
              (ValueTypeOperand)1.0,
              JxWxNCell.data(),
              ldaSizes.data(),
              basisDataInCellRangeClassicalBlock.data(),
              ldbSizes.data(),
              (ValueTypeOperand)0.0,
              basisOverlapStartPtrInCellBlock,
              ldcSizes.data(),
              linAlgOpContext);
            // ------------------- Classical - Classical Block --------------------
  
            // ------------------- Classical - Enrichment Block --------------------
            if(calculateWings)
            {
            // Pristine - with classical 
            JxWxNCellSize = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = 1;
                nSizes[iCell]   = dofsPerCellInCellBlock[iCell];
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
              }

              JxWxNCell.resize(JxWxNCellSize, 0);

              linearAlgebra::blasLapack::scaleStridedVarBatched<
                ValueTypeOperator,
                ValueTypeOperator,
                memorySpace>(
                numCellsInBlock,
                linearAlgebra::blasLapack::Layout::ColMajor,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                strideA.data(),
                strideB.data(),
                strideC.data(),
                mSizes.data(),
                nSizes.data(),
                kSizes.data(),
                cellJxWValuesEnrichmentBlockEnrichmentPtr + cellQuadStartIdsEnrichmentBlockEnrichment,
                basisDataInCellRangeEnrichmentBlockEnrichment.data(),
                JxWxNCell.data(),
                linAlgOpContext);

            // CE block
            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'C');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = dofsPerCellCFE;
                nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                ldaSizes[iCell] = dofsPerCellInCellBlock[iCell];
                ldbSizes[iCell] = nSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = dofsPerCellInCellBlock[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_colOffset[iCell]; 
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                            ValueTypeOperand,
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
                  (ValueTypeOperand)1.0,
                  JxWxNCell.data(),
                  ldaSizes.data(),
                  quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
                  ldbSizes.data(),
                  (ValueTypeOperand)0.0,
                  basisOverlapStartPtrInCellBlock + dofsPerCellInCellBlock[0] * dofsPerCellCFE,
                  ldcSizes.data(),
                  linAlgOpContext);

              // EC block
            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'T');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell]   = dofsPerCellCFE;
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = dofsPerCellInCellBlock[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * dofsPerCellInCellBlock[iCell];
                strideC[iCell]  = strideC_full[iCell];
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                            ValueTypeOperand,
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
                  (ValueTypeOperand)1.0,
                  quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
                  ldaSizes.data(),
                  JxWxNCell.data(),
                  ldbSizes.data(),
                  (ValueTypeOperand)0.0,
                  basisOverlapStartPtrInCellBlock + dofsPerCellCFE,
                  ldcSizes.data(),
                  linAlgOpContext);

              // CiNi interpolated - with classical 
              JxWxNCellSize = 0;
              for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                {
                  mSizes[iCell]   = 1;
                  nSizes[iCell]   = dofsPerCellCFE;
                  kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                  strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                  strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                  strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                  JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
                }

                JxWxNCell.resize(JxWxNCellSize,
                                 0);

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  memorySpace>(
                  numCellsInBlock,
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::ScalarOp::Identity,
                  linearAlgebra::blasLapack::ScalarOp::Identity,
                  strideA.data(),
                  strideB.data(),
                  strideC.data(),
                  mSizes.data(),
                  nSizes.data(),
                  kSizes.data(),
                  cellJxWValuesEnrichmentBlockClassicalPtr + cellQuadStartIdsEnrichmentBlockClassical,
                  basisDataInCellRangeEnrichmentBlockClassical.data(),
                  JxWxNCell.data(),
                  linAlgOpContext);

                // CE block
                std::fill(transA.begin(), transA.end(), 'N');
                std::fill(transB.begin(), transB.end(), 'C');

                for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                  {
                    mSizes[iCell]   = dofsPerCellCFE;
                    nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                    kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                    ldaSizes[iCell] = mSizes[iCell];
                    ldbSizes[iCell] = nSizes[iCell];
                    ldcSizes[iCell] = dofsPerCellInCellBlock[iCell]; 
                    strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                    strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                    strideC[iCell]  = strideC_colOffset[iCell];
                  }

                linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                                ValueTypeOperator,
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
                  (ValueTypeOperand)-1.0,
                  JxWxNCell.data(),
                  ldaSizes.data(),
                  classicalComponentInQuadValuesEC.data(),
                  ldbSizes.data(),
                  (ValueTypeOperator)1.0,
                  basisOverlapStartPtrInCellBlock + dofsPerCellInCellBlock[0] * dofsPerCellCFE,
                  ldcSizes.data(),
                  linAlgOpContext);

              // EC block
                std::fill(transA.begin(), transA.end(), 'N');
                std::fill(transB.begin(), transB.end(), 'T');

                for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                  {
                    mSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                    nSizes[iCell]   = dofsPerCellCFE;
                    kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                    ldaSizes[iCell] = mSizes[iCell];
                    ldbSizes[iCell] = nSizes[iCell];
                    ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                    strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                    strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                    strideC[iCell]  = strideC_full[iCell];
                  }

                linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                                ValueTypeOperator,
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
                  (ValueTypeOperand)-1.0,
                  classicalComponentInQuadValuesEC.data(),                  
                  ldaSizes.data(),
                  JxWxNCell.data(),                  
                  ldbSizes.data(),
                  (ValueTypeOperator)1.0,
                  basisOverlapStartPtrInCellBlock + dofsPerCellCFE,
                  ldcSizes.data(),
                  linAlgOpContext);
            }
            // ------------------- Classical - Enrichment Block --------------------

            // ------------------- Enrichment - Enrichment Block --------------------
            // pristine with pristine
            JxWxNCellSize = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = 1;
                nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
              }
              JxWxNCell.resize(JxWxNCellSize, 0);

              linearAlgebra::blasLapack::scaleStridedVarBatched<
                ValueTypeOperator,
                ValueTypeOperator,
                memorySpace>(
                numCellsInBlock,
                linearAlgebra::blasLapack::Layout::ColMajor,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                strideA.data(),
                strideB.data(),
                strideC.data(),
                mSizes.data(),
                nSizes.data(),
                kSizes.data(),
                cellJxWValuesEnrichmentBlockEnrichmentPtr + cellQuadStartIdsEnrichmentBlockEnrichment,
                quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
                JxWxNCell.data(),
                linAlgOpContext);

            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'C');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = mSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_colOffset[iCell];
              }

              linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                              ValueTypeOperand,
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
                (ValueTypeOperand)1.0,
                JxWxNCell.data(),
                ldaSizes.data(),
                quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
                ldbSizes.data(),
                (ValueTypeOperand)0.0,
                basisOverlapStartPtrInCellBlock + dofsPerCellInCellBlock[0] * dofsPerCellCFE + dofsPerCellCFE,
                ldcSizes.data(),
                linAlgOpContext);

              // interpolated ci's in Ni_classicalQuadrature of Mc = d
              // * interpolated ci's in Ni_classicalQuadrature of Mc =
              // d
            JxWxNCellSize = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = 1;
                nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
              }
              JxWxNCell.resize(JxWxNCellSize,0);

              linearAlgebra::blasLapack::scaleStridedVarBatched<
                ValueTypeOperator,
                ValueTypeOperator,
                memorySpace>(
                numCellsInBlock,
                linearAlgebra::blasLapack::Layout::ColMajor,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                strideA.data(),
                strideB.data(),
                strideC.data(),
                mSizes.data(),
                nSizes.data(),
                kSizes.data(),
                cellJxWValuesEnrichmentBlockClassicalPtr + cellQuadStartIdsEnrichmentBlockClassical,
                classicalComponentInQuadValuesEC.data(),
                JxWxNCell.data(),
                linAlgOpContext);


            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'C');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = nSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell]; 
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_colOffset[iCell];
              }

              linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                              ValueTypeOperator,
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
                (ValueTypeOperand)1.0,
                JxWxNCell.data(),
                ldaSizes.data(),
                classicalComponentInQuadValuesEC.data(),
                ldbSizes.data(),
                (ValueTypeOperator)1.0,
                basisOverlapStartPtrInCellBlock + dofsPerCellInCellBlock[0] * dofsPerCellCFE + dofsPerCellCFE,
                ldcSizes.data(),
                linAlgOpContext);

              // Ni_pristine* interpolated ci's in
              // Ni_classicalQuadratureOfPristine at quadpoints

              JxWxNCellSize = 0;
              for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                {
                  mSizes[iCell]   = 1;
                  nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                  kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                  strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                  strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                  strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                  JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
                }
              JxWxNCell.resize(JxWxNCellSize, 0);  
              
              linearAlgebra::blasLapack::scaleStridedVarBatched<
                ValueTypeOperator,
                ValueTypeOperator,
                memorySpace>(
                numCellsInBlock,
                linearAlgebra::blasLapack::Layout::ColMajor,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                linearAlgebra::blasLapack::ScalarOp::Identity,
                strideA.data(),
                strideB.data(),
                strideC.data(),
                mSizes.data(),
                nSizes.data(),
                kSizes.data(),
                cellJxWValuesEnrichmentBlockEnrichmentPtr + cellQuadStartIdsEnrichmentBlockEnrichment,
                quadValuesInAllCellsEnrichment.data() + cumulativeQuadEnrichBlockEnrichxenrichInCell,
                JxWxNCell.data(),
                linAlgOpContext);

            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'C');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = nSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_colOffset[iCell];
              }

              linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                              ValueTypeOperator,
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
                (ValueTypeOperand)-1.0,
                classicalComponentInQuadValuesEE.data(),
                ldaSizes.data(),
                JxWxNCell.data(),
                ldbSizes.data(),
                (ValueTypeOperator)1.0,
                basisOverlapStartPtrInCellBlock + dofsPerCellInCellBlock[0] * dofsPerCellCFE + dofsPerCellCFE,
                ldcSizes.data(),
                linAlgOpContext);

            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'T');

              for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
                {
                mSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell]   = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell]   = nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = nSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = strideC_colOffset[iCell];
              }

              linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                              ValueTypeOperator,
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
                (ValueTypeOperand)-1.0,
                JxWxNCell.data(),
                ldaSizes.data(),
                classicalComponentInQuadValuesEE.data(),
                ldbSizes.data(),
                (ValueTypeOperator)1.0,
                basisOverlapStartPtrInCellBlock + dofsPerCellInCellBlock[0] * dofsPerCellCFE + dofsPerCellCFE,
                ldcSizes.data(),
                linAlgOpContext);

            for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
              {
                cellStartIdsBasisOverlap[cellStartId + iCell] =
                  numCumulativeDofsxDofsCellsInBlock;
                numCumulativeDofsxDofsCellsInBlock +=
                  dofsPerCellInCellBlock[iCell] * dofsPerCellInCellBlock[iCell];
                cumulativeCoeffsInCellRange +=
                  dofsPerCellCFE * numEnrichmentIdsInCellBlock[iCell];
                cumulativeQuadEnrichBlockEnrichxenrichInCell +=
                  numEnrichmentIdsInCellBlock[iCell] *
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
              }
          }
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

            // utils::MemoryStorage<size_type, memorySpace> cellsInBlockNumDoFs(
            //   numCellsInBlock);
            // cellsInBlockNumDoFs.copyFrom(cellsInBlockNumDoFsSTL);

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
                                      // cellsInBlockNumDoFs,
                                      cellsInBlockNumCumulativeDoFs,
                                      xCellValues);

            std::vector<char> transA(numCellsInBlock, 'N');
            std::vector<char> transB(numCellsInBlock, 'N');

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
                mSizes[iCell]   = numVecs;
                nSizes[iCell]   = cellsInBlockNumDoFsSTL[iCell];
                kSizes[iCell]   = cellsInBlockNumDoFsSTL[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = kSizes[iCell];
                ldcSizes[iCell] = mSizes[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = mSizes[iCell] * nSizes[iCell];
              }

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
              memorySpace>::
              addCellWiseDataToFieldData(yCellValues,
                                         numVecs,
                                         cellLocalIdsStartPtrY +
                                           cellLocalIdsOffset,
                                         // cellsInBlockNumDoFs,
                                         cellsInBlockNumCumulativeDoFs,
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
      OrthoEFEOverlapOperatorContextInternal::computeBasisOverlapMatrixBlocked<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockBasisDataStorage,
             enrichmentBlockEnrichmentBasisDataStorage,
             enrichmentBlockClassicalBasisDataStorage,
             d_basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell,
             BasisDataStorageDefaults::CELL_BATCH_SIZE,
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

      std::shared_ptr<Storage> basisOverlap;
      OrthoEFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockGLLBasisDataStorage,
             enrichmentBlockBasisDataStorage,
             basisOverlap,
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

      const size_type numCumulativeDofsCells =
        std::accumulate(locallyOwnedCellsNumDoFsSTL.begin(),
                        locallyOwnedCellsNumDoFsSTL.end(),
                        0);

      d_diagonal =
        std::make_shared<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>(
          d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      // Create the diagonal of the classical block matrix which is diagonal for
      // GLL with spectral quadrature
      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(basisOverlap->data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           numCumulativeDofsCells,
                                           d_diagonal->data());

      d_feBasisManager->getConstraints().distributeChildToParent(*d_diagonal,
                                                                 1);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      d_diagonal->accumulateAddLocallyOwned();

      d_diagonal->updateGhostValues();

      d_feBasisManager->getConstraints().setConstrainedNodesToZero(*d_diagonal,
                                                                   1);

      utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
        basisOverlapHost(basisOverlap->size());
      basisOverlapHost.template copyFrom<memorySpace>(basisOverlap->data());

      // Now form the enrichment block matrix.
      if (d_isEnrichAtomBlockDiagonalApprox)
        {
          // utils::MemoryStorage<ValueTypeOperator, memorySpace>
          //   basisOverlapEnrichmentBlockExact(d_nglobalEnrichmentIds *
          //     d_nglobalEnrichmentIds);

          utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
            basisOverlapEnrichmentBlock(d_nglobalEnrichmentIds *
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
                      // *(basisOverlapEnrichmentBlockExact.data() +
                      //   enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                      //   enrichmentVecInCell[k]) +=
                      //   *(basisOverlap->data() + cumulativeBasisDataInCells
                      //   +
                      //     (numCellClassicalDofs + nCellEnrichmentDofs) *
                      //       (numCellClassicalDofs + j) +
                      //     numCellClassicalDofs + k);

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
                            *(basisOverlapHost.data() +
                              cumulativeBasisDataInCells +
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

          // int err = utils::mpi::MPIAllreduce<memorySpace>(
          //   utils::mpi::MPIInPlace,
          //   basisOverlapEnrichmentBlockExact.data(),
          //   basisOverlapEnrichmentBlockExact.size(),
          //   utils::mpi::MPIDouble,
          //   utils::mpi::MPISum,
          //   d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
          // std::pair<bool, std::string> mpiIsSuccessAndMsg =
          //   utils::mpi::MPIErrIsSuccessAndMsg(err);
          // utils::throwException(mpiIsSuccessAndMsg.first,
          //                       "MPI Error:" + mpiIsSuccessAndMsg.second);

          auto err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            basisOverlapEnrichmentBlock.data(),
            basisOverlapEnrichmentBlock.size(),
            utils::mpi::Types<ValueTypeOperator>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
          auto mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
          utils::throwException(mpiIsSuccessAndMsg.first,
                                "MPI Error:" + mpiIsSuccessAndMsg.second);


          // basisOverlapEnrichmentAtomBlock = basisOverlapEnrichmentBlock;
          // utils::MemoryStorage<double, memorySpace> eigenValuesMemSpace(
          //   d_nglobalEnrichmentIds);

          // linearAlgebra::blasLapack::heevd<ValueType, memorySpace>(
          //   linearAlgebra::blasLapack::Job::NoVec,
          //   linearAlgebra::blasLapack::Uplo::Lower,
          //   d_nglobalEnrichmentIds,
          //   basisOverlapEnrichmentAtomBlock.data(),
          //   d_nglobalEnrichmentIds,
          //   eigenValuesMemSpace.data(),
          //   *d_diagonal->getLinAlgOpContext());

          // auto minAbsValueOfEigenVecAtomBlock =
          // *(eigenValuesMemSpace.data());

          global_size_type globalEnrichmentStartId =
            efeBDH->getGlobalRanges()[1].first;

          std::pair<global_size_type, global_size_type> locOwnEidPair{
            efeBDH->getLocallyOwnedRanges()[1].first - globalEnrichmentStartId,
            efeBDH->getLocallyOwnedRanges()[1].second -
              globalEnrichmentStartId};

          global_size_type nlocallyOwnedEnrichmentIds =
            locOwnEidPair.second - locOwnEidPair.first;

          d_atomBlockEnrichmentOverlap.resize(nlocallyOwnedEnrichmentIds *
                                              nlocallyOwnedEnrichmentIds);

          utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
            atomBlockEnrichmentOverlapHost(nlocallyOwnedEnrichmentIds *
                                           nlocallyOwnedEnrichmentIds);

          for (global_size_type i = 0; i < nlocallyOwnedEnrichmentIds; i++)
            {
              for (global_size_type j = 0; j < nlocallyOwnedEnrichmentIds; j++)
                {
                  *(atomBlockEnrichmentOverlapHost.data() +
                    i * nlocallyOwnedEnrichmentIds + j) =
                    *(basisOverlapEnrichmentBlock.data() +
                      (i + locOwnEidPair.first) * d_nglobalEnrichmentIds +
                      (j + locOwnEidPair.first));
                }
            }

          d_atomBlockEnrichmentOverlap
            .template copyFrom<utils::MemorySpace::HOST>(
              atomBlockEnrichmentOverlapHost.data());

          // int rank;
          // utils::mpi::MPICommRank(
          //   d_feBasisManager->getMPIPatternP2P()->mpiCommunicator(), &rank);

          // utils::ConditionalOStream rootCout(std::cout);
          // rootCout.setCondition(rank == 0);

          // ValueTypeOperator normMexact = 0;
          // for (int i = 0; i < basisOverlapEnrichmentBlock.size(); i++)
          //   {
          //     *(basisOverlapEnrichmentBlock.data() + i) =
          //       *(basisOverlapEnrichmentBlockExact.data() + i) -
          //       *(basisOverlapEnrichmentBlock.data() + i);

          //     normMexact += *(basisOverlapEnrichmentBlockExact.data()
          //     + i) * *(basisOverlapEnrichmentBlockExact.data() + i);
          //   }
          // normMexact = std::sqrt(normMexact);

          // linearAlgebra::blasLapack::heevd<ValueType, memorySpace>(
          //   linearAlgebra::blasLapack::Job::Vec,
          //   linearAlgebra::blasLapack::Uplo::Lower,
          //   d_nglobalEnrichmentIds,
          //   basisOverlapEnrichmentBlock.data(),
          //   d_nglobalEnrichmentIds,
          //   eigenValuesMemSpace.data(),
          //   *d_diagonal->getLinAlgOpContext());

          // ValueType tolerance = 1e-10;

          // ValueType eigValShift = std::min(*eigenValuesMemSpace.data(), 0.) -
          // tolerance; utils::throwException(minAbsValueOfEigenVecAtomBlock +
          // eigValShift > 0,
          //   "The min eigenvalue of AtomBlockDiagonal is less than shift with
          //   the residual. values: " +
          //     std::to_string(minAbsValueOfEigenVecAtomBlock) + " " +
          //     std::to_string(eigValShift));

          // for(int i = 0; i < eigenValuesMemSpace.size() ;i++)
          //   *(eigenValuesMemSpace.data() + i) -= eigValShift;

          // size_type  i = d_nglobalEnrichmentIds-1;
          // while(i > 0)
          // {
          //   ValueTypeOperand sumEigVal = 0;
          //   for(size_type j = i ; j < d_nglobalEnrichmentIds ; j++)
          //   {
          //     sumEigVal += *(eigenValuesMemSpace.data() + j) *
          //       *(eigenValuesMemSpace.data() + j);
          //   }
          //   if(std::sqrt(sumEigVal)/normMexact < 1e-6)
          //     break;
          //   i--;
          //   d_rank++;
          // }

          // rootCout << "Rank of Residual M Enrichment block matrix: " <<
          // d_rank << "\n";

          // d_residualEnrichOverlapEigenVec.resize(nlocallyOwnedEnrichmentIds *
          //                                             d_rank,
          //                                           0);
          // d_residualEnrichOverlapEigenVal.resize(d_rank, 0);

          // for (int i = 0; i < d_rank; i++)
          //   {
          //     basisOverlapEnrichmentBlock.template copyTo<memorySpace>(
          //       d_residualEnrichOverlapEigenVec.begin(),
          //       nlocallyOwnedEnrichmentIds,
          //       d_nglobalEnrichmentIds * (i + d_nglobalEnrichmentIds -
          //       d_rank) +
          //         locOwnEidPair.first, // srcoffset
          //       nlocallyOwnedEnrichmentIds *
          //         i); // dstoffset ; col - d_rank, row , N_locowned

          //     *(d_residualEnrichOverlapEigenVal.data() + i) =
          //       *(eigenValuesMemSpace.data() + (i + d_nglobalEnrichmentIds -
          //       d_rank)) - eigValShift;
          //   }
        }
      else
        {
          d_basisOverlapEnrichmentBlock = std::make_shared<
            utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
            d_nglobalEnrichmentIds * d_nglobalEnrichmentIds);

          std::vector<ValueTypeOperator> basisOverlapEnrichmentBlockSTL(
            d_nglobalEnrichmentIds * d_nglobalEnrichmentIds, 0);

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
                      *(basisOverlapEnrichmentBlockSTL.data() +
                        enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                        enrichmentVecInCell[k]) +=
                        *(basisOverlapHost.data() + cumulativeBasisDataInCells +
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
            utils::mpi::MPIInPlace,
            basisOverlapEnrichmentBlockSTL.data(),
            basisOverlapEnrichmentBlockSTL.size(),
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

      std::shared_ptr<Storage> basisOverlap;
      OrthoEFEOverlapOperatorContextInternal::computeBasisOverlapMatrixBlocked<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockGLLBasisDataStorage,
             enrichmentBlockEnrichmentBasisDataStorage,
             enrichmentBlockClassicalBasisDataStorage,
             basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell,
             BasisDataStorageDefaults::CELL_BATCH_SIZE,
             false);

      std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                         0);
      std::copy(numCellDofs.begin(),
                numCellDofs.begin() + numLocallyOwnedCells,
                locallyOwnedCellsNumDoFsSTL.begin());

      utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
        numLocallyOwnedCells);
      locallyOwnedCellsNumDoFs.template copyFrom(locallyOwnedCellsNumDoFsSTL);

      const size_type numCumulativeDofsCells =
        std::accumulate(locallyOwnedCellsNumDoFsSTL.begin(),
                        locallyOwnedCellsNumDoFsSTL.end(),
                        0);

      d_diagonal =
        std::make_shared<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>(
          d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      // Create the diagonal of the classical block matrix which is diagonal for
      // GLL with spectral quadrature
      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(basisOverlap->data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           numCumulativeDofsCells,
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

      utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
        basisOverlapHost(basisOverlap->size());
      basisOverlapHost.template copyFrom<memorySpace>(basisOverlap->data());

      if (d_isEnrichAtomBlockDiagonalApprox)
        {
          std::pair<global_size_type, global_size_type> locOwnPair =
            efebasisDofHandler.getEnrichmentIdsPartition()
              ->locallyOwnedEnrichmentIds();

          std::vector<global_size_type> ghostVec =
            efebasisDofHandler.getEnrichmentIdsPartition()
              ->ghostEnrichmentIds();

          std::shared_ptr<
            const utils::mpi::MPIPatternP2P<utils::MemorySpace::HOST>>
            mpiPatternP2P = std::make_shared<
              const utils::mpi::MPIPatternP2P<utils::MemorySpace::HOST>>(
              std::vector<std::pair<global_size_type, global_size_type>>{
                locOwnPair},
              ghostVec,
              d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());

          global_size_type globalEnrichmentStartId =
            efeBDH->getGlobalRanges()[1].first;

          std::pair<global_size_type, global_size_type> locOwnEidPair{
            efeBDH->getLocallyOwnedRanges()[1].first - globalEnrichmentStartId,
            efeBDH->getLocallyOwnedRanges()[1].second -
              globalEnrichmentStartId};

          global_size_type nlocallyOwnedEnrichmentIds =
            locOwnEidPair.second - locOwnEidPair.first;

          size_type nLocalEnrichmentIds =
            ghostVec.size() + nlocallyOwnedEnrichmentIds;

          d_atomBlockEnrichmentOverlap.resize(nlocallyOwnedEnrichmentIds *
                                              nlocallyOwnedEnrichmentIds);

          utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
            atomBlockEnrichmentOverlapHost(nlocallyOwnedEnrichmentIds *
                                           nlocallyOwnedEnrichmentIds);

          global_size_type enrichBatchSize = 5000;
          for (global_size_type enrichStartId = 0;
               enrichStartId < d_nglobalEnrichmentIds;
               enrichStartId += enrichBatchSize)
            {
              const size_type enrichEndId =
                std::min(enrichStartId + enrichBatchSize,
                         d_nglobalEnrichmentIds);
              const size_type numEnrichInBatch = enrichEndId - enrichStartId;

              linearAlgebra::MultiVector<ValueType, utils::MemorySpace::HOST>
                basisOverlapEnrichmentBlock(
                  mpiPatternP2P,
                  linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST,
                  numEnrichInBatch);

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
                          if (enrichmentVecInCell[k] >= enrichStartId &&
                              enrichmentVecInCell[k] < enrichEndId)
                            {
                              basis::EnrichmentIdAttribute eIdAttrj =
                                efeBDH->getEnrichmentIdsPartition()
                                  ->getEnrichmentIdAttribute(
                                    enrichmentVecInCell[j]);

                              basis::EnrichmentIdAttribute eIdAttrk =
                                efeBDH->getEnrichmentIdsPartition()
                                  ->getEnrichmentIdAttribute(
                                    enrichmentVecInCell[k]);

                              if (eIdAttrj.atomId == eIdAttrk.atomId)
                                {
                                  *(basisOverlapEnrichmentBlock.data() +
                                    mpiPatternP2P->globalToLocal(
                                      enrichmentVecInCell[j]) *
                                      numEnrichInBatch +
                                    (enrichmentVecInCell[k] - enrichStartId)) +=
                                    *(basisOverlapHost.data() +
                                      cumulativeBasisDataInCells +
                                      (numCellClassicalDofs +
                                       nCellEnrichmentDofs) *
                                        (numCellClassicalDofs + j) +
                                      numCellClassicalDofs + k);
                                }
                            }
                        }
                    }
                  cumulativeBasisDataInCells +=
                    utils::mathFunctions::sizeTypePow((nCellEnrichmentDofs +
                                                       numCellClassicalDofs),
                                                      2);
                  cellId += 1;
                }

              basisOverlapEnrichmentBlock.accumulateAddLocallyOwned();

              for (global_size_type i = 0; i < nlocallyOwnedEnrichmentIds; i++)
                {
                  for (global_size_type j = 0; j < nlocallyOwnedEnrichmentIds;
                       j++)
                    {
                      if ((j + locOwnEidPair.first) >= enrichStartId &&
                          (j + locOwnEidPair.first) < enrichEndId)
                        *(atomBlockEnrichmentOverlapHost.data() +
                          i * nlocallyOwnedEnrichmentIds + j) =
                          *(basisOverlapEnrichmentBlock.data() +
                            mpiPatternP2P->globalToLocal(i +
                                                         locOwnEidPair.first) *
                              numEnrichInBatch +
                            (j + locOwnEidPair.first) - enrichStartId);
                    }
                }
            }

          d_atomBlockEnrichmentOverlap
            .template copyFrom<utils::MemorySpace::HOST>(
              atomBlockEnrichmentOverlapHost.data());

          // utils::MemoryStorage<ValueTypeOperator, memorySpace>
          //   basisOverlapEnrichmentBlock(d_nglobalEnrichmentIds *
          //                                 d_nglobalEnrichmentIds,
          //                               0);

          // size_type cellId                     = 0;
          // size_type cumulativeBasisDataInCells = 0;
          // for (auto enrichmentVecInCell :
          //      efebasisDofHandler.getEnrichmentIdsPartition()
          //        ->overlappingEnrichmentIdsInCells())
          //   {
          //     size_type nCellEnrichmentDofs = enrichmentVecInCell.size();
          //     for (unsigned int j = 0; j < nCellEnrichmentDofs; j++)
          //       {
          //         for (unsigned int k = 0; k < nCellEnrichmentDofs; k++)
          //           {
          //             basis::EnrichmentIdAttribute eIdAttrj =
          //               efeBDH->getEnrichmentIdsPartition()
          //                 ->getEnrichmentIdAttribute(enrichmentVecInCell[j]);

          //             basis::EnrichmentIdAttribute eIdAttrk =
          //               efeBDH->getEnrichmentIdsPartition()
          //                 ->getEnrichmentIdAttribute(enrichmentVecInCell[k]);

          //             if (eIdAttrj.atomId == eIdAttrk.atomId)
          //               {
          //                 *(basisOverlapEnrichmentBlock.data() +
          //                   enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
          //                   enrichmentVecInCell[k]) +=
          //                   *(basisOverlap->data() +
          //                     cumulativeBasisDataInCells +
          //                     (numCellClassicalDofs + nCellEnrichmentDofs) *
          //                       (numCellClassicalDofs + j) +
          //                     numCellClassicalDofs + k);
          //               }
          //           }
          //       }
          //     cumulativeBasisDataInCells +=
          //     utils::mathFunctions::sizeTypePow(
          //       (nCellEnrichmentDofs + numCellClassicalDofs), 2);
          //     cellId += 1;
          //   }

          // auto err = utils::mpi::MPIAllreduce<memorySpace>(
          //   utils::mpi::MPIInPlace,
          //   basisOverlapEnrichmentBlock.data(),
          //   basisOverlapEnrichmentBlock.size(),
          //   utils::mpi::MPIDouble,
          //   utils::mpi::MPISum,
          //   d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
          // auto mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
          // utils::throwException(mpiIsSuccessAndMsg.first,
          //                       "MPI Error:" + mpiIsSuccessAndMsg.second);

          // // global_size_type globalEnrichmentStartId =
          // //   efeBDH->getGlobalRanges()[1].first;

          // // std::pair<global_size_type, global_size_type> locOwnEidPair{
          // //   efeBDH->getLocallyOwnedRanges()[1].first -
          // globalEnrichmentStartId,
          // //   efeBDH->getLocallyOwnedRanges()[1].second -
          // //     globalEnrichmentStartId};

          // // global_size_type nlocallyOwnedEnrichmentIds =
          // //   locOwnEidPair.second - locOwnEidPair.first;

          // // d_atomBlockEnrichmentOverlap.resize(nlocallyOwnedEnrichmentIds *
          // //                                     nlocallyOwnedEnrichmentIds);

          // for (global_size_type i = 0; i < nlocallyOwnedEnrichmentIds; i++)
          //   {
          //     for (global_size_type j = 0; j < nlocallyOwnedEnrichmentIds;
          //     j++)
          //       {
          //         *(d_atomBlockEnrichmentOverlap.data() +
          //           i * nlocallyOwnedEnrichmentIds + j) =
          //           *(basisOverlapEnrichmentBlock.data() +
          //             (i + locOwnEidPair.first) * d_nglobalEnrichmentIds +
          //             (j + locOwnEidPair.first));
          //       }
          //   }

          // for(int i = 0 ; i < atomBlockEnrichmentOverlap.size() ; i++)
          // {
          //   if(std::abs( *(atomBlockEnrichmentOverlap.data() + i) -
          //   *(d_atomBlockEnrichmentOverlap.data() + i)) > 1e-12)
          //   {
          //     std::cout << i << "\t" << *(atomBlockEnrichmentOverlap.data() +
          //     i) << "\t" << *(d_atomBlockEnrichmentOverlap.data() + i) <<
          //     std::flush << "\n";
          //   }
          // }
        }
      else
        {
          d_basisOverlapEnrichmentBlock = std::make_shared<
            utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
            d_nglobalEnrichmentIds * d_nglobalEnrichmentIds);

          std::vector<ValueTypeOperator> basisOverlapEnrichmentBlockSTL(
            d_nglobalEnrichmentIds * d_nglobalEnrichmentIds, 0);

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
                      *(basisOverlapEnrichmentBlockSTL.data() +
                        enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                        enrichmentVecInCell[k]) +=
                        *(basisOverlapHost.data() + cumulativeBasisDataInCells +
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
            utils::mpi::MPIInPlace,
            basisOverlapEnrichmentBlockSTL.data(),
            basisOverlapEnrichmentBlockSTL.size(),
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

          if (!d_isEnrichAtomBlockDiagonalApprox)
            {
              utils::MemoryStorage<ValueTypeOperand, memorySpace>
                XenrichedGlobalVec(d_nglobalEnrichmentIds * numComponents),
                YenrichedGlobalVec(d_nglobalEnrichmentIds * numComponents);

              utils::MemoryStorage<ValueTypeOperand, utils::MemorySpace::HOST>
                XenrichedGlobalVecTmp(d_nglobalEnrichmentIds * numComponents);

              XenrichedGlobalVecTmp.template copyFrom<memorySpace>(
                X.begin(),
                nlocallyOwnedEnrichmentIds * numComponents,
                nlocallyOwnedClassicalIds * numComponents,
                ((d_feBasisManager->getLocallyOwnedRanges()[1].first) -
                 (d_efebasisDofHandler->getGlobalRanges()[0].second)) *
                  numComponents);

              int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                XenrichedGlobalVecTmp.data(),
                XenrichedGlobalVecTmp.size(),
                utils::mpi::Types<ValueTypeOperand>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
              std::pair<bool, std::string> mpiIsSuccessAndMsg =
                utils::mpi::MPIErrIsSuccessAndMsg(err);
              utils::throwException(mpiIsSuccessAndMsg.first,
                                    "MPI Error:" + mpiIsSuccessAndMsg.second);

              XenrichedGlobalVec.template copyFrom<utils::MemorySpace::HOST>(
                XenrichedGlobalVecTmp);

              // Do dgemm

              ValueType alpha = 1.0;
              ValueType beta  = 0.0;

              linearAlgebra::blasLapack::
                gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
                  'N',
                  'T',
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
              utils::MemoryStorage<ValueTypeOperand, memorySpace>
                XenrichedLocalVec(nlocallyOwnedEnrichmentIds * numComponents),
                YenrichedLocalVec(nlocallyOwnedEnrichmentIds * numComponents);

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
                    'N',
                    'N',
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

              YenrichedLocalVec.template copyTo<memorySpace>(
                Y.begin(),
                nlocallyOwnedEnrichmentIds * numComponents,
                0,
                nlocallyOwnedClassicalIds * numComponents);
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
      if (d_isMassLumping)
        utils::throwException(
          false,
          "Could not getBasisOverlapInAllCells if Masslumping is done in Overlap Operator. ");
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
      if (d_isMassLumping)
        utils::throwException(
          false,
          "Could not getBasisOverlapInCell if Masslumping is done in Overlap Operator. ");
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
      if (d_isMassLumping)
        utils::throwException(
          false,
          "Could not getBasisOverlap if Masslumping is done in Overlap Operator. ");
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
