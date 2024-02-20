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

namespace dftefe
{
  namespace basis
  {
    namespace EFEOverlapOperatorContextInternal
    {
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeBasisOverlapMatrix(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &feBasisDataStorage,
        std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
          &                     basisOverlap,
        std::vector<size_type> &cellStartIdsBasisOverlap,
        std::vector<size_type> &dofsInCellVec)
      {
        std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                                 ValueTypeOperator,
                                                 memorySpace,
                                                 dim>>
          feBDH = std::dynamic_pointer_cast<
            const EFEBasisDofHandler<ValueTypeOperand,
                                     ValueTypeOperator,
                                     memorySpace,
                                     dim>>(
            feBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          feBDH != nullptr,
          "Could not cast BasisDofHandler to EFEBasisDofHandler "
          "in EFEOverlapOperatorContext");

        const size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
        dofsInCellVec.resize(numLocallyOwnedCells, 0);
        cellStartIdsBasisOverlap.resize(numLocallyOwnedCells, 0);
        size_type cumulativeBasisOverlapId = 0;

        size_type       cellId           = 0;
        size_type       basisOverlapSize = 0;
        const size_type feOrder          = feBDH->getFEOrder(cellId);

        // NOTE: cellId 0 passed as we assume only H refined in this function
        size_type dofsPerCell;

        auto locallyOwnedCellIter = feBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != feBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = feBDH->nCellDofs(cellId);
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

        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCells = feBasisDataStorage.getBasisDataInAllCells();

        size_type cumulativeQuadPoints = 0, cumulativeDofQuadPointsOffset = 0;

        locallyOwnedCellIter = feBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != feBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = dofsInCellVec[cellIndex];
            size_type nQuadPointInCell =
              feBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValues =
              feBasisDataStorage.getQuadratureRuleContainer()->getCellJxW(
                cellIndex);

            const ValueTypeOperator *cumulativeDofQuadPoints =
              basisDataInAllCells.data() + cumulativeDofQuadPointsOffset;

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisOverlapTmpIter = 0.0;
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *basisOverlapTmpIter +=
                          *(cumulativeDofQuadPoints + nQuadPointInCell * iNode +
                            qPoint) *
                          *(cumulativeDofQuadPoints + nQuadPointInCell * jNode +
                            qPoint) *
                          cellJxWValues[qPoint];
                      }
                    basisOverlapTmpIter++;
                  }
              }

            cellStartIdsBasisOverlap[cellIndex] = cumulativeBasisOverlapId;
            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
            cumulativeQuadPoints += nQuadPointInCell;
            cumulativeDofQuadPointsOffset += nQuadPointInCell * dofsPerCell;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(), basisOverlap->data(), basisOverlapTmp.data());
      }

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
        std::vector<size_type> &dofsInCellVec)
      {
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          cfeBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            cfeBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          cfeBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in EFEOverlapOperatorContext");

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
          "in EFEOverlapOperatorContext");

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

        bool isConstantDofsAndQuadPointsInCellCFE = false;
        quadrature::QuadratureFamily quadFamily =
          cfeBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        if ((quadFamily == quadrature::QuadratureFamily::GAUSS ||
             quadFamily == quadrature::QuadratureFamily::GLL) &&
            !cfeBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCellCFE = true;

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

        const utils::MemoryStorage<ValueTypeOperator, memorySpace> &
          basisDataInAllCellsCFE = cfeBasisDataStorage.getBasisDataInAllCells();
        const utils::MemoryStorage<ValueTypeOperator, memorySpace> &
          basisDataInAllCellsEFE = efeBasisDataStorage.getBasisDataInAllCells();

        size_type cumulativeCFEDofQuadPointsOffset = 0,
                  cumulativeEFEDofQuadPointsOffset = 0;

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
              basisDataInAllCellsCFE.data() + cumulativeCFEDofQuadPointsOffset;

            const ValueTypeOperator *cumulativeEFEDofQuadPoints =
              basisDataInAllCellsEFE.data() + cumulativeEFEDofQuadPointsOffset;

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
                                nQuadPointInCellCFE * iNode + qPoint) *
                              *(cumulativeCFEDofQuadPoints +
                                nQuadPointInCellCFE * jNode + qPoint) *
                              cellJxWValuesCFE[qPoint];
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEFE;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeEFEDofQuadPoints +
                                nQuadPointInCellEFE * iNode + qPoint) *
                              *(cumulativeEFEDofQuadPoints +
                                nQuadPointInCellEFE * jNode + qPoint) *
                              cellJxWValuesEFE[qPoint];
                          }
                      }
                    basisOverlapTmpIter++;
                  }
              }

            cellStartIdsBasisOverlap[cellIndex] = cumulativeBasisOverlapId;
            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
            if (!isConstantDofsAndQuadPointsInCellCFE)
              cumulativeCFEDofQuadPointsOffset +=
                nQuadPointInCellCFE * dofsPerCellCFE;
            cumulativeEFEDofQuadPointsOffset +=
              nQuadPointInCellEFE * dofsPerCell;
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
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
          &                     basisOverlap,
        std::vector<size_type> &cellStartIdsBasisOverlap,
        std::vector<size_type> &dofsInCellVec)
      {
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          ccfeBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            classicalBlockBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          ccfeBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in EFEOverlapOperatorContext for the Classical data storage of classical dof block.");

        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          ecfeBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            enrichmentBlockClassicalBasisDataStorage->getBasisDofHandler());
        utils::throwException(
          ecfeBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in EFEOverlapOperatorContext for the Classical data storage of enrichment dof blocks.");

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
          "in EFEOverlapOperatorContext for the Enrichment data storage of enrichment dof blocks.");

        utils::throwException(
          ccfeBDH->getTriangulation() == ecfeBDH->getTriangulation() &&
            ccfeBDH->getFEOrder(0) == ecfeBDH->getFEOrder(0) &&
            ccfeBDH->getTriangulation() == eefeBDH->getTriangulation() &&
            ccfeBDH->getFEOrder(0) == eefeBDH->getFEOrder(0),
          "The EFEBasisDataStorage and and Classical FEBasisDataStorage have different triangulation or FEOrder"
          "in EFEOverlapOperatorContext.");

        utils::throwException(
          eefeBDH->isOrthogonalized(),
          "The Enrcihment data storage of enrichment dof blocks should have isOrthogonalized as true in EFEOverlapOperatorContext.");

        std::shared_ptr<
          const EnrichmentClassicalInterfaceSpherical<ValueTypeOperator,
                                                      memorySpace,
                                                      dim>>
          eci = eefeBDH->getEnrichmentClassicalInterface();

        // interpolate the ci 's to the Mc=d classical quadRuleAttr quadpoints

        FEBasisOperations<ValueTypeOperator,
                          ValueTypeOperator,
                          memorySpace,
                          dim>
          basisOp1(enrichmentBlockClassicalBasisDataStorage,
                   L2ProjectionDefaults::MAX_CELL_TIMES_NUMVECS);

        size_type nTotalEnrichmentIds =
          eci->getEnrichmentIdsPartition()->nTotalEnrichmentIds();
        quadrature::QuadratureValuesContainer<ValueTypeOperator, memorySpace>
          basisInterfaceCoeffClassicalQuadRuleValues(
            enrichmentBlockClassicalBasisDataStorage
              ->getQuadratureRuleContainer(),
            nTotalEnrichmentIds,
            (ValueTypeOperator)0);

        basisOp1.interpolate(eci->getBasisInterfaceCoeff(),
                             *eci->getCFEBasisManager(),
                             basisInterfaceCoeffClassicalQuadRuleValues);

        // interpolate the ci 's to the enrichment quadRuleAttr quadpoints

        quadrature::QuadratureValuesContainer<ValueTypeOperator, memorySpace>
          basisInterfaceCoeffEnrichmentQuadRuleValues(
            enrichmentBlockEnrichmentBasisDataStorage
              .getQuadratureRuleContainer(),
            nTotalEnrichmentIds,
            (ValueTypeOperator)0);

        const EFEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorageEFE = dynamic_cast<
            const EFEBasisDataStorage<ValueTypeOperator, memorySpace> &>(
            enrichmentBlockEnrichmentBasisDataStorage);
        utils::throwException(
          &enrichmentBlockEnrichmentBasisDataStorageEFE != nullptr,
          "Could not cast FEBasisDataStorage to EFEBasisDataStorage "
          "in EFEOverlapOperatorContext for enrichmentBlockEnrichmentBasisDataStorage.");

        basisInterfaceCoeffEnrichmentQuadRuleValues =
          enrichmentBlockEnrichmentBasisDataStorageEFE
            .getEnrichmentFunctionClassicalComponentQuadValues();

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

        bool isConstantDofsAndQuadPointsInCellClassicalBlock           = false,
             isConstantDofsAndQuadPointsInCellEnrichmentBlockClassical = false;
        quadrature::QuadratureFamily quadFamilyClassicalBlock =
          classicalBlockBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        quadrature::QuadratureFamily quadFamilyEnrichmentBlockClassical =
          enrichmentBlockClassicalBasisDataStorage->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        if ((quadFamilyClassicalBlock == quadrature::QuadratureFamily::GAUSS ||
             quadFamilyClassicalBlock == quadrature::QuadratureFamily::GLL) &&
            !ccfeBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCellClassicalBlock = true;
        if ((quadFamilyEnrichmentBlockClassical ==
               quadrature::QuadratureFamily::GAUSS ||
             quadFamilyEnrichmentBlockClassical ==
               quadrature::QuadratureFamily::GLL) &&
            !ecfeBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCellEnrichmentBlockClassical = true;

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

        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsClassicalBlock =
            classicalBlockBasisDataStorage.getBasisDataInAllCells();
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsEnrichmentBlockClassical =
            enrichmentBlockClassicalBasisDataStorage->getBasisDataInAllCells();
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsEnrichmentBlockEnrichment =
            enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInAllCells();

        size_type cumulativeClassicalDofQuadPointsOffset                 = 0,
                  cumulativeEnrichmentBlockClassicalDofQuadPointsOffset  = 0,
                  cumulativeEnrichmentBlockEnrichmentDofQuadPointsOffset = 0;

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
                ->getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEnrichmentBlockClassical =
              enrichmentBlockClassicalBasisDataStorage
                ->getQuadratureRuleContainer()
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
              basisDataInAllCellsClassicalBlock.data() +
              cumulativeClassicalDofQuadPointsOffset;

            const ValueTypeOperator
              *cumulativeEnrichmentBlockClassicalDofQuadPoints =
                basisDataInAllCellsEnrichmentBlockClassical.data() +
                cumulativeEnrichmentBlockClassicalDofQuadPointsOffset;

            const ValueTypeOperator
              *cumulativeEnrichmentBlockEnrichmentDofQuadPoints =
                basisDataInAllCellsEnrichmentBlockEnrichment.data() +
                cumulativeEnrichmentBlockEnrichmentDofQuadPointsOffset;

            std::vector<ValueTypeOperator>
              basisInterfaceCoeffClassicalQuadRuleValuesInCell(0),
              basisInterfaceCoeffEnrichmentQuadRuleValuesInCell(0);
            basisInterfaceCoeffClassicalQuadRuleValuesInCell.resize(
              nTotalEnrichmentIds * nQuadPointInCellEnrichmentBlockClassical);
            basisInterfaceCoeffEnrichmentQuadRuleValuesInCell.resize(
              nTotalEnrichmentIds * nQuadPointInCellEnrichmentBlockEnrichment);

            basisInterfaceCoeffClassicalQuadRuleValues
              .template getCellValues<utils::MemorySpace::HOST>(
                cellIndex,
                basisInterfaceCoeffClassicalQuadRuleValuesInCell.data());
            basisInterfaceCoeffEnrichmentQuadRuleValues
              .template getCellValues<utils::MemorySpace::HOST>(
                cellIndex,
                basisInterfaceCoeffEnrichmentQuadRuleValuesInCell.data());

            std::vector<utils::Point> quadRealPointsVec =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellRealPoints(cellIndex);

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisOverlapTmpIter = 0.0;
                    // Ni_classical* Ni_classical of the classicalBlockBasisData
                    if (iNode < dofsPerCellCFE && jNode < dofsPerCellCFE)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellClassicalBlock;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeClassicalBlockDofQuadPoints +
                                nQuadPointInCellClassicalBlock * iNode +
                                qPoint) *
                              *(cumulativeClassicalBlockDofQuadPoints +
                                nQuadPointInCellClassicalBlock * jNode +
                                qPoint) *
                              cellJxWValuesClassicalBlock[qPoint];
                          }
                      }

                    else if (iNode >= dofsPerCellCFE && jNode < dofsPerCellCFE)
                      {
                        size_type enrichmentId =
                          eci->getEnrichmentId(cellIndex,
                                               iNode - dofsPerCellCFE);
                        ValueTypeOperator NpiNcj = (ValueTypeOperator)0,
                                          NciNcj = (ValueTypeOperator)0;
                        // Ni_pristine*Ni_classical at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NpiNcj +=
                              eefeBDH->getEnrichmentValue(
                                cellIndex,
                                iNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints +
                                nQuadPointInCellEnrichmentBlockEnrichment *
                                  jNode +
                                qPoint) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // Ni_classical using Mc = d quadrature * interpolated
                        // ci's in Ni_classicalQuadrature of Mc = d
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockClassical;
                             qPoint++)
                          {
                            NciNcj +=
                              *(basisInterfaceCoeffClassicalQuadRuleValuesInCell
                                  .data() +
                                nTotalEnrichmentIds * qPoint + enrichmentId) *
                              *(cumulativeEnrichmentBlockClassicalDofQuadPoints +
                                nQuadPointInCellEnrichmentBlockClassical *
                                  jNode +
                                qPoint) *
                              cellJxWValuesEnrichmentBlockClassical[qPoint];
                          }
                        *basisOverlapTmpIter += NpiNcj - NciNcj;
                      }

                    else if (iNode < dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
                        size_type enrichmentId =
                          eci->getEnrichmentId(cellIndex,
                                               jNode - dofsPerCellCFE);
                        ValueTypeOperator NciNpj = (ValueTypeOperator)0,
                                          NciNcj = (ValueTypeOperator)0;
                        // Ni_pristine*Ni_classical at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NciNpj +=
                              *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints +
                                nQuadPointInCellEnrichmentBlockEnrichment *
                                  iNode +
                                qPoint) *
                              eefeBDH->getEnrichmentValue(
                                cellIndex,
                                jNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // Ni_classical using Mc = d quadrature * interpolated
                        // ci's in Ni_classicalQuadrature of Mc = d
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockClassical;
                             qPoint++)
                          {
                            NciNcj +=
                              *(cumulativeEnrichmentBlockClassicalDofQuadPoints +
                                nQuadPointInCellEnrichmentBlockClassical *
                                  iNode +
                                qPoint) *
                              *(basisInterfaceCoeffClassicalQuadRuleValuesInCell
                                  .data() +
                                nTotalEnrichmentIds * qPoint + enrichmentId) *
                              cellJxWValuesEnrichmentBlockClassical[qPoint];
                          }
                        *basisOverlapTmpIter += NciNpj - NciNcj;
                      }

                    else if (iNode >= dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
                        size_type enrichmentIdi =
                          eci->getEnrichmentId(cellIndex,
                                               iNode - dofsPerCellCFE);
                        size_type enrichmentIdj =
                          eci->getEnrichmentId(cellIndex,
                                               jNode - dofsPerCellCFE);

                        ValueTypeOperator NpiNpj = (ValueTypeOperator)0,
                                          NciNpj = (ValueTypeOperator)0,
                                          NpiNcj = (ValueTypeOperator)0,
                                          NciNcj = (ValueTypeOperator)0;
                        // Ni_pristine*Ni_pristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NpiNpj +=
                              eefeBDH->getEnrichmentValue(
                                cellIndex,
                                iNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              eefeBDH->getEnrichmentValue(
                                cellIndex,
                                jNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // Ni_pristine* interpolated ci's in
                        // Ni_classicalQuadratureOfPristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NciNpj +=
                              *(basisInterfaceCoeffEnrichmentQuadRuleValuesInCell
                                  .data() +
                                nTotalEnrichmentIds * qPoint + enrichmentIdi) *
                              eefeBDH->getEnrichmentValue(
                                cellIndex,
                                jNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // Ni_pristine* interpolated ci's in
                        // Ni_classicalQuadratureOfPristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NpiNcj +=
                              eefeBDH->getEnrichmentValue(
                                cellIndex,
                                iNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              *(basisInterfaceCoeffEnrichmentQuadRuleValuesInCell
                                  .data() +
                                nTotalEnrichmentIds * qPoint + enrichmentIdj) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // interpolated ci's in Ni_classicalQuadrature of Mc = d
                        // * interpolated ci's in Ni_classicalQuadrature of Mc =
                        // d
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockClassical;
                             qPoint++)
                          {
                            NciNcj +=
                              *(basisInterfaceCoeffClassicalQuadRuleValuesInCell
                                  .data() +
                                nTotalEnrichmentIds * qPoint + enrichmentIdi) *
                              *(basisInterfaceCoeffClassicalQuadRuleValuesInCell
                                  .data() +
                                nTotalEnrichmentIds * qPoint + enrichmentIdj) *
                              cellJxWValuesEnrichmentBlockClassical[qPoint];
                          }
                        *basisOverlapTmpIter +=
                          NpiNpj - NpiNcj - NciNpj + NciNcj;
                      }
                    basisOverlapTmpIter++;
                  }
              }

            cellStartIdsBasisOverlap[cellIndex] = cumulativeBasisOverlapId;
            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
            if (!isConstantDofsAndQuadPointsInCellClassicalBlock)
              cumulativeClassicalDofQuadPointsOffset +=
                nQuadPointInCellClassicalBlock * dofsPerCellCFE;
            if (!isConstantDofsAndQuadPointsInCellEnrichmentBlockClassical)
              cumulativeEnrichmentBlockClassicalDofQuadPointsOffset +=
                nQuadPointInCellEnrichmentBlockClassical * dofsPerCellCFE;
            cumulativeEnrichmentBlockEnrichmentDofQuadPointsOffset +=
              nQuadPointInCellEnrichmentBlockEnrichment * dofsPerCell;
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

            EFEOverlapOperatorContextInternal::storeSizes(
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

    } // end of namespace EFEOverlapOperatorContextInternal


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::
      EFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManagerX,
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManagerY,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             feBasisDataStorage,
        const size_type maxCellTimesNumVecs)
      : d_feBasisManagerX(&feBasisManagerX)
      , d_feBasisManagerY(&feBasisManagerY)
      , d_maxCellTimesNumVecs(maxCellTimesNumVecs)
      , d_cellStartIdsBasisOverlap(0)
      , d_efeBasisDataStorage(&feBasisDataStorage)
      , d_cfeBasisDataStorage(&feBasisDataStorage)
    {
      utils::throwException(
        &(feBasisManagerX.getBasisDofHandler()) ==
          &(feBasisManagerY.getBasisDofHandler()),
        "feBasisManager of X and Y vectors are not from same basisDofhandler");

      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
        basisOverlap;
      EFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(feBasisDataStorage,
             basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell);
      d_basisOverlap = basisOverlap;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::
      EFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManagerX,
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManagerY,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &cfeBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             efeBasisDataStorage,
        const size_type maxCellTimesNumVecs)
      : d_feBasisManagerX(&feBasisManagerX)
      , d_feBasisManagerY(&feBasisManagerY)
      , d_maxCellTimesNumVecs(maxCellTimesNumVecs)
      , d_cellStartIdsBasisOverlap(0)
      , d_efeBasisDataStorage(&efeBasisDataStorage)
      , d_cfeBasisDataStorage(&cfeBasisDataStorage)
    {
      utils::throwException(
        &(feBasisManagerX.getBasisDofHandler()) ==
          &(feBasisManagerY.getBasisDofHandler()),
        "feBasisManager of X and Y vectors are not from same basisDofhandler");

      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
        basisOverlap;
      EFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(cfeBasisDataStorage,
             efeBasisDataStorage,
             basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell);
      d_basisOverlap = basisOverlap;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::
      EFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManagerX,
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManagerY,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeOperator, memorySpace>>
                        enrichmentBlockClassicalBasisDataStorage,
        const size_type maxCellTimesNumVecs)
      : d_feBasisManagerX(&feBasisManagerX)
      , d_feBasisManagerY(&feBasisManagerY)
      , d_maxCellTimesNumVecs(maxCellTimesNumVecs)
      , d_cellStartIdsBasisOverlap(0)
      , d_efeBasisDataStorage(&enrichmentBlockEnrichmentBasisDataStorage)
      , d_cfeBasisDataStorage(&classicalBlockBasisDataStorage)
    {
      utils::throwException(
        &(feBasisManagerX.getBasisDofHandler()) ==
          &(feBasisManagerY.getBasisDofHandler()),
        "feBasisManager of X and Y vectors are not from same basisDofhandler");

      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
        basisOverlap;
      EFEOverlapOperatorContextInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockBasisDataStorage,
             enrichmentBlockEnrichmentBasisDataStorage,
             enrichmentBlockClassicalBasisDataStorage,
             basisOverlap,
             d_cellStartIdsBasisOverlap,
             d_dofsInCell);
      d_basisOverlap = basisOverlap;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEOverlapOperatorContext<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<
                    linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                           ValueTypeOperand>,
                    memorySpace> &Y) const
    {
      const size_type numLocallyOwnedCells =
        d_feBasisManagerX->nLocallyOwnedCells();
      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisManagerX->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBeginX =
        d_feBasisManagerX->locallyOwnedCellLocalDofIdsBegin();

      auto itCellLocalIdsBeginY =
        d_feBasisManagerY->locallyOwnedCellLocalDofIdsBegin();

      const size_type numVecs = X.getNumberComponents();

      // get handle to constraints
      const basis::ConstraintsLocal<
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>,
        memorySpace> &constraintsX = d_feBasisManagerX->getConstraints();

      const basis::ConstraintsLocal<
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>,
        memorySpace> &constraintsY = d_feBasisManagerY->getConstraints();

      X.updateGhostValues();
      // update the child nodes based on the parent nodes
      constraintsX.distributeParentToChild(X, numVecs);

      // access cell-wise discrete Overlap operator
      const utils::MemoryStorage<ValueTypeOperator, memorySpace>
        &basisOverlapInAllCells = *d_basisOverlap;

      const size_type cellBlockSize = d_maxCellTimesNumVecs / numVecs;
      Y.setValue(0.0);

      //
      // perform Ax on the local part of A and x
      // (A = discrete Overlap operator)
      //
      EFEOverlapOperatorContextInternal::computeAxCellWiseLocal(
        basisOverlapInAllCells,
        X.begin(),
        Y.begin(),
        numVecs,
        numLocallyOwnedCells,
        numCellDofs,
        itCellLocalIdsBeginX,
        itCellLocalIdsBeginY,
        cellBlockSize,
        *(X.getLinAlgOpContext()));

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      constraintsY.distributeChildToParent(Y, numVecs);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      Y.accumulateAddLocallyOwned();
      Y.updateGhostValues();
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const utils::MemoryStorage<ValueTypeOperator, memorySpace> &
    EFEOverlapOperatorContext<ValueTypeOperator,
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
    EFEOverlapOperatorContext<ValueTypeOperator,
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
    EFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::getBasisOverlap(const size_type cellId,
                                                    const size_type basisId1,
                                                    const size_type basisId2)
      const
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

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const FEBasisDataStorage<ValueTypeOperator, memorySpace> &
    EFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::getEFEBasisDataStorage() const
    {
      return *d_efeBasisDataStorage;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const FEBasisDataStorage<ValueTypeOperator, memorySpace> &
    EFEOverlapOperatorContext<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace,
                              dim>::getCFEBasisDataStorage() const
    {
      return *d_cfeBasisDataStorage;
    }

  } // namespace basis
} // end of namespace dftefe
