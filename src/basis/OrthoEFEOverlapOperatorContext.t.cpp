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
    namespace EFEOverlapOperatorContextInternal
    {
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

        bool isConstantDofsAndQuadPointsInCellClassicalBlock           = false,
             isConstantDofsAndQuadPointsInCellEnrichmentBlockClassical = false;
        quadrature::QuadratureFamily quadFamilyClassicalBlock =
          classicalBlockBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        quadrature::QuadratureFamily quadFamilyEnrichmentBlockClassical =
          enrichmentBlockClassicalBasisDataStorage.getQuadratureRuleContainer()
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
            enrichmentBlockClassicalBasisDataStorage.getBasisDataInAllCells();
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsEnrichmentBlockEnrichment =
            enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInAllCells();

        size_type cumulativeClassicalDofQuadPointsOffset                 = 0,
                  cumulativeEnrichmentBlockClassicalDofQuadPointsOffset  = 0,
                  cumulativeEnrichmentBlockEnrichmentDofQuadPointsOffset = 0;

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

            /**
            std::vector<ValueTypeOperator> classicalComponentInQuadValuesEE(0);

            classicalComponentInQuadValuesEE.resize(
              nQuadPointInCellEnrichmentBlockEnrichment *
            numEnrichmentIdsInCell, (ValueTypeOperator)0);
            **/

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

                dftefe::utils::MemoryStorage<ValueTypeOperator,
                                             utils::MemorySpace::HOST>
                  basisValInCellEC =
                    enrichmentBlockClassicalBasisDataStorage.getBasisDataInCell(
                      cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::Trans,
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockClassical,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  basisValInCellEC.data(),
                  nQuadPointInCellEnrichmentBlockClassical,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEC.data(),
                  numEnrichmentIdsInCell,
                  *eci->getLinAlgOpContext());

                /**
                dftefe::utils::MemoryStorage<ValueTypeOperator,
                                             utils::MemorySpace::HOST>
                  basisValInCellEE =
                    enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInCell(
                      cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::Trans,
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockEnrichment,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  basisValInCellEE.data(),
                  nQuadPointInCellEnrichmentBlockEnrichment,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEE.data(),
                  numEnrichmentIdsInCell,
                  *eci->getLinAlgOpContext());
                **/
              }

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
                        ValueTypeOperator NpiNcj   = (ValueTypeOperator)0,
                                          ciNciNcj = (ValueTypeOperator)0;
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
                            ciNciNcj +=
                              classicalComponentInQuadValuesEC
                                [numEnrichmentIdsInCell * qPoint +
                                 (iNode - dofsPerCellCFE)] *
                              *(cumulativeEnrichmentBlockClassicalDofQuadPoints +
                                nQuadPointInCellEnrichmentBlockClassical *
                                  jNode +
                                qPoint) *
                              cellJxWValuesEnrichmentBlockClassical[qPoint];
                          }
                        *basisOverlapTmpIter += NpiNcj - ciNciNcj;
                      }

                    else if (iNode < dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
                        ValueTypeOperator NciNpj   = (ValueTypeOperator)0,
                                          NcicjNcj = (ValueTypeOperator)0;
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
                            NcicjNcj +=
                              *(cumulativeEnrichmentBlockClassicalDofQuadPoints +
                                nQuadPointInCellEnrichmentBlockClassical *
                                  iNode +
                                qPoint) *
                              classicalComponentInQuadValuesEC
                                [numEnrichmentIdsInCell * qPoint +
                                 (jNode - dofsPerCellCFE)] *
                              cellJxWValuesEnrichmentBlockClassical[qPoint];
                          }
                        *basisOverlapTmpIter += NciNpj - NcicjNcj;
                      }

                    else if (iNode >= dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
                        // Ni_pristine*Ni_pristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints +
                                nQuadPointInCellEnrichmentBlockEnrichment *
                                  iNode +
                                qPoint) *
                              *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints +
                                nQuadPointInCellEnrichmentBlockEnrichment *
                                  jNode +
                                qPoint) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }

                        /**
                        ValueTypeOperator NpiNpj = (ValueTypeOperator)0,
                                          ciNciNpj = (ValueTypeOperator)0,
                                          NpicjNcj = (ValueTypeOperator)0,
                                          ciNcicjNcj = (ValueTypeOperator)0;
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
                            ciNciNpj +=
                              classicalComponentInQuadValuesEE
                                [numEnrichmentIdsInCell * qPoint +
                                 (iNode - dofsPerCellCFE)] *
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
                            NpicjNcj +=
                              eefeBDH->getEnrichmentValue(
                                cellIndex,
                                iNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
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
                        *basisOverlapTmpIter +=
                          NpiNpj - NpicjNcj - ciNciNpj + ciNcicjNcj;
                        **/
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
    OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                   ValueTypeOperand,
                                   memorySpace,
                                   dim>::
      OrthoEFEOverlapOperatorContext(
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
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             enrichmentBlockClassicalBasisDataStorage,
        const size_type maxCellTimesNumVecs)
      : d_feBasisManagerX(&feBasisManagerX)
      , d_feBasisManagerY(&feBasisManagerY)
      , d_maxCellTimesNumVecs(maxCellTimesNumVecs)
      , d_cellStartIdsBasisOverlap(0)
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
    OrthoEFEOverlapOperatorContext<
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
