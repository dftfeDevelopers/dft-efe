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
#include <utils/MathFunctions.h>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <algorithm>
#include <numeric>
// #include <mkl.h>
#include <utils/ConditionalOStream.h>
namespace dftefe
{
  namespace basis
  {
    namespace OEFEAtomBlockOverlapInvOpContextGLLInternal
    {
      // Use this for data storage of orthogonalized EFE only
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeBasisOverlapMatrix(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>> cfeBDH,
        std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                                 ValueTypeOperator,
                                                 memorySpace,
                                                 dim>>                 efeBDH,
        utils::MemoryStorage<ValueTypeOperator, memorySpace> &NiNjInAllCells,
        linearAlgebra::LinAlgOpContext<memorySpace> &         linAlgOpContext)
      {
        std::shared_ptr<
          const EnrichmentClassicalInterfaceSpherical<ValueTypeOperator,
                                                      memorySpace,
                                                      dim>>
          eci = efeBDH->getEnrichmentClassicalInterface();

        size_type nTotalEnrichmentIds =
          eci->getEnrichmentIdsPartition()->nTotalEnrichmentIds();

        // Set up the overlap matrix quadrature storages.

        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        std::vector<size_type> dofsInCellVec(0);
        dofsInCellVec.resize(numLocallyOwnedCells, 0);
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

        NiNjInAllCells.resize(basisOverlapSize, ValueTypeOperator(0));
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeOperator(0));

        auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex           = 0;

        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();

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


        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            const utils::MemoryStorage<ValueTypeOperator, memorySpace>
              &basisDataInCellClassicalBlock =
                classicalBlockGLLBasisDataStorage.getBasisDataInCell(cellIndex);
            // const utils::MemoryStorage<ValueTypeOperator, memorySpace>
            //   &basisDataInCellEnrichmentBlockEnrichment =
            //     enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInCell(cellIndex);

            dofsPerCell = dofsInCellVec[cellIndex];
            size_type nQuadPointInCellClassicalBlock =
              classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesClassicalBlock =
              classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);

            size_type nQuadPointInCellEnrichmentBlockEnrichment =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEnrichmentBlockEnrichment =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);


            size_type nQuadPointInCellEnrichmentBlockClassical =
              enrichmentBlockClassicalBasisDataStorage
                .getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEnrichmentBlockClassical =
              enrichmentBlockClassicalBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);


            const ValueTypeOperator *cumulativeClassicalBlockDofQuadPoints =
              basisDataInCellClassicalBlock.data(); /*GLL Quad rule*/

            // const ValueTypeOperator
            //   *cumulativeEnrichmentBlockEnrichmentDofQuadPoints =
            //     basisDataInCellEnrichmentBlockEnrichment.data();


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

                utils::MemoryStorage<ValueTypeOperator, memorySpace>
                  basisValInCellEC =
                    enrichmentBlockClassicalBasisDataStorage.getBasisDataInCell(
                      cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector
                ValueTypeOperator *B = basisValInCellEC.data();
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

                utils::MemoryStorage<ValueTypeOperator, memorySpace>
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
                  B,
                  dofsPerCell,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEE.data(),
                  numEnrichmentIdsInCell,
                  *eci->getLinAlgOpContext());

                const std::vector<double> &enrichValAtQuadPts =
                  efeBDH->getEnrichmentValue(cellIndex, quadRealPointsVec);
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

            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(),
          NiNjInAllCells.data(),
          basisOverlapTmp.data());
      }

    } // namespace OEFEAtomBlockOverlapInvOpContextGLLInternal

    // Write M^-1 apply on a matrix for GLL with spectral finite element
    // M^-1 does not have a cell structure.

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    OEFEAtomBlockOverlapInvOpContextGLL<ValueTypeOperator,
                                        ValueTypeOperand,
                                        memorySpace,
                                        dim>::
      OEFEAtomBlockOverlapInvOpContextGLL(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
            &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : d_feBasisManager(&feBasisManager)
      , d_linAlgOpContext(linAlgOpContext)
      , d_diagonalInv(d_feBasisManager->getMPIPatternP2P(), linAlgOpContext)
      , d_atomBlockEnrichmentOverlapInv(0)
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
        "The Enrichment functions have to be orthogonalized for this class to do the application of overlap inverse.");

      utils::throwException(
        classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily() == quadrature::QuadratureFamily::GLL,
        "The quadrature rule for integration of Classical FE dofs has to be GLL."
        "Contact developers if extra options are needed.");

      std::shared_ptr<
        const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
        cfeBDH = std::dynamic_pointer_cast<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
          classicalBlockGLLBasisDataStorage.getBasisDofHandler());
      utils::throwException(
        cfeBDH != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler "
        "in OrthoEFEOverlapInverseOperatorContext for the Classical data storage of classical dof block.");

      const EFEBasisDataStorage<ValueTypeOperator, memorySpace>
        &enrichmentBlockBasisDataStorageEFE = dynamic_cast<
          const EFEBasisDataStorage<ValueTypeOperator, memorySpace> &>(
          enrichmentBlockEnrichmentBasisDataStorage);
      utils::throwException(
        &enrichmentBlockBasisDataStorageEFE != nullptr,
        "Could not cast FEBasisDataStorage to EFEBasisDataStorage "
        "in EFEOverlapOperatorContext for enrichmentBlockEnrichmentBasisDataStorage.");

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
        "in OrthoEFEOverlapInverseOperatorContext for the Enrichment data storage of enrichment dof blocks.");

      utils::throwException(
        cfeBDH->getTriangulation() == efeBDH->getTriangulation() &&
          cfeBDH->getFEOrder(0) == efeBDH->getFEOrder(0),
        "The EFEBasisDataStorage and and Classical FEBasisDataStorage have different triangulation or FEOrder"
        "in OrthoEFEOverlapInverseOperatorContext.");

      utils::throwException(
        &efebasisDofHandler == efeBDH.get(),
        "In OrthoEFEOverlapInverseOperatorContext the feBasisManager and enrichmentBlockEnrichmentBasisDataStorage should"
        "come from same basisDofHandler.");


      const size_type numCellClassicalDofs = utils::mathFunctions::sizeTypePow(
        (efebasisDofHandler.getFEOrder(0) + 1), dim);
      d_nglobalEnrichmentIds = efebasisDofHandler.nGlobalEnrichmentNodes();
      d_rank                 = 0; // d_nglobalEnrichmentIds;

      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBegin =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      utils::MemoryStorage<ValueTypeOperator, memorySpace> NiNjInAllCells(0);

      OEFEAtomBlockOverlapInvOpContextGLLInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockGLLBasisDataStorage,
             enrichmentBlockEnrichmentBasisDataStorage,
             enrichmentBlockClassicalBasisDataStorage,
             cfeBDH,
             efeBDH,
             NiNjInAllCells,
             *d_linAlgOpContext);

      // // access cell-wise discrete Laplace operator
      // auto NiNjInAllCells =
      //   efeOverlapOperatorContext.getBasisOverlapInAllCells();

      std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                         0);
      std::copy(numCellDofs.begin(),
                numCellDofs.begin() + numLocallyOwnedCells,
                locallyOwnedCellsNumDoFsSTL.begin());

      utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
        numLocallyOwnedCells);
      locallyOwnedCellsNumDoFs.template copyFrom(locallyOwnedCellsNumDoFsSTL);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      // Create the diagonal of the classical block matrix which is diagonal for
      // GLL with spectral quadrature
      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(NiNjInAllCells.data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           diagonal.data());

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      // NOTE ::: In a global matrix sense this step can be thought as doing
      // a kind of mass lumping. It is seen that doing such mass lumping in
      // overlap inverse made the scfs converge faster . Without this step the
      // HX residual was not dropping below 1e-3 for non-conforming mesh.
      d_feBasisManager->getConstraints().distributeChildToParent(diagonal, 1);

      d_feBasisManager->getConstraints().setConstrainedNodes(diagonal, 1, 1.0);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      diagonal.accumulateAddLocallyOwned();

      diagonal.updateGhostValues();

      linearAlgebra::blasLapack::reciprocalX(diagonal.localSize(),
                                             1.0,
                                             diagonal.data(),
                                             d_diagonalInv.data(),
                                             *(diagonal.getLinAlgOpContext()));

      d_feBasisManager->getConstraints().setConstrainedNodesToZero(
        d_diagonalInv, 1);

      // Now form the enrichment block matrix.
      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
        basisOverlapInvEnrichmentBlockExact = std::make_shared<
          utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          d_nglobalEnrichmentIds * d_nglobalEnrichmentIds);

      utils::MemoryStorage<ValueTypeOperator, memorySpace>
        basisOverlapInvEnrichmentBlock(d_nglobalEnrichmentIds *
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
                  *(basisOverlapInvEnrichmentBlockExact->data() +
                    enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                    enrichmentVecInCell[k]) +=
                    *(NiNjInAllCells.data() + cumulativeBasisDataInCells +
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
                      *(basisOverlapInvEnrichmentBlock.data() +
                        enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                        enrichmentVecInCell[k]) +=
                        *(NiNjInAllCells.data() + cumulativeBasisDataInCells +
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
        basisOverlapInvEnrichmentBlockExact->data(),
        basisOverlapInvEnrichmentBlockExact->size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      linearAlgebra::blasLapack::inverse<ValueTypeOperator, memorySpace>(
        d_nglobalEnrichmentIds,
        basisOverlapInvEnrichmentBlockExact->data(),
        *(d_diagonalInv.getLinAlgOpContext()));

      err = utils::mpi::MPIAllreduce<memorySpace>(
        utils::mpi::MPIInPlace,
        basisOverlapInvEnrichmentBlock.data(),
        basisOverlapInvEnrichmentBlock.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      linearAlgebra::blasLapack::inverse<ValueTypeOperator, memorySpace>(
        d_nglobalEnrichmentIds,
        basisOverlapInvEnrichmentBlock.data(),
        *(d_diagonalInv.getLinAlgOpContext()));

      global_size_type globalEnrichmentStartId =
        efeBDH->getGlobalRanges()[1].first;

      std::pair<global_size_type, global_size_type> locOwnEidPair{
        efeBDH->getLocallyOwnedRanges()[1].first - globalEnrichmentStartId,
        efeBDH->getLocallyOwnedRanges()[1].second - globalEnrichmentStartId};

      global_size_type nlocallyOwnedEnrichmentIds =
        locOwnEidPair.second - locOwnEidPair.first;

      d_atomBlockEnrichmentOverlapInv.resize(nlocallyOwnedEnrichmentIds *
                                             nlocallyOwnedEnrichmentIds);

      for (global_size_type i = 0; i < nlocallyOwnedEnrichmentIds; i++)
        {
          for (global_size_type j = 0; j < nlocallyOwnedEnrichmentIds; j++)
            {
              *(d_atomBlockEnrichmentOverlapInv.data() +
                i * nlocallyOwnedEnrichmentIds + j) =
                *(basisOverlapInvEnrichmentBlock.data() +
                  (i + locOwnEidPair.first) * d_nglobalEnrichmentIds +
                  (j + locOwnEidPair.first));
            }
        }

      int rank;
      utils::mpi::MPICommRank(
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator(), &rank);

      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      // rootCout << "Atom Enrichment Block Inverse Matrix: " << std::endl;
      // for (size_type i = 0; i < d_nglobalEnrichmentIds; i++)
      //   {
      //     rootCout << "[";
      //     for (size_type j = 0; j < d_nglobalEnrichmentIds; j++)
      //       {
      //         rootCout << *(basisOverlapInvEnrichmentBlock.data() +
      //                       i * d_nglobalEnrichmentIds + j)
      //                  << "\t";
      //       }
      //     rootCout << "]" << std::endl;
      //   }

      // if (d_rank != 0)
      //   {
      //     ValueTypeOperator normMInvexact = 0;
      //     for (int i = 0; i < basisOverlapInvEnrichmentBlock.size(); i++)
      //       {
      //         *(basisOverlapInvEnrichmentBlock.data() + i) =
      //           *(basisOverlapInvEnrichmentBlockExact->data() + i) -
      //           *(basisOverlapInvEnrichmentBlock.data() + i);

      //         normMInvexact += *(basisOverlapInvEnrichmentBlockExact->data()
      //         + i) * *(basisOverlapInvEnrichmentBlockExact->data() + i);
      //       }
      //     normMInvexact = std::sqrt(normMInvexact);

      //     utils::MemoryStorage<double, memorySpace> eigenValuesMemSpace(
      //       d_nglobalEnrichmentIds);

      //     linearAlgebra::blasLapack::heevd<ValueType, memorySpace>(
      //       linearAlgebra::blasLapack::Job::Vec,
      //       linearAlgebra::blasLapack::Uplo::Lower,
      //       d_nglobalEnrichmentIds,
      //       basisOverlapInvEnrichmentBlock.data(),
      //       d_nglobalEnrichmentIds,
      //       eigenValuesMemSpace.data(),
      //       *d_diagonalInv.getLinAlgOpContext());

      //     std::vector<ValueTypeOperand> vec(eigenValuesMemSpace.size());
      //     eigenValuesMemSpace.template copyTo<utils::MemorySpace::HOST>(
      //       vec.data(), vec.size(), 0, 0);
      //     std::vector<size_type> indices(vec.size());
      //     std::iota(indices.begin(), indices.end(), 0);
      //     std::sort(indices.begin(),
      //               indices.end(),
      //               [&vec](size_type i1, size_type i2) {
      //                 return std::abs(vec[i1]) > std::abs(vec[i2]);
      //               });

      //     for(size_type i = 0 ; i < d_nglobalEnrichmentIds ; i++)
      //     {
      //       ValueTypeOperand sumEigVal = 0;
      //       for(size_type j = i ; j < d_nglobalEnrichmentIds ; j++)
      //       {
      //         sumEigVal += *(eigenValuesMemSpace.data() + indices[j]) *
      //         *(eigenValuesMemSpace.data() + indices[j]);
      //       }
      //       if(std::sqrt(sumEigVal)/normMInvexact < 1e-6)
      //       {
      //         d_rank = i;
      //         break;
      //       }
      //     }

      //     rootCout << "Rank of Residual MInv Enrichment block matrix: " <<
      //     d_rank << "\n";

      //     d_residualEnrichOverlapInvEigenVec.resize(nlocallyOwnedEnrichmentIds
      //     *
      //                                                 d_rank,
      //                                               0);
      //     d_residualEnrichOverlapInvEigenVal.resize(d_rank, 0);

      //     for (int i = 0; i < d_rank; i++)
      //       {
      //         basisOverlapInvEnrichmentBlock.template copyTo<memorySpace>(
      //           d_residualEnrichOverlapInvEigenVec.begin(),
      //           nlocallyOwnedEnrichmentIds,
      //           d_nglobalEnrichmentIds * indices[i] +
      //             locOwnEidPair.first, // srcoffset
      //           nlocallyOwnedEnrichmentIds *
      //             i); // dstoffset ; col - d_rank, row , N_locowned

      //         *(d_residualEnrichOverlapInvEigenVal.data() + i) =
      //           *(eigenValuesMemSpace.data() + indices[i]);
      //       }

      //     //   rootCout << "\n\n\nEigenValues: \n";
      //     // for(int i = 0 ; i < eigenValuesMemSpace.size() ; i++)
      //     // {
      //     //   rootCout << *(eigenValuesMemSpace.data() + indices[i]) << ",";
      //     // }
      //     // rootCout << "\n\n\n";
      //   }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    OEFEAtomBlockOverlapInvOpContextGLL<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<ValueType, memorySpace> &       Y,
                  bool updateGhostX,
                  bool updateGhostY) const
    {
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
        d_diagonalInv.localSize(),
        d_diagonalInv.data(),
        X.begin(),
        Y.begin(),
        *(d_linAlgOpContext));

      // enrichment ids

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
            d_atomBlockEnrichmentOverlapInv.data(),
            nlocallyOwnedEnrichmentIds,
            beta,
            YenrichedLocalVec.begin(),
            numComponents,
            *d_linAlgOpContext);

      // if (d_rank != 0)
      //   {
      //     // rank d_rank approx

      //     if (nlocallyOwnedEnrichmentIds > 0)
      //       linearAlgebra::blasLapack::
      //         gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
      //           linearAlgebra::blasLapack::Layout::ColMajor,
      //           linearAlgebra::blasLapack::Op::NoTrans,
      //           linearAlgebra::blasLapack::Op::NoTrans,
      //           numComponents,
      //           d_rank,
      //           nlocallyOwnedEnrichmentIds,
      //           alpha,
      //           XenrichedLocalVec.data(),
      //           numComponents,
      //           d_residualEnrichOverlapInvEigenVec.data(),
      //           nlocallyOwnedEnrichmentIds,
      //           beta,
      //           dotProds.begin(),
      //           numComponents,
      //           *d_linAlgOpContext);

      //     utils::mpi::MPIDatatype mpiDatatype =
      //       utils::mpi::Types<ValueType>::getMPIDatatype();
      //     utils::mpi::MPIAllreduce<memorySpace>(
      //       utils::mpi::MPIInPlace,
      //       dotProds.data(),
      //       dotProds.size(),
      //       utils::mpi::MPIDouble,
      //       utils::mpi::MPISum,
      //       (X.getMPIPatternP2P())->mpiCommunicator());

      //     size_type m = 1, n = numComponents, k = d_rank;
      //     size_type stride = 0;

      //     linearAlgebra::blasLapack::
      //       scaleStridedVarBatched<ValueType, ValueType, memorySpace>(
      //         1,
      //         linearAlgebra::blasLapack::Layout::ColMajor,
      //         linearAlgebra::blasLapack::ScalarOp::Identity,
      //         linearAlgebra::blasLapack::ScalarOp::Identity,
      //         &stride,
      //         &stride,
      //         &stride,
      //         &m,
      //         &n,
      //         &k,
      //         d_residualEnrichOverlapInvEigenVal.data(),
      //         dotProds.data(),
      //         dotProds.data(),
      //         *d_linAlgOpContext);

      //     beta = 1.0;

      //     if (nlocallyOwnedEnrichmentIds > 0)
      //       linearAlgebra::blasLapack::
      //         gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
      //           linearAlgebra::blasLapack::Layout::ColMajor,
      //           linearAlgebra::blasLapack::Op::NoTrans,
      //           linearAlgebra::blasLapack::Op::Trans,
      //           numComponents,
      //           nlocallyOwnedEnrichmentIds,
      //           d_rank,
      //           alpha,
      //           dotProds.data(),
      //           numComponents,
      //           d_residualEnrichOverlapInvEigenVec.data(),
      //           nlocallyOwnedEnrichmentIds,
      //           beta,
      //           YenrichedLocalVec.begin(),
      //           numComponents,
      //           *d_linAlgOpContext);
      //   }

      YenrichedLocalVec.template copyTo<memorySpace>(
        Y.begin(),
        nlocallyOwnedEnrichmentIds * numComponents,
        0,
        nlocallyOwnedClassicalIds * numComponents);

      Y.updateGhostValues();

      // function to do a static condensation to send the constraint nodes
      // to its parent nodes
      d_feBasisManager->getConstraints().distributeChildToParent(
        Y, Y.getNumberComponents());

      // Function to update the ghost values of the Y
      if (updateGhostY)
        Y.updateGhostValues();
    }
  } // namespace basis
} // namespace dftefe
