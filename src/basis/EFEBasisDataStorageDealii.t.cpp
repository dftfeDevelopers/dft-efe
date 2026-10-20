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

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <vector>
#include <unordered_map>
#include <utils/OptimizedIndexSet.h>
#include <climits>
#include <utils/Exceptions.h>
#include <utils/MathFunctions.h>
#include <utils/DataTypeOverloads.h>
#include "DealiiConversions.h"
#include <basis/TriangulationCellDealii.h>
#include <basis/ConstraintsLocal.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRule.h>
#include <quadrature/QuadratureRuleContainer.h>
#include <utility>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
#include <basis/ParentToChildCellsManagerDealii.h>
#include <basis/Defaults.h>

namespace dftefe
{
  namespace basis
  {
    namespace EFEBasisDataStorageDealiiInternal
    {
      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      getClassicalComponentBasisValuesInCellAtQuadOEFE(
        const size_type                                      cellIndex,
        const size_type                                      nQuadPointInCell,
        std::vector<ValueTypeBasisData>                      coeffsInCell,
        std::shared_ptr<const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>> efeBDH,
        ValueTypeBasisData *                                 basisValInCellPtr,
        std::vector<ValueTypeBasisData> &classicalComponentInQuadValues)
      {
        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(cellIndex) + 1),
                                            dim);
        size_type numEnrichmentIdsInCell =
          efeBDH->nCellDofs(cellIndex) - classicalDofsPerCell;

        // dftefe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        //   basisValInCell =
        //   cfeBasisDataStorage->getBasisDataInCell(cellIndex);

        // ValueTypeBasisData *B = basisValInCell.data();
        // Do a gemm (\Sigma c_i N_i^classical)
        // and get the quad values in std::vector

        linearAlgebra::blasLapack::gemm<ValueTypeBasisData,
                                        ValueTypeBasisData,
                                        utils::MemorySpace::HOST>(
          'N',
          'N',
          numEnrichmentIdsInCell,
          nQuadPointInCell,
          classicalDofsPerCell,
          (ValueTypeBasisData)1.0,
          coeffsInCell.data(),
          numEnrichmentIdsInCell,
          basisValInCellPtr,
          classicalDofsPerCell,
          (ValueTypeBasisData)0.0,
          classicalComponentInQuadValues.data(),
          numEnrichmentIdsInCell,
          *linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);
      }

      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      getClassicalComponentBasisGradInCellAtQuadOEFE(
        const size_type                                      cellIndex,
        const size_type                                      nQuadPointInCell,
        std::vector<ValueTypeBasisData> &                    coeffsInCell,
        std::shared_ptr<const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>> efeBDH,
        ValueTypeBasisData *                                 basisGradInCell,
        std::vector<ValueTypeBasisData> &classicalComponentInQuadGradients)
      {
        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(cellIndex) + 1),
                                            dim);
        size_type numEnrichmentIdsInCell =
          efeBDH->nCellDofs(cellIndex) - classicalDofsPerCell;

        // saved as cell->quad->dim->node
        // dftefe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        //   basisGradInCell =
        //     cfeBasisDataStorage->getBasisGradientDataInCell(cellIndex);

        // Do a gemm (\Sigma c_i N_i^classical)
        // and get the quad values in std::vector

        // ValueTypeBasisData *B = basisGradInCell.data();

        linearAlgebra::blasLapack::gemm<ValueTypeBasisData,
                                        ValueTypeBasisData,
                                        utils::MemorySpace::HOST>(
          'N',
          'N',
          numEnrichmentIdsInCell,
          nQuadPointInCell * dim,
          classicalDofsPerCell,
          (ValueTypeBasisData)1.0,
          coeffsInCell.data(),
          numEnrichmentIdsInCell,
          basisGradInCell,
          classicalDofsPerCell,
          (ValueTypeBasisData)0.0,
          classicalComponentInQuadGradients
            .data(), // saved as cell->quad->dim->enrichid
          numEnrichmentIdsInCell,
          *linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);
      }

      // This class stores the enriched FE basis data for a h-refined
      // FE mesh and uniform or non-uniform quadrature Gauss/variable/Adaptive
      // rule across all cells in the mesh.

      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeValuesHRefinedSameQuadEveryCell(
        std::shared_ptr<const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>> efeBDH,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisClassQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisEnrichQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientClassQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientEnrichQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisHessianQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &                                         basisOverlap,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                quadratureRuleContainer,
        std::vector<size_type> &nQuadPointsInCell,
        std::vector<size_type> &cellStartIdsBasisClassQuadStorage,
        std::vector<size_type> &cellStartIdsBasisEnrichQuadStorage,
        std::vector<size_type> &cellStartIdsBasisGradientClassQuadStorage,
        std::vector<size_type> &cellStartIdsBasisGradientEnrichQuadStorage,
        std::vector<size_type> &cellStartIdsBasisHessianQuadStorage,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        // false when a cfeBasisDataStorage serves the classical half
        const bool storeClassicalLocally)
      {
        bool numCellsZero = efeBDH->nLocallyOwnedCells() == 0 ? true : false;
        const quadrature::QuadratureFamily quadratureFamily =
          quadratureRuleAttributes.getQuadratureFamily();
        const size_type num1DQuadPoints =
          quadratureRuleAttributes.getNum1DPoints();
        dealii::Quadrature<dim> dealiiQuadratureRule;
        if (quadratureFamily == quadrature::QuadratureFamily::GAUSS)
          {
            dealiiQuadratureRule = dealii::QGauss<dim>(num1DQuadPoints);
          }
        else if (quadratureFamily == quadrature::QuadratureFamily::GLL)
          {
            dealiiQuadratureRule = dealii::QGaussLobatto<dim>(num1DQuadPoints);
          }
        else if (quadratureFamily ==
                 quadrature::QuadratureFamily::GAUSS_SUBDIVIDED)
          {
            if (!numCellsZero)
              {
                // get the parametric points and jxw in each cell according to
                // the attribute.
                size_type                        cellIndex = 0;
                const std::vector<utils::Point> &cellParametricQuadPoints =
                  quadratureRuleContainer->getCellParametricPoints(cellIndex);
                std::vector<dealii::Point<dim, double>>
                  dealiiParametricQuadPoints(0);

                // get the quad weights in each cell
                const std::vector<double> &quadWeights =
                  quadratureRuleContainer->getCellQuadratureWeights(cellIndex);
                convertToDealiiPoint<dim>(cellParametricQuadPoints,
                                          dealiiParametricQuadPoints);

                // Ask dealii to create quad rule in each cell
                dealiiQuadratureRule =
                  dealii::Quadrature<dim>(dealiiParametricQuadPoints,
                                          quadWeights);
              }
          }

        else
          {
            utils::throwException(
              false,
              "In the case of a h-refined finite "
              "element mesh with a uniform quadrature rule, support is provided "
              "only for Gauss and Gauss-Legendre-Lobatto quadrature rule.");
          }

        bool isQuadCartesianTensorStructured =
          quadratureRuleAttributes.isCartesianTensorStructured();
        utils::throwException(
          isQuadCartesianTensorStructured,
          "In the case of a h-refined finite element mesh with a uniform quadrature "
          "rule, storing the classical finite element basis data is only supported "
          " for a Cartesian tensor structured quadrature grid.");

        dealii::UpdateFlags dealiiUpdateFlags;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second ||
            basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreOverlap)
              ->second)
          dealiiUpdateFlags |= dealii::update_values;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreJxW)
              ->second)
          dealiiUpdateFlags |= dealii::update_JxW_values;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          dealiiUpdateFlags |= dealii::update_gradients;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          dealiiUpdateFlags |= dealii::update_hessians;

        // NOTE: cellId 0 passed as we assume h-refine finite element mesh in
        // this function
        const size_type cellId = 0;
        // FEValues outside cell loop because one has same quad in each cell
        dealii::FEValues<dim> dealiiFEValues(efeBDH->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags);
        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        size_type       dofsPerCell = efeBDH->nCellDofs(cellId);
        const size_type feOrder     = efeBDH->getFEOrder(0);
        const size_type nQuadPointInCell =
          numCellsZero ? 0 :
                         quadratureRuleContainer->nCellQuadraturePoints(cellId);
        // utils::mathFunctions::sizeTypePow(num1DQuadPoints, dim);

        nQuadPointsInCell.assign(numLocallyOwnedCells, nQuadPointInCell);
        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisClassQuadStorageTmp(0), basisEnrichQuadStorageTmp(0),
          basisGradientClassQuadStorageTmp(0),
          basisGradientEnrichQuadStorageTmp(0),
          basisHessianQuadStorageTmp(0), basisOverlapTmp(0);

        size_type cellIndex             = 0;
        size_type basisValuesSize       = 0;
        size_type basisClassValuesSize  = 0;
        size_type basisGradientClassValuesSize = 0;
        size_type basisEnrichValuesSize = 0;
        size_type basisOverlapSize      = 0;

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            basisValuesSize += nQuadPointInCell * dofsPerCell;
            basisOverlapSize += dofsPerCell * dofsPerCell;
            cellIndex++;
          }

        // values repeat so one cell is stored; gradients carry the jacobian
        const size_type classValuesSizeInCell =
          nQuadPointInCell *
          utils::mathFunctions::sizeTypePow((feOrder + 1), dim);
        basisClassValuesSize         = classValuesSizeInCell;
        basisGradientClassValuesSize =
          numLocallyOwnedCells * classValuesSizeInCell;
        basisEnrichValuesSize =
          basisValuesSize - basisGradientClassValuesSize;

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            basisClassQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                storeClassicalLocally ? basisClassValuesSize : 0);
            basisClassQuadStorageTmp.resize(storeClassicalLocally ?
                                              basisClassValuesSize :
                                              classValuesSizeInCell,
                                            ValueTypeBasisData(0));
            cellStartIdsBasisClassQuadStorage.resize(numLocallyOwnedCells, 0);
            basisEnrichQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisEnrichValuesSize);
            basisEnrichQuadStorageTmp.resize(basisEnrichValuesSize,
                                             ValueTypeBasisData(0));
            cellStartIdsBasisEnrichQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            basisGradientClassQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                storeClassicalLocally ? basisGradientClassValuesSize * dim : 0);
            basisGradientClassQuadStorageTmp.resize(
              (storeClassicalLocally ? basisGradientClassValuesSize :
                                       classValuesSizeInCell) *
                dim,
              ValueTypeBasisData(0));
            cellStartIdsBasisGradientClassQuadStorage.resize(
              numLocallyOwnedCells, 0);
            basisGradientEnrichQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisEnrichValuesSize * dim);
            basisGradientEnrichQuadStorageTmp.resize(basisEnrichValuesSize * dim,
                                                     ValueTypeBasisData(0));
            cellStartIdsBasisGradientEnrichQuadStorage.resize(
              numLocallyOwnedCells, 0);
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          {
            basisHessianQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisValuesSize * dim * dim);
            basisHessianQuadStorageTmp.resize(basisValuesSize * dim * dim,
                                              ValueTypeBasisData(0));
            cellStartIdsBasisHessianQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreOverlap)
              ->second)
          {
            basisOverlap =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisOverlapSize);
            basisOverlapTmp.resize(basisOverlapSize, ValueTypeBasisData(0));
          }

        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii = nullptr;
        if (numLocallyOwnedCells != 0)
          {
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            utils::throwException(
              feCellDealii != nullptr,
              "Dynamic casting of FECellBase to FECellDealii not successful");
          }

        auto basisOverlapTmpIter = basisOverlapTmp.begin();

        cellIndex                            = 0;
        size_type cumulativeQuadPointsxnDofs = 0;
        size_type cumulativeQuadPointsxnClassDofs  = 0;
        size_type cumulativeQuadPointsxnEnrichDofs = 0;

        // ------- enrichment, pristine, for every cell at once -------
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            efeBDH->getEnrichmentClassicalInterface()
              ->getEnrichmentValuesInCellRangeAtQuadPts(
                *quadratureRuleContainer,
                basisEnrichQuadStorage->data(),
                linAlgOpContext,
                std::make_pair((size_type)0, numLocallyOwnedCells));
            utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
              basisEnrichQuadStorageTmp.size(),
              basisEnrichQuadStorageTmp.data(),
              basisEnrichQuadStorage->data());
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            efeBDH->getEnrichmentClassicalInterface()
              ->getEnrichmentGradientsInCellRangeAtQuadPts(
                *quadratureRuleContainer,
                basisGradientEnrichQuadStorage->data(),
                linAlgOpContext,
                std::make_pair((size_type)0, numLocallyOwnedCells));
            utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
              basisGradientEnrichQuadStorageTmp.size(),
              basisGradientEnrichQuadStorageTmp.data(),
              basisGradientEnrichQuadStorage->data());
          }

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            // Get classical dof numbers
            size_type classicalDofsPerCell = utils::mathFunctions::sizeTypePow(
              (efeBDH->getFEOrder(cellIndex) + 1), dim);

            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            std::vector<utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

            std::vector<size_type> vecClassicalLocalNodeId(0);


            size_type numEnrichmentIdsInCell =
              dofsPerCell - classicalDofsPerCell;

            std::vector<ValueTypeBasisData> classicalComponentInQuadValues(0);

            std::vector<ValueTypeBasisData> classicalComponentInQuadGradients(
              0);

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second ||
                basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreOverlap)
                  ->second)
              classicalComponentInQuadValues.resize(nQuadPointInCell *
                                                      numEnrichmentIdsInCell,
                                                    (ValueTypeBasisData)0);

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              classicalComponentInQuadGradients.resize(
                nQuadPointInCell * numEnrichmentIdsInCell * dim,
                (ValueTypeBasisData)0);

            std::vector<ValueTypeBasisData> coeffsInCell(0);
            if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
              {
                coeffsInCell =
                  efeBDH->getEnrichmentClassicalInterface()
                    ->getClassicalComponentCoeffsInCellOEFE(cellIndex);
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                // one cell only, so the start id stays at zero
                if (storeClassicalLocally)
                  cellStartIdsBasisClassQuadStorage[cellIndex] = 0;
                cellStartIdsBasisEnrichQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnEnrichDofs;
                // one cell either way: written once, or rebuilt per cell
                if (cellIndex == 0 || !storeClassicalLocally)
                  for (size_type iNode = 0; iNode < classicalDofsPerCell;
                       iNode++)
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      *(basisClassQuadStorageTmp.begin() +
                        qPoint * classicalDofsPerCell + iNode
                        /*iNode * nQuadPointInCell + qPoint*/) =
                        dealiiFEValues.shape_value(iNode, qPoint);
                if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
                  {
                    getClassicalComponentBasisValuesInCellAtQuadOEFE<
                      ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>(cellIndex,
                           nQuadPointInCell,
                           coeffsInCell,
                           efeBDH,
                           basisClassQuadStorageTmp.begin(),
                           classicalComponentInQuadValues);
                  }
                for (size_type iNode = classicalDofsPerCell;
                     iNode < dofsPerCell;
                     iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        // pristine value already in place from the bulk pass
                        *(basisEnrichQuadStorageTmp.begin() +
                          cumulativeQuadPointsxnEnrichDofs +
                          qPoint * numEnrichmentIdsInCell +
                          (iNode - classicalDofsPerCell)
                          /*iNode * nQuadPointInCell + qPoint*/) -=
                          classicalComponentInQuadValues
                            [numEnrichmentIdsInCell * qPoint +
                             (iNode - classicalDofsPerCell)];
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreOverlap)
                  ->second)
              {
                if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0 &&
                    !basisStorageAttributesBoolMap
                       .find(BasisStorageAttributes::StoreValues)
                       ->second)
                  {
                    basisClassQuadStorageTmp.resize(classicalDofsPerCell *
                                                      nQuadPointInCell,
                                                    ValueTypeBasisData(0));
                    for (size_type iNode = 0; iNode < classicalDofsPerCell;
                         iNode++)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *(basisClassQuadStorageTmp.begin() +
                              qPoint * classicalDofsPerCell + iNode
                              /*iNode * nQuadPointInCell + qPoint*/) =
                              dealiiFEValues.shape_value(iNode, qPoint);
                          }
                      }

                    if (efeBDH->isOrthogonalized() &&
                        numEnrichmentIdsInCell > 0)
                      {
                        getClassicalComponentBasisValuesInCellAtQuadOEFE<
                          ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>(cellIndex,
                               nQuadPointInCell,
                               coeffsInCell,
                               efeBDH,
                               basisClassQuadStorageTmp.begin(),
                               classicalComponentInQuadValues);
                      }
                  }

                for (size_type iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (size_type jNode = 0; jNode < dofsPerCell; jNode++)
                      {
                        *basisOverlapTmpIter = 0.0;
                        if (iNode < classicalDofsPerCell &&
                            jNode < classicalDofsPerCell)
                          {
                            for (size_type qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  dealiiFEValues.shape_value(iNode, qPoint) *
                                  dealiiFEValues.shape_value(jNode, qPoint) *
                                  dealiiFEValues.JxW(qPoint);
                              }
                          }
                        else if (iNode >= classicalDofsPerCell &&
                                 jNode < classicalDofsPerCell)
                          {
                            for (size_type qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  utils::conjugate(
                                    efeBDH->getEnrichmentValue(
                                      cellIndex,
                                      iNode - classicalDofsPerCell,
                                      quadRealPointsVec[qPoint]) -
                                    classicalComponentInQuadValues
                                      [numEnrichmentIdsInCell * qPoint +
                                       (iNode - classicalDofsPerCell)]) *
                                  dealiiFEValues.shape_value(jNode, qPoint) *
                                  dealiiFEValues.JxW(qPoint);
                                // enriched i * classical j
                              }
                          }
                        else if (iNode < classicalDofsPerCell &&
                                 jNode >= classicalDofsPerCell)
                          {
                            for (size_type qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  (efeBDH->getEnrichmentValue(
                                     cellIndex,
                                     jNode - classicalDofsPerCell,
                                     quadRealPointsVec[qPoint]) -
                                   classicalComponentInQuadValues
                                     [numEnrichmentIdsInCell * qPoint +
                                      (jNode - classicalDofsPerCell)]) *
                                  dealiiFEValues.shape_value(iNode, qPoint) *
                                  dealiiFEValues.JxW(qPoint);
                                // enriched j * classical i
                              }
                          }
                        else
                          {
                            for (size_type qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  utils::conjugate(
                                    efeBDH->getEnrichmentValue(
                                      cellIndex,
                                      iNode - classicalDofsPerCell,
                                      quadRealPointsVec[qPoint]) -
                                    classicalComponentInQuadValues
                                      [numEnrichmentIdsInCell * qPoint +
                                       (iNode - classicalDofsPerCell)]) *
                                  (efeBDH->getEnrichmentValue(
                                     cellIndex,
                                     jNode - classicalDofsPerCell,
                                     quadRealPointsVec[qPoint]) -
                                   classicalComponentInQuadValues
                                     [numEnrichmentIdsInCell * qPoint +
                                      (jNode - classicalDofsPerCell)]) *
                                  dealiiFEValues.JxW(qPoint);
                                // enriched i * enriched j
                              }
                          }
                        basisOverlapTmpIter++;
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                const size_type classGradOffset =
                  storeClassicalLocally ? cumulativeQuadPointsxnClassDofs * dim :
                                          0;
                if (storeClassicalLocally)
                  cellStartIdsBasisGradientClassQuadStorage[cellIndex] =
                    cumulativeQuadPointsxnClassDofs * dim;
                cellStartIdsBasisGradientEnrichQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnEnrichDofs * dim;
                for (size_type iNode = 0; iNode < classicalDofsPerCell; iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (size_type iDim = 0; iDim < dim; iDim++)
                          {
                            auto it =
                              basisGradientClassQuadStorageTmp.begin() +
                              classGradOffset +
                              qPoint * dim * classicalDofsPerCell +
                              iDim * classicalDofsPerCell + iNode;
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim];
                          }
                      }
                  }

                if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
                  {
                    getClassicalComponentBasisGradInCellAtQuadOEFE<
                      ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>(cellIndex,
                           nQuadPointInCell,
                           coeffsInCell,
                           efeBDH,
                           basisGradientClassQuadStorageTmp.begin() +
                             classGradOffset,
                           classicalComponentInQuadGradients);
                  }

                for (size_type iNode = classicalDofsPerCell;
                     iNode < dofsPerCell;
                     iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        // enriched gradient already in place from the bulk pass
                        for (size_type iDim = 0; iDim < dim; iDim++)
                          {
                            auto it =
                              basisGradientEnrichQuadStorageTmp.begin() +
                              cumulativeQuadPointsxnEnrichDofs * dim +
                              qPoint * dim * numEnrichmentIdsInCell +
                              iDim * numEnrichmentIdsInCell +
                              (iNode - classicalDofsPerCell);
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it -= classicalComponentInQuadGradients
                              [numEnrichmentIdsInCell * dim * qPoint +
                               iDim * numEnrichmentIdsInCell +
                               (iNode - classicalDofsPerCell)];
                            // [iDim][numEnrichmentIdsInCell * qPoint +
                            //        (iNode - classicalDofsPerCell)];
                          }
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreHessian)
                  ->second)
              {
                cellStartIdsBasisHessianQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnDofs * dim * dim;
                for (size_type iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto shapeHessian =
                              dealiiFEValues.shape_hessian(iNode, qPoint);
                            for (size_type iDim = 0; iDim < dim; iDim++)
                              {
                                for (size_type jDim = 0; jDim < dim; jDim++)
                                  {
                                    auto it =
                                      basisHessianQuadStorageTmp.begin() +
                                      cumulativeQuadPointsxnDofs * dim * dim +
                                      qPoint * dim * dim * dofsPerCell +
                                      iDim * dim * dofsPerCell +
                                      jDim * dofsPerCell + iNode;
                                    // iDim * dim * dofsPerCell *
                                    //   nQuadPointInCell +
                                    // jDim * dofsPerCell * nQuadPointInCell +
                                    // iNode * nQuadPointInCell + qPoint;
                                    *it = shapeHessian[iDim][jDim];
                                  }
                              }
                          }
                      }
                    else
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            if (efeBDH->isOrthogonalized())
                              {
                                utils::throwException(
                                  false,
                                  "The hessian values are not calculated for OEFE. Contact developers for this.");
                              }

                            auto shapeHessian = efeBDH->getEnrichmentHessian(
                              cellIndex,
                              iNode - classicalDofsPerCell,
                              quadRealPointsVec[qPoint]);
                            // enriched hessian function
                            for (size_type iDim = 0; iDim < dim; iDim++)
                              {
                                for (size_type jDim = 0; jDim < dim; jDim++)
                                  {
                                    auto it =
                                      basisHessianQuadStorageTmp.begin() +
                                      cumulativeQuadPointsxnDofs * dim * dim +
                                      qPoint * dim * dim * dofsPerCell +
                                      iDim * dim * dofsPerCell +
                                      jDim * dofsPerCell + iNode;
                                    // iDim * dim * dofsPerCell *
                                    //   nQuadPointInCell +
                                    // jDim * dofsPerCell * nQuadPointInCell +
                                    // iNode * nQuadPointInCell + qPoint;
                                    *it = shapeHessian[iDim * dim + jDim];
                                  }
                              }
                          }
                      }
                  }
              }
            cellIndex++;
            cumulativeQuadPointsxnDofs += nQuadPointInCell * dofsPerCell;
            cumulativeQuadPointsxnClassDofs +=
              nQuadPointInCell * classicalDofsPerCell;
            cumulativeQuadPointsxnEnrichDofs +=
              nQuadPointInCell * numEnrichmentIdsInCell;
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            if (storeClassicalLocally)
              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(basisClassQuadStorageTmp.size(),
                     basisClassQuadStorage->data(),
                     basisClassQuadStorageTmp.data());
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisEnrichQuadStorageTmp.size(),
              basisEnrichQuadStorage->data(),
              basisEnrichQuadStorageTmp.data());
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            if (storeClassicalLocally)
              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(basisGradientClassQuadStorageTmp.size(),
                     basisGradientClassQuadStorage->data(),
                     basisGradientClassQuadStorageTmp.data());
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisGradientEnrichQuadStorageTmp.size(),
              basisGradientEnrichQuadStorage->data(),
              basisGradientEnrichQuadStorageTmp.data());
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisHessianQuadStorageTmp.size(),
              basisHessianQuadStorage->data(),
              basisHessianQuadStorageTmp.data());
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreOverlap)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisOverlapTmp.size(),
              basisOverlap->data(),
              basisOverlapTmp.data());
          }
      }

      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeGradNiGradNjHRefinedSameQuadEveryCell(
        std::shared_ptr<const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>> efeBDH,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &                                         basisGradNiGradNj,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadratureRuleContainer)

      {
        bool numCellsZero = efeBDH->nLocallyOwnedCells() == 0 ? true : false;
        const quadrature::QuadratureFamily quadratureFamily =
          quadratureRuleAttributes.getQuadratureFamily();
        const size_type num1DQuadPoints =
          quadratureRuleAttributes.getNum1DPoints();
        dealii::Quadrature<dim> dealiiQuadratureRule;
        if (quadratureFamily == quadrature::QuadratureFamily::GAUSS)
          {
            dealiiQuadratureRule = dealii::QGauss<dim>(num1DQuadPoints);
          }
        else if (quadratureFamily == quadrature::QuadratureFamily::GLL)
          {
            dealiiQuadratureRule = dealii::QGaussLobatto<dim>(num1DQuadPoints);
          }
        else if (quadratureFamily ==
                 quadrature::QuadratureFamily::GAUSS_SUBDIVIDED)
          {
            if (!numCellsZero)
              {
                // get the parametric points and jxw in each cell according to
                // the attribute.
                size_type                        cellIndex = 0;
                const std::vector<utils::Point> &cellParametricQuadPoints =
                  quadratureRuleContainer->getCellParametricPoints(cellIndex);
                std::vector<dealii::Point<dim, double>>
                  dealiiParametricQuadPoints(0);

                // get the quad weights in each cell
                const std::vector<double> &quadWeights =
                  quadratureRuleContainer->getCellQuadratureWeights(cellIndex);
                convertToDealiiPoint<dim>(cellParametricQuadPoints,
                                          dealiiParametricQuadPoints);

                // Ask dealii to create quad rule in each cell
                dealiiQuadratureRule =
                  dealii::Quadrature<dim>(dealiiParametricQuadPoints,
                                          quadWeights);
              }
          }

        else
          {
            utils::throwException(
              false,
              "In the case of a h-refined finite "
              "element mesh with a uniform quadrature rule, support is provided "
              "only for Gauss and Gauss-Legendre-Lobatto quadrature rule.");
          }

        bool isQuadCartesianTensorStructured =
          quadratureRuleAttributes.isCartesianTensorStructured();
        utils::throwException(
          isQuadCartesianTensorStructured,
          "In the case of a h-refined finite element mesh with a uniform quadrature "
          "rule, storing the classical finite element basis data is only supported "
          " for a Cartesian tensor structured quadrature grid.");

        dealii::UpdateFlags dealiiUpdateFlags =
          dealii::update_gradients | dealii::update_JxW_values;

        // NOTE: cellId 0 passed as we assume h-refine finite element mesh in
        // this function
        size_type             cellId = 0;
        dealii::FEValues<dim> dealiiFEValues(efeBDH->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags);

        const size_type nQuadPointInCell =
          numCellsZero ? 0 :
                         quadratureRuleContainer->nCellQuadraturePoints(cellId);

        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        std::vector<ValueTypeBasisData> basisGradNiGradNjTmp(0);

        size_type dofsPerCell        = 0;
        size_type cellIndex          = 0;
        size_type basisStiffnessSize = 0;

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            basisStiffnessSize += dofsPerCell * dofsPerCell;
            cellIndex++;
          }


        basisGradNiGradNj = std::make_shared<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>(
          basisStiffnessSize);
        basisGradNiGradNjTmp.resize(basisStiffnessSize, ValueTypeBasisData(0));
        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii = nullptr;
        if (numLocallyOwnedCells != 0)
          {
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            utils::throwException(
              feCellDealii != nullptr,
              "Dynamic casting of FECellBase to FECellDealii not successful");
          }
        auto basisGradNiGradNjTmpIter = basisGradNiGradNjTmp.begin();
        cellIndex                     = 0;

        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisGradientClassQuadStorageTmp(0);

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            // Get classical dof numbers
            size_type classicalDofsPerCell = utils::mathFunctions::sizeTypePow(
              (efeBDH->getFEOrder(cellIndex) + 1), dim);

            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            std::vector<utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

            std::vector<size_type> vecClassicalLocalNodeId(0);

            size_type numEnrichmentIdsInCell =
              dofsPerCell - classicalDofsPerCell;

            std::vector<ValueTypeBasisData> classicalComponentInQuadGradients(
              0);

            classicalComponentInQuadGradients.resize(nQuadPointInCell *
                                                       numEnrichmentIdsInCell *
                                                       dim,
                                                     (ValueTypeBasisData)0);

            if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
              {
                basisGradientClassQuadStorageTmp.resize(
                  classicalDofsPerCell * nQuadPointInCell * dim,
                  ValueTypeBasisData(0));
                for (size_type iNode = 0; iNode < classicalDofsPerCell; iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (size_type iDim = 0; iDim < dim; iDim++)
                          {
                            auto it =
                              basisGradientClassQuadStorageTmp.begin() +
                              qPoint * dim * classicalDofsPerCell +
                              iDim * classicalDofsPerCell + iNode;
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim];
                          }
                      }
                  }
                std::vector<ValueTypeBasisData> coeffsInCell =
                  efeBDH->getEnrichmentClassicalInterface()
                    ->getClassicalComponentCoeffsInCellOEFE(cellIndex);

                getClassicalComponentBasisGradInCellAtQuadOEFE<
                  ValueTypeBasisCoeff,
                  ValueTypeBasisData,
                  memorySpace,
                  dim>(cellIndex,
                       nQuadPointInCell,
                       coeffsInCell,
                       efeBDH,
                       basisGradientClassQuadStorageTmp.begin(),
                       classicalComponentInQuadGradients);
              }


            for (size_type iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (size_type jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisGradNiGradNjTmpIter = 0.0;
                    if (iNode < classicalDofsPerCell &&
                        jNode < classicalDofsPerCell)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *basisGradNiGradNjTmpIter +=
                              (dealiiFEValues.shape_grad(iNode, qPoint) *
                               dealiiFEValues.shape_grad(jNode, qPoint)) *
                              dealiiFEValues.JxW(qPoint);
                          }
                      }
                    else if (iNode >= classicalDofsPerCell &&
                             jNode < classicalDofsPerCell)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto enrichmentDerivative =
                              efeBDH->getEnrichmentDerivative(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto classicalDerivative =
                              dealiiFEValues.shape_grad(jNode, qPoint);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (size_type k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  utils::conjugate(
                                    enrichmentDerivative[k] -
                                    classicalComponentInQuadGradients
                                      [numEnrichmentIdsInCell * dim * qPoint +
                                       k * numEnrichmentIdsInCell +
                                       (iNode - classicalDofsPerCell)]) *
                                    //  [k][numEnrichmentIdsInCell * qPoint +
                                    //      (iNode - classicalDofsPerCell)]) *
                                    classicalDerivative[k];
                              }
                            *basisGradNiGradNjTmpIter +=
                              dotProd * dealiiFEValues.JxW(qPoint);
                            // enriched i * classical j
                          }
                      }
                    else if (iNode < classicalDofsPerCell &&
                             jNode >= classicalDofsPerCell)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto enrichmentDerivative =
                              efeBDH->getEnrichmentDerivative(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto classicalDerivative =
                              dealiiFEValues.shape_grad(iNode, qPoint);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (size_type k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  (enrichmentDerivative[k] -
                                   classicalComponentInQuadGradients
                                     [numEnrichmentIdsInCell * dim * qPoint +
                                      k * numEnrichmentIdsInCell +
                                      (jNode - classicalDofsPerCell)]) *
                                    //  [k][numEnrichmentIdsInCell * qPoint +
                                    //      (jNode - classicalDofsPerCell)]) *
                                    classicalDerivative[k];
                              }
                            *basisGradNiGradNjTmpIter +=
                              dotProd * dealiiFEValues.JxW(qPoint);
                            // enriched j * classical i
                          }
                      }
                    else
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto enrichmentDerivativei =
                              efeBDH->getEnrichmentDerivative(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto enrichmentDerivativej =
                              efeBDH->getEnrichmentDerivative(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (size_type k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  utils::conjugate(
                                    enrichmentDerivativei[k] -
                                    classicalComponentInQuadGradients
                                      [numEnrichmentIdsInCell * dim * qPoint +
                                       k * numEnrichmentIdsInCell +
                                       (iNode - classicalDofsPerCell)]) *
                                    //  [k][numEnrichmentIdsInCell * qPoint +
                                    //      (iNode - classicalDofsPerCell)]) *
                                    (enrichmentDerivativej[k] -
                                     classicalComponentInQuadGradients
                                       [numEnrichmentIdsInCell * dim * qPoint +
                                        k * numEnrichmentIdsInCell +
                                        (jNode - classicalDofsPerCell)]);
                                //  [k][numEnrichmentIdsInCell * qPoint +
                                //      (jNode - classicalDofsPerCell)]);
                              }
                            *basisGradNiGradNjTmpIter +=
                              dotProd * dealiiFEValues.JxW(qPoint);
                            // enriched i * enriched j
                          }
                        // std::cout << *basisGradNiGradNjTmpIter << " ";
                      }
                    basisGradNiGradNjTmpIter++;
                  }
              }

            cellIndex++;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisGradNiGradNjTmp.size(),
          basisGradNiGradNj->data(),
          basisGradNiGradNjTmp.data());
      }

      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeValuesHRefinedAdaptiveQuad(
        std::shared_ptr<const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>> efeBDH,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisClassQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisEnrichQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientClassQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientEnrichQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisHessianQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &                                         basisOverlap,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                quadratureRuleContainer,
        std::vector<size_type> &nQuadPointsInCell,
        std::vector<size_type> &cellStartIdsBasisClassQuadStorage,
        std::vector<size_type> &cellStartIdsBasisEnrichQuadStorage,
        std::vector<size_type> &cellStartIdsBasisGradientClassQuadStorage,
        std::vector<size_type> &cellStartIdsBasisGradientEnrichQuadStorage,
        std::vector<size_type> &cellStartIdsBasisHessianQuadStorage,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        // false when a cfeBasisDataStorage serves the classical half
        const bool storeClassicalLocally,
        // isFast selects, per cell, how the dealii::FEValues used below is
        // obtained
        const bool isFast = true)
      {
        // Assert if QuadFamily is not having variable quadpoints in each cell
        const quadrature::QuadratureFamily quadratureFamily =
          quadratureRuleAttributes.getQuadratureFamily();
        if (!((quadratureFamily ==
               quadrature::QuadratureFamily::GAUSS_VARIABLE) ||
              (quadratureFamily ==
               quadrature::QuadratureFamily::GLL_VARIABLE) ||
              (quadratureFamily == quadrature::QuadratureFamily::ADAPTIVE)))
          {
            utils::throwException(
              false,
              "For storing of basis data for enriched finite element basis "
              "on a variable quadrature rule across cells, the underlying "
              "quadrature family has to be quadrature::QuadratureFamily::GAUSS_VARIABLE "
              "or quadrature::QuadratureFamily::GLL_VARIABLE or quadrature::QuadratureFamily::ADAPTIVE");
          }

        // init updateflags of dealii data structure storage
        // The same things will be stored in enriched and classical dofs.
        dealii::UpdateFlags dealiiUpdateFlags;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second ||
            basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreOverlap)
              ->second)
          dealiiUpdateFlags |= dealii::update_values;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreJxW)
              ->second)
          dealiiUpdateFlags |= dealii::update_JxW_values;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            dealiiUpdateFlags |= dealii::update_gradients;
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          {
            dealiiUpdateFlags |= dealii::update_hessians;
          }
        const size_type feOrder              = efeBDH->getFEOrder(0);
        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        size_type       dofsPerCell          = 0;

        // Create temporary data structure for Value Storage
        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisClassQuadStorageTmp(0), basisEnrichQuadStorageTmp(0),
          basisGradientClassQuadStorageTmp(0),
          basisGradientEnrichQuadStorageTmp(0),
          basisHessianQuadStorageTmp(0), basisOverlapTmp(0);

        // Find total quadpoints in the processor
        nQuadPointsInCell.resize(numLocallyOwnedCells, 0);
        const size_type nTotalQuadPoints =
          quadratureRuleContainer->nQuadraturePoints();

        size_type cellIndex             = 0;
        size_type basisValuesSize       = 0;
        size_type basisClassValuesSize  = 0;
        size_type basisGradientClassValuesSize = 0;
        size_type basisEnrichValuesSize = 0;
        size_type basisOverlapSize      = 0;
        size_type nQuadPointInCell = 0;
        size_type maxQuadPointsInCell = 0;

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            // Find quad pts in each cell and store in nQuadPointsInCell
            nQuadPointInCell =
              quadratureRuleContainer->nCellQuadraturePoints(cellIndex);
            nQuadPointsInCell[cellIndex] = nQuadPointInCell;
            basisValuesSize += nQuadPointInCell * dofsPerCell;
            basisOverlapSize += dofsPerCell * dofsPerCell;
            maxQuadPointsInCell =
              std::max(maxQuadPointsInCell, nQuadPointInCell);
            cellIndex++;
          }

        // the classical count is the same in every cell
        basisClassValuesSize =
          nTotalQuadPoints *
          utils::mathFunctions::sizeTypePow((feOrder + 1), dim);
        basisGradientClassValuesSize = basisClassValuesSize;
        basisEnrichValuesSize = basisValuesSize - basisClassValuesSize;
        // without the block kept, only the largest cell is held at a time
        const size_type classValuesSizeInCell =
          maxQuadPointsInCell *
          utils::mathFunctions::sizeTypePow((feOrder + 1), dim);

        // Initialize the host and tmp vector for storing basis values in
        // a flattened array. Also cell start ids for the flattened array.
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            basisClassQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                storeClassicalLocally ? basisClassValuesSize : 0);
            basisClassQuadStorageTmp.resize(storeClassicalLocally ?
                                              basisClassValuesSize :
                                              classValuesSizeInCell,
                                            ValueTypeBasisData(0));
            cellStartIdsBasisClassQuadStorage.resize(numLocallyOwnedCells, 0);
            basisEnrichQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisEnrichValuesSize);
            basisEnrichQuadStorageTmp.resize(basisEnrichValuesSize,
                                             ValueTypeBasisData(0));
            cellStartIdsBasisEnrichQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        // Initialize the host and tmp vector for storing basis gradient in
        // a flattened array. Also cell start ids for the flattened array.
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            basisGradientClassQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                storeClassicalLocally ? basisGradientClassValuesSize * dim : 0);
            basisGradientClassQuadStorageTmp.resize(
              (storeClassicalLocally ? basisGradientClassValuesSize :
                                       classValuesSizeInCell) *
                dim,
              ValueTypeBasisData(0));
            cellStartIdsBasisGradientClassQuadStorage.resize(
              numLocallyOwnedCells, 0);
            basisGradientEnrichQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisEnrichValuesSize * dim);
            basisGradientEnrichQuadStorageTmp.resize(basisEnrichValuesSize * dim,
                                                     ValueTypeBasisData(0));
            cellStartIdsBasisGradientEnrichQuadStorage.resize(
              numLocallyOwnedCells, 0);
          }

        // Initialize the host and tmp vector for storing basis hessian in
        // a flattened array. Also cell start ids for the flattened array.
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          {
            basisHessianQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisValuesSize * dim * dim);
            basisHessianQuadStorageTmp.resize(basisValuesSize * dim * dim,
                                              ValueTypeBasisData(0));
            cellStartIdsBasisHessianQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        // Initialize the host and tmp vector for storing basis overlap values
        // in a flattened array.
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreOverlap)
              ->second)
          {
            basisOverlap =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisOverlapSize);
            basisOverlapTmp.resize(basisOverlapSize, ValueTypeBasisData(0));
          }

        auto basisOverlapTmpIter = basisOverlapTmp.begin();

        // Init cell iters and storage iters
        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii = nullptr;
        if (numLocallyOwnedCells != 0)
          {
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            utils::throwException(
              feCellDealii != nullptr,
              "Dynamic casting of FECellBase to FECellDealii not successful");
          }

        cellIndex = 0;

        // get the dealii FiniteElement object
        std::shared_ptr<const dealii::DoFHandler<dim>> dealiiDofHandler =
          efeBDH->getDoFHandler();

        // Used only when isFast: holds one dealii::FEValues per *distinct*
        // quadrature rule (i.e. distinct set of points and weights)
        // encountered so far, not one per cell. Adaptive quadrature
        // typically leaves the bulk of cells (away from a
        // refined/singular region) sharing the exact same (unrefined)
        // rule -- even though quadrature::QuadratureRuleContainer hands
        // back a distinct QuadratureRule *object* per cell regardless, so
        // cells are matched by comparing actual point/weight content, not
        // object identity. Plain vector with a linear scan, since the
        // number of distinct adaptive-quadrature rules across a mesh is
        // expected to be small.
        std::vector<std::pair<std::vector<double>,
                              std::shared_ptr<dealii::FEValues<dim>>>>
          uniqueQuadFEValues;

        size_type cumulativeQuadPointsxnDofs = 0;
        // the split storages need their own running offsets
        size_type cumulativeQuadPointsxnClassDofs  = 0;
        size_type cumulativeQuadPointsxnEnrichDofs = 0;

        // ------- enrichment, pristine, for every cell at once -------
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            efeBDH->getEnrichmentClassicalInterface()
              ->getEnrichmentValuesInCellRangeAtQuadPts(
                *quadratureRuleContainer,
                basisEnrichQuadStorage->data(),
                linAlgOpContext,
                std::make_pair((size_type)0, numLocallyOwnedCells));
            utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
              basisEnrichQuadStorageTmp.size(),
              basisEnrichQuadStorageTmp.data(),
              basisEnrichQuadStorage->data());
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            efeBDH->getEnrichmentClassicalInterface()
              ->getEnrichmentGradientsInCellRangeAtQuadPts(
                *quadratureRuleContainer,
                basisGradientEnrichQuadStorage->data(),
                linAlgOpContext,
                std::make_pair((size_type)0, numLocallyOwnedCells));
            utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
              basisGradientEnrichQuadStorageTmp.size(),
              basisGradientEnrichQuadStorageTmp.data(),
              basisGradientEnrichQuadStorage->data());
          }

        // ---- classical, then the OEFE subtraction, overlap and hessian ----
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            // int rank;
            // dftefe::utils::mpi::MPICommRank(efeBDH->getEnrichmentClassicalInterface()->getCFEBasisManager()->getMPIPatternP2P()->mpiCommunicator(),
            // &rank); std::cout << "\n" <<rank << ", ";
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            // Get classical dof numbers
            size_type classicalDofsPerCell = utils::mathFunctions::sizeTypePow(
              (efeBDH->getFEOrder(cellIndex) + 1), dim);

            nQuadPointInCell = nQuadPointsInCell[cellIndex];

            // get the parametric points and jxw in each cell according to
            // the attribute.
            const std::vector<utils::Point> &cellParametricQuadPoints =
              quadratureRuleContainer->getCellParametricPoints(cellIndex);
            std::vector<double> cellJxWValues =
              quadratureRuleContainer->getCellJxW(cellIndex);
            std::vector<dealii::Point<dim, double>> dealiiParametricQuadPoints(
              0);

            // get the quad weights in each cell
            const std::vector<double> &quadWeights =
              quadratureRuleContainer->getCellQuadratureWeights(cellIndex);
            convertToDealiiPoint<dim>(cellParametricQuadPoints,
                                      dealiiParametricQuadPoints);

            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);

            // The only place isFast changes anything: how dealiiFEValues
            // is obtained. isFast == false rebuilds it fresh every cell
            // (the original behaviour); isFast == true looks it up from
            // uniqueQuadFEValues by exact quadrature-content match,
            // building (and remembering) a new one only on a miss.
            // Everything below that reads off dealiiFEValues is identical
            // either way -- the speedup comes entirely from skipping the
            // repeated, expensive dealii::FEValues construction
            // (dofsPerCell x nQuadPoint shape function evaluation) on a
            // cache hit; reinit() itself is cheap.
            std::unique_ptr<dealii::FEValues<dim>> freshFEValues;
            dealii::FEValues<dim> *                dealiiFEValuesPtr = nullptr;
            if (isFast)
              {
                std::vector<double> quadKey =
                  utils::flatten(cellParametricQuadPoints);
                quadKey.insert(quadKey.end(),
                               quadWeights.begin(),
                               quadWeights.end());

                std::shared_ptr<dealii::FEValues<dim>> matchedFEValues;
                for (const auto &entry : uniqueQuadFEValues)
                  if (entry.first == quadKey)
                    {
                      matchedFEValues = entry.second;
                      break;
                    }

                if (!matchedFEValues)
                  {
                    dealii::Quadrature<dim> dealiiQuadratureRule(
                      dealiiParametricQuadPoints, quadWeights);
                    matchedFEValues = std::make_shared<dealii::FEValues<dim>>(
                      efeBDH->getReferenceFE(cellIndex),
                      dealiiQuadratureRule,
                      dealiiUpdateFlags);
                    uniqueQuadFEValues.emplace_back(std::move(quadKey),
                                                    matchedFEValues);
                  }
                matchedFEValues->reinit(feCellDealii->getDealiiFECellIter());
                dealiiFEValuesPtr = matchedFEValues.get();
              }
            else
              {
                dealii::Quadrature<dim> dealiiQuadratureRule(
                  dealiiParametricQuadPoints, quadWeights);
                freshFEValues = std::make_unique<dealii::FEValues<dim>>(
                  efeBDH->getReferenceFE(cellIndex),
                  dealiiQuadratureRule,
                  dealiiUpdateFlags);
                freshFEValues->reinit(feCellDealii->getDealiiFECellIter());
                dealiiFEValuesPtr = freshFEValues.get();
              }
            dealii::FEValues<dim> &dealiiFEValues = *dealiiFEValuesPtr;

            std::vector<utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

            std::vector<size_type> vecClassicalLocalNodeId(0);

            size_type numEnrichmentIdsInCell =
              dofsPerCell - classicalDofsPerCell;

            std::vector<ValueTypeBasisData> classicalComponentInQuadValues(0);

            std::vector<ValueTypeBasisData> classicalComponentInQuadGradients(
              0);

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second ||
                basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreOverlap)
                  ->second)
              classicalComponentInQuadValues.resize(nQuadPointInCell *
                                                      numEnrichmentIdsInCell,
                                                    (ValueTypeBasisData)0);

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              classicalComponentInQuadGradients.resize(
                nQuadPointInCell * numEnrichmentIdsInCell * dim,
                (ValueTypeBasisData)0);

            std::vector<ValueTypeBasisData> coeffsInCell(0);
            if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
              {
                coeffsInCell =
                  efeBDH->getEnrichmentClassicalInterface()
                    ->getClassicalComponentCoeffsInCellOEFE(cellIndex);
              }

            // Store the basis values.
            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                // one cell, reused, when the block is not kept
                const size_type classOffset =
                  storeClassicalLocally ? cumulativeQuadPointsxnClassDofs : 0;
                if (storeClassicalLocally)
                  cellStartIdsBasisClassQuadStorage[cellIndex] =
                    cumulativeQuadPointsxnClassDofs;
                cellStartIdsBasisEnrichQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnEnrichDofs;
                for (size_type iNode = 0; iNode < classicalDofsPerCell; iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *(basisClassQuadStorageTmp.begin() + classOffset +
                          qPoint * classicalDofsPerCell + iNode
                          /*iNode * nQuadPointInCell + qPoint*/) =
                          dealiiFEValues.shape_value(iNode, qPoint);
                      }
                  }
                if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
                  {
                    getClassicalComponentBasisValuesInCellAtQuadOEFE<
                      ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>(cellIndex,
                           nQuadPointInCell,
                           coeffsInCell,
                           efeBDH,
                           basisClassQuadStorageTmp.begin() + classOffset,
                           classicalComponentInQuadValues);
                  }
                for (size_type iNode = classicalDofsPerCell;
                     iNode < dofsPerCell;
                     iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        // pristine value already in place from the bulk pass
                        *(basisEnrichQuadStorageTmp.begin() +
                          cumulativeQuadPointsxnEnrichDofs +
                          qPoint * numEnrichmentIdsInCell +
                          (iNode - classicalDofsPerCell)
                          /*iNode * nQuadPointInCell + qPoint*/) -=
                          classicalComponentInQuadValues
                            [numEnrichmentIdsInCell * qPoint +
                             (iNode - classicalDofsPerCell)];
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreOverlap)
                  ->second)
              {
                if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0 &&
                    !basisStorageAttributesBoolMap
                       .find(BasisStorageAttributes::StoreValues)
                       ->second)
                  {
                    basisClassQuadStorageTmp.resize(classicalDofsPerCell *
                                                      nQuadPointInCell,
                                                    ValueTypeBasisData(0));
                    for (size_type iNode = 0; iNode < classicalDofsPerCell;
                         iNode++)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *(basisClassQuadStorageTmp.begin() +
                              qPoint * classicalDofsPerCell + iNode
                              /*iNode * nQuadPointInCell + qPoint*/) =
                              dealiiFEValues.shape_value(iNode, qPoint);
                          }
                      }

                    if (efeBDH->isOrthogonalized() &&
                        numEnrichmentIdsInCell > 0)
                      {
                        getClassicalComponentBasisValuesInCellAtQuadOEFE<
                          ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>(cellIndex,
                               nQuadPointInCell,
                               coeffsInCell,
                               efeBDH,
                               basisClassQuadStorageTmp.begin(),
                               classicalComponentInQuadValues);
                      }
                  }
                for (size_type iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (size_type jNode = 0; jNode < dofsPerCell; jNode++)
                      {
                        *basisOverlapTmpIter = 0.0;
                        if (iNode < classicalDofsPerCell &&
                            jNode < classicalDofsPerCell)
                          {
                            for (size_type qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  dealiiFEValues.shape_value(iNode, qPoint) *
                                  dealiiFEValues.shape_value(jNode, qPoint) *
                                  cellJxWValues[qPoint];
                              }
                          }
                        else if (iNode >= classicalDofsPerCell &&
                                 jNode < classicalDofsPerCell)
                          {
                            for (size_type qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  utils::conjugate(
                                    efeBDH->getEnrichmentValue(
                                      cellIndex,
                                      iNode - classicalDofsPerCell,
                                      quadRealPointsVec[qPoint]) -
                                    classicalComponentInQuadValues
                                      [numEnrichmentIdsInCell * qPoint +
                                       (iNode - classicalDofsPerCell)]) *
                                  dealiiFEValues.shape_value(jNode, qPoint) *
                                  cellJxWValues[qPoint];
                                // enriched i * classical j
                              }
                          }
                        else if (iNode < classicalDofsPerCell &&
                                 jNode >= classicalDofsPerCell)
                          {
                            for (size_type qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  (efeBDH->getEnrichmentValue(
                                     cellIndex,
                                     jNode - classicalDofsPerCell,
                                     quadRealPointsVec[qPoint]) -
                                   classicalComponentInQuadValues
                                     [numEnrichmentIdsInCell * qPoint +
                                      (jNode - classicalDofsPerCell)]) *
                                  dealiiFEValues.shape_value(iNode, qPoint) *
                                  cellJxWValues[qPoint];
                                // enriched j * classical i
                              }
                          }
                        else
                          {
                            for (size_type qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  utils::conjugate(
                                    efeBDH->getEnrichmentValue(
                                      cellIndex,
                                      iNode - classicalDofsPerCell,
                                      quadRealPointsVec[qPoint]) -
                                    classicalComponentInQuadValues
                                      [numEnrichmentIdsInCell * qPoint +
                                       (iNode - classicalDofsPerCell)]) *
                                  (efeBDH->getEnrichmentValue(
                                     cellIndex,
                                     jNode - classicalDofsPerCell,
                                     quadRealPointsVec[qPoint]) -
                                   classicalComponentInQuadValues
                                     [numEnrichmentIdsInCell * qPoint +
                                      (jNode - classicalDofsPerCell)]) *
                                  cellJxWValues[qPoint];
                                // enriched i * enriched j
                              }
                          }
                        basisOverlapTmpIter++;
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                const size_type classGradOffset =
                  storeClassicalLocally ? cumulativeQuadPointsxnClassDofs * dim :
                                          0;
                if (storeClassicalLocally)
                  cellStartIdsBasisGradientClassQuadStorage[cellIndex] =
                    cumulativeQuadPointsxnClassDofs * dim;
                cellStartIdsBasisGradientEnrichQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnEnrichDofs * dim;
                for (size_type iNode = 0; iNode < classicalDofsPerCell; iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (size_type iDim = 0; iDim < dim; iDim++)
                          {
                            auto it =
                              basisGradientClassQuadStorageTmp.begin() +
                              classGradOffset +
                              qPoint * dim * classicalDofsPerCell +
                              iDim * classicalDofsPerCell + iNode;
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim];
                          }
                      }
                  }

                if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
                  {
                    getClassicalComponentBasisGradInCellAtQuadOEFE<
                      ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>(cellIndex,
                           nQuadPointInCell,
                           coeffsInCell,
                           efeBDH,
                           basisGradientClassQuadStorageTmp.begin() +
                             classGradOffset,
                           classicalComponentInQuadGradients);
                  }

                for (size_type iNode = classicalDofsPerCell;
                     iNode < dofsPerCell;
                     iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        // enriched gradient already in place from the bulk pass
                        for (size_type iDim = 0; iDim < dim; iDim++)
                          {
                            auto it =
                              basisGradientEnrichQuadStorageTmp.begin() +
                              cumulativeQuadPointsxnEnrichDofs * dim +
                              qPoint * dim * numEnrichmentIdsInCell +
                              iDim * numEnrichmentIdsInCell +
                              (iNode - classicalDofsPerCell);
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it -= classicalComponentInQuadGradients
                              [numEnrichmentIdsInCell * dim * qPoint +
                               iDim * numEnrichmentIdsInCell +
                               (iNode - classicalDofsPerCell)];
                            // [iDim][numEnrichmentIdsInCell * qPoint +
                            //        (iNode - classicalDofsPerCell)];
                          }
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreHessian)
                  ->second)
              {
                cellStartIdsBasisHessianQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnDofs * dim * dim;
                for (size_type iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto shapeHessian =
                              dealiiFEValues.shape_hessian(iNode, qPoint);
                            for (size_type iDim = 0; iDim < dim; iDim++)
                              {
                                for (size_type jDim = 0; jDim < dim; jDim++)
                                  {
                                    auto it =
                                      basisHessianQuadStorageTmp.begin() +
                                      cumulativeQuadPointsxnDofs * dim * dim +
                                      qPoint * dim * dim * dofsPerCell +
                                      iDim * dim * dofsPerCell +
                                      jDim * dofsPerCell + iNode;
                                    // iDim * dim * dofsPerCell *
                                    //   nQuadPointInCell +
                                    // jDim * dofsPerCell * nQuadPointInCell +
                                    // iNode * nQuadPointInCell + qPoint;
                                    *it = shapeHessian[iDim][jDim];
                                  }
                              }
                          }
                      }
                    else
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            if (efeBDH->isOrthogonalized())
                              {
                                utils::throwException(
                                  false,
                                  "The hessian values are not calculated for OEFE. Contact developers for this.");
                              }

                            auto shapeHessian = efeBDH->getEnrichmentHessian(
                              cellIndex,
                              iNode - classicalDofsPerCell,
                              quadRealPointsVec[qPoint]);
                            // enriched hessian function
                            for (size_type iDim = 0; iDim < dim; iDim++)
                              {
                                for (size_type jDim = 0; jDim < dim; jDim++)
                                  {
                                    auto it =
                                      basisHessianQuadStorageTmp.begin() +
                                      cumulativeQuadPointsxnDofs * dim * dim +
                                      qPoint * dim * dim * dofsPerCell +
                                      iDim * dim * dofsPerCell +
                                      jDim * dofsPerCell + iNode;
                                    // iDim * dim * dofsPerCell *
                                    //   nQuadPointInCell +
                                    // jDim * dofsPerCell * nQuadPointInCell +
                                    // iNode * nQuadPointInCell + qPoint;
                                    *it = shapeHessian[iDim * dim + jDim];
                                  }
                              }
                          }
                      }
                  }
              }

            cellIndex++;
            cumulativeQuadPointsxnDofs += nQuadPointInCell * dofsPerCell;
            cumulativeQuadPointsxnClassDofs +=
              nQuadPointInCell * classicalDofsPerCell;
            cumulativeQuadPointsxnEnrichDofs +=
              nQuadPointInCell * numEnrichmentIdsInCell;
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            if (storeClassicalLocally)
              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(basisClassQuadStorageTmp.size(),
                     basisClassQuadStorage->data(),
                     basisClassQuadStorageTmp.data());
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisEnrichQuadStorageTmp.size(),
              basisEnrichQuadStorage->data(),
              basisEnrichQuadStorageTmp.data());
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            if (storeClassicalLocally)
              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(basisGradientClassQuadStorageTmp.size(),
                     basisGradientClassQuadStorage->data(),
                     basisGradientClassQuadStorageTmp.data());
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisGradientEnrichQuadStorageTmp.size(),
              basisGradientEnrichQuadStorage->data(),
              basisGradientEnrichQuadStorageTmp.data());
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisHessianQuadStorageTmp.size(),
              basisHessianQuadStorage->data(),
              basisHessianQuadStorageTmp.data());
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreOverlap)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisOverlapTmp.size(),
              basisOverlap->data(),
              basisOverlapTmp.data());
          }
      }

      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeGradNiGradNjHRefinedAdaptiveQuad(
        std::shared_ptr<const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>> efeBDH,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &                                         basisGradNiGradNj,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadratureRuleContainer)
      {
        const quadrature::QuadratureFamily quadratureFamily =
          quadratureRuleAttributes.getQuadratureFamily();
        if (!((quadratureFamily ==
               quadrature::QuadratureFamily::GAUSS_VARIABLE) ||
              (quadratureFamily ==
               quadrature::QuadratureFamily::GLL_VARIABLE) ||
              (quadratureFamily == quadrature::QuadratureFamily::ADAPTIVE)))
          {
            utils::throwException(
              false,
              "For storing of basis data for enriched finite element basis "
              "on a variable quadrature rule across cells, the underlying "
              "quadrature family has to be quadrature::QuadratureFamily::GAUSS_VARIABLE "
              "or quadrature::QuadratureFamily::GLL_VARIABLE or quadrature::QuadratureFamily::ADAPTIVE");
          }


        dealii::UpdateFlags dealiiUpdateFlags =
          dealii::update_gradients | dealii::update_JxW_values;

        // NOTE: cellId 0 passed as we assume h-refined finite element mesh in
        // this function
        const size_type feOrder              = efeBDH->getFEOrder(0);
        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function

        std::vector<ValueTypeBasisData> basisGradNiGradNjTmp(0);

        const size_type nTotalQuadPoints =
          quadratureRuleContainer->nQuadraturePoints();

        size_type dofsPerCell        = 0;
        size_type cellIndex          = 0;
        size_type basisStiffnessSize = 0;

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            basisStiffnessSize += dofsPerCell * dofsPerCell;
            cellIndex++;
          }

        basisGradNiGradNj = std::make_shared<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>(
          basisStiffnessSize);
        basisGradNiGradNjTmp.resize(basisStiffnessSize, ValueTypeBasisData(0));

        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii = nullptr;
        if (numLocallyOwnedCells != 0)
          {
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            utils::throwException(
              feCellDealii != nullptr,
              "Dynamic casting of FECellBase to FECellDealii not successful");
          }

        auto basisGradNiGradNjTmpIter = basisGradNiGradNjTmp.begin();
        cellIndex                     = 0;

        // get the dealii FiniteElement object
        std::shared_ptr<const dealii::DoFHandler<dim>> dealiiDofHandler =
          efeBDH->getDoFHandler();

        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisGradientClassQuadStorageTmp(0);

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            // Get classical dof numbers
            size_type classicalDofsPerCell = utils::mathFunctions::sizeTypePow(
              (efeBDH->getFEOrder(cellIndex) + 1), dim);

            size_type nQuadPointInCell =
              quadratureRuleContainer->nCellQuadraturePoints(cellIndex);
            const std::vector<utils::Point> &cellParametricQuadPoints =
              quadratureRuleContainer->getCellParametricPoints(cellIndex);
            std::vector<double> cellJxWValues =
              quadratureRuleContainer->getCellJxW(cellIndex);
            std::vector<dealii::Point<dim, double>> dealiiParametricQuadPoints(
              0);
            const std::vector<double> &quadWeights =
              quadratureRuleContainer->getCellQuadratureWeights(cellIndex);
            convertToDealiiPoint<dim>(cellParametricQuadPoints,
                                      dealiiParametricQuadPoints);
            dealii::Quadrature<dim> dealiiQuadratureRule(
              dealiiParametricQuadPoints, quadWeights);
            dealii::FEValues<dim> dealiiFEValues(efeBDH->getReferenceFE(
                                                   cellIndex),
                                                 dealiiQuadratureRule,
                                                 dealiiUpdateFlags);
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            std::vector<utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

            std::vector<size_type> vecClassicalLocalNodeId(0);

            size_type numEnrichmentIdsInCell =
              dofsPerCell - classicalDofsPerCell;

            std::vector<ValueTypeBasisData> classicalComponentInQuadGradients(
              0);

            classicalComponentInQuadGradients.resize(nQuadPointInCell *
                                                       numEnrichmentIdsInCell *
                                                       dim,
                                                     (ValueTypeBasisData)0);


            if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
              {
                basisGradientClassQuadStorageTmp.resize(
                  classicalDofsPerCell * nQuadPointInCell * dim,
                  ValueTypeBasisData(0));
                for (size_type iNode = 0; iNode < classicalDofsPerCell; iNode++)
                  {
                    for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (size_type iDim = 0; iDim < dim; iDim++)
                          {
                            auto it =
                              basisGradientClassQuadStorageTmp.begin() +
                              qPoint * dim * classicalDofsPerCell +
                              iDim * classicalDofsPerCell + iNode;
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim];
                          }
                      }
                  }
                std::vector<ValueTypeBasisData> coeffsInCell =
                  efeBDH->getEnrichmentClassicalInterface()
                    ->getClassicalComponentCoeffsInCellOEFE(cellIndex);

                getClassicalComponentBasisGradInCellAtQuadOEFE<
                  ValueTypeBasisCoeff,
                  ValueTypeBasisData,
                  memorySpace,
                  dim>(cellIndex,
                       nQuadPointInCell,
                       coeffsInCell,
                       efeBDH,
                       basisGradientClassQuadStorageTmp.begin(),
                       classicalComponentInQuadGradients);
              }

            for (size_type iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (size_type jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisGradNiGradNjTmpIter = 0.0;
                    if (iNode < classicalDofsPerCell &&
                        jNode < classicalDofsPerCell)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *basisGradNiGradNjTmpIter +=
                              (dealiiFEValues.shape_grad(iNode, qPoint) *
                               dealiiFEValues.shape_grad(jNode, qPoint)) *
                              cellJxWValues[qPoint];
                          }
                      }
                    else if (iNode >= classicalDofsPerCell &&
                             jNode < classicalDofsPerCell)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto enrichmentDerivative =
                              efeBDH->getEnrichmentDerivative(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto classicalDerivative =
                              dealiiFEValues.shape_grad(jNode, qPoint);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (size_type k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  utils::conjugate(
                                    enrichmentDerivative[k] -
                                    classicalComponentInQuadGradients
                                      [numEnrichmentIdsInCell * dim * qPoint +
                                       k * numEnrichmentIdsInCell +
                                       (iNode - classicalDofsPerCell)]) *
                                    //  [k][numEnrichmentIdsInCell * qPoint +
                                    //      (iNode - classicalDofsPerCell)]) *
                                    classicalDerivative[k];
                              }
                            *basisGradNiGradNjTmpIter +=
                              dotProd * cellJxWValues[qPoint];
                            // enriched i * classical j
                          }
                      }
                    else if (iNode < classicalDofsPerCell &&
                             jNode >= classicalDofsPerCell)
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto enrichmentDerivative =
                              efeBDH->getEnrichmentDerivative(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto classicalDerivative =
                              dealiiFEValues.shape_grad(iNode, qPoint);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (size_type k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  (enrichmentDerivative[k] -
                                   classicalComponentInQuadGradients
                                     [numEnrichmentIdsInCell * dim * qPoint +
                                      k * numEnrichmentIdsInCell +
                                      (jNode - classicalDofsPerCell)]) *
                                    //  [k][numEnrichmentIdsInCell * qPoint +
                                    //      (jNode - classicalDofsPerCell)]) *
                                    classicalDerivative[k];
                              }
                            *basisGradNiGradNjTmpIter +=
                              dotProd * cellJxWValues[qPoint];
                            // enriched j * classical i
                          }
                      }
                    else
                      {
                        for (size_type qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto enrichmentDerivativei =
                              efeBDH->getEnrichmentDerivative(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto enrichmentDerivativej =
                              efeBDH->getEnrichmentDerivative(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (size_type k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  utils::conjugate(
                                    enrichmentDerivativei[k] -
                                    classicalComponentInQuadGradients
                                      [numEnrichmentIdsInCell * dim * qPoint +
                                       k * numEnrichmentIdsInCell +
                                       (iNode - classicalDofsPerCell)]) *
                                    //  [k][numEnrichmentIdsInCell * qPoint +
                                    //      (iNode - classicalDofsPerCell)]) *
                                    (enrichmentDerivativej[k] -
                                     classicalComponentInQuadGradients
                                       [numEnrichmentIdsInCell * dim * qPoint +
                                        k * numEnrichmentIdsInCell +
                                        (jNode - classicalDofsPerCell)]);
                                //  [k][numEnrichmentIdsInCell * qPoint +
                                //      (jNode - classicalDofsPerCell)]);
                              }
                            *basisGradNiGradNjTmpIter +=
                              dotProd * cellJxWValues[qPoint];
                            // enriched i * enriched j
                          }
                      }
                    basisGradNiGradNjTmpIter++;
                  }
              }

            cellIndex++;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisGradNiGradNjTmp.size(),
          basisGradNiGradNj->data(),
          basisGradNiGradNjTmp.data());
      }
    } // namespace EFEBasisDataStorageDealiiInternal


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::
      EFEBasisDataStorageDealii(
        std::shared_ptr<const BasisDofHandler>      efeBDH,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      : d_dofsInCell(0)
      , d_cellStartIdsBasisOverlap(0)
      , d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_basisStorageAttributesBoolMap(basisStorageAttributesBoolMap)
      , d_linAlgOpContext(linAlgOpContext)
      , d_scratch(nullptr)
    {
      d_evaluateBasisData = false;
      d_efeBDH            = std::dynamic_pointer_cast<
        const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                       ValueTypeBasisData,
                                       memorySpace,
                                       dim>>(efeBDH);
      utils::throwException(
        d_efeBDH != nullptr,
        " Could not cast the EFEBasisDofHandler to EFEBasisDofHandlerDealii in EFEBasisDataStorageDealii");

      std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
        d_efeBDH->getDoFHandler();
      const size_type numLocallyOwnedCells = d_efeBDH->nLocallyOwnedCells();
      d_classialDofsInCell =
        utils::mathFunctions::sizeTypePow((d_efeBDH->getFEOrder(0) + 1), dim);
      d_dofsInCell.resize(numLocallyOwnedCells, 0);
      d_cellStartIdsBasisOverlap.resize(numLocallyOwnedCells, 0);
      d_cellStartIdsGradNiGradNj.resize(numLocallyOwnedCells, 0);
      size_type cumulativeBasisOverlapId = 0;
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        {
          d_dofsInCell[iCell]               = d_efeBDH->nCellDofs(iCell);
          d_cellStartIdsBasisOverlap[iCell] = cumulativeBasisOverlapId;

          // Storing this is redundant but can help in readability
          d_cellStartIdsGradNiGradNj[iCell] = d_cellStartIdsBasisOverlap[iCell];

          cumulativeBasisOverlapId +=
            d_efeBDH->nCellDofs(iCell) * d_efeBDH->nCellDofs(iCell);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::
      EFEBasisDataStorageDealii(
        std::shared_ptr<const BasisDofHandler>      efeBDH,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          cfeBasisDataStorage,
        std::shared_ptr<BasisScratch<ValueTypeBasisData, memorySpace>> scratch)
      : EFEBasisDataStorageDealii(efeBDH,
                                  quadratureRuleAttributes,
                                  basisStorageAttributesBoolMap,
                                  linAlgOpContext)
    {
      utils::throwException(
        cfeBasisDataStorage != nullptr,
        "A null cfeBasisDataStorage was given to EFEBasisDataStorageDealii. Use the constructor without it to keep the classical part here.");

      // both must describe the same discretization
      auto cfeBDH = std::dynamic_pointer_cast<
        const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
        cfeBasisDataStorage->getBasisDofHandler());
      utils::throwException(
        cfeBDH != nullptr,
        "The cfeBasisDataStorage given to EFEBasisDataStorageDealii is not built on an FEBasisDofHandler.");
      utils::throwException(
        cfeBDH->getTriangulation() == d_efeBDH->getTriangulation(),
        "The cfeBasisDataStorage given to EFEBasisDataStorageDealii is built on a different triangulation.");
      utils::throwException(
        cfeBDH->nLocallyOwnedCells() == d_efeBDH->nLocallyOwnedCells(),
        "The cfeBasisDataStorage given to EFEBasisDataStorageDealii owns a different number of cells.");
      utils::throwException(
        cfeBDH->getFEOrder(0) == d_efeBDH->getFEOrder(0),
        "The cfeBasisDataStorage given to EFEBasisDataStorageDealii has a different FE order, so its classical dofs per cell do not match.");

      d_cfeBasisDataStorage = cfeBasisDataStorage;
      d_scratch             = scratch;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = true;
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      /**
       * @note We assume a linear mapping from the reference cell
       * to the real cell.
       */
      LinearCellMappingDealii<dim> linearCellMappingDealii;

      size_type num1DQuadPoints = quadratureRuleAttributes.getNum1DPoints();
      quadrature::QuadratureFamily quadFamily =
        quadratureRuleAttributes.getQuadratureFamily();

      if (quadFamily == quadrature::QuadratureFamily::GAUSS)
        {
          std::shared_ptr<quadrature::QuadratureRuleGauss> quadratureRule =
            std::make_shared<quadrature::QuadratureRuleGauss>(dim,
                                                              num1DQuadPoints);
          d_quadratureRuleContainer =
            std::make_shared<quadrature::QuadratureRuleContainer>(
              quadratureRuleAttributes,
              quadratureRule,
              d_efeBDH->getTriangulation(),
              linearCellMappingDealii);

      // checked once here, not in the getBasis*DataInCellRange hot loop
      utils::throwException(
        d_cfeBasisDataStorage == nullptr ||
          d_cfeBasisDataStorage->getQuadratureRuleContainer() ==
            d_quadratureRuleContainer,
        "The cfeBasisDataStorage given to EFEBasisDataStorageDealii was evaluated on a different QuadratureRuleContainer.");
        }
      else if (quadFamily == quadrature::QuadratureFamily::GLL)
        {
          std::shared_ptr<quadrature::QuadratureRuleGLL> quadratureRule =
            std::make_shared<quadrature::QuadratureRuleGLL>(dim,
                                                            num1DQuadPoints);
          d_quadratureRuleContainer =
            std::make_shared<quadrature::QuadratureRuleContainer>(
              quadratureRuleAttributes,
              quadratureRule,
              d_efeBDH->getTriangulation(),
              linearCellMappingDealii);

      // checked once here, not in the getBasis*DataInCellRange hot loop
      utils::throwException(
        d_cfeBasisDataStorage == nullptr ||
          d_cfeBasisDataStorage->getQuadratureRuleContainer() ==
            d_quadratureRuleContainer,
        "The cfeBasisDataStorage given to EFEBasisDataStorageDealii was evaluated on a different QuadratureRuleContainer.");
        }
      else
        utils::throwException<utils::InvalidArgument>(
          false, "Incorrect arguments given for this Quadrature family.");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisClassQuadStorage, basisEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientClassQuadStorage, basisGradientEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisOverlap;

      size_type nTotalEnrichmentIds =
        d_efeBDH->getEnrichmentIdsPartition()->nTotalEnrichmentIds();

      // std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
      //   cfeBasisDataStorage = nullptr;
      /*
            if (d_efeBDH->isOrthogonalized())
              {
                if (!(basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreGradient)
                        ->second ||
                      basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreGradNiGradNj)
                        ->second))
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         true; basisAttrMap[BasisStorageAttributes::StoreGradient]     = false;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    // In HOST !!
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           basisAttrMap);
                  }
                else if (!(basisStorageAttributesBoolMap
                             .find(BasisStorageAttributes::StoreValues)
                             ->second ||
                           basisStorageAttributesBoolMap
                             .find(BasisStorageAttributes::StoreOverlap)
                             ->second))
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         false; basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           basisAttrMap);
                  }
                else
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         true; basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           basisAttrMap);
                  }
              }
      */
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisClassQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisEnrichQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientClassQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientEnrichQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);
      EFEBasisDataStorageDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisClassQuadStorage,
             basisEnrichQuadStorage,
             basisGradientClassQuadStorage,
             basisGradientEnrichQuadStorage,
             basisHessianQuadStorage,
             basisOverlap,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisClassQuadStorage,
             cellStartIdsBasisEnrichQuadStorage,
             cellStartIdsBasisGradientClassQuadStorage,
             cellStartIdsBasisGradientEnrichQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap,
             d_linAlgOpContext,
             d_cfeBasisDataStorage == nullptr);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisClassQuadStorage  = std::move(basisClassQuadStorage);
          // here it only lands the CFE block of a cell range, so it is shared
          if (d_scratch != nullptr)
            d_basisClassQuadStorage = d_scratch->getClassValues();
          d_basisEnrichQuadStorage = std::move(basisEnrichQuadStorage);
          d_cellStartIdsBasisClassQuadStorage =
            cellStartIdsBasisClassQuadStorage;
          d_cellStartIdsBasisEnrichQuadStorage =
            cellStartIdsBasisEnrichQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientClassQuadStorage =
            std::move(basisGradientClassQuadStorage);
          if (d_scratch != nullptr)
            d_basisGradientClassQuadStorage = d_scratch->getClassGradients();
          d_basisGradientEnrichQuadStorage =
            std::move(basisGradientEnrichQuadStorage);
          d_cellStartIdsBasisGradientClassQuadStorage =
            cellStartIdsBasisGradientClassQuadStorage;
          d_cellStartIdsBasisGradientEnrichQuadStorage =
            cellStartIdsBasisGradientEnrichQuadStorage;
        }
      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = std::move(basisHessianQuadStorage);
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreOverlap)
            ->second)
        {
          d_basisOverlap = std::move(basisOverlap);
        }
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            basisGradNiGradNj;
          EFEBasisDataStorageDealiiInternal::
            storeGradNiGradNjHRefinedSameQuadEveryCell<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>(
              d_efeBDH,
              basisGradNiGradNj,
              quadratureRuleAttributes,
              d_quadratureRuleContainer);

          d_basisGradNiGradNj = std::move(basisGradNiGradNj);
        }

      if (basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::StorageReal>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer->getJxW();
          jxwQuadStorage =
            std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                       memorySpace>::StorageReal>(
              jxwVec.size());

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            jxwVec.size(), jxwQuadStorage->data(), jxwVec.data());

          d_JxWStorage = std::move(jxwQuadStorage);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                            quadratureRuleContainer,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = true;
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      /**
       * @note We assume a linear mapping from the reference cell
       * to the real cell.
       */
      LinearCellMappingDealii<dim> linearCellMappingDealii;

      quadrature::QuadratureFamily quadFamily =
        quadratureRuleAttributes.getQuadratureFamily();

      if (quadFamily == quadrature::QuadratureFamily::GAUSS_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::GLL_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::ADAPTIVE ||
          quadFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED)
        d_quadratureRuleContainer = quadratureRuleContainer;
      else
        utils::throwException<utils::InvalidArgument>(
          false, "Incorrect arguments given for this Quadrature family.");

      // checked once here, not in the getBasis*DataInCellRange hot loop
      utils::throwException(
        d_cfeBasisDataStorage == nullptr ||
          d_cfeBasisDataStorage->getQuadratureRuleContainer() ==
            d_quadratureRuleContainer,
        "The cfeBasisDataStorage given to EFEBasisDataStorageDealii was evaluated on a different QuadratureRuleContainer.");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisClassQuadStorage, basisEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientClassQuadStorage, basisGradientEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisOverlap;


      size_type nTotalEnrichmentIds =
        d_efeBDH->getEnrichmentIdsPartition()->nTotalEnrichmentIds();

      // std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
      //   cfeBasisDataStorage = nullptr;
      /*
            if (d_efeBDH->isOrthogonalized())
              {
                if (!(basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreGradient)
                        ->second ||
                      basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreGradNiGradNj)
                        ->second))
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         true; basisAttrMap[BasisStorageAttributes::StoreGradient]     = false;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
                else if (!(basisStorageAttributesBoolMap
                             .find(BasisStorageAttributes::StoreValues)
                             ->second ||
                           basisStorageAttributesBoolMap
                             .find(BasisStorageAttributes::StoreOverlap)
                             ->second))
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         false; basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
                else
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         true; basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
              }
      */
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisClassQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisEnrichQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientClassQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientEnrichQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      if (quadFamily == quadrature::QuadratureFamily::GAUSS_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::GLL_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::ADAPTIVE)
        EFEBasisDataStorageDealiiInternal::storeValuesHRefinedAdaptiveQuad<
          ValueTypeBasisCoeff,
          ValueTypeBasisData,
          memorySpace,
          dim>(d_efeBDH,
               basisClassQuadStorage,
               basisEnrichQuadStorage,
               basisGradientClassQuadStorage,
               basisGradientEnrichQuadStorage,
               basisHessianQuadStorage,
               basisOverlap,
               quadratureRuleAttributes,
               d_quadratureRuleContainer,
               nQuadPointsInCell,
               cellStartIdsBasisClassQuadStorage,
               cellStartIdsBasisEnrichQuadStorage,
               cellStartIdsBasisGradientClassQuadStorage,
               cellStartIdsBasisGradientEnrichQuadStorage,
               cellStartIdsBasisHessianQuadStorage,
               basisStorageAttributesBoolMap,
               d_linAlgOpContext,
               d_cfeBasisDataStorage == nullptr);

      else
        EFEBasisDataStorageDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
          ValueTypeBasisCoeff,
          ValueTypeBasisData,
          memorySpace,
          dim>(d_efeBDH,
               basisClassQuadStorage,
               basisEnrichQuadStorage,
               basisGradientClassQuadStorage,
               basisGradientEnrichQuadStorage,
               basisHessianQuadStorage,
               basisOverlap,
               quadratureRuleAttributes,
               d_quadratureRuleContainer,
               nQuadPointsInCell,
               cellStartIdsBasisClassQuadStorage,
               cellStartIdsBasisEnrichQuadStorage,
               cellStartIdsBasisGradientClassQuadStorage,
               cellStartIdsBasisGradientEnrichQuadStorage,
               cellStartIdsBasisHessianQuadStorage,
               basisStorageAttributesBoolMap,
               d_linAlgOpContext,
               d_cfeBasisDataStorage == nullptr);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisClassQuadStorage  = std::move(basisClassQuadStorage);
          // here it only lands the CFE block of a cell range, so it is shared
          if (d_scratch != nullptr)
            d_basisClassQuadStorage = d_scratch->getClassValues();
          d_basisEnrichQuadStorage = std::move(basisEnrichQuadStorage);
          d_cellStartIdsBasisClassQuadStorage =
            cellStartIdsBasisClassQuadStorage;
          d_cellStartIdsBasisEnrichQuadStorage =
            cellStartIdsBasisEnrichQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientClassQuadStorage =
            std::move(basisGradientClassQuadStorage);
          if (d_scratch != nullptr)
            d_basisGradientClassQuadStorage = d_scratch->getClassGradients();
          d_basisGradientEnrichQuadStorage =
            std::move(basisGradientEnrichQuadStorage);
          d_cellStartIdsBasisGradientClassQuadStorage =
            cellStartIdsBasisGradientClassQuadStorage;
          d_cellStartIdsBasisGradientEnrichQuadStorage =
            cellStartIdsBasisGradientEnrichQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = std::move(basisHessianQuadStorage);
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreOverlap)
            ->second)
        {
          d_basisOverlap = std::move(basisOverlap);
        }
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            basisGradNiGradNj;

          if (quadFamily == quadrature::QuadratureFamily::GAUSS_VARIABLE ||
              quadFamily == quadrature::QuadratureFamily::GLL_VARIABLE ||
              quadFamily == quadrature::QuadratureFamily::ADAPTIVE)
            EFEBasisDataStorageDealiiInternal::
              storeGradNiGradNjHRefinedAdaptiveQuad<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>(
                d_efeBDH,
                basisGradNiGradNj,
                quadratureRuleAttributes,
                d_quadratureRuleContainer);
          else
            EFEBasisDataStorageDealiiInternal::
              storeGradNiGradNjHRefinedSameQuadEveryCell<ValueTypeBasisCoeff,
                                                         ValueTypeBasisData,
                                                         memorySpace,
                                                         dim>(
                d_efeBDH,
                basisGradNiGradNj,
                quadratureRuleAttributes,
                d_quadratureRuleContainer);

          d_basisGradNiGradNj = std::move(basisGradNiGradNj);
        }

      if (basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::StorageReal>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer->getJxW();
          jxwQuadStorage =
            std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                       memorySpace>::StorageReal>(
              jxwVec.size());

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            jxwVec.size(), jxwQuadStorage->data(), jxwVec.data());

          d_JxWStorage = std::move(jxwQuadStorage);
        }
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::vector<std::shared_ptr<const quadrature::QuadratureRule>>
                                            quadratureRuleVec,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = true;
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      /**
       * @note We assume a linear mapping from the reference cell
       * to the real cell.
       */
      LinearCellMappingDealii<dim> linearCellMappingDealii;

      quadrature::QuadratureFamily quadFamily =
        quadratureRuleAttributes.getQuadratureFamily();

      if (quadFamily == quadrature::QuadratureFamily::GAUSS_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::GLL_VARIABLE)
        d_quadratureRuleContainer =
          std::make_shared<quadrature::QuadratureRuleContainer>(
            quadratureRuleAttributes,
            quadratureRuleVec,
            d_efeBDH->getTriangulation(),
            linearCellMappingDealii);
      else
        utils::throwException<utils::InvalidArgument>(
          false, "Incorrect arguments given for this Quadrature family.");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisClassQuadStorage, basisEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientClassQuadStorage, basisGradientEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisOverlap;

      // std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
      //   cfeBasisDataStorage = nullptr;
      /*
            if (d_efeBDH->isOrthogonalized())
              {
                if (!(basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreGradient)
                        ->second ||
                      basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreGradNiGradNj)
                        ->second))
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         true; basisAttrMap[BasisStorageAttributes::StoreGradient]     = false;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
                else if (!(basisStorageAttributesBoolMap
                             .find(BasisStorageAttributes::StoreValues)
                             ->second ||
                           basisStorageAttributesBoolMap
                             .find(BasisStorageAttributes::StoreOverlap)
                             ->second))
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         false; basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
                else
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         true; basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
              }
      */
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisClassQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisEnrichQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientClassQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientEnrichQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      EFEBasisDataStorageDealiiInternal::storeValuesHRefinedAdaptiveQuad<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisClassQuadStorage,
             basisEnrichQuadStorage,
             basisGradientClassQuadStorage,
             basisGradientEnrichQuadStorage,
             basisHessianQuadStorage,
             basisOverlap,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisClassQuadStorage,
             cellStartIdsBasisEnrichQuadStorage,
             cellStartIdsBasisGradientClassQuadStorage,
             cellStartIdsBasisGradientEnrichQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap,
             d_linAlgOpContext,
             d_cfeBasisDataStorage == nullptr);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisClassQuadStorage  = std::move(basisClassQuadStorage);
          // here it only lands the CFE block of a cell range, so it is shared
          if (d_scratch != nullptr)
            d_basisClassQuadStorage = d_scratch->getClassValues();
          d_basisEnrichQuadStorage = std::move(basisEnrichQuadStorage);
          d_cellStartIdsBasisClassQuadStorage =
            cellStartIdsBasisClassQuadStorage;
          d_cellStartIdsBasisEnrichQuadStorage =
            cellStartIdsBasisEnrichQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientClassQuadStorage =
            std::move(basisGradientClassQuadStorage);
          if (d_scratch != nullptr)
            d_basisGradientClassQuadStorage = d_scratch->getClassGradients();
          d_basisGradientEnrichQuadStorage =
            std::move(basisGradientEnrichQuadStorage);
          d_cellStartIdsBasisGradientClassQuadStorage =
            cellStartIdsBasisGradientClassQuadStorage;
          d_cellStartIdsBasisGradientEnrichQuadStorage =
            cellStartIdsBasisGradientEnrichQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = std::move(basisHessianQuadStorage);
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreOverlap)
            ->second)
        {
          d_basisOverlap = std::move(basisOverlap);
        }
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            basisGradNiGradNj;

          EFEBasisDataStorageDealiiInternal::
            storeGradNiGradNjHRefinedAdaptiveQuad<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>(
              d_efeBDH,
              basisGradNiGradNj,
              quadratureRuleAttributes,
              d_quadratureRuleContainer);
          d_basisGradNiGradNj = std::move(basisGradNiGradNj);
        }

      if (basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::StorageReal>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer->getJxW();
          jxwQuadStorage =
            std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                       memorySpace>::StorageReal>(
              jxwVec.size());

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            jxwVec.size(), jxwQuadStorage->data(), jxwVec.data());

          d_JxWStorage = std::move(jxwQuadStorage);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRule>
          baseQuadratureRuleAdaptive,
        std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
          &                                 functions,
        const std::vector<double> &         absoluteTolerances,
        const std::vector<double> &         relativeTolerances,
        const std::vector<double> &         integralThresholds,
        const double                        smallestCellVolume,
        const size_type                     maxRecursion,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = true;
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      /**
       * @note We assume a linear mapping from the reference cell
       * to the real cell.
       */
      LinearCellMappingDealii<dim>         linearCellMappingDealii;
      ParentToChildCellsManagerDealii<dim> parentToChildCellsManagerDealii;

      quadrature::QuadratureFamily quadFamily =
        quadratureRuleAttributes.getQuadratureFamily();

      if (quadFamily == quadrature::QuadratureFamily::ADAPTIVE)
        {
          d_quadratureRuleContainer =
            std::make_shared<quadrature::QuadratureRuleContainer>(
              quadratureRuleAttributes,
              baseQuadratureRuleAdaptive,
              d_efeBDH->getTriangulation(),
              linearCellMappingDealii,
              parentToChildCellsManagerDealii,
              functions,
              absoluteTolerances,
              relativeTolerances,
              integralThresholds,
              smallestCellVolume,
              maxRecursion);

      // checked once here, not in the getBasis*DataInCellRange hot loop
      utils::throwException(
        d_cfeBasisDataStorage == nullptr ||
          d_cfeBasisDataStorage->getQuadratureRuleContainer() ==
            d_quadratureRuleContainer,
        "The cfeBasisDataStorage given to EFEBasisDataStorageDealii was evaluated on a different QuadratureRuleContainer.");
        }
      else
        utils::throwException<utils::InvalidArgument>(
          false, "Incorrect arguments given for this Quadrature family.");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisClassQuadStorage, basisEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientClassQuadStorage, basisGradientEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisOverlap;


      size_type nTotalEnrichmentIds =
        d_efeBDH->getEnrichmentIdsPartition()->nTotalEnrichmentIds();

      // std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
      //   cfeBasisDataStorage = nullptr;
      /*
            if (d_efeBDH->isOrthogonalized())
              {
                if (!(basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreGradient)
                        ->second ||
                      basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreGradNiGradNj)
                        ->second))
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         true; basisAttrMap[BasisStorageAttributes::StoreGradient]     = false;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
                else if (!(basisStorageAttributesBoolMap
                             .find(BasisStorageAttributes::StoreValues)
                             ->second ||
                           basisStorageAttributesBoolMap
                             .find(BasisStorageAttributes::StoreOverlap)
                             ->second))
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         false; basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
                else
                  {
                    BasisStorageAttributesBoolMap basisAttrMap;
                    basisAttrMap[BasisStorageAttributes::StoreValues]       =
         true; basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
                    basisAttrMap[BasisStorageAttributes::StoreHessian]      =
         false; basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
                    basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] =
         false; basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


                    // Set up the FE Basis Data Storage
                    cfeBasisDataStorage =
                      std::make_shared<CFEBasisDataStorageDealii<ValueTypeBasisData,
                                                                 ValueTypeBasisData,
                                                                 memorySpace,
                                                                 dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
              }
      */
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisClassQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisEnrichQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientClassQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientEnrichQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      EFEBasisDataStorageDealiiInternal::storeValuesHRefinedAdaptiveQuad<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisClassQuadStorage,
             basisEnrichQuadStorage,
             basisGradientClassQuadStorage,
             basisGradientEnrichQuadStorage,
             basisHessianQuadStorage,
             basisOverlap,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisClassQuadStorage,
             cellStartIdsBasisEnrichQuadStorage,
             cellStartIdsBasisGradientClassQuadStorage,
             cellStartIdsBasisGradientEnrichQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap,
             d_linAlgOpContext,
             d_cfeBasisDataStorage == nullptr);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisClassQuadStorage  = std::move(basisClassQuadStorage);
          // here it only lands the CFE block of a cell range, so it is shared
          if (d_scratch != nullptr)
            d_basisClassQuadStorage = d_scratch->getClassValues();
          d_basisEnrichQuadStorage = std::move(basisEnrichQuadStorage);
          d_cellStartIdsBasisClassQuadStorage =
            cellStartIdsBasisClassQuadStorage;
          d_cellStartIdsBasisEnrichQuadStorage =
            cellStartIdsBasisEnrichQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientClassQuadStorage =
            std::move(basisGradientClassQuadStorage);
          if (d_scratch != nullptr)
            d_basisGradientClassQuadStorage = d_scratch->getClassGradients();
          d_basisGradientEnrichQuadStorage =
            std::move(basisGradientEnrichQuadStorage);
          d_cellStartIdsBasisGradientClassQuadStorage =
            cellStartIdsBasisGradientClassQuadStorage;
          d_cellStartIdsBasisGradientEnrichQuadStorage =
            cellStartIdsBasisGradientEnrichQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = std::move(basisHessianQuadStorage);
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreOverlap)
            ->second)
        {
          d_basisOverlap = std::move(basisOverlap);
        }
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            basisGradNiGradNj;

          EFEBasisDataStorageDealiiInternal::
            storeGradNiGradNjHRefinedAdaptiveQuad<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>(
              d_efeBDH,
              basisGradNiGradNj,
              quadratureRuleAttributes,
              d_quadratureRuleContainer);
          d_basisGradNiGradNj = std::move(basisGradNiGradNj);
        }

      if (basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::StorageReal>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer->getJxW();
          jxwQuadStorage =
            std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                       memorySpace>::StorageReal>(
              jxwVec.size());

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            jxwVec.size(), jxwQuadStorage->data(), jxwVec.data());

          d_JxWStorage = std::move(jxwQuadStorage);
        }
    }

    //------------------OTHER FNS -----------------------------
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisDataInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreValues)
          ->second,
        "Basis values are not evaluated for the given QuadratureRuleAttributes");
      // stored apart, so use getBasisDataInCellRange into your own buffer
      utils::throwException(
        false,
        "getBasisDataInAllCells() is not implemented in EFEBasisDataStorageDealii");
      return *d_basisEnrichQuadStorage;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisGradientDataInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradient)
          ->second,
        "Basis Gradients are not evaluated for the given QuadratureRuleAttributes");
      // see getBasisDataInAllCells
      utils::throwException(
        false,
        "getBasisGradientDataInAllCells() is not implemented in EFEBasisDataStorageDealii");
      return *d_basisGradientEnrichQuadStorage;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisHessianDataInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreHessian)
          ->second,
        "Basis Hessians are not evaluated for the given QuadratureRuleAttributes");
      return *(d_basisHessianQuadStorage);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::StorageReal &
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getJxWInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
          ->second,
        "JxW values are not stored for the given QuadratureRuleAttributes");
      return *(d_JxWStorage);
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisDataInCell(const size_type cellId)
      const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreValues)
          ->second,
        "Basis values are not evaluated for the given QuadratureRuleAttributes");
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      std::pair<size_type, size_type> cellPair(cellId, cellId + 1);
      getBasisDataInCellRange(cellPair, returnValue);

      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisDataStorageDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisDataInCellRange(std::pair<size_type, size_type> cellRange,
                                    Storage &basisData) const
    {
      const size_type numBatch = cellRange.second - cellRange.first;

      if (d_scratch != nullptr)
        d_scratch->acquireStorageScratch(this, d_linAlgOpContext);

      // fetched here, with d_basisClassQuadStorage as the landing buffer
      const bool classFromCfe = d_cfeBasisDataStorage != nullptr;
      if (classFromCfe)
        {
          // another rule would misalign the per cell counts silently
          size_type classSize = 0;
          for (size_type cellId = cellRange.first; cellId < cellRange.second;
               cellId++)
            classSize += d_nQuadPointsIncell[cellId] * d_classialDofsInCell;
          if (d_basisClassQuadStorage->size() < classSize)
            d_basisClassQuadStorage->resize(classSize);
          d_cfeBasisDataStorage->getBasisDataInCellRange(
            cellRange, *d_basisClassQuadStorage);
        }

      std::vector<size_type> batchStrideSrcClass(numBatch);
      std::vector<size_type> batchStrideDst(numBatch);
      std::vector<size_type> batchVecSize(numBatch);
      std::vector<size_type> batchNumVecClass(numBatch, d_classialDofsInCell);
      std::vector<size_type> batchSrcLDClass(numBatch, d_classialDofsInCell);
      std::vector<size_type> batchSrcStartClass(numBatch, 0);
      std::vector<size_type> batchDstLD(numBatch);
      std::vector<size_type> batchDstStartClass(numBatch, 0);
      std::vector<size_type> batchStrideSrcEnrich(numBatch);
      std::vector<size_type> batchNumVecEnrich(numBatch);
      std::vector<size_type> batchSrcLDEnrich(numBatch);
      std::vector<size_type> batchSrcStartEnrich(numBatch, 0);
      std::vector<size_type> batchDstStartEnrich(numBatch,
                                                 d_classialDofsInCell);

      for (size_type ibatch = 0; ibatch < numBatch; ++ibatch)
        {
          const size_type cellId       = cellRange.first + ibatch;
          const size_type nQuad        = d_nQuadPointsIncell[cellId];
          const size_type nDofs        = d_dofsInCell[cellId];
          const size_type nEnriched    = nDofs - d_classialDofsInCell;
          batchVecSize[ibatch]         = nQuad;
          batchDstLD[ibatch]           = nDofs;
          batchStrideDst[ibatch]       = nDofs * nQuad;
          // zero when the classical block holds a single cell
          batchStrideSrcClass[ibatch] =
            classFromCfe ?
              nQuad * d_classialDofsInCell :
              ((cellId + 1 < d_dofsInCell.size()) ?
                 (d_cellStartIdsBasisClassQuadStorage[cellId + 1] -
                  d_cellStartIdsBasisClassQuadStorage[cellId]) :
                 0);
          batchNumVecEnrich[ibatch]    = nEnriched;
          batchSrcLDEnrich[ibatch]     = nEnriched;
          batchStrideSrcEnrich[ibatch] = nEnriched * nQuad;
        }

      linearAlgebra::blasLapack::varBatchedStridedBlockCopy(
        numBatch,
        batchStrideSrcClass.data(),
        batchStrideDst.data(),
        batchVecSize.data(),
        batchNumVecClass.data(),
        batchSrcLDClass.data(),
        batchSrcStartClass.data(),
        batchDstLD.data(),
        batchDstStartClass.data(),
        d_basisClassQuadStorage->data() +
          (classFromCfe ?
             0 :
             d_cellStartIdsBasisClassQuadStorage[cellRange.first]),
        basisData.data(),
        d_linAlgOpContext);

      linearAlgebra::blasLapack::varBatchedStridedBlockCopy(
        numBatch,
        batchStrideSrcEnrich.data(),
        batchStrideDst.data(),
        batchVecSize.data(),
        batchNumVecEnrich.data(),
        batchSrcLDEnrich.data(),
        batchSrcStartEnrich.data(),
        batchDstLD.data(),
        batchDstStartEnrich.data(),
        d_basisEnrichQuadStorage->data() + d_cellStartIdsBasisEnrichQuadStorage[cellRange.first],
        basisData.data(),
        d_linAlgOpContext);

      if (d_scratch != nullptr)
        d_scratch->releaseStorageScratch();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisGradientDataInCell(const size_type
                                                                 cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradient)
          ->second,
        "Basis gradient values are not evaluated for the given QuadratureRuleAttributes");
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId] * dim;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      std::pair<size_type, size_type> cellPair(cellId, cellId + 1);
      getBasisGradientDataInCellRange(cellPair, returnValue);
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::
      getBasisGradientDataInCellRange(std::pair<size_type, size_type> cellRange,
                                      Storage &basisGradientData) const
    {
      const size_type numBatch = cellRange.second - cellRange.first;

      if (d_scratch != nullptr)
        d_scratch->acquireStorageScratch(this, d_linAlgOpContext);

      // as getBasisDataInCellRange, with the gradient buffer
      const bool classFromCfe = d_cfeBasisDataStorage != nullptr;
      if (classFromCfe)
        {
          // another rule would misalign the per cell counts silently
          size_type classSize = 0;
          for (size_type cellId = cellRange.first; cellId < cellRange.second;
               cellId++)
            classSize +=
              d_nQuadPointsIncell[cellId] * d_classialDofsInCell * dim;
          if (d_basisGradientClassQuadStorage->size() < classSize)
            d_basisGradientClassQuadStorage->resize(classSize);
          d_cfeBasisDataStorage->getBasisGradientDataInCellRange(
            cellRange, *d_basisGradientClassQuadStorage);
        }

      std::vector<size_type> batchStrideSrcClass(numBatch);
      std::vector<size_type> batchStrideDst(numBatch);
      std::vector<size_type> batchVecSize(numBatch);
      std::vector<size_type> batchNumVecClass(numBatch, d_classialDofsInCell);
      std::vector<size_type> batchSrcLDClass(numBatch, d_classialDofsInCell);
      std::vector<size_type> batchSrcStartClass(numBatch, 0);
      std::vector<size_type> batchDstLD(numBatch);
      std::vector<size_type> batchDstStartClass(numBatch, 0);
      std::vector<size_type> batchStrideSrcEnrich(numBatch);
      std::vector<size_type> batchNumVecEnrich(numBatch);
      std::vector<size_type> batchSrcLDEnrich(numBatch);
      std::vector<size_type> batchSrcStartEnrich(numBatch, 0);
      std::vector<size_type> batchDstStartEnrich(numBatch,
                                                 d_classialDofsInCell);

      for (size_type ibatch = 0; ibatch < numBatch; ++ibatch)
        {
          const size_type cellId       = cellRange.first + ibatch;
          const size_type nQuad        = d_nQuadPointsIncell[cellId];
          const size_type nDofs        = d_dofsInCell[cellId];
          const size_type nEnriched    = nDofs - d_classialDofsInCell;
          batchVecSize[ibatch]         = nQuad * dim;
          batchDstLD[ibatch]           = nDofs;
          batchStrideDst[ibatch]       = nDofs * nQuad * dim;
          batchStrideSrcClass[ibatch] =
            classFromCfe ?
              nQuad * d_classialDofsInCell * dim :
              ((cellId + 1 < d_dofsInCell.size()) ?
                 (d_cellStartIdsBasisGradientClassQuadStorage[cellId + 1] -
                  d_cellStartIdsBasisGradientClassQuadStorage[cellId]) :
                 0);
          batchNumVecEnrich[ibatch]    = nEnriched;
          batchSrcLDEnrich[ibatch]     = nEnriched;
          batchStrideSrcEnrich[ibatch] = nEnriched * nQuad * dim;
        }

      linearAlgebra::blasLapack::varBatchedStridedBlockCopy(
        numBatch,
        batchStrideSrcClass.data(),
        batchStrideDst.data(),
        batchVecSize.data(),
        batchNumVecClass.data(),
        batchSrcLDClass.data(),
        batchSrcStartClass.data(),
        batchDstLD.data(),
        batchDstStartClass.data(),
        d_basisGradientClassQuadStorage->data() +
          (classFromCfe ?
             0 :
             d_cellStartIdsBasisGradientClassQuadStorage[cellRange.first]),
        basisGradientData.data(),
        d_linAlgOpContext);

      linearAlgebra::blasLapack::varBatchedStridedBlockCopy(
        numBatch,
        batchStrideSrcEnrich.data(),
        batchStrideDst.data(),
        batchVecSize.data(),
        batchNumVecEnrich.data(),
        batchSrcLDEnrich.data(),
        batchSrcStartEnrich.data(),
        batchDstLD.data(),
        batchDstStartEnrich.data(),
        d_basisGradientEnrichQuadStorage->data() + d_cellStartIdsBasisGradientEnrichQuadStorage[cellRange.first],
        basisGradientData.data(),
        d_linAlgOpContext);

      if (d_scratch != nullptr)
        d_scratch->releaseStorageScratch();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisHessianDataInCell(const size_type
                                                                cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreHessian)
          ->second,
        "Basis hessian values are not evaluated for the given QuadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage = d_basisHessianQuadStorage;
      const std::vector<size_type> &cellStartIds =
        d_cellStartIdsBasisHessianQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId] * dim * dim;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
        sizeToCopy,
        basisHessianQuadStorage->data() + cellStartIds[cellId],
        returnValue.data(),
        d_linAlgOpContext);
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::StorageReal
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getJxWInCell(const size_type cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
          ->second,
        "JxW values are not evaluated for the given QuadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::StorageReal>
        jxwQuadStorage = d_JxWStorage;

      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      const size_type               sizeToCopy = nQuadPointsInCell[cellId];
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::StorageReal
        returnValue(sizeToCopy);
      linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
        sizeToCopy,
        jxwQuadStorage->data() +
          d_quadratureRuleContainer->getCellQuadStartId(cellId),
        returnValue.data(),
        d_linAlgOpContext);
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisData(const QuadraturePointAttributes
                                                   &             attributes,
                                                 const size_type basisId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreValues)
          ->second,
        "Basis values are not evaluated for the given QuadraturePointAttributes");
      const quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        *(attributes.quadratureRuleAttributesPtr);
      const size_type cellId      = attributes.cellId;
      const size_type quadPointId = attributes.quadPointId;
      // the classical dofs come first in a cell, so basisId picks the storage
      const bool      isClassical = basisId < d_classialDofsInCell;
      const size_type nEnriched = d_dofsInCell[cellId] - d_classialDofsInCell;
      if (isClassical && d_cfeBasisDataStorage != nullptr)
        return d_cfeBasisDataStorage->getBasisData(attributes, basisId);
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisQuadStorage =
          isClassical ? d_basisClassQuadStorage : d_basisEnrichQuadStorage;
      const std::vector<size_type> &cellStartIds =
        isClassical ? d_cellStartIdsBasisClassQuadStorage :
                      d_cellStartIdsBasisEnrichQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(1);
      linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
        1,
        basisQuadStorage->data() + cellStartIds[cellId] +
          quadPointId * (isClassical ? d_classialDofsInCell : nEnriched) +
          (isClassical ? basisId : basisId - d_classialDofsInCell),
        returnValue.data(),
        d_linAlgOpContext);
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisGradientData(const QuadraturePointAttributes &attributes,
                                 const size_type                  basisId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradient)
          ->second,
        "Basis gradient values are not evaluated for the given QuadraturePointAttributes");
      const quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        *(attributes.quadratureRuleAttributesPtr);
      const size_type cellId      = attributes.cellId;
      const size_type quadPointId = attributes.quadPointId;
      // the classical dofs come first in a cell, so basisId picks the storage
      const bool      isClassical = basisId < d_classialDofsInCell;
      const size_type nEnriched = d_dofsInCell[cellId] - d_classialDofsInCell;
      const size_type dofsInStorage =
        isClassical ? d_classialDofsInCell : nEnriched;
      if (isClassical && d_cfeBasisDataStorage != nullptr)
        return d_cfeBasisDataStorage->getBasisGradientData(attributes, basisId);
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientQuadStorage = isClassical ?
                                     d_basisGradientClassQuadStorage :
                                     d_basisGradientEnrichQuadStorage;
      const std::vector<size_type> &cellStartIds =
        isClassical ? d_cellStartIdsBasisGradientClassQuadStorage :
                      d_cellStartIdsBasisGradientEnrichQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(dim);
      for (size_type iDim = 0; iDim < dim; ++iDim)
        {
          linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
            1,
            basisGradientQuadStorage->data() + cellStartIds[cellId] +
              quadPointId * dofsInStorage * dim + iDim * dofsInStorage +
              (isClassical ? basisId : basisId - d_classialDofsInCell),
            returnValue.data() + iDim,
            d_linAlgOpContext);
        }
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisHessianData(const QuadraturePointAttributes &attributes,
                                const size_type                  basisId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreHessian)
          ->second,
        "Basis hessian values are not evaluated for the given QuadraturePointAttributes");
      const quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        *(attributes.quadratureRuleAttributesPtr);
      const size_type cellId      = attributes.cellId;
      const size_type quadPointId = attributes.quadPointId;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage = d_basisHessianQuadStorage;
      const std::vector<size_type> &cellStartIds =
        d_cellStartIdsBasisHessianQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(dim * dim);
      for (size_type iDim = 0; iDim < dim; ++iDim)
        {
          for (size_type jDim = 0; jDim < dim; ++jDim)
            {
              linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
                1,
                basisHessianQuadStorage->data() + cellStartIds[cellId] +
                  quadPointId * d_dofsInCell[cellId] * dim * dim +
                  (iDim * dim + jDim) * d_dofsInCell[cellId] + basisId,
                returnValue.data() + iDim * dim + jDim,
                d_linAlgOpContext);
            }
        }
      return returnValue;
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisOverlapInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreOverlap)
          ->second,
        "Basis overlap values are not evaluated for the given QuadratureRuleAttributes");
      return *(d_basisOverlap);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisOverlapInCell(const size_type
                                                            cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreOverlap)
          ->second,
        "Basis overlap values are not evaluated for the given QuadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                      basisOverlapStorage = d_basisOverlap;
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
        sizeToCopy,
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId],
        returnValue.data(),
        d_linAlgOpContext);
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisOverlap(const size_type cellId,
                                                    const size_type basisId1,
                                                    const size_type basisId2)
      const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreOverlap)
          ->second,
        "Basis overlap values are not evaluated for the given QuadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisOverlapStorage = d_basisOverlap;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
                      returnValue(1);
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
        sizeToCopy,
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId] +
          basisId1 * d_dofsInCell[cellId] + basisId2,
        returnValue.data(),
        d_linAlgOpContext);
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::deleteBasisData()
    {
      // the classical buffers then belong to the scratch, which other
      // storages still use
      if (d_scratch == nullptr)
        {
          utils::throwException(
            (d_basisClassQuadStorage).use_count() == 1,
            "More than one owner for the basis quadrature storage found in EFEBasisDataStorageDealii. Not safe to delete it.");
          delete (d_basisClassQuadStorage).get();
        }

      utils::throwException(
        (d_basisEnrichQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBasisDataStorageDealii. Not safe to delete it.");
      delete (d_basisEnrichQuadStorage).get();

      if (d_scratch == nullptr)
        {
          utils::throwException(
            (d_basisGradientClassQuadStorage).use_count() == 1,
            "More than one owner for the basis quadrature storage found in EFEBasisDataStorageDealii. Not safe to delete it.");
          delete (d_basisGradientClassQuadStorage).get();
        }

      utils::throwException(
        (d_basisGradientEnrichQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBasisDataStorageDealii. Not safe to delete it.");
      delete (d_basisGradientEnrichQuadStorage).get();

      utils::throwException(
        (d_basisHessianQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBasisDataStorageDealii. Not safe to delete it.");
      delete (d_basisHessianQuadStorage).get();

      utils::throwException(
        (d_basisOverlap).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBasisDataStorageDealii. Not safe to delete it.");
      delete (d_basisOverlap).get();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisDataInCell(const size_type cellId,
                                                       const size_type basisId)
      const
    {
      utils::throwException(
        false,
        "getBasisDataInCell() for a given basisId is not implemented in EFEBasisDataStorageDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisGradientDataInCell(const size_type cellId,
                                       const size_type basisId) const
    {
      utils::throwException(
        false,
        "getBasisGradientDataInCell() for a given basisId is not implemented in EFEBasisDataStorageDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisHessianDataInCell(const size_type cellId,
                                      const size_type basisId) const
    {
      utils::throwException(
        false,
        "getBasisHessianDataInCell() for a given basisId is not implemented in EFEBasisDataStorageDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const quadrature::QuadratureRuleContainer>
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getQuadratureRuleContainer() const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      return (d_quadratureRuleContainer);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<BasisScratch<ValueTypeBasisData, memorySpace>>
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getScratch() const
    {
      return d_scratch;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisGradNiGradNjInCell(const size_type
                                                                 cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradNiGradNj)
          ->second,
        "Basis Grad Ni Grad Nj values are not evaluated for the given QuadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                      basisGradNiGradNj = d_basisGradNiGradNj;
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      linearAlgebra::blasLapack::copyValueType1ArrToValueType2Arr(
        sizeToCopy,
        basisGradNiGradNj->data() + d_cellStartIdsGradNiGradNj[cellId],
        returnValue.data(),
        d_linAlgOpContext);
      return returnValue;
    }

    // get overlap of all the basis functions in all cells
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisGradNiGradNjInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradNiGradNj)
          ->second,
        "Basis Grad Ni Grad Nj values are not evaluated for the given QuadratureRuleAttributes");
      return *(d_basisGradNiGradNj);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const BasisDofHandler>
    EFEBasisDataStorageDealii<ValueTypeBasisCoeff,
                              ValueTypeBasisData,
                              memorySpace,
                              dim>::getBasisDofHandler() const
    {
      return d_efeBDH;
    }

  } // namespace basis
} // namespace dftefe
