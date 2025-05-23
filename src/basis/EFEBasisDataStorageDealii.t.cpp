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
#include "DealiiConversions.h"
#include <basis/TriangulationCellDealii.h>
#include <basis/ConstraintsLocal.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRule.h>
#include <quadrature/QuadratureRuleContainer.h>
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
      std::vector<ValueTypeBasisData>
      getClassicalComponentCoeffsInCellOEFE(
        const size_type                                      cellIndex,
        std::shared_ptr<const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>> efeBDH)
      {
        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap = nullptr;
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeBasisData>>
          *enrichmentIdToInterfaceCoeffMap = nullptr;
        std::shared_ptr<const FEBasisManager<ValueTypeBasisData,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim>>
          cfeBasisManager = nullptr;

        cfeBasisManager =
          std::dynamic_pointer_cast<const FEBasisManager<ValueTypeBasisData,
                                                         ValueTypeBasisData,
                                                         memorySpace,
                                                         dim>>(
            efeBDH->getEnrichmentClassicalInterface()->getCFEBasisManager());

        enrichmentIdToClassicalLocalIdMap =
          &(efeBDH->getEnrichmentClassicalInterface()
              ->getClassicalComponentLocalIdsMap());

        enrichmentIdToInterfaceCoeffMap =
          &(efeBDH->getEnrichmentClassicalInterface()
              ->getClassicalComponentCoeffMap());

        std::vector<size_type> vecClassicalLocalNodeId(0);

        cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                             vecClassicalLocalNodeId);

        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(cellIndex) + 1),
                                            dim);
        size_type numEnrichmentIdsInCell =
          efeBDH->nCellDofs(cellIndex) - classicalDofsPerCell;

        std::vector<ValueTypeBasisData> coeffsInCell(classicalDofsPerCell *
                                                       numEnrichmentIdsInCell,
                                                     0);

        for (size_type cellEnrichId = 0; cellEnrichId < numEnrichmentIdsInCell;
             cellEnrichId++)
          {
            // get the enrichmentIds
            global_size_type enrichmentId =
              efeBDH->getEnrichmentClassicalInterface()->getEnrichmentId(
                cellIndex, cellEnrichId);

            // get the vectors of non-zero localIds and coeffs
            auto iter = enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
            auto it   = enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
            if (iter != enrichmentIdToInterfaceCoeffMap->end() &&
                it != enrichmentIdToClassicalLocalIdMap->end())
              {
                const std::vector<ValueTypeBasisData> &coeffsInLocalIdsMap =
                  iter->second;

                for (size_type i = 0; i < classicalDofsPerCell; i++)
                  {
                    size_type pos   = 0;
                    bool      found = false;
                    it->second.getPosition(vecClassicalLocalNodeId[i],
                                           pos,
                                           found);
                    if (found)
                      {
                        coeffsInCell[numEnrichmentIdsInCell * i +
                                     cellEnrichId] = coeffsInLocalIdsMap[pos];
                      }
                  }
              }
          }
        return coeffsInCell;
      }

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
          linearAlgebra::blasLapack::Layout::ColMajor,
          linearAlgebra::blasLapack::Op::NoTrans,
          linearAlgebra::blasLapack::Op::NoTrans,
          numEnrichmentIdsInCell,
          nQuadPointInCell,
          classicalDofsPerCell,
          (ValueTypeBasisData)1.0,
          coeffsInCell.data(),
          numEnrichmentIdsInCell,
          basisValInCellPtr,
          efeBDH->nCellDofs(cellIndex),
          (ValueTypeBasisData)0.0,
          classicalComponentInQuadValues.data(),
          numEnrichmentIdsInCell,
          *efeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext());
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
          linearAlgebra::blasLapack::Layout::ColMajor,
          linearAlgebra::blasLapack::Op::NoTrans,
          linearAlgebra::blasLapack::Op::NoTrans,
          numEnrichmentIdsInCell,
          nQuadPointInCell * dim,
          classicalDofsPerCell,
          (ValueTypeBasisData)1.0,
          coeffsInCell.data(),
          numEnrichmentIdsInCell,
          basisGradInCell,
          efeBDH->nCellDofs(cellIndex),
          (ValueTypeBasisData)0.0,
          classicalComponentInQuadGradients
            .data(), // saved as cell->quad->dim->enrichid
          numEnrichmentIdsInCell,
          *efeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext());
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
          &basisQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientQuadStorage,
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
        std::vector<size_type> &cellStartIdsBasisQuadStorage,
        std::vector<size_type> &cellStartIdsBasisGradientQuadStorage,
        std::vector<size_type> &cellStartIdsBasisHessianQuadStorage,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
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
                unsigned int                     cellIndex = 0;
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
        const size_type nQuadPointInCell =
          numCellsZero ? 0 :
                         quadratureRuleContainer->nCellQuadraturePoints(cellId);
        // utils::mathFunctions::sizeTypePow(num1DQuadPoints, dim);

        nQuadPointsInCell.resize(numLocallyOwnedCells, nQuadPointInCell);
        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisQuadStorageTmp(0), basisGradientQuadStorageTmp(0),
          basisHessianQuadStorageTmp(0), basisOverlapTmp(0);

        size_type cellIndex        = 0;
        size_type basisValuesSize  = 0;
        size_type basisOverlapSize = 0;

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            basisValuesSize += nQuadPointInCell * dofsPerCell;
            basisOverlapSize += dofsPerCell * dofsPerCell;
            cellIndex++;
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            basisQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisValuesSize);
            basisQuadStorageTmp.resize(basisValuesSize, ValueTypeBasisData(0));
            cellStartIdsBasisQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            basisGradientQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisValuesSize * dim);
            basisGradientQuadStorageTmp.resize(basisValuesSize * dim,
                                               ValueTypeBasisData(0));
            cellStartIdsBasisGradientQuadStorage.resize(numLocallyOwnedCells,
                                                        0);
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

        /*
        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap = nullptr;
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeBasisData>>
          *enrichmentIdToInterfaceCoeffMap = nullptr;
        std::shared_ptr<const FEBasisManager<ValueTypeBasisData,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim>>
          cfeBasisManager = nullptr;

        if (efeBDH->isOrthogonalized())
          {
            cfeBasisManager =
              std::dynamic_pointer_cast<const FEBasisManager<ValueTypeBasisData,
                                                             ValueTypeBasisData,
                                                             memorySpace,
                                                             dim>>(
                efeBDH->getEnrichmentClassicalInterface()
                  ->getCFEBasisManager());

            enrichmentIdToClassicalLocalIdMap =
              &(efeBDH->getEnrichmentClassicalInterface()
                  ->getClassicalComponentLocalIdsMap());

            enrichmentIdToInterfaceCoeffMap =
              &(efeBDH->getEnrichmentClassicalInterface()
                  ->getClassicalComponentCoeffMap());
          }
          */

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
                  getClassicalComponentCoeffsInCellOEFE<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>(cellIndex, efeBDH);

                // if (basisStorageAttributesBoolMap
                //       .find(BasisStorageAttributes::StoreValues)
                //       ->second ||
                //     basisStorageAttributesBoolMap
                //       .find(BasisStorageAttributes::StoreOverlap)
                //       ->second)
                //   {
                //     getClassicalComponentBasisValuesInCellAtQuadOEFE<
                //       ValueTypeBasisCoeff,
                //       ValueTypeBasisData,
                //       memorySpace,
                //       dim>(cellIndex,
                //            nQuadPointInCell,
                //            coeffsInCell,
                //            efeBDH,
                //            cfeBasisDataStorage,
                //            classicalComponentInQuadValues);
                //   }

                // if (basisStorageAttributesBoolMap
                //       .find(BasisStorageAttributes::StoreGradient)
                //       ->second)
                //   {
                //     getClassicalComponentBasisGradInCellAtQuadOEFE<
                //       ValueTypeBasisCoeff,
                //       ValueTypeBasisData,
                //       memorySpace,
                //       dim>(cellIndex,
                //            nQuadPointInCell,
                //            coeffsInCell,
                //            efeBDH,
                //            cfeBasisDataStorage,
                //            classicalComponentInQuadGradients);
                //   }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                cellStartIdsBasisQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnDofs;
                for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *(basisQuadStorageTmp.begin() +
                          cumulativeQuadPointsxnDofs + qPoint * dofsPerCell +
                          iNode
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
                           basisQuadStorageTmp.begin() +
                             cumulativeQuadPointsxnDofs,
                           classicalComponentInQuadValues);
                  }
                for (unsigned int iNode = classicalDofsPerCell;
                     iNode < dofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *(basisQuadStorageTmp.begin() +
                          cumulativeQuadPointsxnDofs + qPoint * dofsPerCell +
                          iNode
                          /*iNode * nQuadPointInCell + qPoint*/) =
                          efeBDH->getEnrichmentValue(
                            cellIndex,
                            iNode - classicalDofsPerCell,
                            quadRealPointsVec[qPoint]) -
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
                    basisQuadStorageTmp.resize(dofsPerCell * nQuadPointInCell,
                                               ValueTypeBasisData(0));
                    for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                         iNode++)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *(basisQuadStorageTmp.begin() +
                              qPoint * dofsPerCell + iNode
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
                               basisQuadStorageTmp.begin(),
                               classicalComponentInQuadValues);
                      }
                  }

                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                      {
                        *basisOverlapTmpIter = 0.0;
                        if (iNode < classicalDofsPerCell &&
                            jNode < classicalDofsPerCell)
                          {
                            for (unsigned int qPoint = 0;
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
                            for (unsigned int qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  (efeBDH->getEnrichmentValue(
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
                            for (unsigned int qPoint = 0;
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
                            for (unsigned int qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  (efeBDH->getEnrichmentValue(
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
                cellStartIdsBasisGradientQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnDofs * dim;
                for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (unsigned int iDim = 0; iDim < dim; iDim++)
                          {
                            auto it = basisGradientQuadStorageTmp.begin() +
                                      cumulativeQuadPointsxnDofs * dim +
                                      qPoint * dim * dofsPerCell +
                                      iDim * dofsPerCell + iNode;
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
                           basisGradientQuadStorageTmp.begin() +
                             cumulativeQuadPointsxnDofs * dim,
                           classicalComponentInQuadGradients);
                  }

                for (unsigned int iNode = classicalDofsPerCell;
                     iNode < dofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad = efeBDH->getEnrichmentDerivative(
                          cellIndex,
                          iNode - classicalDofsPerCell,
                          quadRealPointsVec[qPoint]);
                        // enriched gradient function call
                        for (unsigned int iDim = 0; iDim < dim; iDim++)
                          {
                            auto it = basisGradientQuadStorageTmp.begin() +
                                      cumulativeQuadPointsxnDofs * dim +
                                      qPoint * dim * dofsPerCell +
                                      iDim * dofsPerCell + iNode;
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim] -
                                  classicalComponentInQuadGradients
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
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto shapeHessian =
                              dealiiFEValues.shape_hessian(iNode, qPoint);
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                for (unsigned int jDim = 0; jDim < dim; jDim++)
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                for (unsigned int jDim = 0; jDim < dim; jDim++)
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
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisQuadStorageTmp.size(),
              basisQuadStorage->data(),
              basisQuadStorageTmp.data());
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisGradientQuadStorageTmp.size(),
              basisGradientQuadStorage->data(),
              basisGradientQuadStorageTmp.data());
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
                unsigned int                     cellIndex = 0;
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

        /**
        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap = nullptr;
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeBasisData>>
          *enrichmentIdToInterfaceCoeffMap = nullptr;
        std::shared_ptr<const FEBasisManager<ValueTypeBasisData,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim>>
          cfeBasisManager = nullptr;

        if (efeBDH->isOrthogonalized())
          {
            enrichmentIdToClassicalLocalIdMap =
              &(efeBDH->getEnrichmentClassicalInterface()
                  ->getClassicalComponentLocalIdsMap());

            enrichmentIdToInterfaceCoeffMap =
              &(efeBDH->getEnrichmentClassicalInterface()
                  ->getClassicalComponentCoeffMap());

            cfeBasisManager =
              std::dynamic_pointer_cast<const FEBasisManager<ValueTypeBasisData,
                                                             ValueTypeBasisData,
                                                             memorySpace,
                                                             dim>>(
                efeBDH->getEnrichmentClassicalInterface()
                  ->getCFEBasisManager());
          }
        **/

        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisGradientQuadStorageTmp(0);

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
                basisGradientQuadStorageTmp.resize(dofsPerCell *
                                                     nQuadPointInCell * dim,
                                                   ValueTypeBasisData(0));
                for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (unsigned int iDim = 0; iDim < dim; iDim++)
                          {
                            auto it = basisGradientQuadStorageTmp.begin() +
                                      qPoint * dim * dofsPerCell +
                                      iDim * dofsPerCell + iNode;
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim];
                          }
                      }
                  }
                std::vector<ValueTypeBasisData> coeffsInCell =
                  getClassicalComponentCoeffsInCellOEFE<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>(cellIndex, efeBDH);

                getClassicalComponentBasisGradInCellAtQuadOEFE<
                  ValueTypeBasisCoeff,
                  ValueTypeBasisData,
                  memorySpace,
                  dim>(cellIndex,
                       nQuadPointInCell,
                       coeffsInCell,
                       efeBDH,
                       basisGradientQuadStorageTmp.begin(),
                       classicalComponentInQuadGradients);

                /**
                cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                                     vecClassicalLocalNodeId);

                std::vector<ValueTypeBasisData> coeffsInCell(
                  classicalDofsPerCell * numEnrichmentIdsInCell, 0);

                for (size_type cellEnrichId = 0;
                     cellEnrichId < numEnrichmentIdsInCell;
                     cellEnrichId++)
                  {
                    // get the enrichmentIds
                    global_size_type enrichmentId =
                      efeBDH->getEnrichmentClassicalInterface()
                        ->getEnrichmentId(cellIndex, cellEnrichId);

                    // get the vectors of non-zero localIds and coeffs
                    auto iter =
                      enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
                    auto it =
                      enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
                    if (iter != enrichmentIdToInterfaceCoeffMap->end() &&
                        it != enrichmentIdToClassicalLocalIdMap->end())
                      {
                        const std::vector<ValueTypeBasisData>
                          &coeffsInLocalIdsMap = iter->second;

                        for (size_type i = 0; i < classicalDofsPerCell; i++)
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

                // saved as cell->dim->node->quad
                dftefe::utils::MemoryStorage<ValueTypeBasisData,
                                             utils::MemorySpace::HOST>
                  basisGradInCell =
                    cfeBasisDataStorage->getBasisGradientDataInCell(cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector

                for (size_type iDim = 0; iDim < dim; iDim++)
                  {
                    ValueTypeBasisData *B =
                      basisGradInCell.data() +
                      iDim * nQuadPointInCell * classicalDofsPerCell;

                    std::vector<ValueTypeBasisData> classicalComponentInQuadDim(
                      nQuadPointInCell * numEnrichmentIdsInCell,
                      (ValueTypeBasisData)0);

                    linearAlgebra::blasLapack::gemm<ValueTypeBasisData,
                                                    ValueTypeBasisData,
                                                    utils::MemorySpace::HOST>(
                      linearAlgebra::blasLapack::Layout::ColMajor,
                      linearAlgebra::blasLapack::Op::NoTrans,
                      linearAlgebra::blasLapack::Op::Trans,
                      numEnrichmentIdsInCell,
                      nQuadPointInCell,
                      classicalDofsPerCell,
                      (ValueTypeBasisData)1.0,
                      coeffsInCell.data(),
                      numEnrichmentIdsInCell,
                      B,
                      nQuadPointInCell,
                      (ValueTypeBasisData)0.0,
                      classicalComponentInQuadDim.data(),
                      numEnrichmentIdsInCell,
                      *efeBDH->getEnrichmentClassicalInterface()
                         ->getLinAlgOpContext());

                    classicalComponentInQuadGradients[iDim] =
                      classicalComponentInQuadDim;
                  }
                  **/
              }


            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisGradNiGradNjTmpIter = 0.0;
                    if (iNode < classicalDofsPerCell &&
                        jNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  (enrichmentDerivative[k] -
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                            for (unsigned int k = 0; k < dim; k++)
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  (enrichmentDerivativei[k] -
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
          &basisQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientQuadStorage,
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
        std::vector<size_type> &cellStartIdsBasisQuadStorage,
        std::vector<size_type> &cellStartIdsBasisGradientQuadStorage,
        std::vector<size_type> &cellStartIdsBasisHessianQuadStorage,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
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
          basisQuadStorageTmp(0), basisGradientQuadStorageTmp(0),
          basisHessianQuadStorageTmp(0), basisOverlapTmp(0);

        // Find total quadpoints in the processor
        nQuadPointsInCell.resize(numLocallyOwnedCells, 0);
        const size_type nTotalQuadPoints =
          quadratureRuleContainer->nQuadraturePoints();

        size_type cellIndex        = 0;
        size_type basisValuesSize  = 0;
        size_type basisOverlapSize = 0;
        size_type nQuadPointInCell = 0;

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
            cellIndex++;
          }

        // Initialize the host and tmp vector for storing basis values in
        // a flattened array. Also cell start ids for the flattened array.
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            basisQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisValuesSize);
            basisQuadStorageTmp.resize(basisValuesSize, ValueTypeBasisData(0));
            cellStartIdsBasisQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        // Initialize the host and tmp vector for storing basis gradient in
        // a flattened array. Also cell start ids for the flattened array.
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            basisGradientQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                basisValuesSize * dim);
            basisGradientQuadStorageTmp.resize(basisValuesSize * dim,
                                               ValueTypeBasisData(0));
            cellStartIdsBasisGradientQuadStorage.resize(numLocallyOwnedCells,
                                                        0);
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

        size_type cumulativeQuadPointsxnDofs = 0;

        /**
        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap = nullptr;
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeBasisData>>
          *enrichmentIdToInterfaceCoeffMap = nullptr;
        std::shared_ptr<const FEBasisManager<ValueTypeBasisData,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim>>
          cfeBasisManager = nullptr;


        if (efeBDH->isOrthogonalized())
          {
            enrichmentIdToClassicalLocalIdMap =
              &(efeBDH->getEnrichmentClassicalInterface()
                  ->getClassicalComponentLocalIdsMap());

            enrichmentIdToInterfaceCoeffMap =
              &(efeBDH->getEnrichmentClassicalInterface()
                  ->getClassicalComponentCoeffMap());

            cfeBasisManager =
              std::dynamic_pointer_cast<const FEBasisManager<ValueTypeBasisData,
                                                             ValueTypeBasisData,
                                                             memorySpace,
                                                             dim>>(
                efeBDH->getEnrichmentClassicalInterface()
                  ->getCFEBasisManager());
          }
          **/

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

            // Ask dealii to create quad rule in each cell
            dealii::Quadrature<dim> dealiiQuadratureRule(
              dealiiParametricQuadPoints, quadWeights);

            // Ask dealii for the update flags.
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
                  getClassicalComponentCoeffsInCellOEFE<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>(cellIndex, efeBDH);

                // if (basisStorageAttributesBoolMap
                //       .find(BasisStorageAttributes::StoreValues)
                //       ->second ||
                //     basisStorageAttributesBoolMap
                //       .find(BasisStorageAttributes::StoreOverlap)
                //       ->second)
                //   {
                //     getClassicalComponentBasisValuesInCellAtQuadOEFE<
                //       ValueTypeBasisCoeff,
                //       ValueTypeBasisData,
                //       memorySpace,
                //       dim>(cellIndex,
                //            nQuadPointInCell,
                //            coeffsInCell,
                //            efeBDH,
                //            cfeBasisDataStorage,
                //            classicalComponentInQuadValues);
                //   }

                // if (basisStorageAttributesBoolMap
                //       .find(BasisStorageAttributes::StoreGradient)
                //       ->second)
                //   {
                //     getClassicalComponentBasisGradInCellAtQuadOEFE<
                //       ValueTypeBasisCoeff,
                //       ValueTypeBasisData,
                //       memorySpace,
                //       dim>(cellIndex,
                //            nQuadPointInCell,
                //            coeffsInCell,
                //            efeBDH,
                //            cfeBasisDataStorage,
                //            classicalComponentInQuadGradients);
                //   }

                /**
                cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                                     vecClassicalLocalNodeId);

                std::vector<ValueTypeBasisData> coeffsInCell(
                  classicalDofsPerCell * numEnrichmentIdsInCell, 0);

                for (size_type cellEnrichId = 0;
                     cellEnrichId < numEnrichmentIdsInCell;
                     cellEnrichId++)
                  {
                    // get the enrichmentIds
                    global_size_type enrichmentId =
                      efeBDH->getEnrichmentClassicalInterface()
                        ->getEnrichmentId(cellIndex, cellEnrichId);

                    // // get the vectors of non-zero localIds and coeffs

                    // auto iter =
                    //   enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
                    // DFTEFE_Assert(iter !=
                    //               enrichmentIdToInterfaceCoeffMap->end());
                    // const std::vector<ValueTypeBasisData>
                    // &coeffsInLocalIdsMap =
                    //   iter->second;

                    // for (size_type i = 0; i < classicalDofsPerCell; i++)
                    //   {
                    //     size_type pos   = 0;
                    //     bool      found = false;
                    //     auto      it =
                    // enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
                    //     DFTEFE_Assert(it !=
                    // enrichmentIdToClassicalLocalIdMap->end());
                    //     it->second.getPosition(vecClassicalLocalNodeId[i],
                    //                            pos,
                    //                            found);
                    //     if (found)
                    //       {
                    //         coeffsInCell[numEnrichmentIdsInCell * i +
                    //                      cellEnrichId] =
                    //           coeffsInLocalIdsMap[pos];
                    //       }
                    //   }

                    // get the vectors of non-zero localIds and coeffs
                    auto iter =
                      enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
                    auto it =
                      enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
                    if (iter != enrichmentIdToInterfaceCoeffMap->end() &&
                        it != enrichmentIdToClassicalLocalIdMap->end())
                      {
                        const std::vector<ValueTypeBasisData>
                          &coeffsInLocalIdsMap = iter->second;

                        for (size_type i = 0; i < classicalDofsPerCell; i++)
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

                if (basisStorageAttributesBoolMap
                      .find(BasisStorageAttributes::StoreValues)
                      ->second ||
                    basisStorageAttributesBoolMap
                      .find(BasisStorageAttributes::StoreOverlap)
                      ->second)
                  {
                    dftefe::utils::MemoryStorage<ValueTypeBasisData,
                                                 utils::MemorySpace::HOST>
                      basisValInCell =
                        cfeBasisDataStorage->getBasisDataInCell(cellIndex);

                    // Do a gemm (\Sigma c_i N_i^classical)
                    // and get the quad values in std::vector

                    linearAlgebra::blasLapack::gemm<ValueTypeBasisData,
                                                    ValueTypeBasisData,
                                                    utils::MemorySpace::HOST>(
                      linearAlgebra::blasLapack::Layout::ColMajor,
                      linearAlgebra::blasLapack::Op::NoTrans,
                      linearAlgebra::blasLapack::Op::Trans,
                      numEnrichmentIdsInCell,
                      nQuadPointInCell,
                      classicalDofsPerCell,
                      (ValueTypeBasisData)1.0,
                      coeffsInCell.data(),
                      numEnrichmentIdsInCell,
                      basisValInCell.data(),
                      nQuadPointInCell,
                      (ValueTypeBasisData)0.0,
                      classicalComponentInQuadValues.data(),
                      numEnrichmentIdsInCell,
                      *efeBDH->getEnrichmentClassicalInterface()
                         ->getLinAlgOpContext());
                  }

                if (basisStorageAttributesBoolMap
                      .find(BasisStorageAttributes::StoreGradient)
                      ->second)
                  {
                    // saved as cell->dim->node->quad
                    dftefe::utils::MemoryStorage<ValueTypeBasisData,
                                                 utils::MemorySpace::HOST>
                      basisGradInCell =
                        cfeBasisDataStorage->getBasisGradientDataInCell(
                          cellIndex);

                    // Do a gemm (\Sigma c_i N_i^classical)
                    // and get the quad values in std::vector

                    for (size_type iDim = 0; iDim < dim; iDim++)
                      {
                        ValueTypeBasisData *B =
                          basisGradInCell.data() +
                          iDim * nQuadPointInCell * classicalDofsPerCell;

                        std::vector<ValueTypeBasisData>
                          classicalComponentInQuadDim(nQuadPointInCell *
                                                        numEnrichmentIdsInCell,
                                                      (ValueTypeBasisData)0);

                        linearAlgebra::blasLapack::gemm<
                          ValueTypeBasisData,
                          ValueTypeBasisData,
                          utils::MemorySpace::HOST>(
                          linearAlgebra::blasLapack::Layout::ColMajor,
                          linearAlgebra::blasLapack::Op::NoTrans,
                          linearAlgebra::blasLapack::Op::Trans,
                          numEnrichmentIdsInCell,
                          nQuadPointInCell,
                          classicalDofsPerCell,
                          (ValueTypeBasisData)1.0,
                          coeffsInCell.data(),
                          numEnrichmentIdsInCell,
                          B,
                          nQuadPointInCell,
                          (ValueTypeBasisData)0.0,
                          classicalComponentInQuadDim.data(),
                          numEnrichmentIdsInCell,
                          *efeBDH->getEnrichmentClassicalInterface()
                             ->getLinAlgOpContext());

                        classicalComponentInQuadGradients[iDim] =
                          classicalComponentInQuadDim;
                      }
                  }
                  **/
              }

            // Store the basis values.
            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                cellStartIdsBasisQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnDofs;
                for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *(basisQuadStorageTmp.begin() +
                          cumulativeQuadPointsxnDofs + qPoint * dofsPerCell +
                          iNode
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
                           basisQuadStorageTmp.begin() +
                             cumulativeQuadPointsxnDofs,
                           classicalComponentInQuadValues);
                  }
                for (unsigned int iNode = classicalDofsPerCell;
                     iNode < dofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *(basisQuadStorageTmp.begin() +
                          cumulativeQuadPointsxnDofs + qPoint * dofsPerCell +
                          iNode
                          /*iNode * nQuadPointInCell + qPoint*/) =
                          efeBDH->getEnrichmentValue(
                            cellIndex,
                            iNode - classicalDofsPerCell,
                            quadRealPointsVec[qPoint]) -
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
                    basisQuadStorageTmp.resize(dofsPerCell * nQuadPointInCell,
                                               ValueTypeBasisData(0));
                    for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                         iNode++)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *(basisQuadStorageTmp.begin() +
                              qPoint * dofsPerCell + iNode
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
                               basisQuadStorageTmp.begin(),
                               classicalComponentInQuadValues);
                      }
                  }
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                      {
                        *basisOverlapTmpIter = 0.0;
                        if (iNode < classicalDofsPerCell &&
                            jNode < classicalDofsPerCell)
                          {
                            for (unsigned int qPoint = 0;
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
                            for (unsigned int qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  (efeBDH->getEnrichmentValue(
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
                            for (unsigned int qPoint = 0;
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
                            for (unsigned int qPoint = 0;
                                 qPoint < nQuadPointInCell;
                                 qPoint++)
                              {
                                *basisOverlapTmpIter +=
                                  (efeBDH->getEnrichmentValue(
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
                cellStartIdsBasisGradientQuadStorage[cellIndex] =
                  cumulativeQuadPointsxnDofs * dim;
                for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (unsigned int iDim = 0; iDim < dim; iDim++)
                          {
                            auto it = basisGradientQuadStorageTmp.begin() +
                                      cumulativeQuadPointsxnDofs * dim +
                                      qPoint * dim * dofsPerCell +
                                      iDim * dofsPerCell + iNode;
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
                           basisGradientQuadStorageTmp.begin() +
                             cumulativeQuadPointsxnDofs * dim,
                           classicalComponentInQuadGradients);
                  }

                for (unsigned int iNode = classicalDofsPerCell;
                     iNode < dofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad = efeBDH->getEnrichmentDerivative(
                          cellIndex,
                          iNode - classicalDofsPerCell,
                          quadRealPointsVec[qPoint]);
                        // enriched gradient function call
                        for (unsigned int iDim = 0; iDim < dim; iDim++)
                          {
                            auto it = basisGradientQuadStorageTmp.begin() +
                                      cumulativeQuadPointsxnDofs * dim +
                                      qPoint * dim * dofsPerCell +
                                      iDim * dofsPerCell + iNode;
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim] -
                                  classicalComponentInQuadGradients
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
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto shapeHessian =
                              dealiiFEValues.shape_hessian(iNode, qPoint);
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                for (unsigned int jDim = 0; jDim < dim; jDim++)
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                for (unsigned int jDim = 0; jDim < dim; jDim++)
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
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisQuadStorageTmp.size(),
              basisQuadStorage->data(),
              basisQuadStorageTmp.data());
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisGradientQuadStorageTmp.size(),
              basisGradientQuadStorage->data(),
              basisGradientQuadStorageTmp.data());
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

        /**
        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap = nullptr;
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeBasisData>>
          *enrichmentIdToInterfaceCoeffMap = nullptr;
        std::shared_ptr<const FEBasisManager<ValueTypeBasisData,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim>>
          cfeBasisManager = nullptr;

        if (efeBDH->isOrthogonalized())
          {
            enrichmentIdToClassicalLocalIdMap =
              &(efeBDH->getEnrichmentClassicalInterface()
                  ->getClassicalComponentLocalIdsMap());

            enrichmentIdToInterfaceCoeffMap =
              &(efeBDH->getEnrichmentClassicalInterface()
                  ->getClassicalComponentCoeffMap());

            cfeBasisManager =
              std::dynamic_pointer_cast<const FEBasisManager<ValueTypeBasisData,
                                                             ValueTypeBasisData,
                                                             memorySpace,
                                                             dim>>(
                efeBDH->getEnrichmentClassicalInterface()
                  ->getCFEBasisManager());
          }
        **/

        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisGradientQuadStorageTmp(0);

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
                basisGradientQuadStorageTmp.resize(dofsPerCell *
                                                     nQuadPointInCell * dim,
                                                   ValueTypeBasisData(0));
                for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                     iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (unsigned int iDim = 0; iDim < dim; iDim++)
                          {
                            auto it = basisGradientQuadStorageTmp.begin() +
                                      qPoint * dim * dofsPerCell +
                                      iDim * dofsPerCell + iNode;
                            // iDim * dofsPerCell * nQuadPointInCell +
                            // iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim];
                          }
                      }
                  }
                std::vector<ValueTypeBasisData> coeffsInCell =
                  getClassicalComponentCoeffsInCellOEFE<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>(cellIndex, efeBDH);

                getClassicalComponentBasisGradInCellAtQuadOEFE<
                  ValueTypeBasisCoeff,
                  ValueTypeBasisData,
                  memorySpace,
                  dim>(cellIndex,
                       nQuadPointInCell,
                       coeffsInCell,
                       efeBDH,
                       basisGradientQuadStorageTmp.begin(),
                       classicalComponentInQuadGradients);

                /**
                cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                                     vecClassicalLocalNodeId);

                std::vector<ValueTypeBasisData> coeffsInCell(
                  classicalDofsPerCell * numEnrichmentIdsInCell, 0);

                for (size_type cellEnrichId = 0;
                     cellEnrichId < numEnrichmentIdsInCell;
                     cellEnrichId++)
                  {
                    // get the enrichmentIds
                    global_size_type enrichmentId =
                      efeBDH->getEnrichmentClassicalInterface()
                        ->getEnrichmentId(cellIndex, cellEnrichId);

                    // get the vectors of non-zero localIds and coeffs
                    auto iter =
                      enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
                    auto it =
                      enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
                    if (iter != enrichmentIdToInterfaceCoeffMap->end() &&
                        it != enrichmentIdToClassicalLocalIdMap->end())
                      {
                        const std::vector<ValueTypeBasisData>
                          &coeffsInLocalIdsMap = iter->second;

                        for (size_type i = 0; i < classicalDofsPerCell; i++)
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

                // saved as cell->dim->node->quad
                dftefe::utils::MemoryStorage<ValueTypeBasisData,
                                             utils::MemorySpace::HOST>
                  basisGradInCell =
                    cfeBasisDataStorage->getBasisGradientDataInCell(cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector

                for (size_type iDim = 0; iDim < dim; iDim++)
                  {
                    ValueTypeBasisData *B =
                      basisGradInCell.data() +
                      iDim * nQuadPointInCell * classicalDofsPerCell;

                    std::vector<ValueTypeBasisData> classicalComponentInQuadDim(
                      nQuadPointInCell * numEnrichmentIdsInCell,
                      (ValueTypeBasisData)0);

                    linearAlgebra::blasLapack::gemm<ValueTypeBasisData,
                                                    ValueTypeBasisData,
                                                    utils::MemorySpace::HOST>(
                      linearAlgebra::blasLapack::Layout::ColMajor,
                      linearAlgebra::blasLapack::Op::NoTrans,
                      linearAlgebra::blasLapack::Op::Trans,
                      numEnrichmentIdsInCell,
                      nQuadPointInCell,
                      classicalDofsPerCell,
                      (ValueTypeBasisData)1.0,
                      coeffsInCell.data(),
                      numEnrichmentIdsInCell,
                      B,
                      nQuadPointInCell,
                      (ValueTypeBasisData)0.0,
                      classicalComponentInQuadDim.data(),
                      numEnrichmentIdsInCell,
                      *efeBDH->getEnrichmentClassicalInterface()
                         ->getLinAlgOpContext());

                    classicalComponentInQuadGradients[iDim] =
                      classicalComponentInQuadDim;
                  }
                **/
              }

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisGradNiGradNjTmpIter = 0.0;
                    if (iNode < classicalDofsPerCell &&
                        jNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  (enrichmentDerivative[k] -
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                            for (unsigned int k = 0; k < dim; k++)
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd =
                                  dotProd +
                                  (enrichmentDerivativei[k] -
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
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
      : d_dofsInCell(0)
      , d_cellStartIdsBasisOverlap(0)
      , d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_basisStorageAttributesBoolMap(basisStorageAttributesBoolMap)
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
        }
      else
        utils::throwException<utils::InvalidArgument>(
          false, "Incorrect arguments given for this Quadrature family.");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientQuadStorage;
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
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);
      EFEBasisDataStorageDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisQuadStorage,
             basisGradientQuadStorage,
             basisHessianQuadStorage,
             basisOverlap,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisQuadStorage,
             cellStartIdsBasisGradientQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisQuadStorage             = std::move(basisQuadStorage);
          d_cellStartIdsBasisQuadStorage = cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage = std::move(basisGradientQuadStorage);
          d_cellStartIdsBasisGradientQuadStorage =
            cellStartIdsBasisGradientQuadStorage;
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
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer->getJxW();
          jxwQuadStorage =
            std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                       memorySpace>::Storage>(
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

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientQuadStorage;
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
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      if (quadFamily == quadrature::QuadratureFamily::GAUSS_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::GLL_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::ADAPTIVE)
        EFEBasisDataStorageDealiiInternal::storeValuesHRefinedAdaptiveQuad<
          ValueTypeBasisCoeff,
          ValueTypeBasisData,
          memorySpace,
          dim>(d_efeBDH,
               basisQuadStorage,
               basisGradientQuadStorage,
               basisHessianQuadStorage,
               basisOverlap,
               quadratureRuleAttributes,
               d_quadratureRuleContainer,
               nQuadPointsInCell,
               cellStartIdsBasisQuadStorage,
               cellStartIdsBasisGradientQuadStorage,
               cellStartIdsBasisHessianQuadStorage,
               basisStorageAttributesBoolMap);

      else
        EFEBasisDataStorageDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
          ValueTypeBasisCoeff,
          ValueTypeBasisData,
          memorySpace,
          dim>(d_efeBDH,
               basisQuadStorage,
               basisGradientQuadStorage,
               basisHessianQuadStorage,
               basisOverlap,
               quadratureRuleAttributes,
               d_quadratureRuleContainer,
               nQuadPointsInCell,
               cellStartIdsBasisQuadStorage,
               cellStartIdsBasisGradientQuadStorage,
               cellStartIdsBasisHessianQuadStorage,
               basisStorageAttributesBoolMap);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisQuadStorage             = std::move(basisQuadStorage);
          d_cellStartIdsBasisQuadStorage = cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage = std::move(basisGradientQuadStorage);
          d_cellStartIdsBasisGradientQuadStorage =
            cellStartIdsBasisGradientQuadStorage;
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
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer->getJxW();
          jxwQuadStorage =
            std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                       memorySpace>::Storage>(
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
        basisQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientQuadStorage;
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
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      EFEBasisDataStorageDealiiInternal::storeValuesHRefinedAdaptiveQuad<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisQuadStorage,
             basisGradientQuadStorage,
             basisHessianQuadStorage,
             basisOverlap,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisQuadStorage,
             cellStartIdsBasisGradientQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisQuadStorage             = std::move(basisQuadStorage);
          d_cellStartIdsBasisQuadStorage = cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage = std::move(basisGradientQuadStorage);
          d_cellStartIdsBasisGradientQuadStorage =
            cellStartIdsBasisGradientQuadStorage;
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
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer->getJxW();
          jxwQuadStorage =
            std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                       memorySpace>::Storage>(
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
        const unsigned int                  maxRecursion,
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
        }
      else
        utils::throwException<utils::InvalidArgument>(
          false, "Incorrect arguments given for this Quadrature family.");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientQuadStorage;
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
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      EFEBasisDataStorageDealiiInternal::storeValuesHRefinedAdaptiveQuad<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisQuadStorage,
             basisGradientQuadStorage,
             basisHessianQuadStorage,
             basisOverlap,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisQuadStorage,
             cellStartIdsBasisGradientQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisQuadStorage             = std::move(basisQuadStorage);
          d_cellStartIdsBasisQuadStorage = cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage = std::move(basisGradientQuadStorage);
          d_cellStartIdsBasisGradientQuadStorage =
            cellStartIdsBasisGradientQuadStorage;
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
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer->getJxW();
          jxwQuadStorage =
            std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                       memorySpace>::Storage>(
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
      return *(d_basisQuadStorage);
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
      return *(d_basisGradientQuadStorage);
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
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
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
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                                    basisQuadStorage = d_basisQuadStorage;
      const std::vector<size_type> &cellStartIds =
        d_cellStartIdsBasisQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisQuadStorage->data() + cellStartIds[cellId]);
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
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreValues)
          ->second,
        "Basis values are not evaluated for the given QuadratureRuleAttributes");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                                    basisQuadStorage = d_basisQuadStorage;
      const std::vector<size_type> &cellStartIds =
        d_cellStartIdsBasisQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      size_type                     sizeToCopy        = 0;
      for (size_type cellId = cellRange.first; cellId < cellRange.second;
           cellId++)
        sizeToCopy += nQuadPointsInCell[cellId] * d_dofsInCell[cellId] * dim;
      for (size_type cellId = cellRange.first; cellId < cellRange.second;
           cellId++)
        utils::MemoryTransfer<memorySpace, memorySpace>::copy(
          nQuadPointsInCell[cellId] * d_dofsInCell[cellId],
          basisData.data() + cellStartIds[cellId] -
            cellStartIds[cellRange.first],
          basisQuadStorage->data() + cellStartIds[cellId]);
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
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientQuadStorage = d_basisGradientQuadStorage;
      const std::vector<size_type> &cellStartIds =
        d_cellStartIdsBasisGradientQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId] * dim;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisGradientQuadStorage->data() + cellStartIds[cellId]);
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
      utils::throwException(
        d_evaluateBasisData,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradient)
          ->second,
        "Basis gradient values are not evaluated for the given QuadratureRuleAttributes");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientQuadStorage = d_basisGradientQuadStorage;
      const std::vector<size_type> &cellStartIds =
        d_cellStartIdsBasisGradientQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      size_type                     sizeToCopy        = 0;
      for (size_type cellId = cellRange.first; cellId < cellRange.second;
           cellId++)
        sizeToCopy += nQuadPointsInCell[cellId] * d_dofsInCell[cellId] * dim;
      for (size_type cellId = cellRange.first; cellId < cellRange.second;
           cellId++)
        utils::MemoryTransfer<memorySpace, memorySpace>::copy(
          nQuadPointsInCell[cellId] * d_dofsInCell[cellId] * dim,
          basisGradientData.data() + cellStartIds[cellId] -
            cellStartIds[cellRange.first],
          basisGradientQuadStorage->data() + cellStartIds[cellId]);
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
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisHessianQuadStorage->data() + cellStartIds[cellId]);
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
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        jxwQuadStorage = d_JxWStorage;

      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      const size_type               sizeToCopy = nQuadPointsInCell[cellId];
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        jxwQuadStorage->data() +
          d_quadratureRuleContainer->getCellQuadStartId(cellId));
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
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                                    basisQuadStorage = d_basisQuadStorage;
      const std::vector<size_type> &cellStartIds =
        d_cellStartIdsBasisQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(1);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        1,
        returnValue.data(),
        basisQuadStorage->data() + cellStartIds[cellId] +
          quadPointId * d_dofsInCell[cellId] + basisId);
      // basisId * nQuadPointsInCell[cellId] + quadPointId);
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
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientQuadStorage = d_basisGradientQuadStorage;
      const std::vector<size_type> &cellStartIds =
        d_cellStartIdsBasisGradientQuadStorage;
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(dim);
      for (size_type iDim = 0; iDim < dim; ++iDim)
        {
          utils::MemoryTransfer<memorySpace, memorySpace>::copy(
            1,
            returnValue.data() + iDim,
            basisGradientQuadStorage->data() + cellStartIds[cellId] +
              quadPointId * d_dofsInCell[cellId] * dim +
              iDim * d_dofsInCell[cellId] + basisId);
          // iDim * d_dofsInCell[cellId] * nQuadPointsInCell[cellId] +
          // basisId * nQuadPointsInCell[cellId] + quadPointId);
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
              utils::MemoryTransfer<memorySpace, memorySpace>::copy(
                1,
                returnValue.data() + iDim * dim + jDim,
                basisHessianQuadStorage->data() + cellStartIds[cellId] +
                  quadPointId * d_dofsInCell[cellId] * dim * dim +
                  (iDim * dim + jDim) * d_dofsInCell[cellId] + basisId);
              // (iDim * dim + jDim) * d_dofsInCell[cellId] *
              //   nQuadPointsInCell[cellId] +
              // basisId * nQuadPointsInCell[cellId] + quadPointId);
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
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId]);
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
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId] +
          basisId1 * d_dofsInCell[cellId] + basisId2);
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
      utils::throwException(
        (d_basisQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBasisDataStorageDealii. Not safe to delete it.");
      delete (d_basisQuadStorage).get();

      utils::throwException(
        (d_basisGradientQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBasisDataStorageDealii. Not safe to delete it.");
      delete (d_basisGradientQuadStorage).get();

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
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisGradNiGradNj->data() + d_cellStartIdsGradNiGradNj[cellId]);
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
