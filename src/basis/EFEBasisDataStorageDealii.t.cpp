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
#include <climits>
#include <utils/Exceptions.h>
#include <utils/MathFunctions.h>
#include "DealiiConversions.h"
#include <basis/TriangulationCellDealii.h>
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

namespace dftefe
{
  namespace basis
  {
    namespace EFEBasisDataStorageDealiiInternal
    {
      // This class stores the enriched FE basis data for a h-refined
      // FE mesh and uniform or non-uniform quadrature Gauss/variable/Adaptive
      // rule across all cells in the mesh.

      template <typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeValuesHRefinedSameQuadEveryCell(
        std::shared_ptr<const EFEBasisManagerDealii<dim>> efeBM,
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
          dealii::update_values | dealii::update_JxW_values;
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
        dealii::FEValues<dim> dealiiFEValues(efeBM->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags);
        const size_type numLocallyOwnedCells = efeBM->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        size_type       dofsPerCell = efeBM->nCellDofs(cellId);
        const size_type numQuadPointsPerCell =
          utils::mathFunctions::sizeTypePow(num1DQuadPoints, dim);

        nQuadPointsInCell.resize(numLocallyOwnedCells, numQuadPointsPerCell);
        std::vector<ValueTypeBasisData> basisQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisGradientQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisHessianQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisOverlapTmp(0);

        size_type cellIndex        = 0;
        size_type basisValuesSize  = 0;
        size_type basisOverlapSize = 0;

        auto locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBM->nCellDofs(cellIndex);
            basisValuesSize += numQuadPointsPerCell * dofsPerCell;
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

        basisOverlap = std::make_shared<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>(
          basisOverlapSize);
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeBasisData(0));

        locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii =
          std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
        utils::throwException(
          feCellDealii != nullptr,
          "Dynamic casting of FECellBase to FECellDealii not successful");

        auto basisQuadStorageTmpIter = basisQuadStorageTmp.begin();
        auto basisGradientQuadStorageTmpIter =
          basisGradientQuadStorageTmp.begin();
        auto basisHessianQuadStorageTmpIter =
          basisHessianQuadStorageTmp.begin();
        auto basisOverlapTmpIter = basisOverlapTmp.begin();

        cellIndex                      = 0;
        size_type cumulativeQuadPoints = 0;

        if(efeBM->isOrthgonalized())
        {

            BasisStorageAttributesBoolMap basisAttrMap;
            basisAttrMap[BasisStorageAttributes::StoreValues] = true;
            basisAttrMap[BasisStorageAttributes::StoreGradient] = true;
            basisAttrMap[BasisStorageAttributes::StoreHessian] = false;
            basisAttrMap[BasisStorageAttributes::StoreOverlap] = false;
            basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] = false;
            basisAttrMap[BasisStorageAttributes::StoreJxW] = false;
            basisAttrMap[BasisStorageAttributes::StoreQuadRealPoints] = false;

            // Set up the FE Basis Data Storage
            std::shared_ptr<BasisDataStorage<ValueTypeBasisData, memorySpace>> cfeBasisDataStorage =
              std::make_shared<FEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>>
              (efeBM->getEnrichmentClassicalInterface()->getCFEBasisHandler()->getBasisManager(), quadratureRuleAttributes, basisAttrMap);

            cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes, basisAttrMap);

            FEBasisOperations<ValueTypeBasisData, ValueTypeBasisData, memorySpace, dim> cfeBasisOp(cfeBasisDataStorage, L2ProjectionDefaults::MAX_CELL_TIMES_NUMVECS);

            dftefe::quadrature::QuadratureValuesContainer<ValueTypeBasisData, memorySpace> 
              basisClassicalInterfaceQuadValues(*quadratureRuleContainer, efeBM->totalRanges()-1);

            dftefe::quadrature::QuadratureValuesContainer<ValueTypeBasisData, memorySpace> 
              basisClassicalInterfaceQuadGradients(*quadratureRuleContainer, (efeBM->totalRanges()-1)*dim);
            
            cfeBasisOp.interpolate( efeBM->getEnrichmentClassicalInterface()->getBasisInterfaceCoeff(), 
                                   efeBM->getEnrichmentClassicalInterface()->getBasisInterfaceCoeffConstraint(),
                                   *efeBM->getEnrichmentClassicalInterface()->getCFEBasisHandler(),
                                   quadratureRuleAttributes,  
                                   basisClassicalInterfaceQuadValues);

            cfeBasisOp.interpolateWithBasisGradient( efeBM->getEnrichmentClassicalInterface()->getBasisInterfaceCoeff(), 
                                   efeBM->getEnrichmentClassicalInterface()->getBasisInterfaceCoeffConstraint(),
                                   *efeBM->getEnrichmentClassicalInterface()->getCFEBasisHandler(),
                                   quadratureRuleAttributes, 
                                   basisClassicalInterfaceQuadGradients);

        }

        for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBM->nCellDofs(cellIndex);
            // Get classical dof numbers
            size_type classicalDofsPerCell = utils::mathFunctions::sizeTypePow(
              (efeBM->getFEOrder(cellIndex) + 1), dim);

            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            std::vector<dftefe::utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                cellStartIdsBasisQuadStorage[cellIndex] =
                  cumulativeQuadPoints * dofsPerCell;
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            *basisQuadStorageTmpIter =
                              dealiiFEValues.shape_value(iNode, qPoint);
                            basisQuadStorageTmpIter++;
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {

                            ValueTypeBasisData classicalComponent = 0;
                            if(efeBM->isOrthgonalized())
                            {
                              for(unsigned int i = 0 ; i < basisClassicalInterfaceQuadValues.getNumberComponents() ; i++)
                              {
                                basisClassicalInterfaceQuadValues.getCellQuadValues<dftefe::utils::MemorySpace::HOST>(cellIndex, qPoint, &classicalComponent);
                              }
                            }

                            *basisQuadStorageTmpIter =
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint])-classicalComponent;

                            // std::cout << quadRealPointsVec[qPoint][0] << " "
                            // << quadRealPointsVec[qPoint][1] << " " <<
                            // quadRealPointsVec[qPoint][2] << " " <<
                            // *basisQuadStorageTmpIter << "\n";

                            basisQuadStorageTmpIter++;
                          }
                      }
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
                             qPoint < numQuadPointsPerCell;
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
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]) *
                              dealiiFEValues.shape_value(jNode, qPoint) *
                              dealiiFEValues.JxW(qPoint);
                            // enriched i * classical j
                          }
                      }
                    else if (iNode < classicalDofsPerCell &&
                             jNode >= classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]) *
                              dealiiFEValues.shape_value(iNode, qPoint) *
                              dealiiFEValues.JxW(qPoint);
                            // enriched j * classical i
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]) *
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]) *
                              dealiiFEValues.JxW(qPoint);
                            // enriched i * enriched j
                          }
                      }
                    basisOverlapTmpIter++;
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                cellStartIdsBasisGradientQuadStorage[cellIndex] =
                  cumulativeQuadPoints * dim * dofsPerCell;
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            auto shapeGrad =
                              dealiiFEValues.shape_grad(iNode, qPoint);
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientQuadStorageTmp.begin() +
                                  cumulativeQuadPoints * dim * dofsPerCell +
                                  iDim * dofsPerCell * numQuadPointsPerCell +
                                  iNode * numQuadPointsPerCell + qPoint;
                                *it = shapeGrad[iDim];
                              }
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            auto shapeGrad = efeBM->getEnrichmentDerivative(
                              cellIndex,
                              iNode - classicalDofsPerCell,
                              quadRealPointsVec[qPoint]);
                            // enriched gradient function call
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientQuadStorageTmp.begin() +
                                  cumulativeQuadPoints * dim * dofsPerCell +
                                  iDim * dofsPerCell * numQuadPointsPerCell +
                                  iNode * numQuadPointsPerCell + qPoint;
                                *it = shapeGrad[iDim];
                              }
                          }
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreHessian)
                  ->second)
              {
                cellStartIdsBasisHessianQuadStorage[cellIndex] =
                  cumulativeQuadPoints * dim * dim * dofsPerCell;
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
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
                                      cumulativeQuadPoints * dim * dim *
                                        dofsPerCell +
                                      iDim * dim * dofsPerCell *
                                        numQuadPointsPerCell +
                                      jDim * dofsPerCell *
                                        numQuadPointsPerCell +
                                      iNode * numQuadPointsPerCell + qPoint;
                                    *it = shapeHessian[iDim][jDim];
                                  }
                              }
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            auto shapeHessian = efeBM->getEnrichmentHessian(
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
                                      cumulativeQuadPoints * dim * dim *
                                        dofsPerCell +
                                      iDim * dim * dofsPerCell *
                                        numQuadPointsPerCell +
                                      jDim * dofsPerCell *
                                        numQuadPointsPerCell +
                                      iNode * numQuadPointsPerCell + qPoint;
                                    *it = shapeHessian[iDim * dim + jDim];
                                  }
                              }
                          }
                      }
                  }
              }
            cellIndex++;
            cumulativeQuadPoints += numQuadPointsPerCell;
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

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(), basisOverlap->data(), basisOverlapTmp.data());
      }

      template <typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeGradNiNjHRefinedSameQuadEveryCell(
        std::shared_ptr<const EFEBasisManagerDealii<dim>> efeBM,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &                                         basisGradNiGradNj,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadratureRuleContainer)

      {
        const quadrature::QuadratureFamily quadratureFamily =
          quadratureRuleAttributes.getQuadratureFamily();
        const size_type num1DQuadPoints =
          quadratureRuleAttributes.getNum1DPoints();
        const size_type numQuadPointsPerCell =
          utils::mathFunctions::sizeTypePow(num1DQuadPoints, dim);
        dealii::Quadrature<dim> dealiiQuadratureRule;
        if (quadratureFamily == quadrature::QuadratureFamily::GAUSS)
          {
            dealiiQuadratureRule = dealii::QGauss<dim>(num1DQuadPoints);
          }
        else if (quadratureFamily == quadrature::QuadratureFamily::GLL)
          {
            dealiiQuadratureRule = dealii::QGaussLobatto<dim>(num1DQuadPoints);
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
        dealii::FEValues<dim> dealiiFEValues(efeBM->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags);
        const size_type numLocallyOwnedCells = efeBM->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        std::vector<ValueTypeBasisData> basisGradNiGradNjTmp(0);

        size_type dofsPerCell        = 0;
        size_type cellIndex          = 0;
        size_type basisStiffnessSize = 0;

        auto locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBM->nCellDofs(cellIndex);
            basisStiffnessSize += dofsPerCell * dofsPerCell;
            cellIndex++;
          }


        basisGradNiGradNj = std::make_shared<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>(
          basisStiffnessSize);
        basisGradNiGradNjTmp.resize(basisStiffnessSize, ValueTypeBasisData(0));
        locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii =
          std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
        utils::throwException(
          feCellDealii != nullptr,
          "Dynamic casting of FECellBase to FECellDealii not successful");
        auto basisGradNiGradNjTmpIter  = basisGradNiGradNjTmp.begin();
        cellIndex                      = 0;
        size_type cumulativeQuadPoints = 0;
        for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBM->nCellDofs(cellIndex);
            // Get classical dof numbers
            size_type classicalDofsPerCell = utils::mathFunctions::sizeTypePow(
              (efeBM->getFEOrder(cellIndex) + 1), dim);

            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            std::vector<dftefe::utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisGradNiGradNjTmpIter = 0.0;
                    if (iNode < classicalDofsPerCell &&
                        jNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
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
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            auto enrichmentDerivative =
                              efeBM->getEnrichmentDerivative(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto classicalDerivative =
                              dealiiFEValues.shape_grad(jNode, qPoint);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd = dotProd + enrichmentDerivative[k] *
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
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            auto enrichmentDerivative =
                              efeBM->getEnrichmentDerivative(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto classicalDerivative =
                              dealiiFEValues.shape_grad(iNode, qPoint);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd = dotProd + enrichmentDerivative[k] *
                                                      classicalDerivative[k];
                              }
                            *basisGradNiGradNjTmpIter +=
                              dotProd * dealiiFEValues.JxW(qPoint);
                            // enriched j * classical i
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < numQuadPointsPerCell;
                             qPoint++)
                          {
                            auto enrichmentDerivativei =
                              efeBM->getEnrichmentDerivative(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto enrichmentDerivativej =
                              efeBM->getEnrichmentDerivative(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd = dotProd + enrichmentDerivativei[k] *
                                                      enrichmentDerivativej[k];
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
            cumulativeQuadPoints += numQuadPointsPerCell;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisGradNiGradNjTmp.size(),
          basisGradNiGradNj->data(),
          basisGradNiGradNjTmp.data());
      }

      template <typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeValuesHRefinedAdaptiveQuad(
        std::shared_ptr<const EFEBasisManagerDealii<dim>> efeBM,
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
              "For storing of basis values for enriched finite element basis "
              "on a variable quadrature rule across cells, the underlying "
              "quadrature family has to be quadrature::QuadratureFamily::GAUSS_VARIABLE "
              "or quadrature::QuadratureFamily::GLL_VARIABLE or quadrature::QuadratureFamily::ADAPTIVE");
          }

        // init updateflags of dealii data structure storage
        // The same things will be stored in enriched and classical dofs.
        dealii::UpdateFlags dealiiUpdateFlags =
          dealii::update_values | dealii::update_JxW_values;

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
        const size_type feOrder              = efeBM->getFEOrder(0);
        const size_type numLocallyOwnedCells = efeBM->nLocallyOwnedCells();
        size_type       dofsPerCell          = 0;

        // Create temporary data structure for Value Storage
        std::vector<ValueTypeBasisData> basisQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisGradientQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisHessianQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisOverlapTmp(0);

        // Find total quadpoints in the processor
        nQuadPointsInCell.resize(numLocallyOwnedCells, 0);
        const size_type nTotalQuadPoints =
          quadratureRuleContainer->nQuadraturePoints();

        size_type cellIndex        = 0;
        size_type basisValuesSize  = 0;
        size_type basisOverlapSize = 0;
        size_type nQuadPointInCell = 0;

        auto locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBM->nCellDofs(cellIndex);
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
        basisOverlap = std::make_shared<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>(
          basisOverlapSize);
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeBasisData(0));

        auto basisQuadStorageTmpIter = basisQuadStorageTmp.begin();
        auto basisGradientQuadStorageTmpIter =
          basisGradientQuadStorageTmp.begin();
        auto basisHessianQuadStorageTmpIter =
          basisHessianQuadStorageTmp.begin();
        auto basisOverlapTmpIter = basisOverlapTmp.begin();

        // Init cell iters and storage iters
        locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii =
          std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
        utils::throwException(
          feCellDealii != nullptr,
          "Dynamic casting of FECellBase to FECellDealii not successful");

        cellIndex = 0;

        // get the dealii FiniteElement object
        std::shared_ptr<const dealii::DoFHandler<dim>> dealiiDofHandler =
          efeBM->getDoFHandler();

        size_type cumulativeQuadPoints = 0;
        for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBM->nCellDofs(cellIndex);
            // Get classical dof numbers
            size_type classicalDofsPerCell = utils::mathFunctions::sizeTypePow(
              (efeBM->getFEOrder(cellIndex) + 1), dim);

            nQuadPointInCell = nQuadPointsInCell[cellIndex];

            // get the parametric points and jxw in each cell according to
            // the attribute.
            const std::vector<dftefe::utils::Point> &cellParametricQuadPoints =
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
            dealii::FEValues<dim> dealiiFEValues(efeBM->getReferenceFE(
                                                   cellIndex),
                                                 dealiiQuadratureRule,
                                                 dealiiUpdateFlags);
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            std::vector<dftefe::utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

            // Store the basis values.
            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                cellStartIdsBasisQuadStorage[cellIndex] =
                  cumulativeQuadPoints * dofsPerCell;
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *basisQuadStorageTmpIter =
                              dealiiFEValues.shape_value(iNode, qPoint);
                            basisQuadStorageTmpIter++;
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *basisQuadStorageTmpIter =
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            basisQuadStorageTmpIter++;
                          }
                      }
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
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
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]) *
                              dealiiFEValues.shape_value(jNode, qPoint) *
                              cellJxWValues[qPoint];
                            // enriched i * classical j
                          }
                      }
                    else if (iNode < classicalDofsPerCell &&
                             jNode >= classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]) *
                              dealiiFEValues.shape_value(iNode, qPoint) *
                              cellJxWValues[qPoint];
                            // enriched j * classical i
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]) *
                              efeBM->getEnrichmentValue(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]) *
                              cellJxWValues[qPoint];
                            // enriched i * enriched j
                          }
                      }
                    basisOverlapTmpIter++;
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                cellStartIdsBasisGradientQuadStorage[cellIndex] =
                  cumulativeQuadPoints * dim * dofsPerCell;
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    if (iNode < classicalDofsPerCell)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto shapeGrad =
                              dealiiFEValues.shape_grad(iNode, qPoint);
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientQuadStorageTmp.begin() +
                                  cumulativeQuadPoints * dim * dofsPerCell +
                                  iDim * dofsPerCell * nQuadPointInCell +
                                  iNode * nQuadPointInCell + qPoint;
                                *it = shapeGrad[iDim];
                              }
                          }
                      }
                    else
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto shapeGrad = efeBM->getEnrichmentDerivative(
                              cellIndex,
                              iNode - classicalDofsPerCell,
                              quadRealPointsVec[qPoint]);
                            // enriched gradient function call
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientQuadStorageTmp.begin() +
                                  cumulativeQuadPoints * dim * dofsPerCell +
                                  iDim * dofsPerCell * nQuadPointInCell +
                                  iNode * nQuadPointInCell + qPoint;
                                *it = shapeGrad[iDim];
                              }
                          }
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreHessian)
                  ->second)
              {
                cellStartIdsBasisHessianQuadStorage[cellIndex] =
                  cumulativeQuadPoints * dim * dim * dofsPerCell;
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
                                      cumulativeQuadPoints * dim * dim *
                                        dofsPerCell +
                                      iDim * dim * dofsPerCell *
                                        nQuadPointInCell +
                                      jDim * dofsPerCell * nQuadPointInCell +
                                      iNode * nQuadPointInCell + qPoint;
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
                            auto shapeHessian = efeBM->getEnrichmentHessian(
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
                                      cumulativeQuadPoints * dim * dim *
                                        dofsPerCell +
                                      iDim * dim * dofsPerCell *
                                        nQuadPointInCell +
                                      jDim * dofsPerCell * nQuadPointInCell +
                                      iNode * nQuadPointInCell + qPoint;
                                    *it = shapeHessian[iDim * dim + jDim];
                                  }
                              }
                          }
                      }
                  }
              }

            cellIndex++;
            cumulativeQuadPoints += nQuadPointInCell;
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

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(), basisOverlap->data(), basisOverlapTmp.data());
      }

      template <typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeGradNiGradNjHRefinedAdaptiveQuad(
        std::shared_ptr<const EFEBasisManagerDealii<dim>> efeBM,
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
              "For storing of basis values for classical finite element basis "
              "on a variable quadrature rule across cells, the underlying "
              "quadrature family has to be quadrature::QuadratureFamily::GAUSS_VARIABLE "
              "or quadrature::QuadratureFamily::GLL_VARIABLE or quadrature::QuadratureFamily::ADAPTIVE");
          }


        dealii::UpdateFlags dealiiUpdateFlags =
          dealii::update_gradients | dealii::update_JxW_values;

        // NOTE: cellId 0 passed as we assume h-refined finite element mesh in
        // this function
        const size_type feOrder              = efeBM->getFEOrder(0);
        const size_type numLocallyOwnedCells = efeBM->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function

        std::vector<ValueTypeBasisData> basisGradNiGradNjTmp(0);

        const size_type nTotalQuadPoints =
          quadratureRuleContainer->nQuadraturePoints();

        size_type dofsPerCell        = 0;
        size_type cellIndex          = 0;
        size_type basisStiffnessSize = 0;

        auto locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBM->nCellDofs(cellIndex);
            basisStiffnessSize += dofsPerCell * dofsPerCell;
            cellIndex++;
          }

        basisGradNiGradNj = std::make_shared<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>(
          basisStiffnessSize);
        basisGradNiGradNjTmp.resize(basisStiffnessSize, ValueTypeBasisData(0));

        locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii =
          std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
        utils::throwException(
          feCellDealii != nullptr,
          "Dynamic casting of FECellBase to FECellDealii not successful");

        auto basisGradNiGradNjTmpIter = basisGradNiGradNjTmp.begin();
        cellIndex                     = 0;

        // get the dealii FiniteElement object
        std::shared_ptr<const dealii::DoFHandler<dim>> dealiiDofHandler =
          efeBM->getDoFHandler();

        size_type cumulativeQuadPoints = 0;
        for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBM->nCellDofs(cellIndex);
            // Get classical dof numbers
            size_type classicalDofsPerCell = utils::mathFunctions::sizeTypePow(
              (efeBM->getFEOrder(cellIndex) + 1), dim);

            size_type nQuadPointInCell =
              quadratureRuleContainer->nCellQuadraturePoints(cellIndex);
            const std::vector<dftefe::utils::Point> &cellParametricQuadPoints =
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
            dealii::FEValues<dim> dealiiFEValues(efeBM->getReferenceFE(
                                                   cellIndex),
                                                 dealiiQuadratureRule,
                                                 dealiiUpdateFlags);
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            std::vector<dftefe::utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

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
                              efeBM->getEnrichmentDerivative(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto classicalDerivative =
                              dealiiFEValues.shape_grad(jNode, qPoint);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd = dotProd + enrichmentDerivative[k] *
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
                              efeBM->getEnrichmentDerivative(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto classicalDerivative =
                              dealiiFEValues.shape_grad(iNode, qPoint);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd = dotProd + enrichmentDerivative[k] *
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
                              efeBM->getEnrichmentDerivative(
                                cellIndex,
                                iNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            auto enrichmentDerivativej =
                              efeBM->getEnrichmentDerivative(
                                cellIndex,
                                jNode - classicalDofsPerCell,
                                quadRealPointsVec[qPoint]);
                            ValueTypeBasisData dotProd =
                              (ValueTypeBasisData)0.0;
                            for (unsigned int k = 0; k < dim; k++)
                              {
                                dotProd = dotProd + enrichmentDerivativei[k] *
                                                      enrichmentDerivativej[k];
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
            cumulativeQuadPoints += nQuadPointInCell;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisGradNiGradNjTmp.size(),
          basisGradNiGradNj->data(),
          basisGradNiGradNjTmp.data());
      }
    } // namespace EFEBasisDataStorageDealiiInternal


    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      EFEBasisDataStorageDealii(
        std::shared_ptr<const BasisManager>         efeBM,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
      : d_dofsInCell(0)
      , d_cellStartIdsBasisOverlap(0)
      , d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_basisStorageAttributesBoolMap(basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = false;
      d_efeBM =
        std::dynamic_pointer_cast<const EFEBasisManagerDealii<dim>>(efeBM);
      utils::throwException(
        d_efeBM != nullptr,
        " Could not cast the EFEBasisManager to EFEBasisManagerDealii in EFEBasisDataStorageDealii");

      std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
        d_efeBM->getDoFHandler();
      const size_type numLocallyOwnedCells = d_efeBM->nLocallyOwnedCells();
      d_dofsInCell.resize(numLocallyOwnedCells, 0);
      d_cellStartIdsBasisOverlap.resize(numLocallyOwnedCells, 0);
      d_cellStartIdsGradNiGradNj.resize(numLocallyOwnedCells, 0);
      size_type cumulativeBasisOverlapId = 0;
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        {
          d_dofsInCell[iCell]               = d_efeBM->nCellDofs(iCell);
          d_cellStartIdsBasisOverlap[iCell] = cumulativeBasisOverlapId;

          // Storing this is redundant but can help in readability
          d_cellStartIdsGradNiGradNj[iCell] = d_cellStartIdsBasisOverlap[iCell];

          cumulativeBasisOverlapId +=
            d_efeBM->nCellDofs(iCell) * d_efeBM->nCellDofs(iCell);
        }
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = true;
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
              d_efeBM->getTriangulation(),
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
              d_efeBM->getTriangulation(),
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

      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);
      EFEBasisDataStorageDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBM,
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
          d_basisQuadStorage             = basisQuadStorage;
          d_cellStartIdsBasisQuadStorage = cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage = basisGradientQuadStorage;
          d_cellStartIdsBasisGradientQuadStorage =
            cellStartIdsBasisGradientQuadStorage;
        }
      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = basisHessianQuadStorage;
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      d_basisOverlap      = basisOverlap;
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            basisGradNiNj;
          EFEBasisDataStorageDealiiInternal::
            storeGradNiNjHRefinedSameQuadEveryCell<ValueTypeBasisData,
                                                   memorySpace,
                                                   dim>(
              d_efeBM,
              basisGradNiNj,
              quadratureRuleAttributes,
              d_quadratureRuleContainer);

          d_basisGradNiGradNj = basisGradNiNj;
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

          d_JxWStorage = jxwQuadStorage;
        }
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                            quadratureRuleContainer,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = true;
      /**
       * @note We assume a linear mapping from the reference cell
       * to the real cell.
       */
      LinearCellMappingDealii<dim> linearCellMappingDealii;

      quadrature::QuadratureFamily quadFamily =
        quadratureRuleAttributes.getQuadratureFamily();

      if (quadFamily == quadrature::QuadratureFamily::GAUSS_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::GLL_VARIABLE ||
          quadFamily == quadrature::QuadratureFamily::ADAPTIVE)
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
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      EFEBasisDataStorageDealiiInternal::
        storeValuesHRefinedAdaptiveQuad<ValueTypeBasisData, memorySpace, dim>(
          d_efeBM,
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
          d_basisQuadStorage             = basisQuadStorage;
          d_cellStartIdsBasisQuadStorage = cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage = basisGradientQuadStorage;
          d_cellStartIdsBasisGradientQuadStorage =
            cellStartIdsBasisGradientQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = basisHessianQuadStorage;
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      d_basisOverlap      = basisOverlap;
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            basisGradNiGradNj;

          EFEBasisDataStorageDealiiInternal::
            storeGradNiGradNjHRefinedAdaptiveQuad<ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>(
              d_efeBM,
              basisGradNiGradNj,
              quadratureRuleAttributes,
              d_quadratureRuleContainer);
          d_basisGradNiGradNj = basisGradNiGradNj;
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

          d_JxWStorage = jxwQuadStorage;
        }
    }


    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::vector<std::shared_ptr<const quadrature::QuadratureRule>>
                                            quadratureRuleVec,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = true;
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
            d_efeBM->getTriangulation(),
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
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      EFEBasisDataStorageDealiiInternal::
        storeValuesHRefinedAdaptiveQuad<ValueTypeBasisData, memorySpace, dim>(
          d_efeBM,
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
          d_basisQuadStorage             = basisQuadStorage;
          d_cellStartIdsBasisQuadStorage = cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage = basisGradientQuadStorage;
          d_cellStartIdsBasisGradientQuadStorage =
            cellStartIdsBasisGradientQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = basisHessianQuadStorage;
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      d_basisOverlap      = basisOverlap;
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            basisGradNiGradNj;

          EFEBasisDataStorageDealiiInternal::
            storeGradNiGradNjHRefinedAdaptiveQuad<ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>(
              d_efeBM,
              basisGradNiGradNj,
              quadratureRuleAttributes,
              d_quadratureRuleContainer);
          d_basisGradNiGradNj = basisGradNiGradNj;
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

          d_JxWStorage = jxwQuadStorage;
        }
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRule>
          baseQuadratureRuleAdaptive,
        std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
          &                                 functions,
        const std::vector<double> &         tolerances,
        const std::vector<double> &         integralThresholds,
        const double                        smallestCellVolume,
        const unsigned int                  maxRecursion,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      d_evaluateBasisData = true;
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
              d_efeBM->getTriangulation(),
              linearCellMappingDealii,
              parentToChildCellsManagerDealii,
              functions,
              tolerances,
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
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      EFEBasisDataStorageDealiiInternal::
        storeValuesHRefinedAdaptiveQuad<ValueTypeBasisData, memorySpace, dim>(
          d_efeBM,
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
          d_basisQuadStorage             = basisQuadStorage;
          d_cellStartIdsBasisQuadStorage = cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage = basisGradientQuadStorage;
          d_cellStartIdsBasisGradientQuadStorage =
            cellStartIdsBasisGradientQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = basisHessianQuadStorage;
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      d_basisOverlap      = basisOverlap;
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
            basisGradNiGradNj;

          EFEBasisDataStorageDealiiInternal::
            storeGradNiGradNjHRefinedAdaptiveQuad<ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>(
              d_efeBM,
              basisGradNiGradNj,
              quadratureRuleAttributes,
              d_quadratureRuleContainer);
          d_basisGradNiGradNj = basisGradNiGradNj;
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

          d_JxWStorage = jxwQuadStorage;
        }
    }

    //------------------OTHER FNS -----------------------------
    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisDataInAllCells(const quadrature::QuadratureRuleAttributes
                               &quadratureRuleAttributes) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreValues)
          ->second,
        "Basis values are not evaluated for the given QuadratureRuleAttributes");
      return *(d_basisQuadStorage);
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisGradientDataInAllCells(const quadrature::QuadratureRuleAttributes
                                       &quadratureRuleAttributes) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradient)
          ->second,
        "Basis Gradients are not evaluated for the given QuadratureRuleAttributes");
      return *(d_basisGradientQuadStorage);
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisHessianDataInAllCells(const quadrature::QuadratureRuleAttributes
                                      &quadratureRuleAttributes) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreHessian)
          ->second,
        "Basis Hessians are not evaluated for the given QuadratureRuleAttributes");
      return *(d_basisHessianQuadStorage);
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getJxWInAllCells(const quadrature::QuadratureRuleAttributes
                         &quadratureRuleAttributes) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      utils::throwException(
        d_basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
          ->second,
        "JxW values are not stored for the given QuadratureRuleAttributes");
      return *(d_JxWStorage);
    }


    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisDataInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
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

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisGradientDataInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
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

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisHessianDataInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
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

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getJxWInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
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

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisData(const QuadraturePointAttributes &attributes,
                   const size_type                  basisId) const
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
          basisId * nQuadPointsInCell[cellId] + quadPointId);
      return returnValue;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisGradientData(const QuadraturePointAttributes &attributes,
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
              iDim * d_dofsInCell[cellId] * nQuadPointsInCell[cellId] +
              basisId * nQuadPointsInCell[cellId] + quadPointId);
        }
      return returnValue;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisHessianData(const QuadraturePointAttributes &attributes,
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
                  (iDim * dim + jDim) * d_dofsInCell[cellId] *
                    nQuadPointsInCell[cellId] +
                  basisId * nQuadPointsInCell[cellId] + quadPointId);
            }
        }
      return returnValue;
    }


    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisOverlapInAllCells(const quadrature::QuadratureRuleAttributes
                                  &quadratureRuleAttributes) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreOverlap)
          ->second,
        "Basis overlap values are not evaluated for the given QuadratureRuleAttributes");
      return *(d_basisOverlap);
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisOverlapInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
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

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisOverlap(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId,
        const size_type                             basisId1,
        const size_type                             basisId2) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
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

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      deleteBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes)
    {
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
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

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisDataInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId,
        const size_type                             basisId) const
    {
      utils::throwException(
        false,
        "getBasisDataInCell() for a given basisId is not implemented in EFEBasisDataStorageDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisGradientDataInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId,
        const size_type                             basisId) const
    {
      utils::throwException(
        false,
        "getBasisGradientDataInCell() for a given basisId is not implemented in EFEBasisDataStorageDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisHessianDataInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId,
        const size_type                             basisId) const
    {
      utils::throwException(
        false,
        "getBasisHessianDataInCell() for a given basisId is not implemented in EFEBasisDataStorageDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const quadrature::QuadratureRuleContainer &
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getQuadratureRuleContainer(const quadrature::QuadratureRuleAttributes
                                   &quadratureRuleAttributes) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      return *(d_quadratureRuleContainer);
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisGradNiGradNjInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradNiGradNj)
          ->second,
        "Basis Grad Ni Grad Nj values are not evaluated for the given QuadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                      basisGradNiNj = d_basisGradNiGradNj;
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisGradNiNj->data() + d_cellStartIdsGradNiGradNj[cellId]);
      return returnValue;
    }

    // get overlap of all the basis functions in all cells
    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisGradNiGradNjInAllCells(const quadrature::QuadratureRuleAttributes
                                       &quadratureRuleAttributes) const
    {
      utils::throwException(
        d_evaluateBasisData == true,
        "Cannot call function before calling evaluateBasisData()");
      utils::throwException<utils::InvalidArgument>(
        d_quadratureRuleAttributes == quadratureRuleAttributes,
        "Incorrect quadratureRuleAttributes given.");
      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradNiGradNj)
          ->second,
        "Basis Grad Ni Grad Nj values are not evaluated for the given QuadratureRuleAttributes");
      return *(d_basisGradNiGradNj);
    }

    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const BasisManager &
    EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
      getBasisManager() const
    {
      return *d_efeBM;
    }
  } // namespace basis
} // namespace dftefe
