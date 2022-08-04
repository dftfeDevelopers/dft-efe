
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
 * @author Bikash Kanungo, Vishal Subramanian
 */

#include <utils/Exceptions.h>
#include <utils/MathFunctions.h>
#include "DealiiConversions.h"
#include <basis/TriangulationCellDealii.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <quadrature/QuadratureAttributes.h>
namespace dftefe
{
  namespace basis
  {
    namespace FEBasisDataStorageDealiiInternal
    {
      //
      // stores the classical FE basis data for a h-refined FE mesh
      // (i.e., uniform p in all elements) and for a uniform quadrature
      // Gauss or Gauss-Legendre-Lobatto (GLL) quadrature
      // rule across all the cells in the mesh.
      //
      template <typename ValueType,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeValuesHRefinedSameQuadEveryCell(
        std::shared_ptr<const FEBasisManagerDealii<dim>> feBM,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &basisQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &basisGradientQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &basisHessianQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &                                         basisOverlap,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::vector<size_type> &                    nQuadPointsInCell,
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
        const size_type       cellId = 0;
        dealii::FEValues<dim> dealiiFEValues(feBM->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags);
        const size_type       numLocallyOwnedCells = feBM->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        const size_type dofsPerCell = feBM->nCellDofs(cellId);
        const size_type numQuadPointsPerCell =
          utils::mathFunctions::sizeTypePow(num1DQuadPoints, dim);
        const size_type nDimxDofsPerCellxNumQuad =
          dim * dofsPerCell * numQuadPointsPerCell;
        const size_type nDimSqxDofsPerCellxNumQuad =
          dim * dim * dofsPerCell * numQuadPointsPerCell;
        const size_type DofsPerCellxNumQuad =
          dofsPerCell * numQuadPointsPerCell;

        nQuadPointsInCell.resize(numLocallyOwnedCells, numQuadPointsPerCell);
        std::vector<ValueType> basisQuadStorageTmp(0);
        std::vector<ValueType> basisGradientQuadStorageTmp(0);
        std::vector<ValueType> basisHessianQuadStorageTmp(0);
        std::vector<ValueType> basisOverlapTmp(0);

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            basisQuadStorage = std::make_shared<
              typename BasisDataStorage<ValueType, memorySpace>::Storage>(
              dofsPerCell * numQuadPointsPerCell);
            basisQuadStorageTmp.resize(dofsPerCell * numQuadPointsPerCell,
                                       ValueType(0));
            cellStartIdsBasisQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            basisGradientQuadStorage = std::make_shared<
              typename BasisDataStorage<ValueType, memorySpace>::Storage>(
              numLocallyOwnedCells * nDimxDofsPerCellxNumQuad);
            basisGradientQuadStorageTmp.resize(numLocallyOwnedCells *
                                                 nDimxDofsPerCellxNumQuad,
                                               ValueType(0));
            cellStartIdsBasisGradientQuadStorage.resize(numLocallyOwnedCells,
                                                        0);
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          {
            basisHessianQuadStorage = std::make_shared<
              typename BasisDataStorage<ValueType, memorySpace>::Storage>(
              numLocallyOwnedCells * nDimSqxDofsPerCellxNumQuad);
            basisHessianQuadStorageTmp.resize(numLocallyOwnedCells *
                                                nDimSqxDofsPerCellxNumQuad,
                                              ValueType(0));
            cellStartIdsBasisHessianQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        basisOverlap = std::make_shared<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>(
          numLocallyOwnedCells * dofsPerCell * dofsPerCell);
        basisOverlapTmp.resize(numLocallyOwnedCells * dofsPerCell * dofsPerCell,
                               ValueType(0));
        auto locallyOwnedCellIter = feBM->beginLocallyOwnedCells();
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
        auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex           = 0;
        for (; locallyOwnedCellIter != feBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());
            //
            // NOTE: For a h-refined (i.e., uniform FE order) mesh with the same
            // quadraure rule in all elements, the classical FE basis values
            // remain the same across as in the reference cell (unit
            // n-dimensional cell). Thus, to optimize on memory we only store
            // the classical FE basis values on the first cell
            //
            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second &&
                locallyOwnedCellIter == feBM->beginLocallyOwnedCells())
              {
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < numQuadPointsPerCell;
                         qPoint++)
                      {
                        *basisQuadStorageTmpIter =
                          dealiiFEValues.shape_value(iNode, qPoint);
                        basisQuadStorageTmpIter++;
                      }
                  }
              }

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisOverlapTmpIter = 0.0;
                    for (unsigned int qPoint = 0; qPoint < numQuadPointsPerCell;
                         qPoint++)
                      {
                        *basisOverlapTmpIter +=
                          dealiiFEValues.shape_value(iNode, qPoint) *
                          dealiiFEValues.shape_value(jNode, qPoint) *
                          dealiiFEValues.JxW(qPoint);
                      }
                    basisOverlapTmpIter++;
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                cellStartIdsBasisGradientQuadStorage[cellIndex] =
                  cellIndex * nDimxDofsPerCellxNumQuad;
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < numQuadPointsPerCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (unsigned int iDim = 0; iDim < dim; iDim++)
                          {
                            auto it = basisGradientQuadStorageTmp.begin() +
                                      cellIndex * nDimxDofsPerCellxNumQuad +
                                      iDim * DofsPerCellxNumQuad +
                                      iNode * numQuadPointsPerCell + qPoint;
                            *it = shapeGrad[iDim];
                          }
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreHessian)
                  ->second)
              {
                cellStartIdsBasisHessianQuadStorage[cellIndex] =
                  cellIndex * nDimSqxDofsPerCellxNumQuad;
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < numQuadPointsPerCell;
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
                                  cellIndex * nDimSqxDofsPerCellxNumQuad +
                                  iDim * nDimxDofsPerCellxNumQuad +
                                  jDim * DofsPerCellxNumQuad +
                                  iNode * numQuadPointsPerCell + qPoint;
                                *it = shapeHessian[iDim][jDim];
                              }
                          }
                      }
                  }
              }

            cellIndex++;
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


      template <typename ValueType,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeGradNiNjHRefinedSameQuadEveryCell(
        std::shared_ptr<const FEBasisManagerDealii<dim>> feBM,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &                                         basisGradNiGradNj,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes)

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
        const size_type       cellId = 0;
        dealii::FEValues<dim> dealiiFEValues(feBM->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags);
        const size_type       numLocallyOwnedCells = feBM->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        const size_type        dofsPerCell = feBM->nCellDofs(cellId);
        std::vector<ValueType> basisGradNiGradNjTmp(0);


        basisGradNiGradNj = std::make_shared<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>(
          numLocallyOwnedCells * dofsPerCell * dofsPerCell);
        basisGradNiGradNjTmp.resize(numLocallyOwnedCells * dofsPerCell *
                                      dofsPerCell,
                                    ValueType(0));
        auto locallyOwnedCellIter = feBM->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii =
          std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
        utils::throwException(
          feCellDealii != nullptr,
          "Dynamic casting of FECellBase to FECellDealii not successful");
        auto      basisGradNiGradNjTmpIter = basisGradNiGradNjTmp.begin();
        size_type cellIndex                = 0;
        for (; locallyOwnedCellIter != feBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());
            //
            // NOTE: For a h-refined (i.e., uniform FE order) mesh with the same
            // quadraure rule in all elements, the classical FE basis values
            // remain the same across as in the reference cell (unit
            // n-dimensional cell). Thus, to optimize on memory we only store
            // the classical FE basis values on the first cell
            //

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisGradNiGradNjTmpIter = 0.0;
                    for (unsigned int qPoint = 0; qPoint < numQuadPointsPerCell;
                         qPoint++)
                      {
                        *basisGradNiGradNjTmpIter +=
                          (dealiiFEValues.shape_grad(iNode, qPoint) *
                           dealiiFEValues.shape_grad(jNode, qPoint)) *
                          dealiiFEValues.JxW(qPoint);
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


      //
      // stores the classical FE basis data for a h-refined FE mesh
      // (i.e., uniform p in all elements) and for an adaptive quadrature
      //
      template <typename ValueType,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeValuesHRefinedAdaptiveQuad(
        std::shared_ptr<const FEBasisManagerDealii<dim>> feBM,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &basisQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &basisGradientQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &basisHessianQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
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
        if ((quadratureFamily !=
             quadrature::QuadratureFamily::GAUSS_VARIABLE) ||
            (quadratureFamily != quadrature::QuadratureFamily::GLL_VARIABLE) ||
            (quadratureFamily != quadrature::QuadratureFamily::ADAPTIVE))
          {
            utils::throwException(
              false,
              "For storing of basis values for classical finite element basis "
              "on a variable quadrature rule across cells, the underlying "
              "quadrature family has to be quadrature::QuadratureFamily::GAUSS_VARIABLE "
              "or quadrature::QuadratureFamily::GLL_VARIABLE or quadrature::QuadratureFamily::ADAPTIVE");
          }


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
        // NOTE: cellId 0 passed as we assume h-refined finite element mesh in
        // this function
        const size_type cellId               = 0;
        const size_type feOrder              = feBM->getFEOrder(cellId);
        const size_type numLocallyOwnedCells = feBM->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        const size_type dofsPerCell = feBM->nCellDofs(cellId);

        std::vector<ValueType> basisQuadStorageTmp(0);
        std::vector<ValueType> basisGradientQuadStorageTmp(0);
        std::vector<ValueType> basisHessianQuadStorageTmp(0);
        std::vector<ValueType> basisOverlapTmp(0);
        nQuadPointsInCell.resize(numLocallyOwnedCells, 0);

        const size_type nTotalQuadPoints =
          quadratureRuleContainer->nQuadraturePoints();
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            basisQuadStorage = std::make_shared<
              typename BasisDataStorage<ValueType, memorySpace>::Storage>(
              dofsPerCell * nTotalQuadPoints);
            basisQuadStorageTmp.resize(dofsPerCell * nTotalQuadPoints,
                                       ValueType(0));
            cellStartIdsBasisQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            basisGradientQuadStorage = std::make_shared<
              typename BasisDataStorage<ValueType, memorySpace>::Storage>(
              dofsPerCell * dim * nTotalQuadPoints);
            basisGradientQuadStorageTmp.resize(dofsPerCell * dim *
                                                 nTotalQuadPoints,
                                               ValueType(0));
            cellStartIdsBasisGradientQuadStorage.resize(numLocallyOwnedCells,
                                                        0);
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          {
            basisHessianQuadStorage = std::make_shared<
              typename BasisDataStorage<ValueType, memorySpace>::Storage>(
              dofsPerCell * dim * dim * nTotalQuadPoints);
            basisHessianQuadStorageTmp.resize(dofsPerCell * dim * dim *
                                                nTotalQuadPoints,
                                              ValueType(0));
            cellStartIdsBasisHessianQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        basisOverlap = std::make_shared<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>(
          numLocallyOwnedCells * dofsPerCell * dofsPerCell);
        basisOverlapTmp.resize(numLocallyOwnedCells * dofsPerCell * dofsPerCell,
                               ValueType(0));
        auto locallyOwnedCellIter = feBM->beginLocallyOwnedCells();
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
        auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex           = 0;

        // get the dealii FiniteElement object
        std::shared_ptr<const dealii::DoFHandler<dim>> dealiiDofHandler =
          feBM->getDoFHandler();

        size_type cumulativeQuadPoints = 0;
        for (; locallyOwnedCellIter != feBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            size_type nQuadPointInCell =
              quadratureRuleContainer->nCellQuadraturePoints(cellIndex);
            nQuadPointsInCell[cellIndex] = nQuadPointInCell;
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
            dealii::FEValues<dim> dealiiFEValues(feBM->getReferenceFE(
                                                   cellIndex),
                                                 dealiiQuadratureRule,
                                                 dealiiUpdateFlags);
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());
            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                cellStartIdsBasisQuadStorage[cellIndex] =
                  cumulativeQuadPoints * dofsPerCell;
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *basisQuadStorageTmpIter =
                          dealiiFEValues.shape_value(iNode, qPoint);
                        basisQuadStorageTmpIter++;
                      }
                  }
              }

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisOverlapTmpIter = 0.0;
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *basisOverlapTmpIter +=
                          dealiiFEValues.shape_value(iNode, qPoint) *
                          dealiiFEValues.shape_value(jNode, qPoint) *
                          cellJxWValues[qPoint];
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
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        auto shapeGrad =
                          dealiiFEValues.shape_grad(iNode, qPoint);
                        for (unsigned int iDim = 0; iDim < dim; iDim++)
                          {
                            auto it = basisGradientQuadStorageTmp.begin() +
                                      cumulativeQuadPoints * dim * dofsPerCell +
                                      iDim * dofsPerCell * nQuadPointInCell +
                                      iNode * nQuadPointInCell + qPoint;
                            *it = shapeGrad[iDim];
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
                                  iDim * dim * dofsPerCell * nQuadPointInCell +
                                  jDim * dofsPerCell * nQuadPointInCell +
                                  iNode * nQuadPointInCell + qPoint;
                                *it = shapeHessian[iDim][jDim];
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

      template <typename ValueType,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeGradNiGradNjHRefinedAdaptiveQuad(
        std::shared_ptr<const FEBasisManagerDealii<dim>> feBM,
        std::shared_ptr<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>
          &                                         basisGradNiGradNj,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
          quadratureRuleContainer)
      {
        const quadrature::QuadratureFamily quadratureFamily =
          quadratureRuleAttributes.getQuadratureFamily();
        if ((quadratureFamily !=
             quadrature::QuadratureFamily::GAUSS_VARIABLE) ||
            (quadratureFamily != quadrature::QuadratureFamily::GLL_VARIABLE) ||
            (quadratureFamily != quadrature::QuadratureFamily::ADAPTIVE))
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
        const size_type cellId               = 0;
        const size_type feOrder              = feBM->getFEOrder(cellId);
        const size_type numLocallyOwnedCells = feBM->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        const size_type dofsPerCell = feBM->nCellDofs(cellId);

        std::vector<ValueType> basisGradNiGradNjTmp(0);

        const size_type nTotalQuadPoints =
          quadratureRuleContainer->nQuadraturePoints();


        basisGradNiGradNj = std::make_shared<
          typename BasisDataStorage<ValueType, memorySpace>::Storage>(
          numLocallyOwnedCells * dofsPerCell * dofsPerCell);
        basisGradNiGradNjTmp.resize(numLocallyOwnedCells * dofsPerCell *
                                      dofsPerCell,
                                    ValueType(0));
        auto locallyOwnedCellIter = feBM->beginLocallyOwnedCells();
        std::shared_ptr<FECellDealii<dim>> feCellDealii =
          std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
        utils::throwException(
          feCellDealii != nullptr,
          "Dynamic casting of FECellBase to FECellDealii not successful");

        auto      basisGradNiGradNjTmpIter = basisGradNiGradNjTmp.begin();
        size_type cellIndex                = 0;

        // get the dealii FiniteElement object
        std::shared_ptr<const dealii::DoFHandler<dim>> dealiiDofHandler =
          feBM->getDoFHandler();

        size_type cumulativeQuadPoints = 0;
        for (; locallyOwnedCellIter != feBM->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
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
            dealii::FEValues<dim> dealiiFEValues(feBM->getReferenceFE(
                                                   cellIndex),
                                                 dealiiQuadratureRule,
                                                 dealiiUpdateFlags);
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisGradNiGradNjTmpIter = 0.0;
                    for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                         qPoint++)
                      {
                        *basisGradNiGradNjTmpIter +=
                          (dealiiFEValues.shape_grad(iNode, qPoint) *
                           dealiiFEValues.shape_grad(jNode, qPoint)) *
                          cellJxWValues[qPoint];
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
      /*
            template <typename ValueType,utils::MemorySpace memorySpace,
         size_type dim> void storeValues(std::shared_ptr<const
         FEBasisManagerDealii<dim>> feBM, std::shared_ptr<typename
         BasisDataStorage<ValueType, memorySpace>::Storage> basisQuadStorage,
                        std::shared_ptr<typename BasisDataStorage<ValueType,
         memorySpace>::Storage> basisGradientQuadStorage,
                        std::shared_ptr<typename BasisDataStorage<ValueType,
         memorySpace>::Storage> basisHessianQuadStorage,
                        std::shared_ptr<typename BasisDataStorage<ValueType,
         memorySpace>::Storage> basisOverlap, const
         quadrature::QuadratureRuleAttributes &quadratureRuleAttributes, const
         bool                      storeValues, const bool storeGradients, const
         bool                      storeHessians)
            {
              const quadrature::QuadratureFamily quadratureFamily =
                quadratureRuleAttributes.getQuadratureFamily();
              if ((quadratureFamily == quadrature::QuadratureFamily::GAUSS) ||
                  (quadratureFamily == quadrature::QuadratureFamily::GLL))
                {
                  if (feBM->isHPRefined() == false)
                    {
                      storeValuesHRefinedSameQuadEveryCell
                        <ValueType,memorySpace, dim >(feBM,
                                                           basisQuadStorage,
                                                           basisGradientQuadStorage,
                                                           basisHessianQuadStorage,
                                                           basisOverlap,
                                                           quadratureRuleAttributes,
                                                           storeValues,
                                                           storeGradients,
                                                           storeHessians);
                    }
                }
            }

            */

    } // namespace FEBasisDataStorageDealiiInternal

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      FEBasisDataStorageDealii(
        std::shared_ptr<const BasisManager> feBM,
        //        std::vector<std::shared_ptr<Constraints<ValueType,
        //        memorySpace>>>
        //          constraintsVec,
        const std::vector<quadrature::QuadratureRuleAttributes>
          &                                 quadratureRuleAttribuesVec,
        const QuadAttrToBasisStorageAttrMap quadAttrToBasisStorageAttrMap)
      : d_dofsInCell(0)
      , d_cellStartIdsBasisOverlap(0)
    {
      d_feBM = std::dynamic_pointer_cast<const FEBasisManagerDealii<dim>>(feBM);
      utils::throwException(
        d_feBM != nullptr,
        " Could not cast the FEBasisManager to FEBasisManagerDealii in FEBasisDataStorageDealii");
      //      const size_type numConstraints  = constraintsVec.size();
      const size_type numQuadRuleType = quadratureRuleAttribuesVec.size();
      std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
        d_feBM->getDoFHandler();
      const size_type numLocallyOwnedCells = d_feBM->nLocallyOwnedCells();
      d_dofsInCell.resize(numLocallyOwnedCells, 0);
      d_cellStartIdsBasisOverlap.resize(numLocallyOwnedCells, 0);
      d_cellStartIdsGradNiGradNj.resize(numLocallyOwnedCells, 0);
      size_type cumulativeBasisOverlapId = 0;
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        {
          d_dofsInCell[iCell]               = d_feBM->nCellDofs(iCell);
          d_cellStartIdsBasisOverlap[iCell] = cumulativeBasisOverlapId;

          // Storing this is redundant but can help in readability
          d_cellStartIdsGradNiGradNj[iCell] = d_cellStartIdsBasisOverlap[iCell];

          cumulativeBasisOverlapId +=
            d_feBM->nCellDofs(iCell) * d_feBM->nCellDofs(iCell);
        }

      //      std::vector<const dealii::DoFHandler<dim> *> dofHandlerVec(
      //        numConstraints, dofHandler.get());
      //      std::vector<const dealii::AffineConstraints<ValueType> *>
      //        dealiiAffineConstraintsVec(numConstraints, nullptr);
      //      for (size_type i = 0; i < numConstraints; ++i)
      //        {
      //          std::shared_ptr<
      //            const FEConstraintsDealii<ValueType, memorySpace, dim>>
      //            constraintsDealii = std::dynamic_pointer_cast<
      //              const FEConstraintsDealii<ValueType, memorySpace, dim>>(
      //              constraintsVec[i]);
      //          utils::throwException(
      //            constraintsDealii != nullptr,
      //            " Could not cast the FEConstraintsBase to
      //            FEConstraintsDealii in FEBasisDataStorageDealii");
      //          dealiiAffineConstraintsVec[i] =
      //            &(constraintsDealii->getAffineConstraints());
      //        }

      std::vector<dealii::Quadrature<dim>> dealiiQuadratureTypeVec(0);

      /**
       * @note We assume a linear mapping from the reference cell
       * to the real cell.
       */
      LinearCellMappingDealii<dim> linearCellMappingDealii;

      bool storeValueBool          = false;
      bool storeGradientBool       = false;
      bool storeHessianBool        = false;
      bool storeQuadRealPointsBool = false;
      bool storeJxWBool            = false;
      for (size_type i = 0; i < numQuadRuleType; ++i)
        {
          BasisStorageAttributesBoolMap basisStorageAttributesBoolMap =
            quadAttrToBasisStorageAttrMap.find(quadratureRuleAttribuesVec[i])
              ->second;
          storeValueBool |= basisStorageAttributesBoolMap
                              .find(BasisStorageAttributes::StoreValues)
                              ->second;
          storeGradientBool |= basisStorageAttributesBoolMap
                                 .find(BasisStorageAttributes::StoreGradient)
                                 ->second;
          storeHessianBool |= basisStorageAttributesBoolMap
                                .find(BasisStorageAttributes::StoreHessian)
                                ->second;
          storeJxWBool |= basisStorageAttributesBoolMap
                            .find(BasisStorageAttributes::StoreJxW)
                            ->second;
          storeQuadRealPointsBool |=
            basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreQuadRealPoints)
              ->second;
          size_type num1DQuadPoints =
            quadratureRuleAttribuesVec[i].getNum1DPoints();
          quadrature::QuadratureFamily quadFamily =
            quadratureRuleAttribuesVec[i].getQuadratureFamily();
          if (quadFamily == quadrature::QuadratureFamily::GAUSS)
            {
              dealiiQuadratureTypeVec.push_back(
                dealii::QGauss<dim>(num1DQuadPoints));

              std::shared_ptr<quadrature::QuadratureRuleGauss> quadratureRule =
                std::make_shared<quadrature::QuadratureRuleGauss>(
                  dim, num1DQuadPoints);
              d_quadratureRuleContainer[quadratureRuleAttribuesVec[i]] =
                std::make_shared<quadrature::QuadratureRuleContainer>(
                  quadratureRuleAttribuesVec[i],
                  quadratureRule,
                  d_feBM->getTriangulation(),
                  linearCellMappingDealii);
            }
          else if (quadFamily == quadrature::QuadratureFamily::GLL)
            {
              dealiiQuadratureTypeVec.push_back(
                dealii::QGaussLobatto<dim>(num1DQuadPoints));

              std::shared_ptr<quadrature::QuadratureRuleGLL> quadratureRule =
                std::make_shared<quadrature::QuadratureRuleGLL>(
                  dim, num1DQuadPoints);
              d_quadratureRuleContainer[quadratureRuleAttribuesVec[i]] =
                std::make_shared<quadrature::QuadratureRuleContainer>(
                  quadratureRuleAttribuesVec[i],
                  quadratureRule,
                  d_feBM->getTriangulation(),
                  linearCellMappingDealii);
            }
          else
            utils::throwException<utils::InvalidArgument>(
              false,
              "The constructor of FEBasisDataStorageDealii is implemented "
              "only for QuadratureRuleAttributes created with "
              "QuadratureFamily GAUSS and GLLL.");
        }

      //      typename dealii::MatrixFree<dim>::AdditionalData
      //      dealiiAdditionalData; dealiiAdditionalData.tasks_parallel_scheme =
      //        dealii::MatrixFree<dim>::AdditionalData::partition_partition;
      //      dealii::UpdateFlags dealiiUpdateFlags = dealii::update_default;
      //
      //      if (storeValueBool)
      //        dealiiUpdateFlags |= dealii::update_values;
      //      if (storeGradientBool)
      //        dealiiUpdateFlags |= dealii::update_gradients;
      //      if (storeHessianBool)
      //        dealiiUpdateFlags |= dealii::update_hessians;
      //      if (storeJxWBool)
      //        dealiiUpdateFlags |= dealii::update_JxW_values;
      //      if (storeQuadRealPointsBool)
      //        dealiiUpdateFlags |= dealii::update_quadrature_points;
      //
      //      dealiiAdditionalData.mapping_update_flags = dealiiUpdateFlags;

      //      d_dealiiMatrixFree =
      //        std::make_shared<dealii::MatrixFree<dim, ValueType>>();
      //      d_dealiiMatrixFree->clear();

      /**
       * @note We assume a linear mapping from the reference cell
       * to the real cell and hence we do not require to pass
       * a dealii mapping object to the  MatrixFree reinit function
       */
      //      d_dealiiMatrixFree->reinit(dofHandlerVec,
      //                                 dealiiAffineConstraintsVec,
      //                                 dealiiQuadratureTypeVec,
      //                                 dealiiAdditionalData);

      for (size_type i = 0; i < numQuadRuleType; ++i)
        {
          quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
            quadratureRuleAttribuesVec[i];
          BasisStorageAttributesBoolMap basisStorageAttributesBoolMap =
            quadAttrToBasisStorageAttrMap.find(quadratureRuleAttributes)
              ->second;
          std::shared_ptr<
            typename BasisDataStorage<ValueType, memorySpace>::Storage>
            basisQuadStorage;
          std::shared_ptr<
            typename BasisDataStorage<ValueType, memorySpace>::Storage>
            basisGradientQuadStorage;
          std::shared_ptr<
            typename BasisDataStorage<ValueType, memorySpace>::Storage>
            basisHessianQuadStorage;
          std::shared_ptr<
            typename BasisDataStorage<ValueType, memorySpace>::Storage>
            basisOverlap;


          std::vector<size_type> nQuadPointsInCell(0);
          std::vector<size_type> cellStartIdsBasisQuadStorage(0);
          std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
          std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);
          FEBasisDataStorageDealiiInternal::
            storeValuesHRefinedSameQuadEveryCell<ValueType, memorySpace, dim>(
              d_feBM,
              basisQuadStorage,
              basisGradientQuadStorage,
              basisHessianQuadStorage,
              basisOverlap,
              quadratureRuleAttributes,
              nQuadPointsInCell,
              cellStartIdsBasisQuadStorage,
              cellStartIdsBasisGradientQuadStorage,
              cellStartIdsBasisHessianQuadStorage,
              basisStorageAttributesBoolMap);

          if (basisStorageAttributesBoolMap
                .find(BasisStorageAttributes::StoreValues)
                ->second)
            {
              d_basisQuadStorage[quadratureRuleAttributes] = basisQuadStorage;
              d_cellStartIdsBasisQuadStorage[quadratureRuleAttributes] =
                cellStartIdsBasisQuadStorage;
            }

          if (basisStorageAttributesBoolMap
                .find(BasisStorageAttributes::StoreGradient)
                ->second)
            {
              d_basisGradientQuadStorage[quadratureRuleAttributes] =
                basisGradientQuadStorage;
              d_cellStartIdsBasisGradientQuadStorage[quadratureRuleAttributes] =
                cellStartIdsBasisGradientQuadStorage;
            }
          if (basisStorageAttributesBoolMap
                .find(BasisStorageAttributes::StoreHessian)
                ->second)
            {
              d_basisHessianQuadStorage[quadratureRuleAttributes] =
                basisHessianQuadStorage;
              d_cellStartIdsBasisHessianQuadStorage[quadratureRuleAttributes] =
                cellStartIdsBasisHessianQuadStorage;
            }

          d_basisOverlap[quadratureRuleAttributes]      = basisOverlap;
          d_nQuadPointsIncell[quadratureRuleAttributes] = nQuadPointsInCell;

          if (basisStorageAttributesBoolMap
                .find(BasisStorageAttributes::StoreGradNiGradNj)
                ->second)
            {
              std::shared_ptr<
                typename BasisDataStorage<ValueType, memorySpace>::Storage>
                basisGradNiNj;
              FEBasisDataStorageDealiiInternal::
                storeGradNiNjHRefinedSameQuadEveryCell<ValueType,
                                                       memorySpace,
                                                       dim>(
                  d_feBM, basisGradNiNj, quadratureRuleAttributes);

              d_basisGradNiGradNj[quadratureRuleAttributes] = basisGradNiNj;
            }

          if (basisStorageAttributesBoolMap
                .find(BasisStorageAttributes::StoreJxW)
                ->second)
            {
              std::shared_ptr<
                typename BasisDataStorage<ValueType, memorySpace>::Storage>
                jxwQuadStorage;

              const std::vector<double> &jxwVec =
                d_quadratureRuleContainer[quadratureRuleAttributes]->getJxW();
              jxwQuadStorage = std::make_shared<
                typename BasisDataStorage<ValueType, memorySpace>::Storage>(
                jxwVec.size());

              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(jxwVec.size(), jxwQuadStorage->data(), jxwVec.data());

              d_JxWStorage[quadratureRuleAttributes] = jxwQuadStorage;
            }
        }
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    void
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::evaluateBasisData(
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                                  quadratureRuleContainer,
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
      const BasisStorageAttributesBoolMap         basisStorageAttributesBoolMap)
    {
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        basisQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        basisGradientQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        basisHessianQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
                             basisOverlap;
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisGradientQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      FEBasisDataStorageDealiiInternal::
        storeValuesHRefinedAdaptiveQuad<ValueType, memorySpace, dim>(
          d_feBM,
          basisQuadStorage,
          basisGradientQuadStorage,
          basisHessianQuadStorage,
          basisOverlap,
          quadratureRuleAttributes,
          quadratureRuleContainer,
          nQuadPointsInCell,
          cellStartIdsBasisQuadStorage,
          cellStartIdsBasisGradientQuadStorage,
          cellStartIdsBasisHessianQuadStorage,
          basisStorageAttributesBoolMap);

      d_quadratureRuleContainer[quadratureRuleAttributes] =
        quadratureRuleContainer;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisQuadStorage[quadratureRuleAttributes] = basisQuadStorage;
          d_cellStartIdsBasisQuadStorage[quadratureRuleAttributes] =
            cellStartIdsBasisQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientQuadStorage[quadratureRuleAttributes] =
            basisGradientQuadStorage;
          d_cellStartIdsBasisGradientQuadStorage[quadratureRuleAttributes] =
            cellStartIdsBasisGradientQuadStorage;
        }
      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage[quadratureRuleAttributes] =
            basisHessianQuadStorage;
          d_cellStartIdsBasisHessianQuadStorage[quadratureRuleAttributes] =
            cellStartIdsBasisHessianQuadStorage;
        }

      d_basisOverlap[quadratureRuleAttributes]      = basisOverlap;
      d_nQuadPointsIncell[quadratureRuleAttributes] = nQuadPointsInCell;


      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueType, memorySpace>::Storage>
            basisGradNiGradNj;

          FEBasisDataStorageDealiiInternal::
            storeGradNiGradNjHRefinedAdaptiveQuad<ValueType, memorySpace, dim>(
              d_feBM,
              basisGradNiGradNj,
              quadratureRuleAttributes,
              quadratureRuleContainer);
          d_basisGradNiGradNj[quadratureRuleAttributes] = basisGradNiGradNj;
        }
      if (basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
            ->second)
        {
          std::shared_ptr<
            typename BasisDataStorage<ValueType, memorySpace>::Storage>
            jxwQuadStorage;

          const std::vector<double> &jxwVec =
            d_quadratureRuleContainer[quadratureRuleAttributes]->getJxW();
          jxwQuadStorage = std::make_shared<
            typename BasisDataStorage<ValueType, memorySpace>::Storage>(
            jxwVec.size());

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            jxwVec.size(), jxwQuadStorage->data(), jxwVec.data());

          d_JxWStorage[quadratureRuleAttributes] = jxwQuadStorage;
        }
    }

    //    template <typename ValueType, utils::MemorySpace memorySpace,
    //    size_type dim> std::shared_ptr<const
    //    quadrature::QuadratureRuleContainer>
    //    FEBasisDataStorageDealii<ValueType, memorySpace,
    //    dim>::getCellQuadratureRuleContainer(
    //      const QuadratureRuleAttributes &quadratureRuleAttributes) const
    //    {
    //      utils::throwException(
    //        false,
    //        "  getCellQuadratureRuleContainer is not implemented ");
    //
    //      std::shared_ptr<const quadrature::QuadratureRuleContainer> cellQuad;
    //      return ;
    //
    //    }
    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    const typename BasisDataStorage<ValueType, memorySpace>::Storage &
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisDataInAllCells(const quadrature::QuadratureRuleAttributes
                               &quadratureRuleAttributes) const
    {
      auto it = d_basisQuadStorage.find(quadratureRuleAttributes);
      utils::throwException<utils::InvalidArgument>(
        it != d_basisQuadStorage.end(),
        "Basis values are not evaluated for the given QuadratureRuleAttributes");
      return *(it->second);
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    const typename BasisDataStorage<ValueType, memorySpace>::Storage &
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisGradientDataInAllCells(const quadrature::QuadratureRuleAttributes
                                       &quadratureRuleAttributes) const
    {
      auto it = d_basisGradientQuadStorage.find(quadratureRuleAttributes);
      utils::throwException<utils::InvalidArgument>(
        it != d_basisGradientQuadStorage.end(),
        "Basis gradients are not evaluated for the given QuadratureRuleAttributes");
      return *(it->second);
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    const typename BasisDataStorage<ValueType, memorySpace>::Storage &
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisHessianDataInAllCells(const quadrature::QuadratureRuleAttributes
                                      &quadratureRuleAttributes) const
    {
      auto it = d_basisHessianQuadStorage.find(quadratureRuleAttributes);
      utils::throwException<utils::InvalidArgument>(
        it != d_basisHessianQuadStorage.end(),
        "Basis hessians are not evaluated for the given QuadratureRuleAttributes");
      return *(it->second);
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    const typename BasisDataStorage<ValueType, memorySpace>::Storage &
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getJxWInAllCells(
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes)
      const
    {
      auto it = d_JxWStorage.find(quadratureRuleAttributes);
      utils::throwException<utils::InvalidArgument>(
        it != d_JxWStorage.end(),
        "JxW values are not stored for the given QuadratureRuleAttributes");
      return *(it->second);
    }


    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getBasisDataInCell(
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
      const size_type                             cellId) const
    {
      auto itBasisQuad = d_basisQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itBasisQuad != d_basisQuadStorage.end(),
        "Basis values are not evaluated for the given QuadratureRuleAttributes");
      auto itCellStartIds =
        d_cellStartIdsBasisQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itCellStartIds != d_cellStartIdsBasisQuadStorage.end(),
        "Cell Start Ids not evaluated for the given QuadratureRuleAttributes");
      auto itNQuad = d_nQuadPointsIncell.find(quadratureRuleAttributes);
      utils::throwException(
        itNQuad != d_nQuadPointsIncell.end(),
        "Quad points in cell is not evaluated for the given QuadratureRuleAttributes.");

      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
                                    basisQuadStorage  = itBasisQuad->second;
      const std::vector<size_type> &cellStartIds      = itCellStartIds->second;
      const std::vector<size_type> &nQuadPointsInCell = itNQuad->second;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(
        sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisQuadStorage->data() + cellStartIds[cellId]);
      return returnValue;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisGradientDataInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      auto itBasisGradientQuad =
        d_basisGradientQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itBasisGradientQuad != d_basisGradientQuadStorage.end(),
        "Basis gradient values are not evaluated for the given QuadratureRuleAttributes");
      auto itCellStartIds =
        d_cellStartIdsBasisGradientQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itCellStartIds != d_cellStartIdsBasisGradientQuadStorage.end(),
        "Cell Start Ids not evaluated for the given QuadratureRuleAttributes");
      auto itNQuad = d_nQuadPointsIncell.find(quadratureRuleAttributes);
      utils::throwException(
        itNQuad != d_nQuadPointsIncell.end(),
        "Quad points in cell is not evaluated for the given QuadratureRuleAttributes.");
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        basisGradientQuadStorage                 = itBasisGradientQuad->second;
      const std::vector<size_type> &cellStartIds = itCellStartIds->second;
      const std::vector<size_type> &nQuadPointsInCell = itNQuad->second;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId] * dim;
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(
        sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisGradientQuadStorage->data() + cellStartIds[cellId]);
      return returnValue;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisHessianDataInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      auto itBasisHessianQuad =
        d_basisHessianQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itBasisHessianQuad != d_basisHessianQuadStorage.end(),
        "Basis hessians values are not evaluated for the given QuadratureRuleAttributes");
      auto itCellStartIds =
        d_cellStartIdsBasisHessianQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itCellStartIds != d_cellStartIdsBasisHessianQuadStorage.end(),
        "Cell Start Ids not evaluated for the given QuadratureRuleAttributes");
      auto itNQuad = d_nQuadPointsIncell.find(quadratureRuleAttributes);
      utils::throwException(
        itNQuad != d_nQuadPointsIncell.end(),
        "Quad points in cell is not evaluated for the given QuadratureRuleAttributes.");
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        basisHessianQuadStorage                  = itBasisHessianQuad->second;
      const std::vector<size_type> &cellStartIds = itCellStartIds->second;
      const std::vector<size_type> &nQuadPointsInCell = itNQuad->second;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId] * dim * dim;
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(
        sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisHessianQuadStorage->data() + cellStartIds[cellId]);
      return returnValue;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getJxWInCell(
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
      const size_type                             cellId) const
    {
      auto itJxWQuad = d_JxWStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itJxWQuad != d_JxWStorage.end(),
        "JxW values are not stored for the given QuadratureRuleAttributes");

      auto itNQuad = d_nQuadPointsIncell.find(quadratureRuleAttributes);
      utils::throwException(
        itNQuad != d_nQuadPointsIncell.end(),
        "Quad points in cell is not evaluated for the given QuadratureRuleAttributes.");

      auto itquadRuleContainer =
        d_quadratureRuleContainer.find(quadratureRuleAttributes);
      utils::throwException(
        itquadRuleContainer != d_quadratureRuleContainer.end(),
        "Quad rule container is not stored for the given QuadratureRuleAttributes.");

      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        jxwQuadStorage = itJxWQuad->second;

      const std::vector<size_type> &nQuadPointsInCell = itNQuad->second;
      const size_type               sizeToCopy = nQuadPointsInCell[cellId];
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(
        sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        jxwQuadStorage->data() +
          itquadRuleContainer->getCellQuadStartId(cellId));
      return returnValue;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getBasisData(
      const QuadraturePointAttributes &attributes,
      const size_type                  basisId) const
    {
      const quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        *(attributes.quadratureRuleAttributesPtr);
      const size_type cellId      = attributes.cellId;
      const size_type quadPointId = attributes.quadPointId;
      auto itBasisQuad = d_basisQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itBasisQuad != d_basisQuadStorage.end(),
        "Basis values are not evaluated for the given QuadratureRuleAttributes");
      auto itCellStartIds =
        d_cellStartIdsBasisQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itCellStartIds != d_cellStartIdsBasisQuadStorage.end(),
        "Cell Start Ids not evaluated for the given QuadratureRuleAttributes");
      auto itNQuad = d_nQuadPointsIncell.find(quadratureRuleAttributes);
      utils::throwException(
        itNQuad != d_nQuadPointsIncell.end(),
        "Quad points in cell is not evaluated for the given QuadratureRuleAttributes.");

      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
                                    basisQuadStorage  = itBasisQuad->second;
      const std::vector<size_type> &cellStartIds      = itCellStartIds->second;
      const std::vector<size_type> &nQuadPointsInCell = itNQuad->second;
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(1);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        1,
        returnValue.data(),
        basisQuadStorage->data() + cellStartIds[cellId] +
          basisId * nQuadPointsInCell[cellId] + quadPointId);
      return returnValue;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getBasisGradientData(
      const QuadraturePointAttributes &attributes,
      const size_type                  basisId) const
    {
      const quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        *(attributes.quadratureRuleAttributesPtr);
      const size_type cellId      = attributes.cellId;
      const size_type quadPointId = attributes.quadPointId;
      auto            itBasisGradientQuad =
        d_basisGradientQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itBasisGradientQuad != d_basisGradientQuadStorage.end(),
        "Basis gradient values are not evaluated for the given QuadratureRuleAttributes");
      auto itCellStartIds =
        d_cellStartIdsBasisGradientQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itCellStartIds != d_cellStartIdsBasisGradientQuadStorage.end(),
        "Cell Start Ids not evaluated for the given QuadratureRuleAttributes");
      auto itNQuad = d_nQuadPointsIncell.find(quadratureRuleAttributes);
      utils::throwException(
        itNQuad != d_nQuadPointsIncell.end(),
        "Quad points in cell is not evaluated for the given QuadratureRuleAttributes.");
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        basisGradientQuadStorage                 = itBasisGradientQuad->second;
      const std::vector<size_type> &cellStartIds = itCellStartIds->second;
      const std::vector<size_type> &nQuadPointsInCell = itNQuad->second;
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(
        dim);
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

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getBasisHessianData(
      const QuadraturePointAttributes &attributes,
      const size_type                  basisId) const
    {
      const quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
        *(attributes.quadratureRuleAttributesPtr);
      const size_type cellId      = attributes.cellId;
      const size_type quadPointId = attributes.quadPointId;
      auto            itBasisHessianQuad =
        d_basisHessianQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itBasisHessianQuad != d_basisHessianQuadStorage.end(),
        "Basis hessian values are not evaluated for the given QuadratureRuleAttributes");
      auto itCellStartIds =
        d_cellStartIdsBasisHessianQuadStorage.find(quadratureRuleAttributes);
      utils::throwException(
        itCellStartIds != d_cellStartIdsBasisHessianQuadStorage.end(),
        "Cell Start Ids not evaluated for the given QuadratureRuleAttributes");
      auto itNQuad = d_nQuadPointsIncell.find(quadratureRuleAttributes);
      utils::throwException(
        itNQuad != d_nQuadPointsIncell.end(),
        "Quad points in cell is not evaluated for the given QuadratureRuleAttributes.");
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        basisHessianQuadStorage                  = itBasisHessianQuad->second;
      const std::vector<size_type> &cellStartIds = itCellStartIds->second;
      const std::vector<size_type> &nQuadPointsInCell = itNQuad->second;
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(
        dim * dim);
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


    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    const typename BasisDataStorage<ValueType, memorySpace>::Storage &
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisOverlapInAllCells(const quadrature::QuadratureRuleAttributes
                                  &quadratureRuleAttributes) const
    {
      auto it = d_basisOverlap.find(quadratureRuleAttributes);
      utils::throwException<utils::InvalidArgument>(
        it != d_basisOverlap.end(),
        "Basis overlap is not evaluated for the given QuadratureRuleAttributes");
      return *(it->second);
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisOverlapInCell(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             cellId) const
    {
      auto itBasisOverlap = d_basisOverlap.find(quadratureRuleAttributes);
      utils::throwException(
        itBasisOverlap != d_basisOverlap.end(),
        "Basis overlap is not evaluated for the given quadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
                      basisOverlapStorage = itBasisOverlap->second;
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(
        sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId]);
      return returnValue;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getBasisOverlap(
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
      const size_type                             cellId,
      const size_type                             basisId1,
      const size_type                             basisId2) const
    {
      auto itBasisOverlap = d_basisOverlap.find(quadratureRuleAttributes);
      utils::throwException(
        itBasisOverlap != d_basisOverlap.end(),
        "Basis overlap is not evaluated for the given quadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
        basisOverlapStorage = itBasisOverlap->second;
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(1);
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisOverlapStorage->data() + d_cellStartIdsBasisOverlap[cellId] +
          basisId1 * d_dofsInCell[cellId] + basisId2);
      return returnValue;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    void
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::deleteBasisData(
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes)
    {
      auto itBasisQuad = d_basisQuadStorage.find(quadratureRuleAttributes);
      if (itBasisQuad != d_basisQuadStorage.end())
        {
          utils::throwException(
            (itBasisQuad->second).use_count() == 1,
            "More than one owner for the basis quadrature storage found in FEBasisDataStorageDealii. Not safe to delete it.");
          delete (itBasisQuad->second).get();
          d_basisQuadStorage.erase(itBasisQuad);
        }

      auto itBasisGradientQuad =
        d_basisGradientQuadStorage.find(quadratureRuleAttributes);
      if (itBasisGradientQuad != d_basisGradientQuadStorage.end())
        {
          utils::throwException(
            (itBasisGradientQuad->second).use_count() == 1,
            "More than one owner for the basis gradient quadrature storage found in FEBasisDataStorageDealii. Not safe to delete it.");
          delete (itBasisGradientQuad->second).get();
          d_basisGradientQuadStorage.erase(itBasisGradientQuad);
        }

      auto itBasisHessianQuad =
        d_basisHessianQuadStorage.find(quadratureRuleAttributes);
      if (itBasisHessianQuad != d_basisHessianQuadStorage.end())
        {
          utils::throwException(
            (itBasisHessianQuad->second).use_count() == 1,
            "More than one owner for the basis hessian quadrature storage found in FEBasisDataStorageDealii. Not safe to delete it.");
          delete (itBasisHessianQuad->second).get();
          d_basisHessianQuadStorage.erase(itBasisHessianQuad);
        }

      auto itBasisOverlap = d_basisOverlap.find(quadratureRuleAttributes);
      if (itBasisOverlap != d_basisOverlap.end())
        {
          utils::throwException(
            (itBasisOverlap->second).use_count() == 1,
            "More than one owner for the basis overlap storage found in FEBasisDataStorageDealii. Not safe to delete it.");
          delete (itBasisOverlap->second).get();
          d_basisOverlap.erase(itBasisOverlap);
        }
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getBasisDataInCell(
      const QuadratureRuleAttributes &quadratureRuleAttributes,
      const size_type                 cellId,
      const size_type                 basisId) const
    {
      utils::throwException(
        false,
        "getBasisDataInCell() for a given basisId is not implemented in FEBasisDataStorageDealii");
      typename BasisDataStorage<ValueType, memorySpace>::Storage dummy(0);
      return dummy;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisGradientDataInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId,
        const size_type                 basisId) const
    {
      utils::throwException(
        false,
        "getBasisGradientDataInCell() for a given basisId is not implemented in FEBasisDataStorageDealii");
      typename BasisDataStorage<ValueType, memorySpace>::Storage dummy(0);
      return dummy;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisHessianDataInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId,
        const size_type                 basisId) const
    {
      utils::throwException(
        false,
        "getBasisHessianDataInCell() for a given basisId is not implemented in FEBasisDataStorageDealii");
      typename BasisDataStorage<ValueType, memorySpace>::Storage dummy(0);
      return dummy;
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    const quadrature::QuadratureRuleContainer &
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getQuadratureRuleContainer(const quadrature::QuadratureRuleAttributes
                                   &quadratureRuleAttributes) const
    {
      auto it = d_quadratureRuleContainer.find(quadratureRuleAttributes);
      utils::throwException(
        it != d_quadratureRuleContainer.end(),
        "QuadratureRuleContainer is not provided for the given QuadratureRuleAttributes.");
      return *(it->second);
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    typename BasisDataStorage<ValueType, memorySpace>::Storage
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisGradNiGradNjInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId) const
    {
      auto itGradNiGradNj = d_basisGradNiGradNj.find(quadratureRuleAttributes);
      utils::throwException(
        itGradNiGradNj != d_basisGradNiGradNj.end(),
        "Basis Grad Ni Grad Nj is not evaluated for the given quadratureRuleAttributes");
      std::shared_ptr<
        typename BasisDataStorage<ValueType, memorySpace>::Storage>
                      basisGradNiNj = itGradNiGradNj->second;
      const size_type sizeToCopy = d_dofsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueType, memorySpace>::Storage returnValue(
        sizeToCopy);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        sizeToCopy,
        returnValue.data(),
        basisGradNiNj->data() + d_cellStartIdsGradNiGradNj[cellId]);
      return returnValue;
    }

    // get overlap of all the basis functions in all cells
    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    const typename BasisDataStorage<ValueType, memorySpace>::Storage &
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      getBasisGradNiGradNjInAllCells(
        const QuadratureRuleAttributes &quadratureRuleAttributes) const
    {
      auto it = d_basisGradNiGradNj.find(quadratureRuleAttributes);
      utils::throwException<utils::InvalidArgument>(
        it != d_basisGradNiGradNj.end(),
        "Basis Grad Ni Grad Nj is not evaluated for the given QuadratureRuleAttributes");
      return *(it->second);
    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    const BasisManager &
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::getBasisManager()
      const
    {
      return *d_feBM;
    }

  } // end of namespace basis
} // end of namespace dftefe
