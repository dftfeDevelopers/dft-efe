
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

#include <utils/Exceptions.h>
#include <utils/MathFunctions.h>
#include "DealiiConversions.h"
#include <basis/TriangulationCellDealii.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <quadrature/QuadratureAttributes.h>
#include <basis/ParentToChildCellsManagerDealii.h>
namespace dftefe
{
  namespace basis
  {
    namespace CFEBDSOnTheFlyComputeDealiiInternal
    {
      template <typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeJacobianInvTimesGradPara(
        std::pair<size_type, size_type> cellRange,
        const std::vector<size_type> &  dofsInCell,
        const std::vector<size_type> &  nQuadPointsInCell,
        const std::vector<std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>>
          &basisGradientParaCellQuadStorage,
        const std::vector<std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>>
          &                           basisJacobianInvQuadStorage,
        const std::vector<size_type> &cellStartIdsBasisJacobianInvQuadStorage,
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
          &                                          tmpGradientBlock,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
          &basisGradientData)
      {
        size_type numCellsInBlock = cellRange.second - cellRange.first;
        tmpGradientBlock.setValue(0);
        size_type numMats = 0;
        for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
          {
            numMats += 1;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
          memoryTransfer;

        std::vector<linearAlgebra::blasLapack::ScalarOp> scalarOpA(
          numMats, linearAlgebra::blasLapack::ScalarOp::Identity);
        std::vector<linearAlgebra::blasLapack::ScalarOp> scalarOpB(
          numMats, linearAlgebra::blasLapack::ScalarOp::Identity);
        std::vector<size_type> mSizesTmp(numMats, 0);
        std::vector<size_type> nSizesTmp(numMats, 0);
        std::vector<size_type> kSizesTmp(numMats, 0);
        std::vector<size_type> strideATmp(numMats, 0);
        std::vector<size_type> strideBTmp(numMats, 0);
        std::vector<size_type> strideCTmp(numMats, 0);

        for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
          {
            size_type index   = iCell;
            mSizesTmp[index]  = dofsInCell[iCell];
            nSizesTmp[index]  = dim;
            kSizesTmp[index]  = nQuadPointsInCell[iCell];
            strideATmp[index] = 0;
            strideBTmp[index] = kSizesTmp[index] * nSizesTmp[index];
            strideCTmp[index] =
              mSizesTmp[index] * nSizesTmp[index] * kSizesTmp[index];
          }

        utils::MemoryStorage<size_type, memorySpace> mSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> nSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> kSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> strideA(numMats);
        utils::MemoryStorage<size_type, memorySpace> strideB(numMats);
        utils::MemoryStorage<size_type, memorySpace> strideC(numMats);
        memoryTransfer.copy(numMats, mSizes.data(), mSizesTmp.data());
        memoryTransfer.copy(numMats, nSizes.data(), nSizesTmp.data());
        memoryTransfer.copy(numMats, kSizes.data(), kSizesTmp.data());
        memoryTransfer.copy(numMats, strideA.data(), strideATmp.data());
        memoryTransfer.copy(numMats, strideB.data(), strideBTmp.data());
        memoryTransfer.copy(numMats, strideC.data(), strideCTmp.data());

        for (size_type iDim = 0; iDim < dim; iDim++)
          {
            ValueTypeBasisData *A =
              basisGradientParaCellQuadStorage[iDim]->data();
            ValueTypeBasisData *B =
              basisJacobianInvQuadStorage[iDim]->data() +
              cellStartIdsBasisJacobianInvQuadStorage[cellRange.first];
            ValueTypeBasisData *C = tmpGradientBlock.data();

            linearAlgebra::blasLapack::scaleStridedVarBatched<
              ValueTypeBasisData,
              ValueTypeBasisData,
              memorySpace>(numMats,
                           scalarOpA.data(),
                           scalarOpA.data(),
                           strideA.data(),
                           strideB.data(),
                           strideC.data(),
                           mSizes.data(),
                           nSizes.data(),
                           kSizes.data(),
                           A,
                           B,
                           C,
                           linAlgOpContext);

            ValueTypeBasisData alpha = 1.0;
            linearAlgebra::blasLapack::
              axpby<ValueTypeBasisData, ValueTypeBasisData, memorySpace>(
                tmpGradientBlock.size(),
                alpha,
                tmpGradientBlock.data(),
                alpha,
                basisGradientData.data(),
                basisGradientData.data(),
                linAlgOpContext);

            tmpGradientBlock.setValue(0);
          }
      }

      //
      // stores the classical FE basis data for a h-refined FE mesh
      // (i.e., uniform p in all elements) and for a uniform quadrature
      // Gauss or Gauss-Legendre-Lobatto (GLL) quadrature
      // rule across all the cells in the mesh.
      //
      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      storeValuesHRefinedSameQuadEveryCell(
        std::shared_ptr<
          const CFEBasisDofHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>>
          feBDH,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisQuadStorage,
        std::vector<std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>>
          &basisGradientParaCellQuadStorage,
        std::vector<std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>>
          &basisJacobianInvQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &                                         basisHessianQuadStorage,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                quadratureRuleContainer,
        std::vector<size_type> &nQuadPointsInCell,
        std::vector<size_type> &cellStartIdsBasisQuadStorage,
        std::vector<size_type> &cellStartIdsBasisJacobianInvQuadStorage,
        std::vector<size_type> &cellStartIdsBasisHessianQuadStorage,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
      {
        // for processors where there are no cells
        bool numCellsZero = feBDH->nLocallyOwnedCells() == 0 ? true : false;
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
          dealii::update_values | dealii::update_JxW_values;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          dealiiUpdateFlags |= dealii::update_inverse_jacobians;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          dealiiUpdateFlags |= dealii::update_hessians;

        // NOTE: cellId 0 passed as we assume h-refine finite element mesh in
        // this function
        const size_type cellId = 0;
        // get real cell feValues
        dealii::FEValues<dim> dealiiFEValues(feBDH->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags);

        dealii::UpdateFlags dealiiUpdateFlagsPara;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          dealiiUpdateFlagsPara = dealii::update_gradients;
        // This is for getting the gradient in parametric cell
        dealii::FEValues<dim> dealiiFEValuesPara(feBDH->getReferenceFE(cellId),
                                                 dealiiQuadratureRule,
                                                 dealiiUpdateFlagsPara);

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            dealii::Triangulation<dim> referenceCell;
            dealii::GridGenerator::hyper_cube(referenceCell, 0., 1.);
            dealiiFEValuesPara.reinit(referenceCell.begin());
          }

        const size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        const size_type dofsPerCell = feBDH->nCellDofs(cellId);
        const size_type nQuadPointsPerCell =
          numCellsZero ? 0 :
                         quadratureRuleContainer->nCellQuadraturePoints(cellId);

        const size_type nDimxDofsPerCellxNumQuad =
          dim * dofsPerCell * nQuadPointsPerCell;
        const size_type nDimSqxDofsPerCellxNumQuad =
          dim * dim * dofsPerCell * nQuadPointsPerCell;
        const size_type nDimxNumQuad        = dim * nQuadPointsPerCell;
        const size_type nDimSqxNumQuad      = dim * dim * nQuadPointsPerCell;
        const size_type DofsPerCellxNumQuad = dofsPerCell * nQuadPointsPerCell;

        nQuadPointsInCell.resize(numLocallyOwnedCells, nQuadPointsPerCell);
        std::vector<ValueTypeBasisData> basisQuadStorageTmp(0);
        std::vector<std::vector<ValueTypeBasisData>>
          basisJacobianInvQuadStorageTmp(dim);
        std::vector<std::vector<ValueTypeBasisData>>
          basisGradientParaCellQuadStorageTmp(dim);
        std::vector<ValueTypeBasisData> basisHessianQuadStorageTmp(0);

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            basisQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                dofsPerCell * nQuadPointsPerCell);
            basisQuadStorageTmp.resize(dofsPerCell * nQuadPointsPerCell,
                                       ValueTypeBasisData(0));
            cellStartIdsBasisQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            for (size_type iDim = 0; iDim < dim; iDim++)
              {
                basisJacobianInvQuadStorage[iDim] = std::make_shared<
                  typename BasisDataStorage<ValueTypeBasisData,
                                            memorySpace>::Storage>(
                  numLocallyOwnedCells * nDimxNumQuad);
                basisJacobianInvQuadStorageTmp[iDim].resize(
                  numLocallyOwnedCells * nDimxNumQuad);
                basisGradientParaCellQuadStorage[iDim] = std::make_shared<
                  typename BasisDataStorage<ValueTypeBasisData,
                                            memorySpace>::Storage>(
                  DofsPerCellxNumQuad);
                basisGradientParaCellQuadStorageTmp[iDim].resize(
                  DofsPerCellxNumQuad);
              }
            cellStartIdsBasisJacobianInvQuadStorage.resize(numLocallyOwnedCells,
                                                           0);
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          {
            std::cout
              << "Store Hessian is not memory optimized in CFEOnTheFlyComputeDealii.h. Contact developers for making it optimal.";
            basisHessianQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                numLocallyOwnedCells * nDimSqxDofsPerCellxNumQuad);
            basisHessianQuadStorageTmp.resize(numLocallyOwnedCells *
                                                nDimSqxDofsPerCellxNumQuad,
                                              ValueTypeBasisData(0));
            cellStartIdsBasisHessianQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        auto locallyOwnedCellIter = feBDH->beginLocallyOwnedCells();
        // Do dynamic cast if there is a dof in the processor
        std::shared_ptr<FECellDealii<dim>> feCellDealii = nullptr;
        if (numLocallyOwnedCells != 0)
          {
            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            utils::throwException(
              feCellDealii != nullptr,
              "Dynamic casting of FECellBase to FECellDealii not successful");
          }

        size_type cellIndex = 0;
        for (; locallyOwnedCellIter != feBDH->endLocallyOwnedCells();
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
                locallyOwnedCellIter == feBDH->beginLocallyOwnedCells())
              {
                for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                  {
                    for (unsigned int qPoint = 0; qPoint < nQuadPointsPerCell;
                         qPoint++)
                      {
                        auto it = basisQuadStorageTmp.begin() +
                                  cellIndex * DofsPerCellxNumQuad +
                                  iNode * nQuadPointsPerCell + qPoint;
                        *it = dealiiFEValues.shape_value(iNode, qPoint);
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                cellStartIdsBasisJacobianInvQuadStorage[cellIndex] =
                  cellIndex * nDimxNumQuad;
                if (locallyOwnedCellIter == feBDH->beginLocallyOwnedCells())
                  {
                    for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointsPerCell;
                             qPoint++)
                          {
                            auto shapeGrad =
                              dealiiFEValuesPara.shape_grad(iNode, qPoint);
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientParaCellQuadStorageTmp[iDim]
                                    .begin() +
                                  cellIndex * DofsPerCellxNumQuad +
                                  iNode * nQuadPointsPerCell + qPoint;
                                *it = shapeGrad[iDim];
                              }
                          }
                      }
                  }
                auto &mappingJacInv = dealiiFEValues.get_inverse_jacobians();
                for (unsigned int iQuad = 0; iQuad < nQuadPointsPerCell;
                     ++iQuad)
                  {
                    for (unsigned int iDim = 0; iDim < dim; iDim++)
                      {
                        for (unsigned int jDim = 0; jDim < dim; jDim++)
                          {
                            auto it =
                              basisJacobianInvQuadStorageTmp[iDim].begin() +
                              cellIndex * nDimxNumQuad +
                              nQuadPointsPerCell * jDim + iQuad;
                            *it = mappingJacInv[iQuad][iDim][jDim];
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
                    for (unsigned int qPoint = 0; qPoint < nQuadPointsPerCell;
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
                                  iNode * nQuadPointsPerCell + qPoint;
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
            for (size_type iDim = 0; iDim < dim; iDim++)
              {
                utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                  copy(basisGradientParaCellQuadStorageTmp[iDim].size(),
                       basisGradientParaCellQuadStorage[iDim]->data(),
                       basisGradientParaCellQuadStorageTmp[iDim].data());

                utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                  copy(basisJacobianInvQuadStorageTmp[iDim].size(),
                       basisJacobianInvQuadStorage[iDim]->data(),
                       basisJacobianInvQuadStorageTmp[iDim].data());
              }
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
      }
    } // namespace CFEBDSOnTheFlyComputeDealiiInternal

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::
      CFEBDSOnTheFlyComputeDealii(
        std::shared_ptr<const BasisDofHandler>      feBDH,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap,
        const size_type                     maxCellBlock,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      : d_dofsInCell(0)
      , d_quadratureRuleAttributes(quadratureRuleAttributes)
      , d_basisStorageAttributesBoolMap(basisStorageAttributesBoolMap)
      , d_maxCellBlock(maxCellBlock)
      , d_linAlgOpContext(linAlgOpContext)
      , d_basisGradientParaCellQuadStorage(dim, nullptr)
      , d_basisJacobianInvQuadStorage(dim, nullptr)
    {
      d_evaluateBasisData = false;
      d_feBDH             = std::dynamic_pointer_cast<
        const CFEBasisDofHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
        feBDH);
      utils::throwException(
        d_feBDH != nullptr,
        " Could not cast the FEBasisDofHandler to CFEBasisDofHandlerDealii in CFEBDSOnTheFlyComputeDealii");
      //      const size_type numConstraints  = constraintsVec.size();
      // const size_type numQuadRuleType = quadratureRuleAttributesVec.size();
      std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
        d_feBDH->getDoFHandler();
      const size_type numLocallyOwnedCells = d_feBDH->nLocallyOwnedCells();
      d_dofsInCell.resize(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        {
          d_dofsInCell[iCell] = d_feBDH->nCellDofs(iCell);
        }
      d_tmpGradientBlock = nullptr;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
              d_feBDH->getTriangulation(),
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
              d_feBDH->getTriangulation(),
              linearCellMappingDealii);
        }
      else
        utils::throwException<utils::InvalidArgument>(
          false, "Incorrect arguments given for this Quadrature family.");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisQuadStorage;
      std::vector<std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>>
        basisGradientParaCellQuadStorage(dim, nullptr);
      std::vector<std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>>
        basisJacobianInvQuadStorage(dim, nullptr);
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage;

      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisJacobianInvQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);
      CFEBDSOnTheFlyComputeDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_feBDH,
             basisQuadStorage,
             basisGradientParaCellQuadStorage,
             basisJacobianInvQuadStorage,
             basisHessianQuadStorage,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisQuadStorage,
             cellStartIdsBasisJacobianInvQuadStorage,
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
          for (size_type iDim = 0; iDim < dim; iDim++)
            {
              d_basisGradientParaCellQuadStorage[iDim] =
                basisGradientParaCellQuadStorage[iDim];
              d_basisJacobianInvQuadStorage[iDim] =
                basisJacobianInvQuadStorage[iDim];
            }
          d_cellStartIdsBasisJacobianInvQuadStorage =
            cellStartIdsBasisJacobianInvQuadStorage;
          d_tmpGradientBlock = std::make_shared<Storage>(
            d_dofsInCell[0] * nQuadPointsInCell[0] * dim * d_maxCellBlock);
          //   size_type gradientParaCellSize =
          //     basisGradientParaCellQuadStorage->size();
          //   for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
          //     {
          //       d_tmpGradientBlock->template copyFrom<memorySpace>(
          //         basisGradientParaCellQuadStorage->data(),
          //         gradientParaCellSize,
          //         0,
          //         gradientParaCellSize * iCell);
          //     }
        }
      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = basisHessianQuadStorage;
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreOverlap)
            ->second)
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "Basis Overlap not implemented in CFEBDSOnTheFlyComputeDealii");
        }
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "Basis GradNiGradNj not implemented in CFEBDSOnTheFlyComputeDealii");
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

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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

      if (quadFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED)
        d_quadratureRuleContainer = quadratureRuleContainer;
      else
        utils::throwException<utils::InvalidArgument>(
          false,
          "Incorrect arguments given for this Quadrature family. On the fly computation is not available for non-uniform quadrature rule in cells.");

      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisQuadStorage;
      std::vector<std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>>
        basisGradientParaCellQuadStorage(dim, nullptr);
      std::vector<std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>>
        basisJacobianInvQuadStorage(dim, nullptr);
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                             basisHessianQuadStorage;
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisJacobianInvQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      CFEBDSOnTheFlyComputeDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_feBDH,
             basisQuadStorage,
             basisGradientParaCellQuadStorage,
             basisJacobianInvQuadStorage,
             basisHessianQuadStorage,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisQuadStorage,
             cellStartIdsBasisJacobianInvQuadStorage,
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
          for (size_type iDim = 0; iDim < dim; iDim++)
            {
              d_basisGradientParaCellQuadStorage[iDim] =
                basisGradientParaCellQuadStorage[iDim];
              d_basisJacobianInvQuadStorage[iDim] =
                basisJacobianInvQuadStorage[iDim];
            }
          d_cellStartIdsBasisJacobianInvQuadStorage =
            cellStartIdsBasisJacobianInvQuadStorage;
          d_tmpGradientBlock = std::make_shared<Storage>(
            d_dofsInCell[0] * nQuadPointsInCell[0] * dim * d_maxCellBlock);
          //   size_type gradientParaCellSize =
          //     basisGradientParaCellQuadStorage->size();
          //   for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
          //     {
          //       d_tmpGradientBlock->template copyFrom<memorySpace>(
          //         basisGradientParaCellQuadStorage->data(),
          //         gradientParaCellSize,
          //         0,
          //         gradientParaCellSize * iCell);
          //     }
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = basisHessianQuadStorage;
          d_cellStartIdsBasisHessianQuadStorage =
            cellStartIdsBasisHessianQuadStorage;
        }

      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "Basis GradNiGradNj not implemented in CFEBDSOnTheFlyComputeDealii");
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


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::vector<std::shared_ptr<const quadrature::QuadratureRule>>
                                            quadratureRuleVec,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
    {
      utils::throwException<utils::InvalidArgument>(
        false,
        "On the fly computation is not available for non-uniform quadrature rule in cells.");
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
      utils::throwException<utils::InvalidArgument>(
        false,
        "On the fly computation is not available for non-uniform/adaptive quadrature rule in cells.");
    }

    //------------------OTHER FNS -----------------------------

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisDataInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData,
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisGradientDataInAllCells() const
    {
      utils::throwException(
        false,
        "getBasisGradientDataInAllCells() is not implemented in CFEBDSOnTheFlyComputeDealii");
      // typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
      // dummy(
      //   0);
      return *d_tmpGradientBlock;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisHessianDataInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData,
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getJxWInAllCells() const
    {
      utils::throwException(
        d_evaluateBasisData,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
          ->second,
        "JxW values are not stored for the given QuadratureRuleAttributes");
      return *(d_JxWStorage);
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisDataInCell(const size_type cellId)
      const
    {
      utils::throwException(
        d_evaluateBasisData,
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
    CFEBDSOnTheFlyComputeDealii<
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisGradientDataInCell(const size_type
                                                                   cellId) const
    {
      utils::throwException(
        d_evaluateBasisData,
        "Cannot call function before calling evaluateBasisData()");

      utils::throwException(
        d_basisStorageAttributesBoolMap
          .find(BasisStorageAttributes::StoreGradient)
          ->second,
        "Basis gradient values are not evaluated for the given QuadratureRuleAttributes");

      const size_type sizeToCopy =
        d_nQuadPointsIncell[cellId] * d_dofsInCell[cellId] * dim;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);

      std::pair<size_type, size_type> cellPair(cellId, cellId + 1);

      CFEBDSOnTheFlyComputeDealiiInternal::
        computeJacobianInvTimesGradPara<ValueTypeBasisData, memorySpace, dim>(
          cellPair,
          d_dofsInCell,
          d_nQuadPointsIncell,
          d_basisGradientParaCellQuadStorage,
          d_basisJacobianInvQuadStorage,
          d_cellStartIdsBasisJacobianInvQuadStorage,
          returnValue,
          d_linAlgOpContext,
          returnValue);

      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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

      basisGradientData.setValue(0);
      std::shared_ptr<Storage> tmpGradientBlock = nullptr;
      if ((cellRange.second - cellRange.first) > d_maxCellBlock)
        {
          tmpGradientBlock = std::make_shared<Storage>(
            d_dofsInCell[0] * d_nQuadPointsIncell[0] * dim *
            (cellRange.second - cellRange.first));
          //   size_type gradientParaCellSize =
          //     d_basisGradientParaCellQuadStorage->size();
          //   for (size_type iCell = 0;
          //        iCell < (cellRange.second - cellRange.first);
          //        ++iCell)
          //     {
          //       tmpGradientBlock->template copyFrom<memorySpace>(
          //         d_basisGradientParaCellQuadStorage->data(),
          //         gradientParaCellSize,
          //         0,
          //         gradientParaCellSize * iCell);
          //     }
        }
      else
        {
          tmpGradientBlock = d_tmpGradientBlock;
        }
      CFEBDSOnTheFlyComputeDealiiInternal::
        computeJacobianInvTimesGradPara<ValueTypeBasisData, memorySpace, dim>(
          cellRange,
          d_dofsInCell,
          d_nQuadPointsIncell,
          d_basisGradientParaCellQuadStorage,
          d_basisJacobianInvQuadStorage,
          d_cellStartIdsBasisJacobianInvQuadStorage,
          *tmpGradientBlock,
          d_linAlgOpContext,
          basisGradientData);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisHessianDataInCell(const size_type
                                                                  cellId) const
    {
      utils::throwException(
        d_evaluateBasisData,
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getJxWInCell(const size_type cellId) const
    {
      utils::throwException(
        d_evaluateBasisData,
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisData(const QuadraturePointAttributes &attributes,
                         const size_type                  basisId) const
    {
      utils::throwException(
        d_evaluateBasisData,
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
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisGradientData(const QuadraturePointAttributes &attributes,
                                 const size_type                  basisId) const
    {
      utils::throwException(
        d_evaluateBasisData,
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
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        basisGradientQuadStorage(getBasisGradientDataInCell(cellId));
      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(dim);
      for (size_type iDim = 0; iDim < dim; ++iDim)
        {
          utils::MemoryTransfer<memorySpace, memorySpace>::copy(
            1,
            returnValue.data() + iDim,
            basisGradientQuadStorage.data() +
              quadPointId * d_dofsInCell[cellId] * dim +
              iDim * d_dofsInCell[cellId] + basisId);
        }
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisHessianData(const QuadraturePointAttributes &attributes,
                                const size_type                  basisId) const
    {
      utils::throwException(
        d_evaluateBasisData,
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
            }
        }
      return returnValue;
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisOverlapInAllCells() const
    {
      utils::throwException<utils::InvalidArgument>(
        false, "Basis Overlap not implemented in CFEBDSOnTheFlyComputeDealii");
      // typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
      // dummy(
      //   0);
      return *d_tmpGradientBlock;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisOverlapInCell(const size_type
                                                              cellId) const
    {
      utils::throwException<utils::InvalidArgument>(
        false, "Basis Overlap not implemented in CFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisOverlap(const size_type cellId,
                                                      const size_type basisId1,
                                                      const size_type basisId2)
      const
    {
      utils::throwException<utils::InvalidArgument>(
        false, "Basis Overlap not implemented in CFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::deleteBasisData()
    {
      utils::throwException(
        (d_basisQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in CFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
      delete (d_basisQuadStorage).get();

      for (size_type iDim = 0; iDim < dim; iDim++)
        {
          utils::throwException(
            (d_basisJacobianInvQuadStorage[iDim]).use_count() == 1,
            "More than one owner for the basis quadrature storage found in CFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
          delete (d_basisJacobianInvQuadStorage[iDim]).get();

          utils::throwException(
            (d_basisGradientParaCellQuadStorage[iDim]).use_count() == 1,
            "More than one owner for the basis quadrature storage found in CFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
          delete (d_basisGradientParaCellQuadStorage[iDim]).get();
        }

      utils::throwException(
        (d_basisHessianQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in CFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
      delete (d_basisHessianQuadStorage).get();

      d_tmpGradientBlock->resize(0);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisDataInCell(const size_type cellId,
                                                         const size_type
                                                           basisId) const
    {
      utils::throwException(
        false,
        "getBasisDataInCell() for a given basisId is not implemented in CFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisGradientDataInCell(const size_type cellId,
                                       const size_type basisId) const
    {
      utils::throwException(
        false,
        "getBasisGradientDataInCell() for a given basisId is not implemented in CFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisHessianDataInCell(const size_type cellId,
                                      const size_type basisId) const
    {
      utils::throwException(
        false,
        "getBasisHessianDataInCell() for a given basisId is not implemented in CFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const quadrature::QuadratureRuleContainer>
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getQuadratureRuleContainer() const
    {
      utils::throwException(
        d_evaluateBasisData,
        "Cannot call function before calling evaluateBasisData()");

      return d_quadratureRuleContainer;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisGradNiGradNjInCell(const size_type
                                                                   cellId) const
    {
      utils::throwException<utils::InvalidArgument>(
        false,
        "Basis GradNiGradNj not implemented in CFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    // get overlap of all the basis functions in all cells
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisGradNiGradNjInAllCells() const
    {
      utils::throwException<utils::InvalidArgument>(
        false,
        "Basis GradNiGradNj not implemented in CFEBDSOnTheFlyComputeDealii");
      // typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
      // dummy(
      //   0);
      return *d_tmpGradientBlock;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const BasisDofHandler>
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisDofHandler() const
    {
      return d_feBDH;
    }
  } // namespace basis
} // namespace dftefe
