
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
        const std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &                           basisJacobianInvQuadStorage,
        const std::vector<size_type> &cellStartIdsBasisJacobianInvQuadStorage,
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
          &                                          tmpGradientBlock,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
          &basisGradientData)
      {
        size_type numCellsInBlock = cellRange.second - cellRange.first;

        size_type numMats = 0;
        for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
          {
            for (size_type iQuad = 0; iQuad < nQuadPointsInCell[iCell]; ++iQuad)
              {
                numMats += 1;
              }
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
          memoryTransfer;

        std::vector<char>      transA(numMats, 'N');
        std::vector<char>      transB(numMats, 'N');
        std::vector<size_type> mSizesTmp(numMats, 0);
        std::vector<size_type> nSizesTmp(numMats, 0);
        std::vector<size_type> kSizesTmp(numMats, 0);
        std::vector<size_type> ldaSizesTmp(numMats, 0);
        std::vector<size_type> ldbSizesTmp(numMats, 0);
        std::vector<size_type> ldcSizesTmp(numMats, 0);
        std::vector<size_type> strideATmp(numMats, 0);
        std::vector<size_type> strideBTmp(numMats, 0);
        std::vector<size_type> strideCTmp(numMats, 0);

        for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
          {
            for (size_type iQuad = 0; iQuad < nQuadPointsInCell[iCell]; ++iQuad)
              {
                size_type index    = iCell * nQuadPointsInCell[iCell] + iQuad;
                mSizesTmp[index]   = dofsInCell[iCell];
                nSizesTmp[index]   = dim;
                kSizesTmp[index]   = dim;
                ldaSizesTmp[index] = mSizesTmp[index];
                ldbSizesTmp[index] = kSizesTmp[index];
                ldcSizesTmp[index] = mSizesTmp[index];
                strideATmp[index]  = mSizesTmp[index] * kSizesTmp[index];
                strideBTmp[index]  = kSizesTmp[index] * nSizesTmp[index];
                strideCTmp[index]  = mSizesTmp[index] * nSizesTmp[index];
              }
          }

        utils::MemoryStorage<size_type, memorySpace> mSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> nSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> kSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> ldaSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> ldbSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> ldcSizes(numMats);
        utils::MemoryStorage<size_type, memorySpace> strideA(numMats);
        utils::MemoryStorage<size_type, memorySpace> strideB(numMats);
        utils::MemoryStorage<size_type, memorySpace> strideC(numMats);
        memoryTransfer.copy(numMats, mSizes.data(), mSizesTmp.data());
        memoryTransfer.copy(numMats, nSizes.data(), nSizesTmp.data());
        memoryTransfer.copy(numMats, kSizes.data(), kSizesTmp.data());
        memoryTransfer.copy(numMats, ldaSizes.data(), ldaSizesTmp.data());
        memoryTransfer.copy(numMats, ldbSizes.data(), ldbSizesTmp.data());
        memoryTransfer.copy(numMats, ldcSizes.data(), ldcSizesTmp.data());
        memoryTransfer.copy(numMats, strideA.data(), strideATmp.data());
        memoryTransfer.copy(numMats, strideB.data(), strideBTmp.data());
        memoryTransfer.copy(numMats, strideC.data(), strideCTmp.data());

        ValueTypeBasisData alpha = 1.0;
        ValueTypeBasisData beta  = 0.0;

        ValueTypeBasisData *B =
          basisJacobianInvQuadStorage->data() +
          cellStartIdsBasisJacobianInvQuadStorage[cellRange.first];
        linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeBasisData,
                                                         ValueTypeBasisData,
                                                         memorySpace>(
          numMats,
          transA.data(),
          transB.data(),
          strideA.data(),
          strideB.data(),
          strideC.data(),
          mSizes.data(),
          nSizes.data(),
          kSizes.data(),
          alpha,
          tmpGradientBlock.data(),
          ldaSizes.data(),
          B,
          ldbSizes.data(),
          beta,
          basisGradientData.data(),
          ldcSizes.data(),
          linAlgOpContext);
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
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientParaCellQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
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
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap,
        dealii::Quadrature<dim> &           dealiiQuadratureRule)
      {
        // for processors where there are no cells
        bool numCellsZero = feBDH->nLocallyOwnedCells() == 0 ? true : false;
        const quadrature::QuadratureFamily quadratureFamily =
          quadratureRuleAttributes.getQuadratureFamily();
        const size_type num1DQuadPoints =
          quadratureRuleAttributes.getNum1DPoints();
        // dealii::Quadrature<dim> dealiiQuadratureRule;
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
            const quadrature::QuadratureRuleGaussIterated &quadRule =
              dynamic_cast<const quadrature::QuadratureRuleGaussIterated &>(
                quadratureRuleContainer->getQuadratureRule(0));
            // if (!numCellsZero)
            //   {
            //     // get the parametric points and jxw in each cell according
            //     to
            //     // the attribute.
            //     unsigned int                     cellIndex = 0;
            //     const std::vector<utils::Point> &cellParametricQuadPoints =
            //       quadratureRuleContainer->getCellParametricPoints(cellIndex);
            //     std::vector<dealii::Point<dim, double>>
            //       dealiiParametricQuadPoints(0);

            //     // get the quad weights in each cell
            //     const std::vector<double> &quadWeights =
            //       quadratureRuleContainer->getCellQuadratureWeights(cellIndex);
            //     convertToDealiiPoint<dim>(cellParametricQuadPoints,
            //                               dealiiParametricQuadPoints);

            //     // Ask dealii to create quad rule in each cell
            //     dealiiQuadratureRule =
            //       dealii::Quadrature<dim>(dealiiParametricQuadPoints,
            //                               quadWeights);
            //   }
            dealiiQuadratureRule =
              dealii::QIterated<dim>(dealii::QGauss<1>(quadRule.order1D()),
                                     quadRule.numCopies());
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
        std::shared_ptr<dealii::FEValues<dim>> dealiiFEValuesPara = nullptr;

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            dealiiFEValuesPara = std::make_shared<dealii::FEValues<dim>>(
              feBDH->getReferenceFE(cellId),
              dealiiQuadratureRule,
              dealiiUpdateFlagsPara); // takes time
            dealii::Triangulation<dim> referenceCell;
            dealii::GridGenerator::hyper_cube(referenceCell, 0., 1.);
            dealiiFEValuesPara->reinit(referenceCell.begin()); // takes time
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
        const size_type nDimSqxNumQuad      = dim * dim * nQuadPointsPerCell;
        const size_type DofsPerCellxNumQuad = dofsPerCell * nQuadPointsPerCell;

        nQuadPointsInCell.resize(numLocallyOwnedCells, nQuadPointsPerCell);
        std::vector<ValueTypeBasisData> basisQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisJacobianInvQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisGradientParaCellQuadStorageTmp(0);
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
            basisJacobianInvQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                numLocallyOwnedCells * nDimSqxNumQuad);
            basisJacobianInvQuadStorageTmp.resize(numLocallyOwnedCells *
                                                  nDimSqxNumQuad);
            basisGradientParaCellQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                nDimxDofsPerCellxNumQuad);
            basisGradientParaCellQuadStorageTmp.resize(
              nDimxDofsPerCellxNumQuad);
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
                                  qPoint * dofsPerCell + iNode;
                        // iNode * nQuadPointsPerCell + qPoint;
                        *it = dealiiFEValues.shape_value(iNode, qPoint);
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                cellStartIdsBasisJacobianInvQuadStorage[cellIndex] =
                  cellIndex * nDimSqxNumQuad;
                if (locallyOwnedCellIter == feBDH->beginLocallyOwnedCells())
                  {
                    for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointsPerCell;
                             qPoint++)
                          {
                            auto shapeGrad =
                              dealiiFEValuesPara->shape_grad(iNode, qPoint);
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientParaCellQuadStorageTmp.begin() +
                                  cellIndex * nDimxDofsPerCellxNumQuad +
                                  qPoint * dim * dofsPerCell +
                                  iDim * dofsPerCell + iNode;
                                *it = shapeGrad[iDim];
                              }
                          }
                      }
                  }
                auto &mappingJacInv = dealiiFEValues.get_inverse_jacobians();
                size_type numJacobiansPerCell = nQuadPointsPerCell;
                for (unsigned int iQuad = 0; iQuad < numJacobiansPerCell;
                     ++iQuad)
                  {
                    for (unsigned int iDim = 0; iDim < dim; iDim++)
                      {
                        for (unsigned int jDim = 0; jDim < dim; jDim++)
                          {
                            auto it = basisJacobianInvQuadStorageTmp.begin() +
                                      cellIndex * nDimSqxNumQuad +
                                      iQuad * dim * dim + jDim * dim + iDim;
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
                                  qPoint * dim * dim * dofsPerCell +
                                  iDim * dim * dofsPerCell +
                                  jDim * dofsPerCell + iNode;
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
              basisGradientParaCellQuadStorageTmp.size(),
              basisGradientParaCellQuadStorage->data(),
              basisGradientParaCellQuadStorageTmp.data());

            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisJacobianInvQuadStorageTmp.size(),
              basisJacobianInvQuadStorage->data(),
              basisJacobianInvQuadStorageTmp.data());
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
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientParaCellQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisJacobianInvQuadStorage;
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
             basisStorageAttributesBoolMap,
             d_dealiiQuadratureRule);

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
          d_basisGradientParaCellQuadStorage =
            std::move(basisGradientParaCellQuadStorage);
          d_basisJacobianInvQuadStorage =
            std::move(basisJacobianInvQuadStorage);
          d_cellStartIdsBasisJacobianInvQuadStorage =
            cellStartIdsBasisJacobianInvQuadStorage;
          if (d_maxCellBlock != 1)
            {
              d_tmpGradientBlock = std::make_shared<Storage>(
                d_dofsInCell[0] * nQuadPointsInCell[0] * dim * d_maxCellBlock);
              size_type gradientParaCellSize =
                d_basisGradientParaCellQuadStorage->size();
              for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
                {
                  d_tmpGradientBlock->template copyFrom<memorySpace>(
                    d_basisGradientParaCellQuadStorage->data(),
                    gradientParaCellSize,
                    0,
                    gradientParaCellSize * iCell);
                }
            }
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

          d_JxWStorage = std::move(jxwQuadStorage);
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
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientParaCellQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisJacobianInvQuadStorage;
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
             basisStorageAttributesBoolMap,
             d_dealiiQuadratureRule);

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
          d_basisGradientParaCellQuadStorage =
            std::move(basisGradientParaCellQuadStorage);
          d_basisJacobianInvQuadStorage =
            std::move(basisJacobianInvQuadStorage);
          d_cellStartIdsBasisJacobianInvQuadStorage =
            cellStartIdsBasisJacobianInvQuadStorage;
          if (d_maxCellBlock != 1)
            {
              d_tmpGradientBlock = std::make_shared<Storage>(
                d_dofsInCell[0] * nQuadPointsInCell[0] * dim * d_maxCellBlock);
              size_type gradientParaCellSize =
                d_basisGradientParaCellQuadStorage->size();
              for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
                {
                  d_tmpGradientBlock->template copyFrom<memorySpace>(
                    d_basisGradientParaCellQuadStorage->data(),
                    gradientParaCellSize,
                    0,
                    gradientParaCellSize * iCell);
                }
            }
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreHessian)
            ->second)
        {
          d_basisHessianQuadStorage = std::move(basisHessianQuadStorage);
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

          d_JxWStorage = std::move(jxwQuadStorage);
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
          d_basisJacobianInvQuadStorage,
          d_cellStartIdsBasisJacobianInvQuadStorage,
          *d_basisGradientParaCellQuadStorage,
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

      std::shared_ptr<Storage> tmpGradientBlock = nullptr;
      if ((cellRange.second - cellRange.first) > d_maxCellBlock)
        {
          std::cout
            << "Warning: The cellBlockSize given to "
               "CFEBDSOnTheFlyComputeDealii.getBasisGradientDataInCellRange() "
               "is more than scratch storage. This may cause scratch initilization overheads.";
          tmpGradientBlock = std::make_shared<Storage>(
            d_dofsInCell[0] * d_nQuadPointsIncell[0] * dim *
            (cellRange.second - cellRange.first));
          size_type gradientParaCellSize =
            d_basisGradientParaCellQuadStorage->size();
          for (size_type iCell = 0;
               iCell < (cellRange.second - cellRange.first);
               ++iCell)
            {
              tmpGradientBlock->template copyFrom<memorySpace>(
                d_basisGradientParaCellQuadStorage->data(),
                gradientParaCellSize,
                0,
                gradientParaCellSize * iCell);
            }
        }
      else
        {
          tmpGradientBlock = (d_maxCellBlock != 1) ?
                               d_tmpGradientBlock :
                               d_basisGradientParaCellQuadStorage;
        }
      CFEBDSOnTheFlyComputeDealiiInternal::
        computeJacobianInvTimesGradPara<ValueTypeBasisData, memorySpace, dim>(
          cellRange,
          d_dofsInCell,
          d_nQuadPointsIncell,
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

      utils::throwException(
        (d_basisJacobianInvQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in CFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
      delete (d_basisJacobianInvQuadStorage).get();

      utils::throwException(
        (d_basisGradientParaCellQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in CFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
      delete (d_basisGradientParaCellQuadStorage).get();

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

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const dealii::Quadrature<dim> &
    CFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getDealiiQuadratureRule() const
    {
      return d_dealiiQuadratureRule;
    }

  } // namespace basis
} // namespace dftefe
