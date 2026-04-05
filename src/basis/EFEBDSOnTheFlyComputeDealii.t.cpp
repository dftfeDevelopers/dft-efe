
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
#include <linearAlgebra/Defaults.h>
#include <quadrature/QuadratureAttributes.h>
#include <basis/ParentToChildCellsManagerDealii.h>
#include <basis/CFEBDSOnTheFlyComputeDealii.h>
namespace dftefe
{
  namespace basis
  {
    namespace EFEBDSOnTheFlyComputeDealiiInternal
    {
      template <typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeJacobianInvTimesGradPara(
        std::pair<size_type, size_type> cellRange,
        const size_type                 classicalDofsInCell,
        const std::vector<size_type> &  dofsInCell,
        const std::vector<size_type> &  nQuadPointsInCell,
        // const std::shared_ptr<
        //   typename BasisDataStorage<ValueTypeBasisData,
        //   memorySpace>::Storage>
        //   &
        const ValueTypeBasisData *    basisJacobianInvQuadStorage,
        const std::vector<size_type> &cellStartIdsBasisJacobianInvQuadStorage,
        // typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        //   &
        const ValueTypeBasisData *                   tmpGradientBlock,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        // typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        //   &
        ValueTypeBasisData *basisGradientData)
      {
        size_type numMats = 0;
        for (size_type iCell = cellRange.first; iCell < cellRange.second;
             ++iCell)
          {
            for (size_type iQuad = 0; iQuad < nQuadPointsInCell[iCell]; ++iQuad)
              {
                numMats += 1;
              }
          }

        std::vector<char>      transA(numMats, 'N');
        std::vector<char>      transB(numMats, 'N');
        std::vector<size_type> mSizes(numMats, 0);
        std::vector<size_type> nSizes(numMats, 0);
        std::vector<size_type> kSizes(numMats, 0);
        std::vector<size_type> ldaSizes(numMats, 0);
        std::vector<size_type> ldbSizes(numMats, 0);
        std::vector<size_type> ldcSizes(numMats, 0);
        std::vector<size_type> strideA(numMats, 0);
        std::vector<size_type> strideB(numMats, 0);
        std::vector<size_type> strideC(numMats, 0);

        for (size_type iCell = cellRange.first; iCell < cellRange.second;
             ++iCell)
          {
            for (size_type iQuad = 0; iQuad < nQuadPointsInCell[iCell]; ++iQuad)
              {
                size_type index =
                  (iCell - cellRange.first) * nQuadPointsInCell[iCell] + iQuad;
                mSizes[index]   = classicalDofsInCell;
                nSizes[index]   = dim;
                kSizes[index]   = dim;
                ldaSizes[index] = mSizes[index];
                ldbSizes[index] = kSizes[index];
                ldcSizes[index] = dofsInCell[iCell];
                strideA[index]  = mSizes[index] * kSizes[index];
                strideB[index]  = kSizes[index] * nSizes[index];
                strideC[index]  = dofsInCell[iCell] * nSizes[index];
              }
          }

        ValueTypeBasisData alpha = 1.0;
        ValueTypeBasisData beta  = 0.0;

        const ValueTypeBasisData *B =
          basisJacobianInvQuadStorage /*->data()*/ +
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
          tmpGradientBlock /*.data()*/,
          ldaSizes.data(),
          B,
          ldbSizes.data(),
          beta,
          basisGradientData /*.data()*/,
          ldcSizes.data(),
          linAlgOpContext);
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
        dftefe::utils::MemoryStorage<ValueTypeBasisData,
                                     utils::MemorySpace::HOST> &basisValInCell,
        dftefe::utils::MemoryStorage<ValueTypeBasisData,
                                     utils::MemorySpace::HOST>
          &classicalComponentInQuadValues)
      {
        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(cellIndex) + 1),
                                            dim);
        size_type numEnrichmentIdsInCell =
          efeBDH->nCellDofs(cellIndex) - classicalDofsPerCell;

        ValueTypeBasisData *B = basisValInCell.data();
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
          B,
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
        dftefe::utils::MemoryStorage<ValueTypeBasisData,
                                     utils::MemorySpace::HOST> &basisGradInCell,
        dftefe::utils::MemoryStorage<ValueTypeBasisData,
                                     utils::MemorySpace::HOST>
          &classicalComponentInQuadGradients)
      {
        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(cellIndex) + 1),
                                            dim);
        size_type numEnrichmentIdsInCell =
          efeBDH->nCellDofs(cellIndex) - classicalDofsPerCell;

        // Do a gemm (\Sigma c_i N_i^classical)
        // and get the quad values in std::vector

        ValueTypeBasisData *B = basisGradInCell.data();

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
          B,
          classicalDofsPerCell,
          (ValueTypeBasisData)0.0,
          classicalComponentInQuadGradients
            .data(), // saved as cell->quad->dim->enrichid
          numEnrichmentIdsInCell,
          *linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);
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
        std::shared_ptr<const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpace,
                                                       dim>> efeBDH,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisParaCellClassQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisEnrichQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientParaCellClassQuadStorage,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientEnrichQuadStorage,
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
        std::vector<size_type> &cellStartIdsBasisJacobianInvQuadStorage,
        std::vector<size_type> &cellStartIdsBasisHessianQuadStorage,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
      {
        // for processors where there are no cells
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
          {
            dealiiUpdateFlags |= dealii::update_inverse_jacobians;
            // if (efeBDH->isOrthogonalized())
            //   {
            //     dealiiUpdateFlags |= dealii::update_gradients;
            //   }
          }
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreHessian)
              ->second)
          dealiiUpdateFlags |= dealii::update_hessians;

        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(0) + 1), dim);

        // NOTE: cellId 0 passed as we assume h-refine finite element mesh in
        // this function
        const size_type cellId = 0;
        // get real cell feValues
        dealii::FEValues<dim> dealiiFEValues(efeBDH->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags); // takes time

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
              efeBDH->getReferenceFE(cellId),
              dealiiQuadratureRule,
              dealiiUpdateFlagsPara); // takes time
            dealii::Triangulation<dim> referenceCell;
            dealii::GridGenerator::hyper_cube(referenceCell, 0., 1.);
            dealiiFEValuesPara->reinit(referenceCell.begin()); // takes time
          }

        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        size_type       dofsPerCell = efeBDH->nCellDofs(cellId);
        const size_type nQuadPointInCell =
          numCellsZero ? 0 :
                         quadratureRuleContainer->nCellQuadraturePoints(cellId);

        const size_type nDimSqxNumQuad = dim * dim * nQuadPointInCell;

        nQuadPointsInCell.resize(numLocallyOwnedCells, nQuadPointInCell);
        const std::vector<size_type> classDofsInCell(numLocallyOwnedCells,
                                                     classicalDofsPerCell);
        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisParaCellClassQuadStorageTmp(0),
          basisJacobianInvQuadStorageTmp(0),
          basisGradientParaCellClassQuadStorageTmp(0), tmpGradientInCell(0);
        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisHessianQuadStorageTmp(0);

        utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
          basisEnrichQuadStorageTmp(0), basisGradientEnrichQuadStorageTmp(0);

        size_type cellIndex                = 0;
        size_type basisValuesSize          = 0;
        size_type enrichQuadValStorageSize = 0;

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            basisValuesSize += nQuadPointInCell * dofsPerCell;
            enrichQuadValStorageSize +=
              (dofsPerCell - classicalDofsPerCell) * nQuadPointInCell;
            cellIndex++;
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            basisParaCellClassQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                classicalDofsPerCell * nQuadPointInCell);
            basisParaCellClassQuadStorageTmp.resize(classicalDofsPerCell *
                                                      nQuadPointInCell,
                                                    ValueTypeBasisData(0));
            basisEnrichQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                enrichQuadValStorageSize);
            basisEnrichQuadStorageTmp.resize(enrichQuadValStorageSize,
                                             ValueTypeBasisData(0));
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
            basisGradientParaCellClassQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                classicalDofsPerCell * nQuadPointInCell * dim);
            basisGradientParaCellClassQuadStorageTmp.resize(
              classicalDofsPerCell * nQuadPointInCell * dim);
            if (efeBDH->isOrthogonalized())
              {
                tmpGradientInCell.resize(classicalDofsPerCell *
                                         nQuadPointInCell * dim);
              }
            cellStartIdsBasisJacobianInvQuadStorage.resize(numLocallyOwnedCells,
                                                           0);
            basisGradientEnrichQuadStorage =
              std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                         memorySpace>::Storage>(
                enrichQuadValStorageSize * dim);
            basisGradientEnrichQuadStorageTmp.resize(enrichQuadValStorageSize *
                                                       dim,
                                                     ValueTypeBasisData(0));
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
                basisValuesSize * dim * dim);
            basisHessianQuadStorageTmp.resize(basisValuesSize * dim * dim,
                                              ValueTypeBasisData(0));
            cellStartIdsBasisHessianQuadStorage.resize(numLocallyOwnedCells, 0);
          }

        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
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

        cellIndex                            = 0;
        size_type cumulativeQuadPointsxnDofs = 0;

        // classical storage for quad and gradients , class + enriched for hessian
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            // Get classical dof numbers

            feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(
              *locallyOwnedCellIter);
            dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());

            std::vector<utils::Point> quadRealPointsVec =
              quadratureRuleContainer->getCellRealPoints(cellIndex);

            //
            // NOTE: For a h-refined (i.e., uniform FE order) mesh with the same
            // quadraure rule in all elements, the classical FE basis values
            // remain the same across as in the reference cell (unit
            // n-dimensional cell). Thus, to optimize on memory we only store
            // the classical FE basis values on the first cell
            //
            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                if (locallyOwnedCellIter == efeBDH->beginLocallyOwnedCells())
                  {
                    for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                         iNode++)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto it = basisParaCellClassQuadStorageTmp.begin() +
                                      qPoint * classicalDofsPerCell + iNode;
                            *it = dealiiFEValues.shape_value(iNode, qPoint);
                          }
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                cellStartIdsBasisJacobianInvQuadStorage[cellIndex] =
                  cellIndex * nDimSqxNumQuad;
                if (locallyOwnedCellIter == efeBDH->beginLocallyOwnedCells())
                  {
                    for (unsigned int iNode = 0; iNode < classicalDofsPerCell;
                         iNode++)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto shapeGrad =
                              dealiiFEValuesPara->shape_grad(iNode, qPoint);
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientParaCellClassQuadStorageTmp
                                    .begin() +
                                  qPoint * dim * classicalDofsPerCell +
                                  iDim * classicalDofsPerCell + iNode;
                                *it = shapeGrad[iDim];
                              }
                          }
                      }
                  }
                auto &mappingJacInv = dealiiFEValues.get_inverse_jacobians();
                size_type numJacobiansPerCell = nQuadPointInCell;
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
                  cumulativeQuadPointsxnDofs * dim * dim;
                const std::vector<double> &enrichHessAtQuadPts =
                  efeBDH->getEnrichmentHessian(cellIndex, quadRealPointsVec);
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
                                    *it =
                                      *(enrichHessAtQuadPts.data() +
                                        nQuadPointInCell * iNode * dim +
                                        qPoint * dim * dim + iDim * dim + jDim);
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
              basisParaCellClassQuadStorageTmp.size(),
              basisParaCellClassQuadStorage->data(),
              basisParaCellClassQuadStorageTmp.data());
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisGradientParaCellClassQuadStorageTmp.size(),
              basisGradientParaCellClassQuadStorage->data(),
              basisGradientParaCellClassQuadStorageTmp.data());

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

        // enriched storage for quad and gradients 
        
          bool storeEnrichValues = false, storeEnrichGrad = false;
          if (basisStorageAttributesBoolMap
                .find(BasisStorageAttributes::StoreValues)
                ->second)
            storeEnrichValues = true;

          if (basisStorageAttributesBoolMap
                .find(BasisStorageAttributes::StoreGradient)
                ->second)
            storeEnrichGrad = true;
      std::vector<double> quadValuesInAllCellsEnrichment, quadGradientsInAllCellsEnrichment;
        if(storeEnrichGrad || storeEnrichValues)
        {
          efeBDH->getEnrichmentClassicalInterface()->getEnrichmentDataInAllCellsAtQuadPts(
            storeEnrichValues,
            storeEnrichGrad,
            *quadratureRuleContainer,
            quadValuesInAllCellsEnrichment,
            quadGradientsInAllCellsEnrichment,
            linAlgOpContext);
        }

        cellIndex                            = 0;
        size_type cumulativeEnrichQuadxDof   = 0;
        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            // Get classical dof numbers

            size_type numEnrichmentIdsInCell =
              dofsPerCell - classicalDofsPerCell;

            utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
              classicalComponentInQuadValues(0);

            utils::MemoryStorage<ValueTypeBasisData, utils::MemorySpace::HOST>
              classicalComponentInQuadGradients(0);

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
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
                  efeBDH->getEnrichmentClassicalInterface()->getClassicalComponentCoeffsInCellOEFE(cellIndex);
              }

            //
            // NOTE: For a h-refined (i.e., uniform FE order) mesh with the same
            // quadraure rule in all elements, the classical FE basis values
            // remain the same across as in the reference cell (unit
            // n-dimensional cell). Thus, to optimize on memory we only store
            // the classical FE basis values on the first cell
            //
            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreValues)
                  ->second)
              {
                if (numEnrichmentIdsInCell > 0)
                  {
                    if (efeBDH->isOrthogonalized())
                      {
                        getClassicalComponentBasisValuesInCellAtQuadOEFE<
                          ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>(cellIndex,
                               nQuadPointInCell,
                               coeffsInCell,
                               efeBDH,
                               basisParaCellClassQuadStorageTmp,
                               classicalComponentInQuadValues);
                      }
                    ValueTypeBasisData *iter =
                      classicalComponentInQuadValues.data();
                    // const std::vector<double> &enrichValAtQuadPts =
                    //   efeBDH->getEnrichmentValue(cellIndex, quadRealPointsVec);
                    for (unsigned int iNode = 0; iNode < numEnrichmentIdsInCell;
                         iNode++)
                      {
                        // const std::vector<double> &enrichValAtQuadPts =
                        //   efeBDH->getEnrichmentValue(cellIndex,
                        //                              iNode,
                        //                              quadRealPointsVec);
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            // std::cout << efeBDH->getEnrichmentValue(
                            //     cellIndex,
                            //     iNode,
                            //     quadRealPointsVec[qPoint]) << " " <<
                            //     classicalComponentInQuadValues
                            //     [numEnrichmentIdsInCell * qPoint + iNode] <<
                            //     "\n";
                            *(basisEnrichQuadStorageTmp.data() +
                              cumulativeEnrichQuadxDof +
                              qPoint * numEnrichmentIdsInCell + iNode) =
                              *(quadValuesInAllCellsEnrichment.data() + cumulativeEnrichQuadxDof +
                                nQuadPointInCell * iNode + qPoint)
                              /*enrichValAtQuadPts[qPoint]*/
                              -
                              *(iter + numEnrichmentIdsInCell * qPoint + iNode);
                          }
                      }
                  }
              }

            if (basisStorageAttributesBoolMap
                  .find(BasisStorageAttributes::StoreGradient)
                  ->second)
              {
                cellStartIdsBasisJacobianInvQuadStorage[cellIndex] =
                  cellIndex * nDimSqxNumQuad;
  
                if (numEnrichmentIdsInCell > 0)
                  {
                    if (efeBDH->isOrthogonalized())
                      {
                        computeJacobianInvTimesGradPara<
                          ValueTypeBasisData,
                          utils::MemorySpace::HOST,
                          dim>(std::make_pair(cellIndex, cellIndex + 1),
                               classicalDofsPerCell,
                               classDofsInCell,
                               nQuadPointsInCell,
                               basisJacobianInvQuadStorageTmp.data(),
                               cellStartIdsBasisJacobianInvQuadStorage,
                               basisGradientParaCellClassQuadStorageTmp.data(),
                               *linearAlgebra::LinAlgOpContextDefaults::
                                 LINALG_OP_CONTXT_HOST,
                               tmpGradientInCell.begin());

                        getClassicalComponentBasisGradInCellAtQuadOEFE<
                          ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>(cellIndex,
                               nQuadPointInCell,
                               coeffsInCell,
                               efeBDH,
                               tmpGradientInCell,
                               classicalComponentInQuadGradients);
                      }
                    ValueTypeBasisData *iter =
                      classicalComponentInQuadGradients.data();
                    // const std::vector<double> &enrichGradAtQuadPts =
                    //   efeBDH->getEnrichmentDerivative(cellIndex,
                    //                                   quadRealPointsVec);
                    for (unsigned int iNode = 0; iNode < numEnrichmentIdsInCell;
                         iNode++)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            // auto shapeGrad = efeBDH->getEnrichmentDerivative(
                            //   cellIndex,
                            //   iNode,
                            //   quadRealPointsVec[qPoint]);
                            // enriched gradient function call
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientEnrichQuadStorageTmp.data() +
                                  cumulativeEnrichQuadxDof * dim +
                                  qPoint * dim * numEnrichmentIdsInCell +
                                  iDim * numEnrichmentIdsInCell + iNode;
                                *it = *(quadGradientsInAllCellsEnrichment.data() +
                                        cumulativeEnrichQuadxDof * dim + nQuadPointInCell * iNode * dim +
                                        qPoint * dim + iDim)
                                      /*shapeGrad[iDim]*/
                                      -
                                      *(iter +
                                        numEnrichmentIdsInCell * dim * qPoint +
                                        iDim * numEnrichmentIdsInCell + iNode);
                              }
                          }
                      }
                  }
              }
            cellIndex++;
            cumulativeEnrichQuadxDof +=
              numEnrichmentIdsInCell * nQuadPointInCell;
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreValues)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisEnrichQuadStorageTmp.size(),
              basisEnrichQuadStorage->data(),
              basisEnrichQuadStorageTmp.data());
          }

        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          {
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              basisGradientEnrichQuadStorageTmp.size(),
              basisGradientEnrichQuadStorage->data(),
              basisGradientEnrichQuadStorageTmp.data());
          }
      }
    } // namespace EFEBDSOnTheFlyComputeDealiiInternal

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::
      EFEBDSOnTheFlyComputeDealii(
        std::shared_ptr<const BasisDofHandler>      efeBDH,
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
      d_efeBDH            = std::dynamic_pointer_cast<
        const EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                       ValueTypeBasisData,
                                       memorySpace,
                                       dim>>(efeBDH);
      utils::throwException(
        d_efeBDH != nullptr,
        " Could not cast the FEBasisDofHandler to EFEBasisDofHandlerDealii in EFEBDSOnTheFlyComputeDealii");
      //      const size_type numConstraints  = constraintsVec.size();
      // const size_type numQuadRuleType = quadratureRuleAttributesVec.size();
      std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
        d_efeBDH->getDoFHandler();
      const size_type numLocallyOwnedCells = d_efeBDH->nLocallyOwnedCells();
      d_dofsInCell.resize(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        {
          d_dofsInCell[iCell] = d_efeBDH->nCellDofs(iCell);
        }
      d_tmpGradientBlock = nullptr;
      d_classialDofsInCell =
        utils::mathFunctions::sizeTypePow((d_efeBDH->getFEOrder(0) + 1), dim);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
        basisParaCellClassQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientParaCellClassQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisJacobianInvQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisEnrichQuadStorage;

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
                      std::make_shared<CFEBDSOnTheFlyComputeDealii<ValueTypeBasisData,
                                                                   ValueTypeBasisData,
                                                                   memorySpace,
                                                                   dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap,
                        d_maxCellBlock,
                        d_linAlgOpContext);

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
                      std::make_shared<CFEBDSOnTheFlyComputeDealii<ValueTypeBasisData,
                                                                   ValueTypeBasisData,
                                                                   memorySpace,
                                                                   dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap,
                        d_maxCellBlock,
                        d_linAlgOpContext);

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
                      std::make_shared<CFEBDSOnTheFlyComputeDealii<ValueTypeBasisData,
                                                                   ValueTypeBasisData,
                                                                   memorySpace,
                                                                   dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap,
                        d_maxCellBlock,
                        d_linAlgOpContext);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           basisAttrMap);
                  }
              }
      */
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisJacobianInvQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);
      EFEBDSOnTheFlyComputeDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisParaCellClassQuadStorage,
             basisEnrichQuadStorage,
             basisGradientParaCellClassQuadStorage,
             basisGradientEnrichQuadStorage,
             basisJacobianInvQuadStorage,
             basisHessianQuadStorage,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisJacobianInvQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap,
             d_linAlgOpContext);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisParaCellClassQuadStorage =
            std::move(basisParaCellClassQuadStorage);
          d_basisEnrichQuadStorage = std::move(basisEnrichQuadStorage);
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientParaCellClassQuadStorage =
            std::move(basisGradientParaCellClassQuadStorage);
          d_basisJacobianInvQuadStorage =
            std::move(basisJacobianInvQuadStorage);
          d_cellStartIdsBasisJacobianInvQuadStorage =
            cellStartIdsBasisJacobianInvQuadStorage;
          if (d_maxCellBlock != 1)
            {
              d_tmpGradientBlock = std::make_shared<Storage>(
                d_classialDofsInCell * nQuadPointsInCell[0] * dim *
                d_maxCellBlock);
              // size_type gradientParaCellSize =
              //   d_basisGradientParaCellClassQuadStorage->size();
              // for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
              //   {
              //     d_tmpGradientBlock->template copyFrom<memorySpace>(
              //       d_basisGradientParaCellClassQuadStorage->data(),
              //       gradientParaCellSize,
              //       0,
              //       gradientParaCellSize * iCell);
              //   }

           size_type cumulativeOffset = 0;
            for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
              {
                const size_type nQuad = nQuadPointsInCell[0];
                const size_type nDofs  = d_classialDofsInCell;
                linearAlgebra::blasLapack::stridedBlockCopy(
                    nQuad * dim,               // vecSize: number of quadrature points (slowest)
                    d_classialDofsInCell,          // numVec: number of classical DOFs (fastest)
                    d_classialDofsInCell,          // srcLeadingDim
                    0,                   // srcBlockStartId
                    d_classialDofsInCell,               // dstLeadingDim
                    0,                   // dstBlockStartId
                    d_basisGradientParaCellClassQuadStorage->data(), // src
                    d_tmpGradientBlock->data() + cumulativeOffset,   // dst
                    d_linAlgOpContext);

                cumulativeOffset += nDofs * nQuad * dim;
              }
            }
          d_basisGradientEnrichQuadStorage =
            std::move(basisGradientEnrichQuadStorage);
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
            "Basis Overlap not implemented in EFEBDSOnTheFlyComputeDealii");
        }
      d_nQuadPointsIncell = nQuadPointsInCell;

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradNiGradNj)
            ->second)
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "Basis GradNiGradNj not implemented in EFEBDSOnTheFlyComputeDealii");
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
        basisParaCellClassQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientParaCellClassQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisJacobianInvQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisHessianQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisGradientEnrichQuadStorage;
      std::shared_ptr<
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
        basisEnrichQuadStorage;

      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisJacobianInvQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

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
                      std::make_shared<CFEBDSOnTheFlyComputeDealii<ValueTypeBasisData,
                                                                   ValueTypeBasisData,
                                                                   memorySpace,
                                                                   dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap,
                        d_maxCellBlock,
                        d_linAlgOpContext);

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
                      std::make_shared<CFEBDSOnTheFlyComputeDealii<ValueTypeBasisData,
                                                                   ValueTypeBasisData,
                                                                   memorySpace,
                                                                   dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap,
                        d_maxCellBlock,
                        d_linAlgOpContext);

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
                      std::make_shared<CFEBDSOnTheFlyComputeDealii<ValueTypeBasisData,
                                                                   ValueTypeBasisData,
                                                                   memorySpace,
                                                                   dim>>(
                        d_efeBDH->getEnrichmentClassicalInterface()
                          ->getCFEBasisDofHandler(),
                        quadratureRuleAttributes,
                        basisAttrMap,
                        d_maxCellBlock,
                        d_linAlgOpContext);

                    cfeBasisDataStorage->evaluateBasisData(quadratureRuleAttributes,
                                                           d_quadratureRuleContainer,
                                                           basisAttrMap);
                  }
              }
      */
      EFEBDSOnTheFlyComputeDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisParaCellClassQuadStorage,
             basisEnrichQuadStorage,
             basisGradientParaCellClassQuadStorage,
             basisGradientEnrichQuadStorage,
             basisJacobianInvQuadStorage,
             basisHessianQuadStorage,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisJacobianInvQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap,
             d_linAlgOpContext);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisParaCellClassQuadStorage =
            std::move(basisParaCellClassQuadStorage);
          d_basisEnrichQuadStorage = std::move(basisEnrichQuadStorage);
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientParaCellClassQuadStorage =
            std::move(basisGradientParaCellClassQuadStorage);
          d_basisJacobianInvQuadStorage =
            std::move(basisJacobianInvQuadStorage);
          d_cellStartIdsBasisJacobianInvQuadStorage =
            cellStartIdsBasisJacobianInvQuadStorage;
          if (d_maxCellBlock != 1)
            {
              d_tmpGradientBlock = std::make_shared<Storage>(
                d_classialDofsInCell * nQuadPointsInCell[0] * dim *
                d_maxCellBlock);
              // size_type gradientParaCellSize =
              //   d_basisGradientParaCellClassQuadStorage->size();
              // for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
              //   {
              //     d_tmpGradientBlock->template copyFrom<memorySpace>(
              //       d_basisGradientParaCellClassQuadStorage->data(),
              //       gradientParaCellSize,
              //       0,
              //       gradientParaCellSize * iCell);
              //   }

           size_type cumulativeOffset = 0;
            for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
            {
                const size_type nQuad = nQuadPointsInCell[0];
                const size_type nDofs  = d_classialDofsInCell;
                linearAlgebra::blasLapack::stridedBlockCopy(
                    nQuad * dim,               // vecSize: number of quadrature points (slowest)
                    d_classialDofsInCell,          // numVec: number of classical DOFs (fastest)
                    d_classialDofsInCell,          // srcLeadingDim
                    0,                   // srcBlockStartId
                    d_classialDofsInCell,               // dstLeadingDim
                    0,                   // dstBlockStartId
                    d_basisGradientParaCellClassQuadStorage->data(), // src
                    d_tmpGradientBlock->data() + cumulativeOffset,   // dst
                    d_linAlgOpContext);

                cumulativeOffset += nDofs * nQuad * dim;
              }
            }
          d_basisGradientEnrichQuadStorage =
            std::move(basisGradientEnrichQuadStorage);
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
            "Basis GradNiGradNj not implemented in EFEBDSOnTheFlyComputeDealii");
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisDataInAllCells() const
    {
      utils::throwException(
        false,
        "getBasisDataInAllCells() is not implemented in EFEBDSOnTheFlyComputeDealii");
      return *d_tmpGradientBlock;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage &
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisGradientDataInAllCells() const
    {
      utils::throwException(
        false,
        "getBasisGradientDataInAllCells() is not implemented in EFEBDSOnTheFlyComputeDealii");
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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

      std::pair<size_type, size_type> cellPair(cellId, cellId + 1);

      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      const size_type               sizeToCopy =
        nQuadPointsInCell[cellId] * d_dofsInCell[cellId];
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(sizeToCopy);
      getBasisDataInCellRange(cellPair, returnValue);

      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBDSOnTheFlyComputeDealii<
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

      // size_type cumulativeOffsetEnrichQuad = 0;
      // for (size_type cellId = 0; cellId < cellRange.first; cellId++)
      //   {
      //     cumulativeOffsetEnrichQuad +=
      //       (d_dofsInCell[cellId] - d_classialDofsInCell) *
      //       d_nQuadPointsIncell[cellId];
      //   }
      // size_type cumulativeOffset = 0;
      // for (size_type cellId = cellRange.first; cellId < cellRange.second;
      //      cellId++)
      //   {
      //     for (size_type quadId = 0; quadId < d_nQuadPointsIncell[cellId];
      //          quadId++)
      //       {
      //         basisData.template copyFrom<memorySpace>(
      //           d_basisParaCellClassQuadStorage->data(),
      //           d_classialDofsInCell,
      //           d_classialDofsInCell * quadId,
      //           cumulativeOffset + d_dofsInCell[cellId] * quadId);

      //         if (d_dofsInCell[cellId] - d_classialDofsInCell > 0)
      //           basisData.template copyFrom<memorySpace>(
      //             d_basisEnrichQuadStorage->data(),
      //             d_dofsInCell[cellId] - d_classialDofsInCell,
      //             cumulativeOffsetEnrichQuad +
      //               (d_dofsInCell[cellId] - d_classialDofsInCell) * quadId,
      //             cumulativeOffset + d_dofsInCell[cellId] * quadId +
      //               d_classialDofsInCell);
      //       }
      //     cumulativeOffset +=
      //       d_dofsInCell[cellId] * d_nQuadPointsIncell[cellId];
      //     cumulativeOffsetEnrichQuad +=
      //       (d_dofsInCell[cellId] - d_classialDofsInCell) *
      //       d_nQuadPointsIncell[cellId];
      //   }

      size_type cumulativeOffsetEnrichQuad = 0;
      for (size_type cellId = 0; cellId < cellRange.first; cellId++)
        {
          cumulativeOffsetEnrichQuad +=
            (d_dofsInCell[cellId] - d_classialDofsInCell) *
            d_nQuadPointsIncell[cellId];
        }
        size_type cumulativeOffset = 0;
        for (size_type cellId = cellRange.first; cellId < cellRange.second; cellId++)
        {
            const size_type nQuad = d_nQuadPointsIncell[cellId];
            const size_type nDofs  = d_dofsInCell[cellId];
            const size_type nEnriched = nDofs - d_classialDofsInCell;

            linearAlgebra::blasLapack::stridedBlockCopy(
                nQuad,               // vecSize: number of quadrature points (slowest)
                d_classialDofsInCell,          // numVec: number of classical DOFs (fastest)
                d_classialDofsInCell,          // srcLeadingDim
                0,                   // srcBlockStartId
                nDofs,               // dstLeadingDim
                0,                   // dstBlockStartId
                d_basisParaCellClassQuadStorage->data(), // src
                basisData.data() + cumulativeOffset, // dst
                d_linAlgOpContext);

            if (nEnriched > 0)
            {
              linearAlgebra::blasLapack::stridedBlockCopy(
                  nQuad,               // vecSize: number of quadrature points
                  nEnriched,           // numVec: number of enriched DOFs
                  nEnriched,           // srcLeadingDim
                  0,                   // srcBlockStartId
                  nDofs,               // dstLeadingDim
                  d_classialDofsInCell,          // dstBlockStartId
                  d_basisEnrichQuadStorage->data() + cumulativeOffsetEnrichQuad, // src
                  basisData.data() + cumulativeOffset,             // dst
                  d_linAlgOpContext);
            }

            cumulativeOffset += nDofs * nQuad;
            cumulativeOffsetEnrichQuad += nEnriched * nQuad;
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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

      getBasisGradientDataInCellRange(cellPair, returnValue);

      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
               "EFEBDSOnTheFlyComputeDealii.getBasisGradientDataInCellRange() "
               "is more than scratch storage. This may cause scratch initilization overheads.";
          tmpGradientBlock = std::make_shared<Storage>(
            d_dofsInCell[0] * d_nQuadPointsIncell[0] * dim *
            (cellRange.second - cellRange.first));

          // size_type gradientParaCellSize =
          //   d_basisGradientParaCellClassQuadStorage->size();
          // for (size_type iCell = 0;
          //      iCell < (cellRange.second - cellRange.first);
          //      ++iCell)
          //   {
          //     tmpGradientBlock->template copyFrom<memorySpace>(
          //       d_basisGradientParaCellClassQuadStorage->data(),
          //       gradientParaCellSize,
          //       0,
          //       gradientParaCellSize * iCell);
          //   }

           size_type cumulativeOffset = 0;
          for (size_type cellId = cellRange.first; cellId < cellRange.second; cellId++)
          {
                const size_type nQuad = d_nQuadPointsIncell[cellId];
                const size_type nDofs  = d_classialDofsInCell;
                linearAlgebra::blasLapack::stridedBlockCopy(
                    nQuad * dim,               // vecSize: number of quadrature points (slowest)
                    d_classialDofsInCell,          // numVec: number of classical DOFs (fastest)
                    d_classialDofsInCell,          // srcLeadingDim
                    0,                   // srcBlockStartId
                    d_classialDofsInCell,               // dstLeadingDim
                    0,                   // dstBlockStartId
                    d_basisGradientParaCellClassQuadStorage->data(), // src
                    tmpGradientBlock->data() + cumulativeOffset,   // dst
                    d_linAlgOpContext);

                cumulativeOffset += nDofs * nQuad * dim;
            }
        }
      else
        {
          tmpGradientBlock = (d_maxCellBlock != 1) ?
                               d_tmpGradientBlock :
                               d_basisGradientParaCellClassQuadStorage;
        }
      EFEBDSOnTheFlyComputeDealiiInternal::
        computeJacobianInvTimesGradPara<ValueTypeBasisData, memorySpace, dim>(
          cellRange,
          d_classialDofsInCell,
          d_dofsInCell,
          d_nQuadPointsIncell,
          d_basisJacobianInvQuadStorage->data(),
          d_cellStartIdsBasisJacobianInvQuadStorage,
          tmpGradientBlock->data(),
          d_linAlgOpContext,
          basisGradientData.data());

      size_type cumulativeOffsetEnrichQuad = 0;
      for (size_type cellId = 0; cellId < cellRange.first; cellId++)
        {
          cumulativeOffsetEnrichQuad +=
            (d_dofsInCell[cellId] - d_classialDofsInCell) *
            d_nQuadPointsIncell[cellId] * dim;
        }
      // size_type cumulativeOffset = 0;
      // for (size_type cellId = cellRange.first; cellId < cellRange.second;
      //      cellId++)
      //   {
      //     if (d_dofsInCell[cellId] - d_classialDofsInCell > 0)
      //       {
      //         for (size_type quadId = 0; quadId < d_nQuadPointsIncell[cellId];
      //              quadId++)
      //           {
      //             for (size_type iDim = 0; iDim < dim; iDim++)
      //               {
      //                 basisGradientData.template copyFrom<memorySpace>(
      //                   d_basisGradientEnrichQuadStorage->data(),
      //                   (d_dofsInCell[cellId] - d_classialDofsInCell),
      //                   cumulativeOffsetEnrichQuad +
      //                     (d_dofsInCell[cellId] - d_classialDofsInCell) * dim *
      //                       quadId +
      //                     iDim * (d_dofsInCell[cellId] - d_classialDofsInCell),
      //                   cumulativeOffset + d_dofsInCell[cellId] * dim * quadId +
      //                     d_dofsInCell[cellId] * iDim + d_classialDofsInCell);
      //               }
      //           }
      //       }
      //     cumulativeOffset +=
      //       d_dofsInCell[cellId] * d_nQuadPointsIncell[cellId] * dim;
      //     cumulativeOffsetEnrichQuad +=
      //       (d_dofsInCell[cellId] - d_classialDofsInCell) *
      //       d_nQuadPointsIncell[cellId] * dim;
      //   }

      size_type cumulativeOffset = 0;
      for (size_type cellId = cellRange.first; cellId < cellRange.second; cellId++)
      {
        const size_type nDofs = d_dofsInCell[cellId];
        const size_type nEnriched = nDofs - d_classialDofsInCell;
        const size_type nQuad = d_nQuadPointsIncell[cellId];
        if (nEnriched > 0)
        {
          linearAlgebra::blasLapack::stridedBlockCopy(
              nQuad * dim,          // vecSize: quad * dim (slowest)
              nEnriched,            // numVec: DOFs (fastest)
              nEnriched,            // srcLeadingDim: DOFs per quad/dim
              0,                    // srcBlockStartId
              nDofs,          // dstLeadingDim: total DOFs * dim
              d_classialDofsInCell, // dstBlockStartId: after classical DOFs
              d_basisGradientEnrichQuadStorage->data() + cumulativeOffsetEnrichQuad, // src
              basisGradientData.data() + cumulativeOffset,   // dst
              d_linAlgOpContext);
        }
        cumulativeOffset += nDofs * nQuad * dim;
        cumulativeOffsetEnrichQuad += nEnriched * nQuad * dim;
      }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
    EFEBDSOnTheFlyComputeDealii<
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
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        basisQuadStorage(getBasisDataInCell(cellId));

      const std::vector<size_type> &nQuadPointsInCell = d_nQuadPointsIncell;
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
        returnValue(1);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(
        1,
        returnValue.data(),
        basisQuadStorage.data() + quadPointId * d_dofsInCell[cellId] + basisId);
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBDSOnTheFlyComputeDealii<
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
    EFEBDSOnTheFlyComputeDealii<
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisOverlapInAllCells() const
    {
      utils::throwException<utils::InvalidArgument>(
        false, "Basis Overlap not implemented in EFEBDSOnTheFlyComputeDealii");
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisOverlapInCell(const size_type
                                                              cellId) const
    {
      utils::throwException<utils::InvalidArgument>(
        false, "Basis Overlap not implemented in EFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisOverlap(const size_type cellId,
                                                      const size_type basisId1,
                                                      const size_type basisId2)
      const
    {
      utils::throwException<utils::InvalidArgument>(
        false, "Basis Overlap not implemented in EFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::deleteBasisData()
    {
      utils::throwException(
        (d_basisParaCellClassQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
      delete (d_basisParaCellClassQuadStorage).get();

      utils::throwException(
        (d_basisJacobianInvQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
      delete (d_basisJacobianInvQuadStorage).get();

      utils::throwException(
        (d_basisGradientParaCellClassQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
      delete (d_basisGradientParaCellClassQuadStorage).get();

      utils::throwException(
        (d_basisHessianQuadStorage).use_count() == 1,
        "More than one owner for the basis quadrature storage found in EFEBDSOnTheFlyComputeDealii. Not safe to delete it.");
      delete (d_basisHessianQuadStorage).get();

      d_tmpGradientBlock->resize(0);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisDataInCell(const size_type cellId,
                                                         const size_type
                                                           basisId) const
    {
      utils::throwException(
        false,
        "getBasisDataInCell() for a given basisId is not implemented in EFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBDSOnTheFlyComputeDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisGradientDataInCell(const size_type cellId,
                                       const size_type basisId) const
    {
      utils::throwException(
        false,
        "getBasisGradientDataInCell() for a given basisId is not implemented in EFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage
    EFEBDSOnTheFlyComputeDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisHessianDataInCell(const size_type cellId,
                                      const size_type basisId) const
    {
      utils::throwException(
        false,
        "getBasisHessianDataInCell() for a given basisId is not implemented in EFEBDSOnTheFlyComputeDealii");
      typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage dummy(
        0);
      return dummy;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const quadrature::QuadratureRuleContainer>
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisGradNiGradNjInCell(const size_type
                                                                   cellId) const
    {
      utils::throwException<utils::InvalidArgument>(
        false,
        "Basis GradNiGradNj not implemented in EFEBDSOnTheFlyComputeDealii");
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisGradNiGradNjInAllCells() const
    {
      utils::throwException<utils::InvalidArgument>(
        false,
        "Basis GradNiGradNj not implemented in EFEBDSOnTheFlyComputeDealii");
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
    EFEBDSOnTheFlyComputeDealii<ValueTypeBasisCoeff,
                                ValueTypeBasisData,
                                memorySpace,
                                dim>::getBasisDofHandler() const
    {
      return d_efeBDH;
    }
  } // namespace basis
} // namespace dftefe
