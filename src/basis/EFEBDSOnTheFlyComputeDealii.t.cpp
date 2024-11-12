
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
        size_type numMats = 0;
        for (size_type iCell = cellRange.first; iCell < cellRange.second;
             ++iCell)
          {
            for (size_type iQuad = 0; iQuad < nQuadPointsInCell[iCell]; ++iQuad)
              {
                numMats += 1;
              }
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
          memoryTransfer;

        std::vector<linearAlgebra::blasLapack::Op> transA(
          numMats, linearAlgebra::blasLapack::Op::NoTrans);
        std::vector<linearAlgebra::blasLapack::Op> transB(
          numMats, linearAlgebra::blasLapack::Op::NoTrans);
        std::vector<size_type> mSizesTmp(numMats, 0);
        std::vector<size_type> nSizesTmp(numMats, 0);
        std::vector<size_type> kSizesTmp(numMats, 0);
        std::vector<size_type> ldaSizesTmp(numMats, 0);
        std::vector<size_type> ldbSizesTmp(numMats, 0);
        std::vector<size_type> ldcSizesTmp(numMats, 0);
        std::vector<size_type> strideATmp(numMats, 0);
        std::vector<size_type> strideBTmp(numMats, 0);
        std::vector<size_type> strideCTmp(numMats, 0);

        for (size_type iCell = cellRange.first; iCell < cellRange.second;
             ++iCell)
          {
            for (size_type iQuad = 0; iQuad < nQuadPointsInCell[iCell]; ++iQuad)
              {
                size_type index =
                  (iCell - cellRange.first) * nQuadPointsInCell[iCell] + iQuad;
                mSizesTmp[index]   = classicalDofsInCell;
                nSizesTmp[index]   = dim;
                kSizesTmp[index]   = dim;
                ldaSizesTmp[index] = mSizesTmp[index];
                ldbSizesTmp[index] = kSizesTmp[index];
                ldcSizesTmp[index] = dofsInCell[iCell];
                strideATmp[index]  = mSizesTmp[index] * kSizesTmp[index];
                strideBTmp[index]  = kSizesTmp[index] * nSizesTmp[index];
                strideCTmp[index]  = dofsInCell[iCell] * nSizesTmp[index];
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
          linearAlgebra::blasLapack::Layout::ColMajor,
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
        std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                         cfeBasisDataStorage,
        std::vector<ValueTypeBasisData> &classicalComponentInQuadValues)
      {
        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(cellIndex) + 1),
                                            dim);
        size_type numEnrichmentIdsInCell =
          efeBDH->nCellDofs(cellIndex) - classicalDofsPerCell;

        dftefe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
          basisValInCell = cfeBasisDataStorage->getBasisDataInCell(cellIndex);

        ValueTypeBasisData *B = basisValInCell.data();
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
          B,
          classicalDofsPerCell,
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
        std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                         cfeBasisDataStorage,
        std::vector<ValueTypeBasisData> &classicalComponentInQuadGradients)
      {
        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(cellIndex) + 1),
                                            dim);
        size_type numEnrichmentIdsInCell =
          efeBDH->nCellDofs(cellIndex) - classicalDofsPerCell;

        // saved as cell->quad->dim->node
        dftefe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
          basisGradInCell =
            cfeBasisDataStorage->getBasisGradientDataInCell(cellIndex);

        // Do a gemm (\Sigma c_i N_i^classical)
        // and get the quad values in std::vector

        ValueTypeBasisData *B = basisGradInCell.data();

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
          B,
          classicalDofsPerCell,
          (ValueTypeBasisData)0.0,
          classicalComponentInQuadGradients
            .data(), // saved as cell->quad->dim->enrichid
          numEnrichmentIdsInCell,
          *efeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext());
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
        std::unordered_map<
          size_type,
          std::shared_ptr<typename BasisDataStorage<ValueTypeBasisData,
                                                    memorySpace>::Storage>>
          &basisEnrichQuadStorageMap,
        std::shared_ptr<
          typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
          &basisGradientParaCellClassQuadStorage,
        std::unordered_map<
          size_type,
          std::shared_ptr<typename BasisDataStorage<ValueTypeBasisData,
                                                    memorySpace>::Storage>>
          &basisGradientEnrichQuadStorageMap,
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
        std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          cfeBasisDataStorage = nullptr)
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

        size_type classicalDofsPerCell =
          utils::mathFunctions::sizeTypePow((efeBDH->getFEOrder(0) + 1), dim);

        // NOTE: cellId 0 passed as we assume h-refine finite element mesh in
        // this function
        const size_type cellId = 0;
        // get real cell feValues
        dealii::FEValues<dim> dealiiFEValues(efeBDH->getReferenceFE(cellId),
                                             dealiiQuadratureRule,
                                             dealiiUpdateFlags);

        dealii::UpdateFlags dealiiUpdateFlagsPara;
        if (basisStorageAttributesBoolMap
              .find(BasisStorageAttributes::StoreGradient)
              ->second)
          dealiiUpdateFlagsPara = dealii::update_gradients;
        // This is for getting the gradient in parametric cell
        dealii::FEValues<dim> dealiiFEValuesPara(efeBDH->getReferenceFE(cellId),
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

        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        // NOTE: cellId 0 passed as we assume only H refined in this function
        size_type       dofsPerCell = efeBDH->nCellDofs(cellId);
        const size_type nQuadPointInCell =
          numCellsZero ? 0 :
                         quadratureRuleContainer->nCellQuadraturePoints(cellId);

        const size_type nDimSqxNumQuad = dim * dim * nQuadPointInCell;

        nQuadPointsInCell.resize(numLocallyOwnedCells, nQuadPointInCell);
        std::vector<ValueTypeBasisData> basisParaCellClassQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisJacobianInvQuadStorageTmp(0);
        std::vector<ValueTypeBasisData>
          basisGradientParaCellClassQuadStorageTmp(0);
        std::vector<ValueTypeBasisData> basisHessianQuadStorageTmp(0);

        std::unordered_map<size_type, std::vector<ValueTypeBasisData>>
          basisEnrichQuadStorageMapTmp, basisGradientEnrichQuadStorageMapTmp;

        size_type cellIndex       = 0;
        size_type basisValuesSize = 0;

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = efeBDH->nCellDofs(cellIndex);
            basisValuesSize += nQuadPointInCell * dofsPerCell;
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

            std::vector<size_type> vecClassicalLocalNodeId(0);


            size_type numEnrichmentIdsInCell =
              dofsPerCell - classicalDofsPerCell;

            std::vector<ValueTypeBasisData> classicalComponentInQuadValues(0);

            std::vector<ValueTypeBasisData> classicalComponentInQuadGradients(
              0);

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


            if (efeBDH->isOrthogonalized() && numEnrichmentIdsInCell > 0)
              {
                std::vector<ValueTypeBasisData> coeffsInCell =
                  getClassicalComponentCoeffsInCellOEFE<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>(cellIndex, efeBDH);

                if (basisStorageAttributesBoolMap
                      .find(BasisStorageAttributes::StoreValues)
                      ->second)
                  {
                    getClassicalComponentBasisValuesInCellAtQuadOEFE<
                      ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>(cellIndex,
                           nQuadPointInCell,
                           coeffsInCell,
                           efeBDH,
                           cfeBasisDataStorage,
                           classicalComponentInQuadValues);
                  }

                if (basisStorageAttributesBoolMap
                      .find(BasisStorageAttributes::StoreGradient)
                      ->second)
                  {
                    getClassicalComponentBasisGradInCellAtQuadOEFE<
                      ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      memorySpace,
                      dim>(cellIndex,
                           nQuadPointInCell,
                           coeffsInCell,
                           efeBDH,
                           cfeBasisDataStorage,
                           classicalComponentInQuadGradients);
                  }
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

                if (numEnrichmentIdsInCell > 0)
                  {
                    basisEnrichQuadStorageMapTmp[cellIndex].resize(
                      nQuadPointInCell * numEnrichmentIdsInCell);
                    for (unsigned int iNode = 0; iNode < numEnrichmentIdsInCell;
                         iNode++)
                      {
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
                            *(basisEnrichQuadStorageMapTmp[cellIndex].data() +
                              qPoint * numEnrichmentIdsInCell + iNode) =
                              efeBDH->getEnrichmentValue(
                                cellIndex, iNode, quadRealPointsVec[qPoint]) -
                              classicalComponentInQuadValues
                                [numEnrichmentIdsInCell * qPoint + iNode];
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
                              dealiiFEValuesPara.shape_grad(iNode, qPoint);
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

                if (numEnrichmentIdsInCell > 0)
                  {
                    basisGradientEnrichQuadStorageMapTmp[cellIndex].resize(
                      dim * nQuadPointInCell * numEnrichmentIdsInCell);
                    for (unsigned int iNode = 0; iNode < numEnrichmentIdsInCell;
                         iNode++)
                      {
                        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                             qPoint++)
                          {
                            auto shapeGrad = efeBDH->getEnrichmentDerivative(
                              cellIndex, iNode, quadRealPointsVec[qPoint]);
                            // enriched gradient function call
                            for (unsigned int iDim = 0; iDim < dim; iDim++)
                              {
                                auto it =
                                  basisGradientEnrichQuadStorageMapTmp
                                    [cellIndex]
                                      .data() +
                                  qPoint * dim * numEnrichmentIdsInCell +
                                  iDim * numEnrichmentIdsInCell + iNode;
                                *it = shapeGrad[iDim] -
                                      classicalComponentInQuadGradients
                                        [numEnrichmentIdsInCell * dim * qPoint +
                                         iDim * numEnrichmentIdsInCell + iNode];
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
              basisParaCellClassQuadStorageTmp.size(),
              basisParaCellClassQuadStorage->data(),
              basisParaCellClassQuadStorageTmp.data());

            for (auto &it : basisEnrichQuadStorageMapTmp)
              {
                basisEnrichQuadStorageMap[it.first] = std::make_shared<
                  typename BasisDataStorage<ValueTypeBasisData,
                                            memorySpace>::Storage>(
                  it.second.size());
                utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                  copy(it.second.size(),
                       basisEnrichQuadStorageMap[it.first]->data(),
                       it.second.data());
              }
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

            for (auto &it : basisGradientEnrichQuadStorageMapTmp)
              {
                basisGradientEnrichQuadStorageMap[it.first] = std::make_shared<
                  typename BasisDataStorage<ValueTypeBasisData,
                                            memorySpace>::Storage>(
                  it.second.size());
                utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                  copy(it.second.size(),
                       basisGradientEnrichQuadStorageMap[it.first]->data(),
                       it.second.data());
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

      size_type nTotalEnrichmentIds =
        d_efeBDH->getEnrichmentIdsPartition()->nTotalEnrichmentIds();

      std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        cfeBasisDataStorage = nullptr;

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
              basisAttrMap[BasisStorageAttributes::StoreValues]       = true;
              basisAttrMap[BasisStorageAttributes::StoreGradient]     = false;
              basisAttrMap[BasisStorageAttributes::StoreHessian]      = false;
              basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
              basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] = false;
              basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


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
              basisAttrMap[BasisStorageAttributes::StoreValues]       = false;
              basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
              basisAttrMap[BasisStorageAttributes::StoreHessian]      = false;
              basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
              basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] = false;
              basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


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
              basisAttrMap[BasisStorageAttributes::StoreValues]       = true;
              basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
              basisAttrMap[BasisStorageAttributes::StoreHessian]      = false;
              basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
              basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] = false;
              basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


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

      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisJacobianInvQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);
      EFEBDSOnTheFlyComputeDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisParaCellClassQuadStorage,
             d_basisEnrichQuadStorageMap,
             basisGradientParaCellClassQuadStorage,
             d_basisGradientEnrichQuadStorageMap,
             basisJacobianInvQuadStorage,
             basisHessianQuadStorage,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisJacobianInvQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap,
             cfeBasisDataStorage);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisParaCellClassQuadStorage = basisParaCellClassQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientParaCellClassQuadStorage =
            basisGradientParaCellClassQuadStorage;
          d_basisJacobianInvQuadStorage = basisJacobianInvQuadStorage;
          d_cellStartIdsBasisJacobianInvQuadStorage =
            cellStartIdsBasisJacobianInvQuadStorage;
          d_tmpGradientBlock = std::make_shared<Storage>(
            d_classialDofsInCell * nQuadPointsInCell[0] * dim * d_maxCellBlock);
          size_type gradientParaCellSize =
            basisGradientParaCellClassQuadStorage->size();
          for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
            {
              d_tmpGradientBlock->template copyFrom<memorySpace>(
                basisGradientParaCellClassQuadStorage->data(),
                gradientParaCellSize,
                0,
                gradientParaCellSize * iCell);
            }
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

          d_JxWStorage = jxwQuadStorage;
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
      std::vector<size_type> nQuadPointsInCell(0);
      std::vector<size_type> cellStartIdsBasisJacobianInvQuadStorage(0);
      std::vector<size_type> cellStartIdsBasisHessianQuadStorage(0);

      std::shared_ptr<FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        cfeBasisDataStorage = nullptr;

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
              basisAttrMap[BasisStorageAttributes::StoreValues]       = true;
              basisAttrMap[BasisStorageAttributes::StoreGradient]     = false;
              basisAttrMap[BasisStorageAttributes::StoreHessian]      = false;
              basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
              basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] = false;
              basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


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
              basisAttrMap[BasisStorageAttributes::StoreValues]       = false;
              basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
              basisAttrMap[BasisStorageAttributes::StoreHessian]      = false;
              basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
              basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] = false;
              basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


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
              basisAttrMap[BasisStorageAttributes::StoreValues]       = true;
              basisAttrMap[BasisStorageAttributes::StoreGradient]     = true;
              basisAttrMap[BasisStorageAttributes::StoreHessian]      = false;
              basisAttrMap[BasisStorageAttributes::StoreOverlap]      = false;
              basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] = false;
              basisAttrMap[BasisStorageAttributes::StoreJxW]          = false;


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

      EFEBDSOnTheFlyComputeDealiiInternal::storeValuesHRefinedSameQuadEveryCell<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_efeBDH,
             basisParaCellClassQuadStorage,
             d_basisEnrichQuadStorageMap,
             basisGradientParaCellClassQuadStorage,
             d_basisGradientEnrichQuadStorageMap,
             basisJacobianInvQuadStorage,
             basisHessianQuadStorage,
             quadratureRuleAttributes,
             d_quadratureRuleContainer,
             nQuadPointsInCell,
             cellStartIdsBasisJacobianInvQuadStorage,
             cellStartIdsBasisHessianQuadStorage,
             basisStorageAttributesBoolMap,
             cfeBasisDataStorage);

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreValues)
            ->second)
        {
          d_basisParaCellClassQuadStorage = basisParaCellClassQuadStorage;
        }

      if (basisStorageAttributesBoolMap
            .find(BasisStorageAttributes::StoreGradient)
            ->second)
        {
          d_basisGradientParaCellClassQuadStorage =
            basisGradientParaCellClassQuadStorage;
          d_basisJacobianInvQuadStorage = basisJacobianInvQuadStorage;
          d_cellStartIdsBasisJacobianInvQuadStorage =
            cellStartIdsBasisJacobianInvQuadStorage;
          d_tmpGradientBlock = std::make_shared<Storage>(
            d_classialDofsInCell * nQuadPointsInCell[0] * dim * d_maxCellBlock);
          size_type gradientParaCellSize =
            basisGradientParaCellClassQuadStorage->size();
          for (size_type iCell = 0; iCell < d_maxCellBlock; ++iCell)
            {
              d_tmpGradientBlock->template copyFrom<memorySpace>(
                basisGradientParaCellClassQuadStorage->data(),
                gradientParaCellSize,
                0,
                gradientParaCellSize * iCell);
            }
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

          d_JxWStorage = jxwQuadStorage;
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

      size_type cumulativeOffset = 0;
      for (size_type cellId = cellRange.first; cellId < cellRange.second;
           cellId++)
        {
          auto iter = d_basisEnrichQuadStorageMap.find(cellId);
          for (size_type quadId = 0; quadId < d_nQuadPointsIncell[cellId];
               quadId++)
            {
              basisData.template copyFrom<memorySpace>(
                d_basisParaCellClassQuadStorage->data(),
                d_classialDofsInCell,
                d_classialDofsInCell * quadId,
                cumulativeOffset + d_dofsInCell[cellId] * quadId);
              if (iter != d_basisEnrichQuadStorageMap.end())
                {
                  basisData.template copyFrom<memorySpace>(
                    iter->second->data(),
                    d_dofsInCell[cellId] - d_classialDofsInCell,
                    (d_dofsInCell[cellId] - d_classialDofsInCell) * quadId,
                    cumulativeOffset + d_dofsInCell[cellId] * quadId +
                      d_classialDofsInCell);
                }
            }
          cumulativeOffset +=
            d_dofsInCell[cellId] * d_nQuadPointsIncell[cellId];
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
          size_type gradientParaCellSize =
            d_basisGradientParaCellClassQuadStorage->size();
          for (size_type iCell = 0;
               iCell < (cellRange.second - cellRange.first);
               ++iCell)
            {
              tmpGradientBlock->template copyFrom<memorySpace>(
                d_basisGradientParaCellClassQuadStorage->data(),
                gradientParaCellSize,
                0,
                gradientParaCellSize * iCell);
            }
        }
      else
        {
          tmpGradientBlock = d_tmpGradientBlock;
        }
      EFEBDSOnTheFlyComputeDealiiInternal::
        computeJacobianInvTimesGradPara<ValueTypeBasisData, memorySpace, dim>(
          cellRange,
          d_classialDofsInCell,
          d_dofsInCell,
          d_nQuadPointsIncell,
          d_basisJacobianInvQuadStorage,
          d_cellStartIdsBasisJacobianInvQuadStorage,
          *tmpGradientBlock,
          d_linAlgOpContext,
          basisGradientData);

      size_type cumulativeOffset = 0;
      for (size_type cellId = cellRange.first; cellId < cellRange.second;
           cellId++)
        {
          auto iter = d_basisGradientEnrichQuadStorageMap.find(cellId);
          if (iter != d_basisGradientEnrichQuadStorageMap.end())
            {
              for (size_type quadId = 0; quadId < d_nQuadPointsIncell[cellId];
                   quadId++)
                {
                  for (size_type iDim = 0; iDim < dim; iDim++)
                    {
                      basisGradientData.template copyFrom<memorySpace>(
                        iter->second->data(),
                        (d_dofsInCell[cellId] - d_classialDofsInCell),
                        (d_dofsInCell[cellId] - d_classialDofsInCell) * dim *
                            quadId +
                          iDim * (d_dofsInCell[cellId] - d_classialDofsInCell),
                        cumulativeOffset + d_dofsInCell[cellId] * dim * quadId +
                          d_dofsInCell[cellId] * iDim + d_classialDofsInCell);
                    }
                }
            }
          cumulativeOffset +=
            d_dofsInCell[cellId] * d_nQuadPointsIncell[cellId] * dim;
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
