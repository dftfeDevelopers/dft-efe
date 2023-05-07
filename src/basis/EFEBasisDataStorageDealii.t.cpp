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

namespace dftefe
{
    namespace basis
    {
        namespace EFEBasisDataStorgeDealiiInternal
        {
            //This class stores the enriched FE basis data for a h-refined
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
                std::vector<size_type> &                    nQuadPointsInCell,
                std::vector<size_type> &cellStartIdsBasisQuadStorage,
                std::vector<size_type> &cellStartIdsBasisGradientQuadStorage,
                std::vector<size_type> &cellStartIdsBasisHessianQuadStorage,
                const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
            {
                utils::throwException(
                    false,
                    "This feature is not yet implemented in dft-efe for an enriched basis."
                    "Contact the developers for this.");
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
                const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes)

            {
                utils::throwException(
                    false,
                    "This feature is not yet implemented in dft-efe for an enriched basis."
                    "Contact the developers for this.");
            }

            template <typename ValueTypeBasisData,
                      utils::MemorySpace memorySpace    
                      size_type         dim>
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
                        &basisOverlap,
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
                const quuadrature::QuadratureFamily quadratureFamily = 
                    quadratureRuleAttributes.getQuadratureFamily();
                if((quadratureFamily != quadrature::QuadratureFamily::GAUSS_VARIABLE)||
                    (quadratureFamily != quadrature::QuadratureFamily::GLL_VARIABLE) ||
                    (quadratureFamily != quadrature::QuadratureFamily::ADAPTIVE))
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

                if(basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreGradient)->second)
                {
                    dealiiUpdateFlags |= dealii::update_gradients;
                }
                if (basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreHessian)->second)
                {
                    dealiiUpdateFlags |= dealii::update_hessians;
                }
                const size_type feOrder = efeBM->getFEOrder(0);
                const size_type numLocallyOwnedCells = efeBM->nLocallyOwnedCells();
                size_type dofsPerCell = 0;

                // Create temporary data structure for Value Storage
                std::vector<ValueTypeBasisData> basisQuadStorageTmp(0);
                std::vector<ValueTypeBasisData> basisGradientQuadStorageTmp(0);
                std::vector<ValueTypeBasisData> basisHessianQuadStorageTmp(0);
                std::vector<ValueTypeBasisData> basisOverlapTmp(0);

                // Find total quadpoints in the processor
                nQuadPointsInCell.resize(numLocallyOwnedCells, 0);
                const size_type nTotalQuadPoints =
                quadratureRuleContainer->nQuadraturePoints();

                size_type cellIndex = 0;
                size_type basisValuesSize = 0;
                size_type basisOverlapSize = 0;
                size_type nQuadPointInCell = 0;

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
                    basisQuadStorageTmp.resize(basisValuesSize,
                                            ValueTypeBasisData(0));
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

                // Initialize the host and tmp vector for storing basis overlap values in 
                // a flattened array. 
                basisOverlap = std::make_shared<
                typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>(
                basisOverlapSize);
                basisOverlapTmp.resize(basisOverlapSize,
                                    ValueTypeBasisData(0));

                auto basisQuadStorageTmpIter = basisQuadStorageTmp.begin();
                auto basisGradientQuadStorageTmpIter =
                basisGradientQuadStorageTmp.begin();
                auto basisHessianQuadStorageTmpIter =
                basisHessianQuadStorageTmp.begin();
                auto      basisOverlapTmpIter = basisOverlapTmp.begin();

                // Init cell iters and storage iters                
                auto locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
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
                    //Get classical dof numbers
                    size_type classicalDofsPerCell =  
                        utils::mathFunctions::sizeTypePow((efeBM->getFEOrder(cellIndex)+1),dim);

                    nQuadPointInCell = nQuadPointsInCell[cellIndex];

                    // get the parametric points and jxw in each cell according to
                    // the attribute.
                    const std::vector<dftefe::utils::Point> &cellParametricQuadPoints =
                    quadratureRuleContainer->getCellParametricPoints(cellIndex);
                    std::vector<double> cellJxWValues =
                    quadratureRuleContainer->getCellJxW(cellIndex);
                    std::vector<dealii::Point<dim, double>> dealiiParametricQuadPoints(
                    0);

                    //get the quad weights in each cell
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
                        d_dofHandler->get_fe().n_dofs_per_cell();
                        cellStartIdsBasisQuadStorage[cellIndex] =
                        cumulativeQuadPoints * dofsPerCell;
                        for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                        {
                            if(iNode < classicalDofsPerCell)
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
                                    efeBM->getEnrichmentValue
                                        (cellIndex, iNode - classicalDofsPerCell, 
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
                            if( iNode < classicalDofsPerCell && jNode < classicalDofsPerCell)
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
                            else if( iNode >= classicalDofsPerCell && jNode < classicalDofsPerCell)
                            {
                                for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                                    qPoint++)
                                {
                                    *basisOverlapTmpIter +=
                                    efeBM->getEnrichmentValue
                                        (cellIndex, iNode - classicalDofsPerCell, 
                                        quadRealPointsVec[qPoint]) * 
                                        dealiiFEValues.shape_value(jNode, qPoint) *
                                        cellJxWValues[qPoint];
                                    // enriched i * classical j
                                }
                            }
                            else if( iNode < classicalDofsPerCell && jNode >= classicalDofsPerCell)
                            {
                                for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                                    qPoint++)
                                {
                                    *basisOverlapTmpIter +=
                                    efeBM->getEnrichmentValue
                                        (cellIndex, jNode - classicalDofsPerCell, 
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
                                    efeBM->getEnrichmentValue
                                        (cellIndex, iNode - classicalDofsPerCell, 
                                        quadRealPointsVec[qPoint]) * 
                                    efeBM->getEnrichmentValue
                                        (cellIndex, jNode - classicalDofsPerCell, 
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
                            if(iNode < classicalDofsPerCell)
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
                            else
                            {
                                for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                                    qPoint++)
                                {
                                    auto shapeGrad =
                                    efeBM->getEnrichmentDerivative
                                        (cellIndex, jNode - classicalDofsPerCell, 
                                        quadRealPointsVec[qPoint]);
                                    // enriched gradient function call
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
                    }

                    if (basisStorageAttributesBoolMap
                        .find(BasisStorageAttributes::StoreHessian)
                        ->second)
                    {
                        cellStartIdsBasisHessianQuadStorage[cellIndex] =
                        cumulativeQuadPoints * dim * dim * dofsPerCell;
                        for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                        {
                            if(iNode < classicalDofsPerCell)
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
                            else
                            {
                                for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                                    qPoint++)
                                {
                                    auto shapeHessian =
                                    efeBM->getEnrichmentHessian
                                        (cellIndex, jNode - classicalDofsPerCell, 
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
                                            iDim * dim * dofsPerCell * nQuadPointInCell +
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
                const size_type feOrder              = efeBM->getFEOrder(0);
                const size_type numLocallyOwnedCells = efeBM->nLocallyOwnedCells();
                // NOTE: cellId 0 passed as we assume only H refined in this function
                
                std::vector<ValueTypeBasisData> basisGradNiGradNjTmp(0);

                const size_type nTotalQuadPoints =
                quadratureRuleContainer->nQuadraturePoints();

                size_type dofsPerCell = 0;
                size_type cellIndex = 0;
                size_type basisStiffnessSize = 0;

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
                basisGradNiGradNjTmp.resize(basisStiffnessSize,
                                            ValueTypeBasisData(0));

                auto locallyOwnedCellIter = efeBM->beginLocallyOwnedCells();
                std::shared_ptr<FECellDealii<dim>> feCellDealii =
                std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
                utils::throwException(
                feCellDealii != nullptr,
                "Dynamic casting of FECellBase to FECellDealii not successful");

                auto      basisGradNiGradNjTmpIter = basisGradNiGradNjTmp.begin();
                size_type cellIndex                = 0;

                // get the dealii FiniteElement object
                std::shared_ptr<const dealii::DoFHandler<dim>> dealiiDofHandler =
                efeBM->getDoFHandler();

                size_type cumulativeQuadPoints = 0;
                for (; locallyOwnedCellIter != efeBM->endLocallyOwnedCells();
                    ++locallyOwnedCellIter)
                {
                    dofsPerCell = efeBM->nCellDofs(cellIndex);
                    //Get classical dof numbers
                    size_type classicalDofsPerCell =  
                        utils::mathFunctions::sizeTypePow((efeBM->getFEOrder(cellIndex)+1),dim);

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

                    for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
                    {
                        for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                        {
                            *basisGradNiGradNjTmpIter = 0.0;
                            if( iNode < classicalDofsPerCell && jNode < classicalDofsPerCell)
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
                            else if( iNode >= classicalDofsPerCell && jNode < classicalDofsPerCell)
                            {
                                for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                                    qPoint++)
                                {
                                    *basisGradNiGradNjTmpIter +=
                                    efeBM->getEnrichmentDerivative
                                        (cellIndex, iNode - classicalDofsPerCell, 
                                        quadRealPointsVec[qPoint]) * 
                                        dealiiFEValues.shape_grad(jNode, qPoint) *
                                        cellJxWValues[qPoint];
                                    // enriched i * classical j
                                }
                            }
                            else if( iNode < classicalDofsPerCell && jNode >= classicalDofsPerCell)
                            {
                                for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                                    qPoint++)
                                {
                                    *basisGradNiGradNjTmpIter +=
                                    efeBM->getEnrichmentDerivative
                                        (cellIndex, jNode - classicalDofsPerCell, 
                                        quadRealPointsVec[qPoint]) * 
                                        dealiiFEValues.shape_grad(iNode, qPoint) *
                                        cellJxWValues[qPoint];                                    
                                    // enriched j * classical i
                                }
                            }
                            else
                            {
                                for (unsigned int qPoint = 0; qPoint < nQuadPointInCell;
                                    qPoint++)
                                {
                                    *basisGradNiGradNjTmpIter +=
                                    efeBM->getEnrichmentDerivative
                                        (cellIndex, iNode - classicalDofsPerCell, 
                                        quadRealPointsVec[qPoint]) * 
                                    efeBM->getEnrichmentDerivative
                                        (cellIndex, jNode - classicalDofsPerCell, 
                                        quadRealPointsVec[qPoint]) * 
                                        cellJxWValues[qPoint];                                    
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
        } // End of efebasisdatastorageinternal


        template <typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
        EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
        EFEBasisDataStorageDealii(
            std::shared_ptr<const BasisManager> efeBM,
            const std::vector<quadrature::QuadratureRuleAttributes>
            &                                 quadratureRuleAttribuesVec,
            const QuadAttrToBasisStorageAttrMap quadAttrToBasisStorageAttrMap,
            std::vector<std::shared_ptr<const QuadratureRule>> 
                                &quadratureRuleVec = {},
            std::shared_ptr<const QuadratureRule>  baseQuadratureRuleAdaptive = nullptr,
            std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>>
                                &functions = {},
            const std::vector<double> &tolerances = {},
            const std::vector<double> &integralThresholds = {},
            const double               smallestCellVolume = 1e-12,
            const unsigned int         maxRecursion       = 100)
        : d_dofsInCell(0)
        , d_cellStartIdsBasisOverlap(0)
        {
            d_efeBM = std::dynamic_pointer_cast<const EFEBasisManagerDealii<dim>>(efeBM);
            utils::throwException(
                d_efeBM != nullptr,
                " Could not cast the EFEBasisManager to EFEBasisManagerDealii in EFEBasisDataStorageDealii");
            const size_type numQuadRuleType = quadratureRuleAttribuesVec.size();
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

            std::vector<dealii::Quadrature<dim>> dealiiQuadratureTypeVec(0);

            /**
            * @note We assume a linear mapping from the reference cell
            * to the real cell.
            */
            LinearCellMappingDealii<dim> linearCellMappingDealii;
            std::shared_ptr<ParentToChildCellsManagerBase> parentToChildCellsManagerDealii = 
                std::make_shared<parentToChildCellsManagerDealii<dim>>();

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
                        d_efeBM->getTriangulation(),
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
                        d_efeBM->getTriangulation(),
                        linearCellMappingDealii);
                }
                else if (quadFamily == quadrature::QuadratureFamily::GAUSS_VARIABLE)
                {
                    dealiiQuadratureTypeVec.push_back(
                        dealii::QGauss<dim>(num1DQuadPoints));

                    d_quadratureRuleContainer[quadratureRuleAttribuesVec[i]] =
                        std::make_shared<quadrature::QuadratureRuleContainer>(
                        quadratureRuleAttribuesVec[i],
                        quadratureRuleVec,
                        d_efeBM->getTriangulation(),
                        linearCellMappingDealii);
                }
                else if (quadFamily == quadrature::QuadratureFamily::GLL_VARIABLE)
                {
                    dealiiQuadratureTypeVec.push_back(
                        dealii::QGaussLobatto<dim>(num1DQuadPoints));

                    d_quadratureRuleContainer[quadratureRuleAttribuesVec[i]] =
                        std::make_shared<quadrature::QuadratureRuleContainer>(
                        quadratureRuleAttribuesVec[i],
                        quadratureRuleVec,
                        d_efeBM->getTriangulation(),
                        linearCellMappingDealii);
                }
                else if (quadFamily == quadrature::QuadratureFamily::ADAPTIVE)
                {
                    dealiiQuadratureTypeVec.push_back(
                        dealii::QGaussLobatto<dim>(num1DQuadPoints));

                    d_quadratureRuleContainer[quadratureRuleAttribuesVec[i]] =
                        std::make_shared<quadrature::QuadratureRuleContainer>(
                        quadratureRuleAttribuesVec[i],
                        baseQuadratureRuleAdaptive,
                        d_efeBM->getTriangulation(),
                        linearCellMappingDealii,
                        *(parentToChildCellsManagerDealii),
                        functions,
                        tolerances,
                        integralThresholds,
                        smallestCellVolume,
                        maxRecursion);
                }
                else
                    utils::throwException<utils::InvalidArgument>(
                    false,
                    "Invalid QuadratureFamily type given.");
            }

            for (size_type i = 0; i < numQuadRuleType; ++i)
            {
                quadrature::QuadratureRuleAttributes quadratureRuleAttributes =
                    quadratureRuleAttribuesVec[i];
                quadrature::QuadratureFamily quadFamily =
                    quadratureRuleAttributes.getQuadratureFamily();
                BasisStorageAttributesBoolMap basisStorageAttributesBoolMap =
                    quadAttrToBasisStorageAttrMap.find(quadratureRuleAttributes)
                    ->second;

                if (quadFamily == quadrature::QuadratureFamily::GAUSS || 
                    quadFamily == quadrature::QuadratureFamily::GLL)
                {
                    evaluateBasisData(
                        quadratureRuleAttributes,
                        basisStorageAttributesBoolMap);
                }

                if (quadFamily == quadrature::QuadratureFamily::GAUSS_VARIABLE || 
                    quadFamily == quadrature::QuadratureFamily::GLL_VARIABLE ||
                    quadFamily == quadrature::QuadratureFamily::ADAPTIVE)
                {
                    evaluateBasisData(
                        d_quadratureRuleContainer[quadratureRuleAttributes],
                        quadratureRuleAttributes,
                        basisStorageAttributesBoolMap);
                }
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
                storeValuesHRefinedSameQuadEveryCell<ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>(
                d_efeBM,
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
                std::shared_ptr<typename BasisDataStorage<ValueTypeBasisData,
                                                            memorySpace>::Storage>
                    basisGradNiNj;
                EFEBasisDataStorageDealiiInternal::
                    storeGradNiNjHRefinedSameQuadEveryCell<ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>(
                    d_efeBM, basisGradNiNj, quadratureRuleAttributes);

                d_basisGradNiGradNj[quadratureRuleAttributes] = basisGradNiNj;
            }

            if (basisStorageAttributesBoolMap
                    .find(BasisStorageAttributes::StoreJxW)
                    ->second)
            {
                std::shared_ptr<typename BasisDataStorage<ValueTypeBasisData,
                                                            memorySpace>::Storage>
                    jxwQuadStorage;

                const std::vector<double> &jxwVec =
                    d_quadratureRuleContainer[quadratureRuleAttributes]->getJxW();
                jxwQuadStorage = std::make_shared<
                    typename BasisDataStorage<ValueTypeBasisData,
                                            memorySpace>::Storage>(jxwVec.size());

                utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                    copy(jxwVec.size(), jxwQuadStorage->data(), jxwVec.data());

                d_JxWStorage[quadratureRuleAttributes] = jxwQuadStorage;
            }
        }

        template <typename ValueTypeBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
        void
        EFEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>::
        evaluateBasisData(
            std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                                        quadratureRuleContainer,
            const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
            const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
        {
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
                typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                basisGradNiGradNj;

            EFEBasisDataStorageDealiiInternal::
                storeGradNiGradNjHRefinedAdaptiveQuad<ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>(d_efeBM,
                                                        basisGradNiGradNj,
                                                        quadratureRuleAttributes,
                                                        quadratureRuleContainer);
            d_basisGradNiGradNj[quadratureRuleAttributes] = basisGradNiGradNj;
            }
        if (basisStorageAttributesBoolMap.find(BasisStorageAttributes::StoreJxW)
                ->second)
            {
            std::shared_ptr<
                typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage>
                jxwQuadStorage;

            const std::vector<double> &jxwVec =
                d_quadratureRuleContainer[quadratureRuleAttributes]->getJxW();
            jxwQuadStorage =
                std::make_shared<typename BasisDataStorage<ValueTypeBasisData,
                                                        memorySpace>::Storage>(
                jxwVec.size());

            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
                jxwVec.size(), jxwQuadStorage->data(), jxwVec.data());

            d_JxWStorage[quadratureRuleAttributes] = jxwQuadStorage;
            }
        }
    }
}