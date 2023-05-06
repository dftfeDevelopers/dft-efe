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

            // This class takes quadruleattribute(GAUSS)

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
                        "For storing of basis values for classical finite element basis "
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
                const size_type cellId = 0;
                const size_type feOrder = efeBM->getFEOrder(cellId);
                const size_type numLocallyOwnedCells = efeBM->nLocallyOwnedCells();
                
                size_type dofsPerCell = efeBM->nCellDofs(cellId);

                // Create temporary data structure for Value Storage
                std::vector<ValueTypeBasisData> basisQuadStorageTmp(0);
                std::vector<ValueTypeBasisData> basisGradientQuadStorageTmp(0);
                std::vector<ValueTypeBasisData> basisHessianQuadStorageTmp(0);
                std::vector<ValueTypeBasisData> basisOverlapTmp(0);

                nQuadPointsInCell.resize(numLocallyOwnedCells, 0);
                const size_type nTotalQuadPoints = quadratureRuleContainer->nQuadraturePoints();
                if(basisStorageAttributesB)

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
            }
    }
}