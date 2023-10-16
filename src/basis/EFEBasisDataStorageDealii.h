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

#ifndef dftefeEFEBasisDataStorageDealii_h
#define dftefeEFEBasisDataStorageDealii_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <basis/BasisDataStorage.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/FEBasisOperations.h>
#include <basis/EFEBasisDataStorage.h>
#include <basis/EFEBasisManagerDealii.h>
#include <basis/EFEBasisManager.h>
#include <basis/LinearCellMappingDealii.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleGLL.h>
#include <quadrature/QuadratureRuleAdaptive.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <memory>
#include <map>
#include <vector>

namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class to store and access data for enriched fe basis,
     * such as the basis function values on a quadrature grid, the overlap
     * matrix of the basis, etc.
     */
    template <typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class EFEBasisDataStorageDealii
      : public EFEBasisDataStorage<ValueTypeBasisData, memorySpace>
    {
    public:
      using QuadraturePointAttributes = quadrature::QuadraturePointAttributes;
      using QuadratureRuleAttributes  = quadrature::QuadratureRuleAttributes;
      using Storage =
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage;

      EFEBasisDataStorageDealii(
        std::shared_ptr<const BasisManager>         efeBM,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap);

      ~EFEBasisDataStorageDealii() = default;

      const BasisManager &
      getBasisManager() const override;

      void
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
        override;

      void
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                            quadratureRuleContainer,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
        override;

      void
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::vector<std::shared_ptr<const quadrature::QuadratureRule>>
                                            quadratureRuleVec,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap)
        override;

      void
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
        override;

      void
      deleteBasisData() override;

      // functions to get data for a basis function on a given quad point in a
      // cell
      Storage
      getBasisData(const QuadraturePointAttributes &attributes,
                   const size_type                  basisId) const override;
      Storage
      getBasisGradientData(const QuadraturePointAttributes &attributes,
                           const size_type basisId) const override;
      Storage
      getBasisHessianData(const QuadraturePointAttributes &attributes,
                          const size_type basisId) const override;

      // functions to get data for a basis function on all quad points in a cell
      // The evaluation of functions
      Storage
      getBasisDataInCell(
        const size_type                 cellId,
        const size_type                 basisId) const override;
      Storage
      getBasisGradientDataInCell(
        const size_type                 cellId,
        const size_type                 basisId) const override;
      Storage
      getBasisHessianDataInCell(
        const size_type                 cellId,
        const size_type                 basisId) const override;

      // functions to get data for all basis functions on all quad points in a
      // cell
      Storage
      getBasisDataInCell(const size_type                 cellId) const override;
      Storage
      getBasisGradientDataInCell(const size_type                 cellId) const override;
      Storage
      getBasisHessianDataInCell(const size_type                 cellId) const override;

      Storage
      getJxWInCell(const size_type                 cellId) const override;

      // functions to get data for all basis functions on all quad points in all
      // cells
      const Storage &
      getBasisDataInAllCells() const override;
      const Storage &
      getBasisGradientDataInAllCells() const override;
      const Storage &
      getBasisHessianDataInAllCells() const override;

      const Storage &
      getJxWInAllCells()
        const override;

      // get overlap of two basis functions in a cell
      Storage
      getBasisOverlap(const size_type                 cellId,
                      const size_type                 basisId1,
                      const size_type                 basisId2) const override;

      // get overlap of all the basis functions in a cell
      Storage
      getBasisOverlapInCell(const size_type                 cellId) const override;

      // get overlap of all the basis functions in all cells
      const Storage &
      getBasisOverlapInAllCells() const override;

      // get the laplace operator in a cell
      Storage
      getBasisGradNiGradNjInCell(const size_type                 cellId) const override;

      // get laplace operator in all cells
      const Storage &
      getBasisGradNiGradNjInAllCells() const override;

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
      getQuadratureRuleContainer() const override;

    private:
      bool                                              d_evaluateBasisData;
      std::shared_ptr<const EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>> d_efeBM;
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                    d_quadratureRuleContainer;
      QuadratureRuleAttributes      d_quadratureRuleAttributes;
      BasisStorageAttributesBoolMap d_basisStorageAttributesBoolMap;
      std::shared_ptr<Storage>      d_basisQuadStorage;
      std::shared_ptr<Storage>      d_JxWStorage;
      std::shared_ptr<Storage>      d_basisGradNiGradNj;
      std::shared_ptr<Storage>      d_basisGradientQuadStorage;
      std::shared_ptr<Storage>      d_basisHessianQuadStorage;
      std::shared_ptr<Storage>      d_basisOverlap;
      std::vector<size_type>        d_dofsInCell;
      std::vector<size_type>        d_cellStartIdsBasisOverlap;
      std::vector<size_type>        d_nQuadPointsIncell;
      std::vector<size_type>        d_cellStartIdsBasisQuadStorage;
      std::vector<size_type>        d_cellStartIdsBasisGradientQuadStorage;
      std::vector<size_type>        d_cellStartIdsBasisHessianQuadStorage;
      std::vector<size_type>        d_cellStartIdsGradNiGradNj;
    };
  } // end of namespace basis
} // end of namespace dftefe
#include <basis/EFEBasisDataStorageDealii.t.cpp>
#endif
