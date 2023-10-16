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

#ifndef dftefeEFEBasisDataStorage_h
#define dftefeEFEBasisDataStorage_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/FEBasisManagerDealii.h>
#include <basis/FEBasisManager.h>
#include <basis/FEConstraintsDealii.h>
#include <basis/LinearCellMappingDealii.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleGLL.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <memory>
#include <map>
#include <vector>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class to store and access data for a given basis,
     * such as the basis function values on a quadrature grid, the overlap
     * matrix of the basis, etc.
     */
    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace>
    class EFEBasisDataStorage
      : public FEBasisDataStorage<ValueTypeBasisData, memorySpace>
    {
    public:
      using QuadraturePointAttributes = quadrature::QuadraturePointAttributes;
      using QuadratureRuleAttributes  = quadrature::QuadratureRuleAttributes;
      using Storage =
        typename BasisDataStorage<ValueTypeBasisData, memorySpace>::Storage;


      virtual const BasisManager &
      getBasisManager() const = 0;

      virtual void
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap) = 0;

      virtual void
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                            quadratureRuleContainer,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap) = 0;

      virtual void
      evaluateBasisData(
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        std::vector<std::shared_ptr<const quadrature::QuadratureRule>>
                                            quadratureRuleVec,
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap) = 0;

      virtual void
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
        const BasisStorageAttributesBoolMap basisStorageAttributesBoolMap) = 0;

      virtual void
      deleteBasisData() = 0;



      //      std::shared_ptr<const quadrature::QuadratureRuleContainer>
      //      getCellQuadratureRuleContainer(std::shared_ptr<Storage>>
      //        const QuadratureRuleAttributes &quadratureRuleAttributes) const
      //        override;
      // functions to get data for a basis function on a given quad point in a
      // cell
      virtual Storage
      getBasisData(const QuadraturePointAttributes &attributes,
                   const size_type                  basisId) const = 0;
      virtual Storage
      getBasisGradientData(const QuadraturePointAttributes &attributes,
                           const size_type                  basisId) const = 0;
      virtual Storage
      getBasisHessianData(const QuadraturePointAttributes &attributes,
                          const size_type                  basisId) const = 0;

      // functions to get data for a basis function on all quad points in a cell
      virtual Storage
      getBasisDataInCell(
        const size_type                 cellId,
        const size_type                 basisId) const = 0;
      virtual Storage
      getBasisGradientDataInCell(
        const size_type                 cellId,
        const size_type                 basisId) const = 0;
      virtual Storage
      getBasisHessianDataInCell(
        const size_type                 cellId,
        const size_type                 basisId) const = 0;

      // functions to get data for all basis functions on all quad points in a
      // cell
      virtual Storage
      getBasisDataInCell(
        const size_type                 cellId) const = 0;
      virtual Storage
      getBasisGradientDataInCell(
        const size_type                 cellId) const = 0;
      virtual Storage
      getBasisHessianDataInCell(
        const size_type                 cellId) const = 0;

      virtual Storage
      getJxWInCell(
                   const size_type                 cellId) const = 0;

      // functions to get data for all basis functions on all quad points in all
      // cells
      virtual const Storage &
      getBasisDataInAllCells() const = 0;
      virtual const Storage &
      getBasisGradientDataInAllCells() const = 0;
      virtual const Storage &
      getBasisHessianDataInAllCells() const = 0;

      virtual const Storage &
      getJxWInAllCells() const = 0;

      // get overlap of two basis functions in a cell
      virtual Storage
      getBasisOverlap(
                      const size_type                 cellId,
                      const size_type                 basisId1,
                      const size_type                 basisId2) const = 0;

      // get overlap of all the basis functions in a cell
      virtual Storage
      getBasisOverlapInCell(
        const size_type                 cellId) const = 0;

      // get the laplace operator in a cell
      virtual Storage
      getBasisGradNiGradNjInCell(
        const size_type                 cellId) const = 0;

      // get laplace operator in all cells
      virtual const Storage &
      getBasisGradNiGradNjInAllCells() const = 0;

      // get overlap of all the basis functions in all cells
      virtual const Storage &
      getBasisOverlapInAllCells() const = 0;

      virtual std::shared_ptr<const quadrature::QuadratureRuleContainer>
      getQuadratureRuleContainer() const = 0;

    }; // end of EFEBasisDataStorage
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeEFEBasisDataStorage_h
