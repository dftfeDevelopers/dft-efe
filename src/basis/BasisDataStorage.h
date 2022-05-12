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
 * @author Bikash Kanungo
 */

#ifndef dftefeBasisDataStorage_h
#define dftefeBasisDataStorage_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <quadrature/QuadratureRuleContainer.h>
#include <quadrature/QuadratureAttributes.h>
#include <basis/BasisManager.h>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class to store and access data for a given basis,
     * such as the basis function values on a quadrature grid, the overlap
     * matrix of the basis, etc.
     */
    template <typename ValueType, utils::MemorySpace memorySpace>
    class BasisDataStorage
    {
    public:
      //
      // typedefs
      //
      using Storage   = dftefe::utils::MemoryStorage<ValueType, memorySpace>;
      using pointer   = typename Storage::pointer;
      using reference = typename Storage::reference;
      using const_reference           = typename Storage::const_reference;
      using iterator                  = typename Storage::iterator;
      using const_iterator            = typename Storage::const_iterator;
      using QuadraturePointAttributes = quadrature::QuadraturePointAttributes;
      using QuadratureRuleAttributes  = quadrature::QuadratureRuleAttributes;

    public:
      virtual ~BasisDataStorage() = default;
      virtual void
      evaluateBasisData(
        std::shared_ptr<const quadrature::QuadratureRuleContainer>
                                        quadratureContainer,
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const bool                      storeValues,
        const bool                      storeGradient,
        const bool                      storeHessian,
        const bool                      storeOverlap) = 0;
      virtual void
      deleteBasisData(
        const QuadratureRuleAttributes &quadratureRuleAttributes) = 0;

      //      virtual std::shared_ptr<const quadrature::QuadratureRuleContainer>
      //      getCellQuadratureRuleContainer(
      //        const QuadratureRuleAttributes &quadratureRuleAttributes) const
      //        = 0;


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
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId,
        const size_type                 basisId) const = 0;
      virtual Storage
      getBasisGradientDataInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId,
        const size_type                 basisId) const = 0;
      virtual Storage
      getBasisHessianDataInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId,
        const size_type                 basisId) const = 0;

      // functions to get data for all basis functions on all quad points in a
      // cell
      virtual Storage
      getBasisDataInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId) const = 0;
      virtual Storage
      getBasisGradientDataInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId) const = 0;
      virtual Storage
      getBasisHessianDataInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId) const = 0;

      // functions to get data for all basis functions on all quad points in all
      // cells
      virtual const Storage &
      getBasisDataInAllCells(
        const QuadratureRuleAttributes &quadratureRuleAttributes) const = 0;
      virtual const Storage &
      getBasisGradientDataInAllCells(
        const QuadratureRuleAttributes &quadratureRuleAttributes) const = 0;
      virtual const Storage &
      getBasisHessianDataInAllCells(
        const QuadratureRuleAttributes &quadratureRuleAttributes) const = 0;

      // get overlap of two basis functions in a cell
      virtual Storage
      getBasisOverlap(const QuadratureRuleAttributes &quadratureRuleAttributes,
                      const size_type                 cellId,
                      const size_type                 basisId1,
                      const size_type                 basisId2) const = 0;

      // get overlap of all the basis functions in a cell
      virtual Storage
      getBasisOverlapInCell(
        const QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                 cellId) const = 0;

      // get overlap of all the basis functions in all cells
      virtual const Storage &
      getBasisOverlapInAllCells(
        const QuadratureRuleAttributes &quadratureRuleAttributes) const = 0;

      virtual const quadrature::QuadratureRuleContainer &
      getQuadratureRuleContainer(
        const QuadratureRuleAttributes &quadratureRuleAttributes) const = 0;

    }; // end of BasisDataStorage
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeBasisDataStorage_h
