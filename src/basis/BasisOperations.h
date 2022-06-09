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

#ifndef dftefeBasisOperations_h
#define dftefeBasisOperations_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/ScalarSpatialFunction.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureValuesContainer.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * An abstract class to handle interactions between a basis and a
     * field (e.g., integration of field with basis).
     */
    template <typename ValueType, utils::MemorySpace memorySpace>
    class BasisOperations
    {
    public:
      virtual ~BasisOperations() = default;

      virtual void
      interpolate(
        const Field<ValueType, memorySpace> &       field,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        quadrature::QuadratureValuesContainer<ValueType, memorySpace>
          &quadValuesContainer) const = 0;

      // virtual void
      // integrateWithBasisValues(const ScalarSpatialFunction<ValueType> &f,
      //    const quadrature::QuadratureRuleAttributes &
      //    quadratureRuleAttributes,
      //                         Field<ValueType, memorySpace> &         field)
      //                         const = 0;

      // virtual void
      // integrateWithBasisValues(const QuadratureValuesContainer<ValueType,
      // memorySpace> &f,
      //    const quadrature::QuadratureRuleAttributes &
      //    quadratureRuleAttributes,
      //                         Field<ValueType, memorySpace> &field) const =
      //                         0;

      virtual void
      integrateWithBasisValues(
        const Field<ValueType, memorySpace> &       fieldInput,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        Field<ValueType, memorySpace> &             fieldOutput) const = 0;

    }; // end of BasisOperations
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeBasisOperations_h
