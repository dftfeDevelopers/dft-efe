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

#ifndef dftefeFEBasisOperations_h
#define dftefeFEBasisOperations_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/Field.h>
#include <basis/BasisOperations.h>
#include <basis/BasisDataStorage.h>
#include <basis/FEBasisHandler.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * An abstract class to handle interactions between a basis and a
     * field (e.g., integration of field with basis).
     */
    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    class FEBasisOperations : public BasisOperations<ValueTypeBasisCoeff, memorySpace>
    {
    public:
      FEBasisOperations(
        std::shared_ptr<const BasisDataStorage<ValueTypeBasisData, memorySpace>>
                        basisDataStorage,
        const size_type maxCellTimesFieldBlock);

      ~FEBasisOperations() = default;

      void
      interpolate(
        const Field<ValueTypeBasisCoeff, memorySpace> &       field,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff, memorySpace>
          &quadValuesContainer) const override;
      
      virtual void
      integrateWithBasisValues(
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff, memorySpace> &f,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        Field<ValueTypeBasisCoeff, memorySpace> &             field) const = 0;

      // virtual void
      // integrateWithBasisValues(
      //  const Field<ValueTypeBasisCoeff, memorySpace> &       fieldInput,
      //  const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
      //  Field<ValueTypeBasisCoeff, memorySpace> &             fieldOutput) const
      //  override;

    private:
      std::shared_ptr<const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                d_feBasisDataStorage;
      size_type d_maxCellTimesFieldBlock;

    }; // end of FEBasisOperations
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/FEBasisOperations.t.cpp>
#endif // dftefeBasisOperations_h
