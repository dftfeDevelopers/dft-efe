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
#include <utils/Exceptions.h>
namespace dftefe 
{
  namespace basis
  {
    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
      FEBasisOperations<ValueType, memorySpace, dim>::FEBasisOperations(
	  std::shared_ptr<const BasisDataStorgae<ValueType, memorySpace>> basisDataStorage):
      {
	d_feBasisDataStorage = std::dynamic_pointer_cast<const FEBasisDataStorage<ValueType, memorySpace, dim>>(basisDataStorage);
	utils::throwException(d_feBasisDataStorage != nullptr, 
	    "Could not cast BasisDataStorage to FEBasisDataStorage in the constructor of FEBasisOperations");
      }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
      void
      FEBasisOperations<ValueType, memorySpace, dim>::interpolate(
        const Field<ValueType, memorySpace> &       field,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        quadrarture::QuadratureValuesContainer<ValueType, memorySpace>
          &quadValuesContainer) const
      {
	const BasisHandler<memorySpace> & basisHandler = field.getBasisHandler();
	const BasisManager & basisManagerField = basisHandler.getBasisManager();
	const BasisManager & basisManagerDataStorage = d_feBasisDataStorage->getBasisManager();
	utils::throwException(&basisManagerField == &basisManagerDataStorage,
	    "Mismatch in BasisManager used in Field and BasisDataStorage.");
	const FEBasisManager & feBasisManager = dynamic_cast<const FEBasisManageri &>(basisManagerField);
	utils::throwException(&feBasisManager != nullptr,
	    "Could not cast BasisManager to FEBasisManager in FEBasisOperations.interpolate()");
	const size_type numLocallyOwnedCells = feBasisManagerField.nLocallyOwnedCells();
        	

      }

  } // end of namespace 
}//
