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

#include <linearAlgebra/VectorKernels.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                                 a,
        const VectorBase<ValueType, memorySpace> &u,
        ValueType                                 b,
        const VectorBase<ValueType, memorySpace> &v,
        VectorBase<ValueType, memorySpace> &      w)
    {
      const VectorAttributes &uVectorAttributes = u.getVectorAttributes();
      const VectorAttributes &vVectorAttributes = v.getVectorAttributes();
      const VectorAttributes &wVectorAttributes = w.getVectorAttributes();
      bool                    areCompatible =
        uVectorAttributes.areDistributionCompatible(vVectorAttributes);
      utils::throwException(areCompatible, "Trying to add incompatible vectors. One is a serial
	  vector and the other a distributed vector.");
      areCompatible = vVectorAttributes.areDistributionCompatible(wVectorAttributes);
      utils::throwException(areCompatible, "Trying to add incompatible vectors. One is a serial
	  vector and the other a distributed vector.");
      utils::throwException<utils::LengthError>(
        ((u.size() == v.size()) && (v.size() == w.size())),
        "Mismatch of sizes of the vectors that are added.");
      const size_type uStorageSize = (u.getStorage()).size();
      const size_type vStorageSize = (v.getStorage()).size();
      const size_type wStorageSize = (w.getStorage()).size();
      utils::throwException<utils::LengthError>(
        (uStorageSize == vStorageSize) && (vStorageSize == wStorageSize),
        "Mismatch of sizes of the underlying storages"
        "of the vectors that are added.");
      VectorKernels<ValueType, memorySpace>::add(
        uStorageSize, a, u.data(), b, v.data(), w.data());
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
