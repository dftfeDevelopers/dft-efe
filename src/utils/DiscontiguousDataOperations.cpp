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

#include <utils/DiscontiguousDataOperations.h>
#include <utils/MemoryTransfer.h>
#include <utils/Exceptions.h>
#include <complex>
#include <algorithm>

namespace dftefe
{
  namespace utils
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DiscontiguousDataOperations<ValueType, memorySpace>::
      copyFromDiscontiguousMemory(const ValueType *src,
                                  ValueType *      dst,
                                  const size_type *discontIds,
                                  const size_type  N,
                                  const size_type  nComponents)
    {
      MemoryTransfer<memorySpace, memorySpace> memoryTransfer;
      for (size_type i = 0; i < N; ++i)
        {
          size_type        index  = *(discontIds + i);
          const ValueType *srcTmp = src + index * nComponents;
          ValueType *      dstTmp = i * nComponents;
          memoryTransfer.copy(nComponents, dstTmp, srcTmp);
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DiscontiguousDataOperations<ValueType, memorySpace>::
      copyToDiscontiguousMemory(const ValueType *src,
                                ValueType *      dst,
                                const size_type *discontIds,
                                const size_type  N,
                                const size_type  nComponents)
    {
      MemoryTransfer<memorySpace, memorySpace> memoryTransfer;
      for (size_type i = 0; i < N; ++i)
        {
          size_type        index  = *(discontIds + i);
          const ValueType *srcTmp = i * nComponents;
          ValueType *      dstTmp = dst + index * nComponents;
          memoryTransfer.copy(nComponents, dstTmp, srcTmp);
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DiscontiguousDataOperations<ValueType, memorySpace>::
      addToDiscontiguousMemory(const ValueType *src,
                               ValueType *      dst,
                               const size_type *discontIds,
                               const size_type  N,
                               const size_type  nComponents)
    {
      for (size_type i = 0; i < N; ++i)
        {
          size_type        index  = *(discontIds + i);
          const ValueType *srcTmp = i * nComponents;
          ValueType *      dstTmp = dst + index * nComponents;
          for (size_type j = 0; j < nComponents; ++j)
            *(dstTmp + j) += *(srcTmp + j);
        }
    }

    template class DiscontiguousDataOperations<
      double,
      dftefe::utils::MemorySpace::HOST>;
    template class DiscontiguousDataOperations<
      float,
      dftefe::utils::MemorySpace::HOST>;
    template class DiscontiguousDataOperations<
      std::complex<double>,
      dftefe::utils::MemorySpace::HOST>;
    template class DiscontiguousDataOperations<
      std::complex<float>,
      dftefe::utils::MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE
    template class DiscontiguousDataOperations<
      double,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class DiscontiguousDataOperations<
      float,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class DiscontiguousDataOperations<
      std::complex<double>,
      dftefe::utils::MemorySpace::HOST_PINNED>;
    template class DiscontiguousDataOperations<
      std::complex<float>,
      dftefe::utils::MemorySpace::HOST_PINNED>;
#endif
  } // end of namespace utils
} // end of namespace dftefe
