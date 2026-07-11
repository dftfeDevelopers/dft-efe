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

#ifndef dftefe_HamiltonianSpinBlockCopyKernels_h
#define dftefe_HamiltonianSpinBlockCopyKernels_h

#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <utility>
#include <vector>

namespace dftefe
{
  namespace ksdft
  {
    enum class SpinStorageLayout { DofFastest, SpinFastest };

    template <typename ValueType, utils::MemorySpace memorySpace>
    class HamiltonianSpinBlockCopyKernels
    {
    public:
      /*
       * Copy K source spin blocks into the (S·d_c)×(S·d_c) cell destination.
       * spinIdsFilled[k] = (sRow, sCol): the destination spin block for
       * source block k. Caller must zero dst when K < S².
       *
       * src: [k (0..K-1)][cell c][col j (0..d_c-1)][row i (0..d_c-1)]
       *   flat = k·Σd_c² + cellSrcOffset_c + j·d_c + i
       *
       * dst (DofFastest):
       *   flat = cellDstOffset_c + (sCol·d_c + j)·(S·d_c) + sRow·d_c + i
       *
       * dst (SpinFastest):
       *   flat = cellDstOffset_c + (j·S + sCol)·(S·d_c) + i·S + sRow
       *
       * cellDstOffset_c = Σ_{c'<c} (S·d_{c'})²
       */
      static void
      copyIntoBlock(
        const utils::MemoryStorage<ValueType, memorySpace> &              src,
        utils::MemoryStorage<ValueType, memorySpace> &                    dst,
        size_type                                                          S,
        SpinStorageLayout                                                  layout,
        const std::vector<std::pair<size_type, size_type>> &spinIdsFilled,
        const std::vector<size_type> &                       numCellDofs,
        linearAlgebra::LinAlgOpContext<memorySpace> &        linAlgOpContext);
    };

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    HamiltonianSpinBlockCopyKernels<ValueType, memorySpace>::copyIntoBlock(
      const utils::MemoryStorage<ValueType, memorySpace> &              src,
      utils::MemoryStorage<ValueType, memorySpace> &                    dst,
      size_type                                                          S,
      SpinStorageLayout                                                  layout,
      const std::vector<std::pair<size_type, size_type>> &spinIdsFilled,
      const std::vector<size_type> &                       numCellDofs,
      linearAlgebra::LinAlgOpContext<memorySpace> & /*linAlgOpContext*/)
    {
      const size_type  K = spinIdsFilled.size();
      size_type        basisOverlapSize = 0;
      for (const size_type d : numCellDofs)
        basisOverlapSize += d * d;

      const ValueType *srcPtr = src.begin();
      ValueType *      dstPtr = dst.begin();

      for (size_type k = 0; k < K; ++k)
        {
          const size_type sRow  = spinIdsFilled[k].first;
          const size_type sCol  = spinIdsFilled[k].second;
          size_type       cellSrcOffset = 0;
          size_type       cellDstOffset = 0;
          for (const size_type d : numCellDofs)
            {
              const size_type Sd = S * d;
              for (size_type j = 0; j < d; ++j)
                for (size_type i = 0; i < d; ++i)
                  {
                    const size_type dstIdx =
                      (layout == SpinStorageLayout::DofFastest)
                        ? cellDstOffset + (sCol * d + j) * Sd + sRow * d + i
                        : cellDstOffset + (j * S + sCol) * Sd + i * S + sRow;
                    dstPtr[dstIdx] =
                      srcPtr[k * basisOverlapSize + cellSrcOffset + j * d + i];
                  }
              cellSrcOffset += d * d;
              cellDstOffset += Sd * Sd;
            }
        }
    }

#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class HamiltonianSpinBlockCopyKernels<ValueType,
                                          utils::MemorySpace::DEVICE>
    {
    public:
      static void
      copyIntoBlock(
        const utils::MemoryStorage<ValueType,
                                   utils::MemorySpace::DEVICE> &src,
        utils::MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &dst,
        size_type                                                     S,
        SpinStorageLayout                                             layout,
        const std::vector<std::pair<size_type, size_type>> &spinIdsFilled,
        const std::vector<size_type> &                       numCellDofs,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE> &
          linAlgOpContext);
    };
#endif

  } // namespace ksdft
} // namespace dftefe
#endif // dftefe_HamiltonianSpinBlockCopyKernels_h
