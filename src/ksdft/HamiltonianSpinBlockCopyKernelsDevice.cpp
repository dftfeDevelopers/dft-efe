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

#ifdef DFTEFE_WITH_DEVICE

#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceDataTypeOverloads.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  include <utils/MemoryTransfer.h>
#  include <utils/Exceptions.h>
#  include <complex>
#  include <algorithm>
#  include "HamiltonianSpinBlockCopyKernels.h"

namespace dftefe
{
  namespace ksdft
  {
    namespace
    {
      /*
       * Per-cell kernel for copyIntoBlock.
       * Launched with K·d² threads per cell on the cell's stream.
       * t = k·d² + j·d + i.
       * layout==0: DofFastest dst index = (sCol·d+j)·(S·d)+sRow·d+i
       * layout==1: SpinFastest dst index = (j·S+sCol)·(S·d)+i·S+sRow
       */
      template <typename ValueType>
      DFTEFE_CREATE_KERNEL(
        void,
        copyIntoBlockCellKernel,
        {
          const size_type total = K * d * d;
          for (size_type t = globalThreadId; t < total;
               t += nThreadsPerBlock * nThreadBlock)
            {
              const size_type k    = t / (d * d);
              const size_type ji   = t % (d * d);
              const size_type j    = ji / d;
              const size_type i    = ji % d;
              const size_type sRow = sRowArr[k];
              const size_type sCol = sColArr[k];
              const size_type Sd   = S * d;
              const size_type dstIdx =
                (layout == 0) ?
                  cellDstOffset + (sCol * d + j) * Sd + sRow * d + i :
                  cellDstOffset + (j * S + sCol) * Sd + i * S + sRow;
              dftefe::utils::copyValue(
                dst + dstIdx,
                src[k * basisOverlapSize + cellSrcOffset + j * d + i]);
            }
        },
        size_type        K,
        size_type        S,
        size_type        d,
        size_type        basisOverlapSize,
        size_type        cellSrcOffset,
        size_type        cellDstOffset,
        int              layout,
        const size_type *sRowArr,
        const size_type *sColArr,
        const ValueType *src,
        ValueType *      dst);

    } // namespace

    template <typename ValueType>
    void
    HamiltonianSpinBlockCopyKernels<ValueType, utils::MemorySpace::DEVICE>::
      copyIntoBlock(
        const utils::MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &src,
        utils::MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &      dst,
        size_type                                                          S,
        SpinStorageLayout                                   layout,
        const std::vector<std::pair<size_type, size_type>> &spinIdsFilled,
        const std::vector<size_type> &                      numCellDofs,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
          &linAlgOpContext)
    {
      const size_type K          = spinIdsFilled.size();
      const size_type numStreams = linAlgOpContext.numBlasStreams();
      auto *          streams    = linAlgOpContext.getBlasStreamsVec();
      const size_type C          = numCellDofs.size();
      const auto *    srcPtr = utils::makeDataTypeDeviceCompatible(src.begin());
      auto *          dstPtr = utils::makeDataTypeDeviceCompatible(dst.begin());
      const int layoutInt = (layout == SpinStorageLayout::DofFastest) ? 0 : 1;

      // Copy spin-index mapping to device
      std::vector<size_type> sRowHost(K), sColHost(K);
      for (size_type k = 0; k < K; ++k)
        {
          sRowHost[k] = spinIdsFilled[k].first;
          sColHost[k] = spinIdsFilled[k].second;
        }
      utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE> sRowDev(K);
      utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE> sColDev(K);
      utils::MemoryTransfer<utils::MemorySpace::DEVICE,
                            utils::MemorySpace::HOST>
        memTransH2D;
      memTransH2D.copy(K, sRowDev.begin(), sRowHost.data());
      memTransH2D.copy(K, sColDev.begin(), sColHost.data());

      size_type basisOverlapSize = 0;
      for (size_type c = 0; c < C; ++c)
        basisOverlapSize += numCellDofs[c] * numCellDofs[c];

      size_type cellSrcOffset = 0;
      size_type cellDstOffset = 0;
      for (size_type c = 0; c < C; ++c)
        {
          const size_type d         = numCellDofs[c];
          const size_type sid       = c % numStreams;
          const size_type total     = K * d * d;
          const size_type blockSize = utils::DEVICE_BLOCK_SIZE;
          const size_type grid      = (total + blockSize - 1) / blockSize;

          DFTEFE_LAUNCH_KERNEL(copyIntoBlockCellKernel,
                               grid,
                               blockSize,
                               streams[sid],
                               K,
                               S,
                               d,
                               basisOverlapSize,
                               cellSrcOffset,
                               cellDstOffset,
                               layoutInt,
                               sRowDev.begin(),
                               sColDev.begin(),
                               srcPtr,
                               dstPtr);

          cellSrcOffset += d * d;
          const size_type Sd = S * d;
          cellDstOffset += Sd * Sd;
        }

      for (size_type s = 0; s < numStreams; ++s)
        {
          utils::deviceError_t err = utils::deviceStreamSynchronize(streams[s]);
          DEVICE_API_CHECK(err);
        }
    }

    template class HamiltonianSpinBlockCopyKernels<double,
                                                   utils::MemorySpace::DEVICE>;
    template class HamiltonianSpinBlockCopyKernels<float,
                                                   utils::MemorySpace::DEVICE>;
    template class HamiltonianSpinBlockCopyKernels<std::complex<double>,
                                                   utils::MemorySpace::DEVICE>;
    template class HamiltonianSpinBlockCopyKernels<std::complex<float>,
                                                   utils::MemorySpace::DEVICE>;

  } // namespace ksdft
} // namespace dftefe
#endif // DFTEFE_WITH_DEVICE
