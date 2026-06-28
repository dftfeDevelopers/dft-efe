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

#ifndef dftefeMultivectorScratch_h
#define dftefeMultivectorScratch_h

#include <memory>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/Exceptions.h>
#include <linearAlgebra/MultiVector.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief Shared scratch MultiVectors (large-batch and small-batch variants)
     * that can be reused across ChebyshevFilteredEigenSolver,
     * OrthonormalizationFunctions, RayleighRitzEigenSolver, and
     * KohnShamEigenSolver to avoid redundant GPU allocations.
     *
     * The large-batch buffers are supplied at construction (their size is
     * always known upfront). The small-batch buffers are registered lazily
     * via setXinBatchSmall / setXoutBatchSmall the first time a remainder
     * batch is encountered, and are reused by all subsequent solver calls.
     */
    template <typename ValueType, utils::MemorySpace memorySpace>
    class MultivectorScratch
    {
    public:
      MultivectorScratch(
        std::shared_ptr<MultiVector<ValueType, memorySpace>> XinBatch,
        std::shared_ptr<MultiVector<ValueType, memorySpace>> XoutBatch)
        : d_XinBatch(XinBatch)
        , d_XoutBatch(XoutBatch)
        , d_XinBatchSmall(nullptr)
        , d_XoutBatchSmall(nullptr)
        , d_inUse(false)
      {}

      // --- concurrent-access guard ---

      void
      acquire()
      {
        utils::throwException(
          !d_inUse,
          "MultivectorScratch is already in use. Concurrent access detected.");
        d_inUse = true;
      }

      void
      release()
      {
        d_inUse = false;
      }

      // --- small-batch setters (called once on first remainder batch) ---

      void
      setXinBatchSmall(std::shared_ptr<MultiVector<ValueType, memorySpace>> v)
      {
        d_XinBatchSmall = v;
      }

      void
      setXoutBatchSmall(std::shared_ptr<MultiVector<ValueType, memorySpace>> v)
      {
        d_XoutBatchSmall = v;
      }

      // --- getters ---

      std::shared_ptr<MultiVector<ValueType, memorySpace>>
      getXinBatch() const
      {
        return d_XinBatch;
      }

      std::shared_ptr<MultiVector<ValueType, memorySpace>>
      getXoutBatch() const
      {
        return d_XoutBatch;
      }

      std::shared_ptr<MultiVector<ValueType, memorySpace>>
      getXinBatchSmall() const
      {
        return d_XinBatchSmall;
      }

      std::shared_ptr<MultiVector<ValueType, memorySpace>>
      getXoutBatchSmall() const
      {
        return d_XoutBatchSmall;
      }

      // --- validation helpers ---

      bool
      hasXinBatch() const
      {
        return d_XinBatch != nullptr;
      }

      bool
      hasXoutBatch() const
      {
        return d_XoutBatch != nullptr;
      }

      bool
      hasXinBatchSmall() const
      {
        return d_XinBatchSmall != nullptr;
      }

      bool
      hasXoutBatchSmall() const
      {
        return d_XoutBatchSmall != nullptr;
      }

      size_type
      getXinBatchSize() const
      {
        return d_XinBatch ? d_XinBatch->getNumberComponents() : 0;
      }

      size_type
      getXoutBatchSize() const
      {
        return d_XoutBatch ? d_XoutBatch->getNumberComponents() : 0;
      }

      size_type
      getXinBatchSmallSize() const
      {
        return d_XinBatchSmall ? d_XinBatchSmall->getNumberComponents() : 0;
      }

    private:
      std::shared_ptr<MultiVector<ValueType, memorySpace>> d_XinBatch;
      std::shared_ptr<MultiVector<ValueType, memorySpace>> d_XoutBatch;
      std::shared_ptr<MultiVector<ValueType, memorySpace>> d_XinBatchSmall;
      std::shared_ptr<MultiVector<ValueType, memorySpace>> d_XoutBatchSmall;
      bool                                                 d_inUse;
    };

  } // end of namespace linearAlgebra
} // end of namespace dftefe

#endif // dftefeMultivectorScratch_h
