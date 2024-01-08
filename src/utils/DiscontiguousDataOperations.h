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

#ifndef dftefeDiscontiguousDataOperations_h
#define dftefeDiscontiguousDataOperations_h

#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <utils/TypeConfig.h>

namespace dftefe
{
  namespace utils
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class DiscontiguousDataOperations
    {
    public:
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;

      /**
       * @brief Function to copy a source array \p x to a destination array \p y,
       * where the source is accessed discontiguously but the destination is
       * accessed contiguously. In other words, \f$ y[i] = x[d_i],
       * i=0,1,2,\ldots,N-1\f$, where \f$\{d_0, d_1, d_2, \ldots, d_{N-1}\}\f$
       * define a set of discontiguous indices for \p x. In practice, we extend
       * the above to multi-component case, wherein both \p x and \y contain
       * multiple-components for each index. We assume the components to be
       * stored contiguously for each index. Thus, if there are \f$C\f$
       * components, the function does \f$ y[i*C + j] = x[d_i*C + j],
       * i=0,1,2,\ldots,N-1\f$ and \f$j=0,1,2,\ldots,C-1\f$.
       * @tparam ValueType the type of the data to be copies (e.g., int, double, float, complex<double>, etc.)
       * @tparam memorySpace Memory where both \p x and \p y reside (e.g., utils::MemorySpace::HOST, utils::MemorySpace::DEVICE, etc.)
       * @param[in] src Pointer to the \p x array (source array)
       * @param[out] dst Pointer to the \p y array (destination array)
       * @param[in] discontIds Pointer to array containing the discontiguous
       * indices of the source array (\f$d_i\f$'s in the above description)
       * @param[in] N Non-negative integer specifying the number of
       * discontiguous indices
       * @param[in] nComponents Number of components (i.e., \f$C\f$ in the above
       * description)
       * @note The \p dst array is not allocated internally and hence must be pre-allocted appropriately.
       * That is, dst[i*C + j] must be s a valid memory access for a i's and j's
       * @note No size size consistency is performed on the \p src array and hemce must be pre-allocated appropriately.
       * That is, src[d_i*C + j] must be s a valid memory access for a d_i's and
       * j's
       */
      static void
      copyFromDiscontiguousMemory(const ValueType *src,
                                  ValueType *      dst,
                                  const size_type *discontIds,
                                  const size_type  N,
                                  const size_type  nComponents);

      /**
       * @brief Function to copy a source array \p x to a destination array \p y,
       * where the source is accessed contiguously but the destination is
       * accessed discontiguously. In other words, \f$ y[d_i] = x[i],
       * i=0,1,2,\ldots,N-1\f$, where \f$\{d_0, d_1, d_2, \ldots, d_{N-1}\}\f$
       * define a set of discontiguous indices for \p y. In practice, we extend
       * the above to multi-component case, wherein both \p x and \y contain
       * multiple-components for each index. We assume the components to be
       * stored contiguously for each index. Thus, if there are \f$C\f$
       * components, the function does \f$ y[d_i*C + j] = x[i*C + j],
       * i=0,1,2,\ldots,N-1\f$ and \f$j=0,1,2,\ldots,C-1\f$.
       * @note It assumes that \p y preallocated to a size that is atleast \p N (i.e., number of discontiguous indices)
       * @tparam ValueType the type of the data to be copies (e.g., int, double, float, complex<double>, etc.)
       * @tparam memorySpace Memory where both \p x and \p y reside (e.g., utils::MemorySpace::HOST, utils::MemorySpace::DEVICE, etc.)
       * @param[in] src Pointer to the \p x array (source array)
       * @param[out] dst Pointer to the \p y array (destination array)
       * @param[in] discontIds Pointer to array containing the discontiguous
       * indices of the destination array (\f$d_i\f$'s in the above description)
       * @param[in] N Non-negative integer specifying the number of
       * discontiguous indices
       * @param[in] nComponents Number of components (i.e., \f$C\f$ in the above
       * description)
       * @note The \p dst array is not allocated internally and hence must be pre-allocted appropriately.
       * That is, dst[d_i*C + j] must be a valid memory access for all d_i's and
       * j's
       * @note No size size consistency is performed on the \p src array and must be pre-allocated appropriately.
       * That is, src[i*C + j] must be a valid memory access for all i's and j's
       */
      static void
      copyToDiscontiguousMemory(const ValueType *src,
                                ValueType *      dst,
                                const size_type *discontIds,
                                const size_type  N,
                                const size_type  nComponents);

      /**
       * @brief Function to add a source array \p x to a destination array \p y,
       * where the source is accessed contiguously but the destination is
       * accessed discontiguously. In other words, \f$ y[d_i] = y[d_i] + x[i],
       * i=0,1,2,\ldots,N-1\f$, where \f$\{d_0, d_1, d_2, \ldots, d_{N-1}\}\f$
       * define a set of discontiguous indices for \p y. In practice, we extend
       * the above to multi-component case, wherein both \p x and \y contain
       * multiple-components for each index. We assume the components to be
       * stored contiguously for each index. Thus, if there are \f$C\f$
       * components, the function does \f$ y[d_i*C + j] = y[d_i*C + j] + x[i*C +
       * j], i=0,1,2,\ldots,N-1\f$ and \f$j=0,1,2,\ldots,C-1\f$.
       * @note It assumes that \p y preallocated to a size that is atleast \p N (i.e., number of discontiguous indices)
       * @tparam ValueType the type of the data to be copies (e.g., int, double, float, complex<double>, etc.)
       * @tparam memorySpace Memory where both \p x and \p y reside (e.g., utils::MemorySpace::HOST, utils::MemorySpace::DEVICE, etc.)
       * @param[in] src Pointer to the \p x array (source array)
       * @param[out] dst Pointer to the \p y array (destination array)
       * @param[in] discontIds Pointer to array containing the discontiguous
       * indices of the destination array (\f$d_i\f$'s in the above description)
       * @param[in] N Non-negative integer specifying the number of
       * discontiguous indices
       * @param[in] nComponents Number of components (i.e., \f$C\f$ in the above
       * description)
       * @note The \p dst array is not allocated internally and hence must be pre-allocted appropriately.
       * That is, dst[d_i*C + j] must be a valid memory access for all d_i's and
       * j's
       * @note No size size consistency is performed on the \p src array and must be pre-allocated appropriately.
       * That is, src[i*C + j] must be a valid memory access for all i's and j's
       */
      static void
      addToDiscontiguousMemory(const ValueType *src,
                               ValueType *      dst,
                               const size_type *discontIds,
                               const size_type  N,
                               const size_type  nComponents);
    }; // end of class DiscontiguousDataOperations


// partial template specialization for DEVICE
#ifdef DFTEFE_WITH_DEVICE
    template <typename ValueType>
    class DiscontiguousDataOperations<ValueType, utils::MemorySpace::DEVICE>
    {
    public:
      using SizeTypeVector =
        utils::MemoryStorage<size_type, MemorySpace::DEVICE>;

      /**
       * @brief Function to copy a source array \p x to a destination array \p y,
       * where the source is accessed discontiguously but the destination is
       * accessed contiguously. In other words, \f$ y[i] = x[d_i],
       * i=0,1,2,\ldots,N-1\f$, where \f$\{d_0, d_1, d_2, \ldots, d_{N-1}\}\f$
       * define a set of discontiguous indices for \p x. In practice, we extend
       * the above to multi-component case, wherein both \p x and \y contain
       * multiple-components for each index. We assume the components to be
       * stored contiguously for each index. Thus, if there are \f$C\f$
       * components, the function does \f$ y[i*C + j] = x[d_i*C + j],
       * i=0,1,2,\ldots,N-1\f$ and \f$j=0,1,2,\ldots,C-1\f$.
       * @note It assumes that \p y preallocated to a size that is atleast \p N (i.e., number of discontiguous indices)
       * @tparam ValueType the type of the data to be copies (e.g., int, double, float, complex<double>, etc.)
       * @tparam memorySpace Memory where both \p x and \p y reside (e.g., utils::MemorySpace::HOST, utils::MemorySpace::DEVICE, etc.)
       * @param[in] src Pointer to the \p x array (source array)
       * @param[out] dst Pointer to the \p y array (destination array)
       * @param[in] discontIds Pointer to array containing the discontiguous
       * indices of the source array (\f$d_i\f$'s in the above description)
       * @param[in] N Non-negative integer specifying the number of
       * discontiguous indices
       * @param[in] nComponents Number of components (i.e., \f$C\f$ in the above
       * description)
       * @note The \p dst array is not allocated internally and hence must be pre-allocted appropriately.
       * That is, dst[i*C + j] must be s a valid memory access for a i's and j's
       * @note No size size consistency is performed on the \p src array and hemce must be pre-allocated appropriately.
       * That is, src[d_i*C + j] must be s a valid memory access for a d_i's and
       * j's
       */
      static void
      copyFromDiscontiguousMemory(const ValueType *src,
                                  ValueType *      dst,
                                  const size_type *discontIds,
                                  const size_type  N,
                                  const size_type  nComponents);

      /**
       * @brief Function to copy a source array \p x to a destination array \p y,
       * where the source is accessed contiguously but the destination is
       * accessed discontiguously. In other words, \f$ y[d_i] = x[i],
       * i=0,1,2,\ldots,N-1\f$, where \f$\{d_0, d_1, d_2, \ldots, d_{N-1}\}\f$
       * define a set of discontiguous indices for \p y. In practice, we extend
       * the above to multi-component case, wherein both \p x and \y contain
       * multiple-components for each index. We assume the components to be
       * stored contiguously for each index. Thus, if there are \f$C\f$
       * components, the function does \f$ y[d_i*C + j] = x[i*C + j],
       * i=0,1,2,\ldots,N-1\f$ and \f$j=0,1,2,\ldots,C-1\f$.
       * @note It assumes that \p y preallocated to a size that is atleast \p N (i.e., number of discontiguous indices)
       * @tparam ValueType the type of the data to be copies (e.g., int, double, float, complex<double>, etc.)
       * @tparam memorySpace Memory where both \p x and \p y reside (e.g., utils::MemorySpace::HOST, utils::MemorySpace::DEVICE, etc.)
       * @param[in] src Pointer to the \p x array (source array)
       * @param[out] dst Pointer to the \p y array (destination array)
       * @param[in] discontIds Pointer to array containing the discontiguous
       * indices of the destination array (\f$d_i\f$'s in the above description)
       * @param[in] N Non-negative integer specifying the number of
       * discontiguous indices
       * @param[in] nComponents Number of components (i.e., \f$C\f$ in the above
       * description)
       * @note The \p dst array is not allocated internally and hence must be pre-allocted appropriately.
       * That is, dst[d_i*C + j] must be a valid memory access for all d_i's and
       * j's
       * @note No size size consistency is performed on the \p src array and must be pre-allocated appropriately.
       * That is, src[i*C + j] must be a valid memory access for all i's and j's
       */
      static void
      copyToDiscontiguosMemory(const ValueType *src,
                               ValueType *      dst,
                               const size_type *discontIds,
                               const size_type  N,
                               const size_type  nComponents);

      /**
       * @brief Function to add a source array \p x to a destination array \p y,
       * where the source is accessed contiguously but the destination is
       * accessed discontiguously. In other words, \f$ y[d_i] = y[d_i] + x[i],
       * i=0,1,2,\ldots,N-1\f$, where \f$\{d_0, d_1, d_2, \ldots, d_{N-1}\}\f$
       * define a set of discontiguous indices for \p y. In practice, we extend
       * the above to multi-component case, wherein both \p x and \y contain
       * multiple-components for each index. We assume the components to be
       * stored contiguously for each index. Thus, if there are \f$C\f$
       * components, the function does \f$ y[d_i*C + j] = y[d_i*C + j] + x[i*C +
       * j], i=0,1,2,\ldots,N-1\f$ and \f$j=0,1,2,\ldots,C-1\f$.
       * @note It assumes that \p y preallocated to a size that is atleast \p N (i.e., number of discontiguous indices)
       * @tparam ValueType the type of the data to be copies (e.g., int, double, float, complex<double>, etc.)
       * @tparam memorySpace Memory where both \p x and \p y reside (e.g., utils::MemorySpace::HOST, utils::MemorySpace::DEVICE, etc.)
       * @param[in] src Pointer to the \p x array (source array)
       * @param[out] dst Pointer to the \p y array (destination array)
       * @param[in] discontIds Pointer to array containing the discontiguous
       * indices of the destination array (\f$d_i\f$'s in the above description)
       * @param[in] nComponents Number of components (i.e., \f$C\f$ in the above
       * description)
       * @param[in] N Non-negative integer specifying the number of
       * discontiguous indices
       * @param[in] nComponents Number of components (i.e., \f$C\f$ in the above
       * description) That is, dst[d_i*C + j] must be a valid memory access for
       * all d_i's and j's
       * @note No size size consistency is performed on the \p src array and must be pre-allocated appropriately.
       * That is, src[i*C + j] must be a valid memory access for all i's and j's
       */
      static void
      addToDiscontiguosMemory(const ValueType *src,
                              ValueType *      dst,
                              const size_type *discontIds,
                              const size_type  N,
                              const size_type  nComponents);
    }; // end of class DiscontiguousDataOperations for DEVICE
#endif // DFTEFE_WITH_DEVICE

  } // end of namespace utils
} // namespace dftefe
#endif // dftefeDiscontiguousDataOperations_h
