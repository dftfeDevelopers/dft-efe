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
#ifndef dftefeOptimizedIndexSet_h
#define dftefeOptimizedIndexSet_h

#include <utils/TypeConfig.h>
#include <set>
#include <vector>
namespace dftefe
{
  namespace utils
  {
    /*
     * @brief Class to create an optimized index set which
     * creates contiguous sub-ranges within an index set for faster
     * search operation. This is useful when the number of contiguous sub-ranges
     * are fewer compared to the size of the index set. If the number of
     * contiguous sub-ranges competes with the size of the index set (i.e., the
     * index set is very random) then it default to the behavior of an std::set.
     * The default to STL set is governed by the numRangesToSetSizeTol value.
     * That is, if the ratio of number of contiguous ranges to the size of index
     * set exceeds numRangesToSetSizeTol, it defaults to the behavior of STL
     * set. The default value for numRangesToSetSizeTol is 0.1
     *
     * @tparam ValueType The data type of the indices (e.g., unsigned int, unsigned long int)
     */

    template <typename T>
    class OptimizedIndexSet
    {
    public:
      OptimizedIndexSet();
      OptimizedIndexSet(const std::set<T> &inputSet);
      ~OptimizedIndexSet() = default;

      void
      getPosition(const T &index, size_type &pos, bool &found) const;

    private:
      /// Store the number of contiguous ranges in the input set of indices
      size_type d_numContiguousRanges;

      /*
       * Vector of size 2*(d_numContiguousRanges in d_set).
       * The entries are arranged as:
       * <contiguous range1 startId> <continguous range1 endId> <contiguous
       * range2 startId> <continguous range2 endId> ... NOTE: The endId is one
       * past the lastId in the continguous range
       */
      std::vector<T> d_contiguousRanges;

      /// Vector of size d_numContiguousRanges which stores the accumulated
      /// number of elements in d_set prior to the i-th contiguous range
      std::vector<size_type> d_numEntriesBefore;
    };

  } // end of namespace utils

} // end of namespace dftefe
#include <utils/OptimizedIndexSet.t.cpp>
#endif // dftefeOptimizedSet_h
