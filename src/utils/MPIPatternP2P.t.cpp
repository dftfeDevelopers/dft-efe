
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

#include <utils/Exceptions.h>
#include <utils/MPITags.h>
#include <utils/MPIRequestersBase.h>
#include <utils/MPIRequestersNBX.h>
#include <utils/MPIWrapper.h>
#include <utils/MPIErrorCodeHandler.h>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include <memory>
#include <numeric>

namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      namespace 
      {
        template <typename T>
        bool
        comparePairsByFirst(const T &a, const T &b)
        {
          if (a.first == b.first)
            return a.second < b.second;
          else
            return a.first < b.first;
        }

        template <typename T>
        bool
        comparePairsBySecond(const T &a, const T &b)
        {
          if (a.second == b.second)
            return a.first < b.first;
          else
            return a.second < b.second;
        }

        /**
         * @brief Given an array of intervals, returns a sorted array of intervals
         * @param[in] ranges Input array of intervals
         * @param[in] compareByFirst Flag to tell whether to sort the ranges by
         * their start or end points. If true: (a) it compares by the start
         * point of the intervals; (b) if two intervals have the same start
         * point, it then compares using the end points. If false: (a) it
         * compares by end points of the intervals; (b) if two intervals have
         * the same end point, it then compares using the start points.
         * @param[in] ignoreEmptyRanges Flag to tell whether to ignore intervals
         * in \p ranges that are empty. If true, the size of the output \p
         * rangesSorted is the number of non-empty intervals in the input \p
         * ranges. If false, the empty intervals are retained.
         * @param[out] rangesSorted Output sorted array of intervals
         */
        void
        arrangeRanges(
          const std::vector<std::pair<global_size_type, global_size_type>>
            &  ranges,
          bool compareByFirst,
          bool ignoreEmptyRanges,
          std::vector<std::pair<global_size_type, global_size_type>>
            &rangesSorted)
        {
          const size_type nRanges = ranges.size();
          rangesSorted.resize(0);
          rangesSorted.reserve(nRanges);
          if (ignoreEmptyRanges == false)
            {
              rangesSorted = ranges;
            }

          else
            {
              for (unsigned int i = 0; i < nRanges; ++i)
                {
                  const size_type rangeSize =
                    ranges[i].second - ranges[i].first;
                  if (rangeSize != 0)
                    {
                      rangesSorted.push_back(ranges[i]);
                    }
                }
            }

          if (compareByFirst)
            std::sort(rangesSorted.begin(),
                      rangesSorted.end(),
                      comparePairsByFirst<
                        std::pair<global_size_type, global_size_type>>);

          else
            std::sort(rangesSorted.begin(),
                      rangesSorted.end(),
                      comparePairsBySecond<
                        std::pair<global_size_type, global_size_type>>);
        }

        /**
         * @brief Given an array of intervals, returns a sorted array of intervals
         * @param[in] ranges Input array of intervals
         * @param[in] compareByFirst Flag to tell whether to sort the ranges by
         * their start or end points. If true: (a) it compares by the start
         * point of the intervals; (b) if two intervals have the same start
         * point, it then compares using the end points. If false: (a) it
         * compares by end points of the intervals; (b) if two intervals have
         * the same end point, it then compares using the start points.
         * @param[in] ignoreEmptyRanges Flag to tell whether to ignore intervals
         * in \p ranges that are empty. If true, the size of the output \p
         * rangesSorted and \p indexPermutation is the number of non-empty
         * intervals in the input \p ranges. If false, the empty intervals are
         * retained.
         * @param[out] rangesSorted Output sorted array of intervals
         * @param[out] indexPermutation Stores the permutation of the intervals
         * due to sorting. That is, \p indexPermutation[i] tells where the i-th
         * interval in \p rangesSorted resided in the original unsorted \p
         * ranges
         */
        void
        arrangeRanges(
          const std::vector<std::pair<global_size_type, global_size_type>>
            &  ranges,
          bool compareByFirst,
          bool ignoreEmptyRanges,
          std::vector<std::pair<global_size_type, global_size_type>>
            &                     rangesSorted,
          std::vector<size_type> &indexPermutation)
        {
          const size_type               nRanges = ranges.size();
          std::vector<global_size_type> points(nRanges, 0);
          std::vector<
            std::pair<size_type, std::pair<global_size_type, global_size_type>>>
            idAndRanges;
          idAndRanges.reserve(nRanges);

          if (ignoreEmptyRanges == false)
            {
              for (unsigned int i = 0; i < nRanges; ++i)
                {
                  idAndRanges.push_back(std::make_pair(i, ranges[i]));
                }
            }

          else
            {
              for (unsigned int i = 0; i < nRanges; ++i)
                {
                  const size_type rangeSize =
                    ranges[i].second - ranges[i].first;
                  if (rangeSize != 0)
                    {
                      idAndRanges.push_back(std::make_pair(i, ranges[i]));
                    }
                }
            }

          if (compareByFirst)
            {
              std::sort(idAndRanges.begin(),
                        idAndRanges.end(),
                        [](auto &left, auto &right) {
                          return comparePairsByFirst(left.second, right.second);
                        });
            }

          else
            {
              std::sort(idAndRanges.begin(),
                        idAndRanges.end(),
                        [](auto &left, auto &right) {
                          return comparePairsBySecond(left.second,
                                                      right.second);
                        });
            }

          const size_type nNonEmptyRanges = idAndRanges.size();
          rangesSorted.resize(nNonEmptyRanges);
          indexPermutation.resize(nNonEmptyRanges);
          for (unsigned int i = 0; i < nNonEmptyRanges; ++i)
            {
              indexPermutation[i] = idAndRanges[i].first;
              rangesSorted[i]     = idAndRanges[i].second;
            }
        }

        //
        // Checks if an input set of integers are in strictly increasing order.
        // Returns true if they are strictly increasing, else returns false
        //
        bool
        checkStrictlyIncreasing(const std::vector<global_size_type> &vals)
        {
          for (size_type i = 1; i < vals.size(); ++i)
            {
              if (vals[i] <= vals[i - 1])
                return false;
            }

          return true;
        }

        //
        // For a sorted vector of non-overlapping ranges, checks if the union of
        // the ranges for a contiguous set of integers. Returns true if the
        // union forms a contiguous set, else returns false
        //
        bool
        checkContiguity(
          const std::vector<std::pair<global_size_type, global_size_type>>
            &ranges)
        {
          const size_type N = ranges.size();
          for (unsigned int i = 1; i < N; ++i)
            {
              if (ranges[i - 1].second != ranges[i].first)
                return false;
            }
          return true;
        }


        // For a sorted vector of ranges, checks if the union of the ranges
        // for a contiguous set of integers.
        // Returns true if the union forms
        // a contiguous set, else returns false
        bool
        checkContiguity(const std::vector<size_type> &v)
        {
          const size_type N           = v.size();
          bool            returnValue = true;
          for (unsigned int i = 1; i < N; ++i)
            {
              if ((v[i] - 1) != v[i - 1])
                return false;
            }
          return true;
        }

        //
        // Given an array of intervals (i.e., array of pairs) checks if any two
        // intervals overlap/intersect or not. Returns true if there is any
        // overlapping interval. The logic is simple: (i) sort the intervals
        // based on their start (or end) points, (ii) traverse the sorted set of
        // intervals. If the start of the current interval is lower than the end
        // of the previous interval, then the two intervals intersect
        //
        //
        bool
        containsOverlappingRanges(
          const std::vector<std::pair<global_size_type, global_size_type>>
            &ranges)
        {
          const size_type nRanges = ranges.size();
          std::vector<std::pair<global_size_type, global_size_type>>
            rangesSorted(0);
          arrangeRanges(ranges,
                        true,  // compareByFirst,
                        false, // do not ignore empty ranges
                        rangesSorted);

          for (size_type i = 1; i < nRanges; ++i)
            {
              if (rangesSorted[i].first < rangesSorted[i - 1].second)
                return true;
            }

          return false;
        }

        /**
         * @bried Given a sorted array of non-overlapping ranges and a given
         * value, find
         *  1. if the value belongs to any of the ranges
         *  2. the index of the range to which it belongs
         *
         * @note The following assumptions must hold:
         *  1. The ranges are assumed to be half-open [a,b) (i.e., a is
         * included, but b is not).
         *  2. The ranges are assumed to be non-overlapping
         *
         *
         * @param[in] ranges Input sorted ranges
         * @param[in] val Value to search
         * @param[out] found Boolean to store if \p val belongs to any of the
         * input ranges
         * @param[out] rangeId Stores the index of the range to which \p val
         * belongs. It has undefined value if \p val is not present in any of
         * ranges (i.e., if \p found is false)
         */
        template <typename T>
        void
        findRange(const std::vector<std::pair<T, T>> &ranges,
                  const T &                           val,
                  bool &                              found,
                  size_type &                         rangeId)
        {
          const size_type nRanges = ranges.size();
          std::vector<T>  rangesFlattened(2 * nRanges);
          for (size_type i = 0; i < nRanges; ++i)
            {
              rangesFlattened[2 * i]     = ranges[i].first;
              rangesFlattened[2 * i + 1] = ranges[i].second;
            }

          found = false;
          /*
           * The logic used for finding an index is as follows:
           * 1. Find the first the element in rangesFlattened
           *    which is greater than (strictly greater) the input val.
           *    Let's call this element upVal and its position in
           * rangesFlattened as upPos. The complexity of finding it is
           * O(log(nRanges))
           * 2. Since rangesFlattened stores pairs of startId and endId
           *    (endId not inclusive) of contiguous ranges,
           *    any index for which upPos is even (i.e., it corresponds to a
           *    startId) cannot belong to the input ranges. Why? Consider two
           * consequtive ranges [k1,k2) and [k3,k4) where k1 < k2 <= k3 < k4
           * (NOTE: k2 can be equal to k3). If upVal for val corresponds to k3
           * (i.e., startId of a range), then (a) val does not lie in the
           * [k3,k4) as val < upVal (=k3). (b) val cannot lie in [k1,k2),
           * because if it lies in [k1,k2), then upVal should have been be k2
           * (not k3)
           *  3. If upPos is odd (i.e, it corresponds to an endId), then check
           * if the rangeId = upPos/2 (integer part of it) is a non-empty range
           * or not. If rangeId is an non-empty, set found = true, else set
           * found = false
           */

          auto up = std::upper_bound(rangesFlattened.begin(),
                                     rangesFlattened.end(),
                                     val);
          if (up != rangesFlattened.end())
            {
              size_type upPos = std::distance(rangesFlattened.begin(), up);
              if (upPos % 2 == 1)
                {
                  rangeId = upPos / 2;
                  if ((rangesFlattened[2 * rangeId + 1] -
                       rangesFlattened[2 * rangeId]) != 0)
                    found = true;
                }
            }
        }
        /**
         * @brief The current processor has \f$k\f$ different locally owned ranges.
         * For each of the \f$k\f$ ranges, this function collates the locally
         * owned ranges from all the processors that are part of the input \p
         * mpiComm
         *
         */
        void
        getAllOwnedRanges(
          const std::vector<std::pair<global_size_type, global_size_type>>
            &locallyOwnedRanges,
          std::vector<
            std::vector<std::pair<global_size_type, global_size_type>>>
            &            allOwnedRanges,
          const MPIComm &mpiComm)
        {
          int         nprocs = 1;
          int         err    = MPICommSize(mpiComm, &nprocs);
          std::string errMsg = "Error occured while using MPI_Comm_size. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPISuccess, errMsg);
          const size_type nRanges = locallyOwnedRanges.size();
          allOwnedRanges.resize(
            nRanges,
            std::vector<std::pair<global_size_type, global_size_type>>(nprocs));
          for (unsigned int iRange = 0; iRange < nRanges; ++iRange)
            {
              std::vector<int> recvCounts(nprocs, 2);
              std::vector<int> displs(nprocs, 0);
              for (unsigned int i = 0; i < nprocs; ++i)
                displs[i] = 2 * i;

              std::vector<global_size_type> ownedRanges = {
                locallyOwnedRanges[iRange].first,
                locallyOwnedRanges[iRange].second};

              std::vector<global_size_type> ownedRangesAcrossProcs(2 * nprocs);

              MPIAllgatherv<MemorySpace::HOST>(&ownedRanges[0],
                                               2,
                                               MPIUnsignedLong,
                                               &ownedRangesAcrossProcs[0],
                                               &recvCounts[0],
                                               &displs[0],
                                               MPIUnsignedLong,
                                               mpiComm);

              for (size_type iProc = 0; iProc < nprocs; ++iProc)
                {
                  allOwnedRanges[iRange][iProc] =
                    std::make_pair(ownedRangesAcrossProcs[2 * iProc],
                                   ownedRangesAcrossProcs[2 * iProc + 1]);
                }
            }
        }



        /**
         * @brief Let each processor contain \f$k\f$ different locally owned ranges.
         * For each of the \f$k\f$ ranges, let \p allOwnedRanges store the
         * locally owned ranges collated across all the processors. This
         * functions finds the global start and end within each of the \f$k\f$
         * ranges. By global, we mean across all the processors
         */
        void
        getGlobalRangesStartAndEnd(
          const std::vector<
            std::vector<std::pair<global_size_type, global_size_type>>>
            &allOwnedRanges,
          std::vector<std::pair<global_size_type, global_size_type>>
            &rangesGlobalStartAndEnd)
        {
          const size_type nRanges = allOwnedRanges.size();
          const size_type nProcs  = allOwnedRanges[0].size();
          rangesGlobalStartAndEnd.resize(nRanges);
          std::vector<global_size_type> flattenedOwnedRanges(2 * nProcs);
          for (size_type iRange = 0; iRange < nRanges; ++iRange)
            {
              for (size_type iProc = 0; iProc < nProcs; ++iProc)
                {
                  flattenedOwnedRanges[2 * iProc] =
                    allOwnedRanges[iRange][iProc].first;
                  flattenedOwnedRanges[2 * iProc + 1] =
                    allOwnedRanges[iRange][iProc].second;
                }

              global_size_type a =
                *std::min_element(flattenedOwnedRanges.begin(),
                                  flattenedOwnedRanges.end());
              global_size_type b =
                *std::max_element(flattenedOwnedRanges.begin(),
                                  flattenedOwnedRanges.end());

              rangesGlobalStartAndEnd[iRange] = std::make_pair(a, b);
            }
        }

        void
        getGhostIndicesRangeId(
          const std::vector<global_size_type> &ghostIndices,
          const std::vector<std::pair<global_size_type, global_size_type>>
            &                     globalRanges,
          std::vector<size_type> &ghostIndicesRangeId)
        {
          const size_type nRanges = globalRanges.size();

          std::vector<std::pair<global_size_type, global_size_type>>
                                 globalRangesSorted(0);
          std::vector<size_type> globalRangesIndexPermutation(0);
          arrangeRanges(globalRanges,
                        true,  /*compareByFirst*/
                        false, /* do not ignore empty ranges*/
                        globalRangesSorted,
                        globalRangesIndexPermutation);

          const size_type numGhosts = ghostIndices.size();
          ghostIndicesRangeId.resize(numGhosts);
          for (unsigned int iGhost = 0; iGhost < numGhosts; ++iGhost)
            {
              bool      found   = false;
              size_type rangeId = 0;
              findRange(globalRangesSorted,
                        ghostIndices[iGhost],
                        found,
                        rangeId);
              if (found)
                ghostIndicesRangeId[iGhost] =
                  globalRangesIndexPermutation[rangeId];
              else
                throwException<LogicError>(
                  false,
                  "In MPIPatternP2P, cannot find ghost index in any of the global ranges.");
            }
        }



        void
        getGhostProcIdToLocalGhostIndicesMap(
          const std::vector<global_size_type> &ghostIndices,
          const std::vector<
            std::vector<std::pair<global_size_type, global_size_type>>>
            &                           allOwnedRanges,
          const std::vector<size_type> &ghostIndicesRangeId,
          std::map<size_type, std::vector<size_type>>
            &            ghostProcIdToLocalGhostIndices,
          const MPIComm &mpiComm)
        {
          int         nprocs = 1;
          int         err    = MPICommSize(mpiComm, &nprocs);
          std::string errMsg = "Error occured while using MPI_Comm_size. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPISuccess, errMsg);

          const size_type nRanges   = allOwnedRanges.size();
          const size_type numGhosts = ghostIndices.size();
          for (size_type iRange = 0; iRange < nRanges; ++iRange)
            {
              //
              // NOTE: For rangeId iRange, the locally owned ranges need not be
              // ordered as per the processor ranks. That is ranges for
              // processor 0, 1, ...., P-1 given by [N_0,N_1), [N_1, N_2), [N_2,
              // N_3), ..., [N_{P-1},N_P) need not honor the fact that N_0, N_1,
              // ..., N_P are increasing. However, it is more efficient to
              // perform search operations in a sorted vector. Thus, we perform
              // a sort on the end of each locally owned range and also keep
              // track of the indices during the sort
              //

              std::vector<std::pair<global_size_type, global_size_type>>
                                     iRangesSorted(0);
              std::vector<size_type> iRangesProcIdPermutation(0);
              arrangeRanges(allOwnedRanges[iRange],
                            true,  /*compareByFirst*/
                            false, /*do not ignore empty ranges*/
                            iRangesSorted,
                            iRangesProcIdPermutation);

              for (unsigned int iGhost = 0; iGhost < numGhosts; ++iGhost)
                {
                  if (iRange == ghostIndicesRangeId[iGhost])
                    {
                      bool      foundGhost = false;
                      size_type procIdSorted;
                      findRange(iRangesSorted,
                                ghostIndices[iGhost],
                                foundGhost,
                                procIdSorted);
                      if (foundGhost)
                        {
                          const size_type procId =
                            iRangesProcIdPermutation[procIdSorted];
                          ghostProcIdToLocalGhostIndices[procId].push_back(
                            iGhost);
                        }
                      else
                        {
                          std::string msg =
                            "Ghost index " +
                            std::to_string(ghostIndices[iGhost]) +
                            " not found in any of the processors";
                          throwException<LogicError>(false, msg);
                        }
                    }
                }
            }
        }


        struct RangeMetaData
        {
          global_size_type Id;
          size_type        rangeId;
          bool             isRangeStart;
        };

        bool
        compareRangeMetaData(const RangeMetaData &x, const RangeMetaData &y)
        {
          if (x.Id == y.Id)
            return (!x.isRangeStart);
          else
            return x.Id < y.Id;
        }

        std::vector<size_type>
        getOverlappingRangeIds(const std::vector<global_size_type> &ranges)
        {
          size_type                  numRanges = ranges.size() / 2;
          std::vector<RangeMetaData> rangeMetaDataVec(0);
          for (unsigned int i = 0; i < numRanges; ++i)
            {
              RangeMetaData left;
              left.Id           = ranges[2 * i];
              left.rangeId      = i;
              left.isRangeStart = true;

              RangeMetaData right;
              right.Id           = ranges[2 * i + 1];
              right.rangeId      = i;
              right.isRangeStart = false;

              // This check is required to ignore ranges with 0 elements
              if (left.Id != right.Id)
                {
                  rangeMetaDataVec.push_back(left);
                  rangeMetaDataVec.push_back(right);
                }
            }

          std::sort(rangeMetaDataVec.begin(),
                    rangeMetaDataVec.end(),
                    compareRangeMetaData);
          int                    currentOpen = -1;
          bool                   added       = false;
          std::vector<size_type> returnValue(0);
          for (unsigned int i = 0; i < rangeMetaDataVec.size(); ++i)
            {
              size_type rangeId = rangeMetaDataVec[i].rangeId;
              if (rangeMetaDataVec[i].isRangeStart)
                {
                  if (currentOpen == -1)
                    {
                      currentOpen = rangeId;
                      added       = false;
                    }
                  else
                    {
                      if (!added)
                        {
                          returnValue.push_back(currentOpen);
                          added = true;
                        }
                      returnValue.push_back(rangeId);
                      if (ranges[2 * rangeId + 1] > ranges[2 * currentOpen + 1])
                        {
                          currentOpen = rangeId;
                          added       = true;
                        }
                    }
                }
              else
                {
                  if (rangeId == currentOpen)
                    {
                      currentOpen = -1;
                      added       = false;
                    }
                }
            }
          return returnValue;
        }

        void
        checkNumRanges(const size_type nRanges, const MPIComm &mpiComm)
        {
          int         nprocs = 1;
          int         err    = MPICommSize(mpiComm, &nprocs);
          std::string errMsg = "Error occured while using MPI_Comm_size. "
                               "Error code: " +
                               std::to_string(err);
          throwException(err == MPISuccess, errMsg);
          size_type nRangesMax;
          MPIAllreduce<MemorySpace::HOST>(&nRanges,
                                          &nRangesMax,
                                          1,
                                          Types<size_type>::getMPIDatatype(),
                                          MPIMax,
                                          mpiComm);
          throwException<LogicError>(
            nRanges == nRangesMax,
            "Different number of ranges passed to different "
            "processors in MPIPatternP2P ");
        }

        void
        checkGlobalRanges(
          const std::vector<std::pair<global_size_type, global_size_type>>
            &globalRanges)
        {
          std::string msg = "In MPIPatternP2P, found overlap between "
                            "two or more of the global ranges";
          throwException<LogicError>(containsOverlappingRanges(globalRanges) ==
                                       false,
                                     msg);
        }

        void
        checkOwnedRangesAssumptions(
          const std::vector<std::pair<global_size_type, global_size_type>>
            &locallyOwnedRanges,
          const std::vector<
            std::vector<std::pair<global_size_type, global_size_type>>>
            &             allOwnedRanges,
          const size_type processorRank)
        {
          const size_type nRanges = locallyOwnedRanges.size();
          const size_type nProcs  = allOwnedRanges[0].size();
          //
          // for each pair in locallyOwnedRanges (say a and b), check if b >= a
          //
          for (size_type iRange = 0; iRange < nRanges; ++iRange)
            {
              if (locallyOwnedRanges[iRange].second <
                  locallyOwnedRanges[iRange].first)
                {
                  std::string msg = "In processor " +
                                    std::to_string(processorRank) +
                                    ", in one of "
                                    "the locally owned ranges the range start "
                                    "point is greater than its end point.";
                  throwException<LogicError>(false, msg);
                }
            }

          // flatten allOwnedRanges
          std::vector<std::pair<global_size_type, global_size_type>>
            allOwnedRangesFlattened(nRanges * nProcs);
          for (size_type iRange = 0; iRange < nRanges; ++iRange)
            {
              for (size_type iProc = 0; iProc < nProcs; ++iProc)
                allOwnedRangesFlattened[iRange * nProcs + iProc] =
                  allOwnedRanges[iRange][iProc];
            }

          std::string msg = "In MPIPatternP2P, among all locally "
                            "owned ranges collated across all the processors, "
                            " found two or more ranges that overlap.";
          // check if any two ranges in allOwnedRangesFlattened overlap
          throwException<LogicError>(
            containsOverlappingRanges(allOwnedRangesFlattened) == false, msg);

          //
          // check if for each rangeId, the union of the respective
          // locallyOwnedRanges from all the processors form a contiguous set
          //
          for (size_type iRange = 0; iRange < nRanges; ++iRange)
            {
              std::vector<std::pair<global_size_type, global_size_type>>
                                     iRangesSorted(0);
              std::vector<size_type> idPermutation(
                0); // required only for the following function call
              arrangeRanges(allOwnedRanges[iRange],
                            true,  /*compureByFirst*/
                            false, /*do not ignore emty ranges*/
                            iRangesSorted,
                            idPermutation);

              bool isContiguous = checkContiguity(iRangesSorted);
              msg               = "In MPIPatternP2P, the union of the " +
                    std::to_string(iRange) + "-th locallyOwnedRange " +
                    " from all processors does not form a contiguous "
                    "set of integers";
              throwException<LogicError>(isContiguous, msg);
            }

          //
          // check if
          //
        }

        void
        checkGhostIndicesAssumptions(
          const std::vector<global_size_type> &ghostIndices,
          const std::vector<std::pair<global_size_type, global_size_type>>
            &             locallyOwnedRangesSorted,
          const size_type procRank)
        {
          // check if the ghostIndices are in strictly increasing order
          throwException(
            checkStrictlyIncreasing(ghostIndices),
            "In processor " + std::to_string(procRank) +
              ", the ghost indices passed to MPIPatternP2P is not a "
              "strictly increasing set.");

          for (size_type i = 0; i < ghostIndices.size(); ++i)
            {
              bool      found;
              size_type rangeId = 0;
              findRange(locallyOwnedRangesSorted,
                        ghostIndices[i],
                        found,
                        rangeId);
              std::string msg =
                "In processor " + std::to_string(procRank) +
                ", found an overlap between its ghost indices and " +
                "locally owned ranges";
              throwException(found == false, msg);
            }
        }

      } // namespace 

#ifdef DFTEFE_WITH_MPI

      ///
      /// Constructor with MPI for multiple global ranges
      ///
      template <dftefe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(
        const std::vector<std::pair<global_size_type, global_size_type>>
          &                                          locallyOwnedRanges,
        const std::vector<dftefe::global_size_type> &ghostIndices,
        const MPIComm &                              mpiComm)
        : d_locallyOwnedRanges(0)
        , d_ghostIndices(0)
        , d_mpiComm(0)
        , d_nprocs(0)
        , d_myRank(0)
        , d_nGlobalRanges(0)
        , d_locallyOwnedRangesSorted(0)
        , d_locallyOwnedRangesIdPermutation(0)
        , d_allOwnedRanges(0)
        , d_globalRanges(0)
        , d_numLocallyOwnedIndices(0)
        , d_locallyOwnedRangesCumulativePairs(0)
        , d_numGhostIndices(0)
        , d_ghostIndicesOptimizedIndexSet(std::set<global_size_type>())
        , d_ghostIndicesRangeId(0)
        , d_numGhostProcs(0)
        , d_ghostProcIds(0)
        , d_numGhostIndicesInGhostProcs(0)
        , d_localGhostIndicesRanges(0)
        , d_numTargetProcs(0)
        , d_flattenedLocalGhostIndices(0)
        , d_ghostProcLocallyOwnedRangesCumulative(0)
        , d_targetProcIds(0)
        , d_numOwnedIndicesForTargetProcs(0)
        , d_flattenedLocalTargetIndices(0)
        , d_nGlobalIndices(0)
      {
        this->reinit(locallyOwnedRanges, ghostIndices, mpiComm);
      }

      ///
      /// Constructor with MPI for a single global ranges
      ///
      template <dftefe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(
        const std::pair<global_size_type, global_size_type> &locallyOwnedRange,
        const std::vector<dftefe::global_size_type> &        ghostIndices,
        const MPIComm &                                      mpiComm)
        : d_locallyOwnedRanges(0)
        , d_ghostIndices(0)
        , d_mpiComm(0)
        , d_nprocs(0)
        , d_myRank(0)
        , d_nGlobalRanges(0)
        , d_locallyOwnedRangesSorted(0)
        , d_locallyOwnedRangesIdPermutation(0)
        , d_allOwnedRanges(0)
        , d_globalRanges(0)
        , d_numLocallyOwnedIndices(0)
        , d_locallyOwnedRangesCumulativePairs(0)
        , d_numGhostIndices(0)
        , d_ghostIndicesOptimizedIndexSet(std::set<global_size_type>())
        , d_ghostIndicesRangeId(0)
        , d_numGhostProcs(0)
        , d_ghostProcIds(0)
        , d_numGhostIndicesInGhostProcs(0)
        , d_localGhostIndicesRanges(0)
        , d_numTargetProcs(0)
        , d_flattenedLocalGhostIndices(0)
        , d_ghostProcLocallyOwnedRangesCumulative(0)
        , d_targetProcIds(0)
        , d_numOwnedIndicesForTargetProcs(0)
        , d_flattenedLocalTargetIndices(0)
        , d_nGlobalIndices(0)
      {
        std::vector<std::pair<global_size_type, global_size_type>>
          locallyOwnedRanges(1);
        locallyOwnedRanges[0] = locallyOwnedRange;
        this->reinit(locallyOwnedRanges, ghostIndices, mpiComm);
      }


      ///
      /// reinit with MPI
      ///
      template <dftefe::utils::MemorySpace memorySpace>
      void
      MPIPatternP2P<memorySpace>::reinit(
        const std::vector<std::pair<global_size_type, global_size_type>>
          &                                          locallyOwnedRanges,
        const std::vector<dftefe::global_size_type> &ghostIndices,
        const MPIComm &                              mpiComm)
      {
        d_mpiComm          = mpiComm;
        d_myRank           = 0;
        d_nprocs           = 1;
        int         err    = MPICommSize(d_mpiComm, &d_nprocs);
        std::string errMsg = "Error occured while using MPI_Comm_size. "
                             "Error code: " +
                             std::to_string(err);
        throwException(err == MPISuccess, errMsg);

        err    = MPICommRank(d_mpiComm, &d_myRank);
        errMsg = "Error occured while using MPI_Comm_rank. "
                 "Error code: " +
                 std::to_string(err);
        throwException(err == MPISuccess, errMsg);


        d_locallyOwnedRanges = locallyOwnedRanges;
        d_ghostIndices       = ghostIndices;
        d_nGlobalRanges      = d_locallyOwnedRanges.size();

        // check if all processors have the same number of d_nGlobalRanges
        // If not, throw an exception
        checkNumRanges(d_nGlobalRanges, d_mpiComm);

        //
        // store d_allOwnedRanges
        d_allOwnedRanges.clear();
        getAllOwnedRanges(d_locallyOwnedRanges,
                                                 d_allOwnedRanges,
                                                 d_mpiComm);

        // store d_globalRanges
        getGlobalRangesStartAndEnd(d_allOwnedRanges,
                                                          d_globalRanges);

        // check assumptions on global ranges
        checkGlobalRanges(d_globalRanges);

        checkOwnedRangesAssumptions(d_locallyOwnedRanges,
                                                           d_allOwnedRanges,
                                                           d_myRank);

        d_numLocallyOwnedIndices = 0;
        d_locallyOwnedRangesCumulativePairs.resize(d_nGlobalRanges);
        for (unsigned int i = 0; i < d_nGlobalRanges; ++i)
          {
            const size_type start = d_numLocallyOwnedIndices;

            d_numLocallyOwnedIndices +=
              d_locallyOwnedRanges[i].second - d_locallyOwnedRanges[i].first;

            const size_type end = d_numLocallyOwnedIndices;

            d_locallyOwnedRangesCumulativePairs[i] = std::make_pair(start, end);
          }

        // sort d_locallyOwnedRanges
        arrangeRanges(
          d_locallyOwnedRanges,
          true,  // compareByFirst
          false, // do not ignore empty ranges
          d_locallyOwnedRangesSorted,
          d_locallyOwnedRangesIdPermutation);

        checkGhostIndicesAssumptions(
          d_ghostIndices, d_locallyOwnedRangesSorted, d_myRank);

        // get the global range id for each ghost index
        // throws an exception if any ghost index is not present in
        // any of the global ranges.
        getGhostIndicesRangeId(d_ghostIndices,
                                                      d_globalRanges,
                                                      d_ghostIndicesRangeId);

        std::set<global_size_type> ghostIndicesSetSTL;
        std::copy(d_ghostIndices.begin(),
                  d_ghostIndices.end(),
                  std::inserter(ghostIndicesSetSTL, ghostIndicesSetSTL.end()));
        d_ghostIndicesOptimizedIndexSet =
          OptimizedIndexSet<global_size_type>(ghostIndicesSetSTL);

        d_numGhostIndices = d_ghostIndices.size();

        d_nGlobalIndices = 0;
        for (unsigned int i = 0; i < d_globalRanges.size(); ++i)
          {
            d_nGlobalIndices +=
              d_globalRanges[i].second - d_globalRanges[i].first;
          }

        ///////////////////////////////////////////////////
        //////////// Ghost Data Evaluation Begin //////////
        ///////////////////////////////////////////////////
        MemoryTransfer<memorySpace, MemorySpace::HOST> memoryTransfer;

        std::map<size_type, std::vector<size_type>>
          ghostProcIdToLocalGhostIndices;
        getGhostProcIdToLocalGhostIndicesMap(
          d_ghostIndices,
          d_allOwnedRanges,
          d_ghostIndicesRangeId,
          ghostProcIdToLocalGhostIndices,
          d_mpiComm);

        d_numGhostProcs = ghostProcIdToLocalGhostIndices.size();
        d_ghostProcIds.resize(d_numGhostProcs);
        d_numGhostIndicesInGhostProcs.resize(d_numGhostProcs);
        d_localGhostIndicesRanges.resize(2 * d_numGhostProcs);

        std::vector<size_type> flattenedLocalGhostIndicesTmp(0);
        auto                   it = ghostProcIdToLocalGhostIndices.begin();
        unsigned int           iGhostProc = 0;
        size_type              offset     = 0;
        for (; it != ghostProcIdToLocalGhostIndices.end(); ++it)
          {
            d_ghostProcIds[iGhostProc] = it->first;
            const std::vector<size_type> localGhostIndicesInGhostProc =
              it->second;

            const size_type nLocalGhostInGhostProc =
              localGhostIndicesInGhostProc.size();
            d_numGhostIndicesInGhostProcs[iGhostProc] = nLocalGhostInGhostProc;

            d_localGhostIndicesRanges[2 * iGhostProc] = offset;
            d_localGhostIndicesRanges[2 * iGhostProc + 1] =
              offset + nLocalGhostInGhostProc;

            //
            // Append localGhostIndicesInGhostProc to
            // flattenedLocalGhostIndicesTmp
            //
            std::copy(localGhostIndicesInGhostProc.begin(),
                      localGhostIndicesInGhostProc.end(),
                      back_inserter(flattenedLocalGhostIndicesTmp));

            offset += nLocalGhostInGhostProc;
            ++iGhostProc;
          }

        std::string msg = "In rank " + std::to_string(d_myRank) +
                          " mismatch of"
                          " the sizes of ghost indices. Expected size: " +
                          std::to_string(d_numGhostIndices) +
                          " Received size: " +
                          std::to_string(flattenedLocalGhostIndicesTmp.size());
        throwException<DomainError>(flattenedLocalGhostIndicesTmp.size() ==
                                      d_numGhostIndices,
                                    msg);


        d_flattenedLocalGhostIndices.resize(d_numGhostIndices);
        if (d_numGhostIndices > 0)
          memoryTransfer.copy(d_numGhostIndices,
                              d_flattenedLocalGhostIndices.begin(),
                              &flattenedLocalGhostIndicesTmp[0]);

        d_ghostProcLocallyOwnedRangesCumulative.resize(
          d_numGhostProcs, std::vector<size_type>(d_nGlobalRanges));
        for (size_type iGhostProc = 0; iGhostProc < d_numGhostProcs;
             ++iGhostProc)
          {
            size_type ghostProcCumulativeLocalCount = 0;
            size_type ghostProcId = d_ghostProcIds[iGhostProc];
            for (size_type iRange = 0; iRange < d_nGlobalRanges; ++iRange)
              {
                d_ghostProcLocallyOwnedRangesCumulative[iGhostProc][iRange] =
                  ghostProcCumulativeLocalCount;

                ghostProcCumulativeLocalCount +=
                  d_allOwnedRanges[iRange][ghostProcId].second -
                  d_allOwnedRanges[iRange][ghostProcId].first;
              }
          }


        ///////////////////////////////////////////////////
        //////////// Ghost Data Evaluation End / //////////
        ///////////////////////////////////////////////////


        ///////////////////////////////////////////////////
        //////////// Target Data Evaluation Begin ////////
        ///////////////////////////////////////////////////
        std::shared_ptr<MPIRequestersBase> mpirequesters =
          std::make_shared<MPIRequestersNBX>(d_ghostProcIds, d_mpiComm);
        d_targetProcIds  = mpirequesters->getRequestingRankIds();
        d_numTargetProcs = d_targetProcIds.size();
        d_numOwnedIndicesForTargetProcs.resize(d_numTargetProcs);

        std::vector<MPIRequest> sendRequests(d_numGhostProcs);
        std::vector<MPIStatus>  sendStatuses(d_numGhostProcs);
        std::vector<MPIRequest> recvRequests(d_numTargetProcs);
        std::vector<MPIStatus>  recvStatuses(d_numTargetProcs);
        const int tag = static_cast<int>(MPITags::MPI_P2P_PATTERN_TAG);
        for (unsigned int iGhostProc = 0; iGhostProc < d_numGhostProcs;
             ++iGhostProc)
          {
            const size_type numGhostIndicesInProc =
              d_numGhostIndicesInGhostProcs[iGhostProc];
            const int ghostProcId = d_ghostProcIds[iGhostProc];
            err = MPIIsend<MemorySpace::HOST>(&numGhostIndicesInProc,
                                              1,
                                              MPIUnsigned,
                                              ghostProcId,
                                              tag,
                                              d_mpiComm,
                                              &sendRequests[iGhostProc]);
            std::string errMsg = "Error occured while using MPI_Isend. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }

        for (unsigned int iTarget = 0; iTarget < d_numTargetProcs; ++iTarget)
          {
            const int targetProcId = d_targetProcIds[iTarget];
            err                    = MPIIrecv<MemorySpace::HOST>(
              &d_numOwnedIndicesForTargetProcs[iTarget],
              1,
              MPIUnsigned,
              targetProcId,
              tag,
              d_mpiComm,
              &recvRequests[iTarget]);
            std::string errMsg = "Error occured while using MPI_Irecv. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }

        if (sendRequests.size() > 0)
          {
            err    = MPIWaitall(d_numGhostProcs,
                             sendRequests.data(),
                             sendStatuses.data());
            errMsg = "Error occured while using MPI_Waitall. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            err    = MPIWaitall(d_numTargetProcs,
                             recvRequests.data(),
                             recvStatuses.data());
            errMsg = "Error occured while using MPI_Waitall. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }

        size_type totalOwnedIndicesForTargetProcs =
          std::accumulate(d_numOwnedIndicesForTargetProcs.begin(),
                          d_numOwnedIndicesForTargetProcs.end(),
                          0);


        std::vector<size_type> flattenedLocalTargetIndicesTmp(
          totalOwnedIndicesForTargetProcs, 0);

        std::vector<size_type> localIndicesForGhostProc(d_numGhostIndices, 0);

        size_type startIndex = 0;
        for (unsigned int iGhostProc = 0; iGhostProc < d_numGhostProcs;
             ++iGhostProc)
          {
            const int numGhostIndicesInProc =
              d_numGhostIndicesInGhostProcs[iGhostProc];
            const int ghostProcId = d_ghostProcIds[iGhostProc];

            // We need to send what is the local index in the ghost processor
            // (i.e., the processor that owns the current processor's ghost
            // index)
            for (unsigned int iIndex = 0; iIndex < numGhostIndicesInProc;
                 ++iIndex)
              {
                const size_type ghostLocalIndex =
                  flattenedLocalGhostIndicesTmp[startIndex + iIndex];

                // throwException<LogicError>(ghostLocalIndex <
                //        d_ghostIndices.size(),
                //        "BUG1");

                const global_size_type ghostGlobalIndex =
                  d_ghostIndices[ghostLocalIndex];
                const size_type ghostIndexRangeId =
                  d_ghostIndicesRangeId[ghostLocalIndex];
                const global_size_type ghostProcRangeStart =
                  d_allOwnedRanges[ghostIndexRangeId][ghostProcId].first;
                localIndicesForGhostProc[startIndex + iIndex] =
                  (size_type)(ghostGlobalIndex - ghostProcRangeStart) +
                  d_ghostProcLocallyOwnedRangesCumulative[iGhostProc]
                                                         [ghostIndexRangeId];

                // throwException<LogicError>(
                //        localIndicesForGhostProc[startIndex + iIndex] <
                //        (d_allOwnedRanges[2 * ghostProcId + 1] -
                //         d_allOwnedRanges[2 * ghostProcId]),
                //        "BUG2");
              }

            err =
              MPIIsend<MemorySpace::HOST>(&localIndicesForGhostProc[startIndex],
                                          numGhostIndicesInProc,
                                          MPIUnsigned,
                                          ghostProcId,
                                          tag,
                                          d_mpiComm,
                                          &sendRequests[iGhostProc]);
            std::string errMsg = "Error occured while using MPI_Isend. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPISuccess, errMsg);
            startIndex += numGhostIndicesInProc;
          }

        startIndex = 0;
        for (unsigned int iTarget = 0; iTarget < d_numTargetProcs; ++iTarget)
          {
            const int targetProcId = d_targetProcIds[iTarget];
            const int numOwnedIndicesForTarget =
              d_numOwnedIndicesForTargetProcs[iTarget];
            err = MPIIrecv<MemorySpace::HOST>(
              &flattenedLocalTargetIndicesTmp[startIndex],
              numOwnedIndicesForTarget,
              MPIUnsigned,
              targetProcId,
              tag,
              d_mpiComm,
              &recvRequests[iTarget]);
            std::string errMsg = "Error occured while using MPI_Irecv. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPISuccess, errMsg);
            startIndex += numOwnedIndicesForTarget;
          }

        if (sendRequests.size() > 0)
          {
            err    = MPIWaitall(d_numGhostProcs,
                             sendRequests.data(),
                             sendStatuses.data());
            errMsg = "Error occured while using MPI_Waitall. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }

        if (recvRequests.size() > 0)
          {
            err    = MPIWaitall(d_numTargetProcs,
                             recvRequests.data(),
                             recvStatuses.data());
            errMsg = "Error occured while using MPI_Waitall. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPISuccess, errMsg);
          }

        for (size_type i = 0; i < totalOwnedIndicesForTargetProcs; ++i)
          {
            throwException<LogicError>(
              flattenedLocalTargetIndicesTmp[i] < d_numLocallyOwnedIndices,
              "In proc " + std::to_string(d_myRank) +
                ", detected local owned "
                "target index to be larger than (nLocallyOwnedIndices-1)" +
                " target index: " +
                std::to_string(flattenedLocalTargetIndicesTmp[i]) +
                ", number of  locally owned indices: " +
                std::to_string(d_numLocallyOwnedIndices));
          }

        d_flattenedLocalTargetIndices.resize(totalOwnedIndicesForTargetProcs);
        if (totalOwnedIndicesForTargetProcs > 0)
          memoryTransfer.copy(totalOwnedIndicesForTargetProcs,
                              d_flattenedLocalTargetIndices.begin(),
                              &flattenedLocalTargetIndicesTmp[0]);

        ///////////////////////////////////////////////////
        //////////// Target Data Evaluation End ////////
        ///////////////////////////////////////////////////
      }

#else // DFTEFE_WITH_MPI


      ///
      /// Constructor without MPI for multiple global ranges
      ///
      template <dftefe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(
        const std::vector<std::pair<global_size_type, global_size_type>>
          &                                          locallyOwnedRanges,
        const std::vector<dftefe::global_size_type> &ghostIndices,
        const MPIComm &                              mpiComm)
        : d_locallyOwnedRanges(0)
        , d_ghostIndices(0)
        , d_mpiComm(0)
        , d_nprocs(0)
        , d_myRank(0)
        , d_nGlobalRanges(0)
        , d_locallyOwnedRangesSorted(0)
        , d_locallyOwnedRangesIdPermutation(0)
        , d_allOwnedRanges(0)
        , d_globalRanges(0)
        , d_numLocallyOwnedIndices(0)
        , d_locallyOwnedRangesCumulativePairs(0)
        , d_numGhostIndices(0)
        , d_ghostIndicesOptimizedIndexSet(std::set<global_size_type>())
        , d_ghostIndicesRangeId(0)
        , d_numGhostProcs(0)
        , d_ghostProcIds(0)
        , d_numGhostIndicesInGhostProcs(0)
        , d_localGhostIndicesRanges(0)
        , d_numTargetProcs(0)
        , d_flattenedLocalGhostIndices(0)
        , d_ghostProcLocallyOwnedRangesCumulative(0)
        , d_targetProcIds(0)
        , d_numOwnedIndicesForTargetProcs(0)
        , d_flattenedLocalTargetIndices(0)
        , d_nGlobalIndices(0)
      {
        this->reinit(locallyOwnedRanges, ghostIndices, mpiComm);
      }

      ///
      /// Constructor without MPI for a single global ranges
      ///
      template <dftefe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(
        const std::pair<global_size_type, global_size_type> &locallyOwnedRange,
        const std::vector<dftefe::global_size_type> &        ghostIndices,
        const MPIComm &                                      mpiComm)
        : d_locallyOwnedRanges(0)
        , d_ghostIndices(0)
        , d_mpiComm(0)
        , d_nprocs(0)
        , d_myRank(0)
        , d_nGlobalRanges(0)
        , d_locallyOwnedRangesSorted(0)
        , d_locallyOwnedRangesIdPermutation(0)
        , d_allOwnedRanges(0)
        , d_globalRanges(0)
        , d_numLocallyOwnedIndices(0)
        , d_locallyOwnedRangesCumulativePairs(0)
        , d_numGhostIndices(0)
        , d_ghostIndicesOptimizedIndexSet(std::set<global_size_type>())
        , d_ghostIndicesRangeId(0)
        , d_numGhostProcs(0)
        , d_ghostProcIds(0)
        , d_numGhostIndicesInGhostProcs(0)
        , d_localGhostIndicesRanges(0)
        , d_numTargetProcs(0)
        , d_flattenedLocalGhostIndices(0)
        , d_ghostProcLocallyOwnedRangesCumulative(0)
        , d_targetProcIds(0)
        , d_numOwnedIndicesForTargetProcs(0)
        , d_flattenedLocalTargetIndices(0)
        , d_nGlobalIndices(0)
      {
        std::vector<std::pair<global_size_type, global_size_type>>
          locallyOwnedRanges(1);
        locallyOwnedRanges[0] = locallyOwnedRange;
        this->reinit(locallyOwnedRanges, ghostIndices, mpiComm);
      }

      ///
      /// reinit without MPI
      ///
      template <dftefe::utils::MemorySpace memorySpace>
      void
      MPIPatternP2P<memorySpace>::reinit(
        const std::vector<std::pair<global_size_type, global_size_type>>
          &                                          locallyOwnedRanges,
        const std::vector<dftefe::global_size_type> &ghostIndices,
        const MPIComm &                              mpiComm)
      {
        d_mpiComm = mpiComm;
        d_myRank  = 0;
        d_nprocs  = 1;

        d_locallyOwnedRanges = locallyOwnedRanges;
        // explcitly make d_ghosIndices to be empty
        d_ghostIndices  = std::vector<global_size_type>(0);
        d_nGlobalRanges = d_locallyOwnedRanges.size();
        d_allOwnedRanges.resize(
          d_nGlobalRanges,
          std::vector<std::pair<global_size_type, global_size_type>>(1));
        for (size_type iRange = 0; iRange < d_nGlobalRanges; ++iRange)
          {
            d_allOwnedRanges[iRange][0] = d_locallyOwnedRanges[iRange];
          }

        d_globalRanges = d_locallyOwnedRanges;

        checkOwnedRangesAssumptions(d_locallyOwnedRanges,
                                                           d_allOwnedRanges,
                                                           d_myRank);

        d_numLocallyOwnedIndices = 0;
        d_locallyOwnedRangesCumulativePairs.resize(d_nGlobalRanges);
        for (unsigned int i = 0; i < d_nGlobalRanges; ++i)
          {
            const size_type start = d_numLocallyOwnedIndices;

            d_numLocallyOwnedIndices +=
              d_locallyOwnedRanges[i].second - d_locallyOwnedRanges[i].first;

            const size_type end = d_numLocallyOwnedIndices;

            d_locallyOwnedRangesCumulativePairs[i] = std::make_pair(start, end);
          }

        // sort d_locallyOwnedRanges
        arrangeRanges(d_locallyOwnedRanges,
                      true,  // compareByFirst
                      false, // do not ignore empty ranges
                      d_locallyOwnedRangesSorted,
                      d_locallyOwnedRangesIdPermutation);

        d_numGhostIndices = 0;
        d_nGlobalIndices  = 0;
        for (unsigned int i = 0; i < d_globalRanges.size(); ++i)
          {
            d_nGlobalIndices +=
              d_globalRanges[i].second - d_globalRanges[i].first;
          }
      }

#endif // DFTEFE_WITH_MPI


      ///
      /// Constructor for a serial case with multiple global ranges
      ///
      template <dftefe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(
        const std::vector<size_type> &sizes)
        : d_locallyOwnedRanges(0)
        , d_ghostIndices(0)
        , d_mpiComm(0)
        , d_nprocs(0)
        , d_myRank(0)
        , d_nGlobalRanges(0)
        , d_locallyOwnedRangesSorted(0)
        , d_locallyOwnedRangesIdPermutation(0)
        , d_allOwnedRanges(0)
        , d_globalRanges(0)
        , d_numLocallyOwnedIndices(0)
        , d_locallyOwnedRangesCumulativePairs(0)
        , d_numGhostIndices(0)
        , d_ghostIndicesOptimizedIndexSet(std::set<global_size_type>())
        , d_ghostIndicesRangeId(0)
        , d_numGhostProcs(0)
        , d_ghostProcIds(0)
        , d_numGhostIndicesInGhostProcs(0)
        , d_localGhostIndicesRanges(0)
        , d_numTargetProcs(0)
        , d_flattenedLocalGhostIndices(0)
        , d_ghostProcLocallyOwnedRangesCumulative(0)
        , d_targetProcIds(0)
        , d_numOwnedIndicesForTargetProcs(0)
        , d_flattenedLocalTargetIndices(0)
        , d_nGlobalIndices(0)
      {
        this->reinit(sizes);
      }

      ///
      /// Constructor for a serial case with single global range
      ///
      template <dftefe::utils::MemorySpace memorySpace>
      MPIPatternP2P<memorySpace>::MPIPatternP2P(const size_type &size)
        : d_locallyOwnedRanges(0)
        , d_ghostIndices(0)
        , d_mpiComm(0)
        , d_nprocs(0)
        , d_myRank(0)
        , d_nGlobalRanges(0)
        , d_locallyOwnedRangesSorted(0)
        , d_locallyOwnedRangesIdPermutation(0)
        , d_allOwnedRanges(0)
        , d_globalRanges(0)
        , d_numLocallyOwnedIndices(0)
        , d_locallyOwnedRangesCumulativePairs(0)
        , d_numGhostIndices(0)
        , d_ghostIndicesOptimizedIndexSet(std::set<global_size_type>())
        , d_ghostIndicesRangeId(0)
        , d_numGhostProcs(0)
        , d_ghostProcIds(0)
        , d_numGhostIndicesInGhostProcs(0)
        , d_localGhostIndicesRanges(0)
        , d_numTargetProcs(0)
        , d_flattenedLocalGhostIndices(0)
        , d_ghostProcLocallyOwnedRangesCumulative(0)
        , d_targetProcIds(0)
        , d_numOwnedIndicesForTargetProcs(0)
        , d_flattenedLocalTargetIndices(0)
        , d_nGlobalIndices(0)
      {
        std::vector<size_type> sizes(1);
        sizes[0] = size;
        this->reinit(sizes);
      }

      //
      // reinit for serial case
      //
      template <dftefe::utils::MemorySpace memorySpace>
      void
      MPIPatternP2P<memorySpace>::reinit(const std::vector<size_type> &sizes)
      {
        d_mpiComm       = MPICommSelf;
        d_myRank        = 0;
        d_nprocs        = 1;
        d_nGlobalRanges = sizes.size();
        d_locallyOwnedRanges.resize(d_nGlobalRanges);
        size_type cumulativeCount = 0;
        for (size_type iRange = 0; iRange < d_nGlobalRanges; ++iRange)
          {
            d_locallyOwnedRanges[iRange].first = cumulativeCount;
            d_locallyOwnedRanges[iRange].second =
              cumulativeCount + sizes[iRange];
            cumulativeCount += sizes[iRange];
          }

        // explcitly make d_ghostIndices to be empty
        d_ghostIndices = std::vector<global_size_type>(0);
        d_allOwnedRanges.resize(
          d_nGlobalRanges,
          std::vector<std::pair<global_size_type, global_size_type>>(1));
        for (size_type iRange = 0; iRange < d_nGlobalRanges; ++iRange)
          {
            d_allOwnedRanges[iRange][0] = d_locallyOwnedRanges[iRange];
          }

        d_globalRanges = d_locallyOwnedRanges;
        checkOwnedRangesAssumptions(d_locallyOwnedRanges,
                                                           d_allOwnedRanges,
                                                           d_myRank);

        d_numLocallyOwnedIndices = 0;
        d_locallyOwnedRangesCumulativePairs.resize(d_nGlobalRanges);
        for (unsigned int i = 0; i < d_nGlobalRanges; ++i)
          {
            const size_type start = d_numLocallyOwnedIndices;

            d_numLocallyOwnedIndices +=
              d_locallyOwnedRanges[i].second - d_locallyOwnedRanges[i].first;

            const size_type end = d_numLocallyOwnedIndices;

            d_locallyOwnedRangesCumulativePairs[i] = std::make_pair(start, end);
          }

        // sort d_locallyOwnedRanges
        arrangeRanges(
          d_locallyOwnedRanges,
          true,  // compareByFirst
          false, // do not ignore empty ranges
          d_locallyOwnedRangesSorted,
          d_locallyOwnedRangesIdPermutation);

        d_numGhostIndices = 0;
        d_nGlobalIndices  = 0;
        for (unsigned int i = 0; i < d_globalRanges.size(); ++i)
          {
            d_nGlobalIndices +=
              d_globalRanges[i].second - d_globalRanges[i].first;
          }
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::nGlobalRanges() const
      {
        return d_nGlobalRanges;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      std::vector<std::pair<global_size_type, global_size_type>>
      MPIPatternP2P<memorySpace>::getGlobalRanges() const
      {
        return d_globalRanges;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      std::vector<std::pair<global_size_type, global_size_type>>
      MPIPatternP2P<memorySpace>::getLocallyOwnedRanges() const
      {
        return d_locallyOwnedRanges;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      std::pair<global_size_type, global_size_type>
      MPIPatternP2P<memorySpace>::getLocallyOwnedRange(size_type rangeId) const
      {
        return d_locallyOwnedRanges[rangeId];
      }

      template <dftefe::utils::MemorySpace memorySpace>
      const std::vector<global_size_type> &
      MPIPatternP2P<memorySpace>::getGhostIndices() const
      {
        return d_ghostIndices;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      const std::vector<size_type> &
      MPIPatternP2P<memorySpace>::getGhostProcIds() const
      {
        return d_ghostProcIds;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      const std::vector<size_type> &
      MPIPatternP2P<memorySpace>::getNumGhostIndicesInProcs() const
      {
        return d_numGhostIndicesInGhostProcs;
      }


      template <dftefe::utils::MemorySpace memorySpace>
      const std::vector<size_type> &
      MPIPatternP2P<memorySpace>::getGhostLocalIndicesRanges() const
      {
        return d_localGhostIndicesRanges;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::getNumGhostIndicesInProc(
        const size_type procId) const
      {
        auto      itProcIds             = d_ghostProcIds.begin();
        auto      itNumGhostIndices     = d_numGhostIndicesInGhostProcs.begin();
        size_type numGhostIndicesInProc = 0;
        for (; itProcIds != d_ghostProcIds.end(); ++itProcIds)
          {
            numGhostIndicesInProc = *itNumGhostIndices;
            if (procId == *itProcIds)
              break;

            ++itNumGhostIndices;
          }

        std::string msg =
          "The processor Id " + std::to_string(procId) +
          " does not contain any ghost indices for the current processor"
          " (i.e., processor Id " +
          std::to_string(d_myRank) + ")";
        throwException<InvalidArgument>(itProcIds != d_ghostProcIds.end(), msg);

        return numGhostIndicesInProc;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      typename MPIPatternP2P<memorySpace>::SizeTypeVector
      MPIPatternP2P<memorySpace>::getGhostLocalIndices(
        const size_type procId) const
      {
        size_type cumulativeIndices     = 0;
        size_type numGhostIndicesInProc = 0;
        auto      itProcIds             = d_ghostProcIds.begin();
        auto      itNumGhostIndices     = d_numGhostIndicesInGhostProcs.begin();
        for (; itProcIds != d_ghostProcIds.end(); ++itProcIds)
          {
            numGhostIndicesInProc = *itNumGhostIndices;
            if (procId == *itProcIds)
              break;

            cumulativeIndices += numGhostIndicesInProc;
            ++itNumGhostIndices;
          }

        std::string msg =
          "The processor Id " + std::to_string(procId) +
          " does not contain any ghost indices for the current processor"
          " (i.e., processor Id " +
          std::to_string(d_myRank) + ")";
        throwException<InvalidArgument>(itProcIds != d_ghostProcIds.end(), msg);

        SizeTypeVector returnValue(numGhostIndicesInProc);
        MemoryTransfer<memorySpace, memorySpace>::copy(
          numGhostIndicesInProc,
          returnValue.begin(),
          d_flattenedLocalGhostIndices.begin() + cumulativeIndices);

        return returnValue;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      const std::vector<size_type> &
      MPIPatternP2P<memorySpace>::getTargetProcIds() const
      {
        return d_targetProcIds;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      const std::vector<size_type> &
      MPIPatternP2P<memorySpace>::getNumOwnedIndicesForTargetProcs() const
      {
        return d_numOwnedIndicesForTargetProcs;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      const typename MPIPatternP2P<memorySpace>::SizeTypeVector &
      MPIPatternP2P<memorySpace>::getOwnedLocalIndicesForTargetProcs() const
      {
        return d_flattenedLocalTargetIndices;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::getNumOwnedIndicesForTargetProc(
        const size_type procId) const
      {
        auto      itProcIds         = d_targetProcIds.begin();
        auto      itNumOwnedIndices = d_numOwnedIndicesForTargetProcs.begin();
        size_type numOwnedIndicesForProc = 0;
        for (; itProcIds != d_targetProcIds.end(); ++itProcIds)
          {
            numOwnedIndicesForProc = *itNumOwnedIndices;
            if (procId == *itProcIds)
              break;

            ++itNumOwnedIndices;
          }

        std::string msg = "There are no owned indices for "
                          " target processor Id " +
                          std::to_string(procId) +
                          " in the current processor"
                          " (i.e., processor Id " +
                          std::to_string(d_myRank) + ")";
        throwException<InvalidArgument>(itProcIds != d_targetProcIds.end(),
                                        msg);
        return numOwnedIndicesForProc;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      typename MPIPatternP2P<memorySpace>::SizeTypeVector
      MPIPatternP2P<memorySpace>::getOwnedLocalIndices(
        const size_type procId) const
      {
        size_type cumulativeIndices      = 0;
        size_type numOwnedIndicesForProc = 0;
        auto      itProcIds              = d_targetProcIds.begin();
        auto      itNumOwnedIndices = d_numOwnedIndicesForTargetProcs.begin();
        for (; itProcIds != d_targetProcIds.end(); ++itProcIds)
          {
            numOwnedIndicesForProc = *itNumOwnedIndices;
            if (procId == *itProcIds)
              break;

            cumulativeIndices += numOwnedIndicesForProc;
            ++itNumOwnedIndices;
          }

        std::string msg = "There are no owned indices for "
                          " target processor Id " +
                          std::to_string(procId) +
                          " in the current processor"
                          " (i.e., processor Id " +
                          std::to_string(d_myRank) + ")";
        throwException<InvalidArgument>(itProcIds != d_targetProcIds.end(),
                                        msg);

        SizeTypeVector returnValue(numOwnedIndicesForProc);
        MemoryTransfer<memorySpace, memorySpace>::copy(
          numOwnedIndicesForProc,
          returnValue.begin(),
          d_flattenedLocalTargetIndices.begin() + cumulativeIndices);

        return returnValue;
      }



      template <dftefe::utils::MemorySpace memorySpace>
      const MPIComm &
      MPIPatternP2P<memorySpace>::mpiCommunicator() const
      {
        return d_mpiComm;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::nmpiProcesses() const
      {
        return d_nprocs;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::thisProcessId() const
      {
        return d_myRank;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      global_size_type
      MPIPatternP2P<memorySpace>::nGlobalIndices() const
      {
        return d_nGlobalIndices;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::localOwnedSize() const
      {
        return d_numLocallyOwnedIndices;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::localOwnedSize(size_type rangeId) const
      {
        return (d_locallyOwnedRanges[rangeId].second -
                d_locallyOwnedRanges[rangeId].first);
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::localGhostSize() const
      {
        return d_numGhostIndices;
      }


      template <dftefe::utils::MemorySpace memorySpace>
      global_size_type
      MPIPatternP2P<memorySpace>::localToGlobal(const size_type localId) const
      {
        global_size_type returnValue = 0;
        if (localId < d_numLocallyOwnedIndices)
          {
            bool      found;
            size_type rangeId;
            findRange(
              d_locallyOwnedRangesCumulativePairs, localId, found, rangeId);
            if (found)
              {
                returnValue =
                  d_locallyOwnedRanges[rangeId].first +
                  (localId -
                   d_locallyOwnedRangesCumulativePairs[rangeId].first);
              }
            else
              {
                std::string msg =
                  "In processor " + std::to_string(d_myRank) +
                  ", the local index " + std::to_string(localId) +
                  " passed to localToGlobal() in MPIPatternP2P is"
                  " supposed to be found in the locally owned ranges" +
                  " but is not found.";
                throwException<LogicError>(false, msg);
              }
          }
        else if (localId < (d_numLocallyOwnedIndices + d_numGhostIndices))
          {
            auto it =
              d_ghostIndices.begin() + (localId - d_numLocallyOwnedIndices);
            returnValue = *it;
          }
        else
          {
            std::string msg =
              "In processor " + std::to_string(d_myRank) +
              ", the local index " + std::to_string(localId) +
              " passed to localToGlobal() in MPIPatternP2P is"
              " larger than number of locally owned plus ghost indices.";
            throwException<InvalidArgument>(false, msg);
          }
        return returnValue;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      std::pair<global_size_type, size_type>
      MPIPatternP2P<memorySpace>::localToGlobalAndRangeId(
        const size_type localId) const
      {
        std::pair<global_size_type, size_type> returnValue;
        if (localId < d_numLocallyOwnedIndices)
          {
            bool      found;
            size_type rangeId;
            findRange(
              d_locallyOwnedRangesCumulativePairs, localId, found, rangeId);
            if (found)
              {
                returnValue.first =
                  d_locallyOwnedRanges[rangeId].first +
                  (localId -
                   d_locallyOwnedRangesCumulativePairs[rangeId].first);
                returnValue.second = rangeId;
              }
            else
              {
                std::string msg =
                  "In processor " + std::to_string(d_myRank) +
                  ", the local index " + std::to_string(localId) +
                  " passed to localToGlobal() in MPIPatternP2P is"
                  " supposed to be found in the locally owned ranges" +
                  " but is not found.";
                throwException<LogicError>(false, msg);
              }
          }
        else if (localId < (d_numLocallyOwnedIndices + d_numGhostIndices))
          {
            const size_type localGhostIndex =
              localId - d_numLocallyOwnedIndices;
            auto it            = d_ghostIndices.begin() + localGhostIndex;
            returnValue.first  = *it;
            returnValue.second = d_ghostIndicesRangeId[localGhostIndex];
          }
        else
          {
            std::string msg =
              "In processor " + std::to_string(d_myRank) +
              ", the local index " + std::to_string(localId) +
              " passed to localToGlobal() in MPIPatternP2P is"
              " larger than number of locally owned plus ghost indices.";
            throwException<InvalidArgument>(false, msg);
          }
        return returnValue;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      size_type
      MPIPatternP2P<memorySpace>::globalToLocal(
        const global_size_type globalId) const
      {
        size_type returnValue = 0;
        bool      found       = false;
        size_type rangeId, rangeIdSorted;

        findRange(d_locallyOwnedRangesSorted,
                                         globalId,
                                         found,
                                         rangeIdSorted);
        if (found)
          {
            rangeId     = d_locallyOwnedRangesIdPermutation[rangeIdSorted];
            returnValue = d_locallyOwnedRangesCumulativePairs[rangeId].first +
                          (globalId - d_locallyOwnedRanges[rangeId].first);
          }
        else
          {
            size_type localGhostIndex;
            d_ghostIndicesOptimizedIndexSet.getPosition(globalId,
                                                        localGhostIndex,
                                                        found);
            if (found)
              {
                returnValue = localGhostIndex + d_numLocallyOwnedIndices;
              }
          }

        std::string msg =
          "In processor " + std::to_string(d_myRank) + ", the global index " +
          std::to_string(globalId) +
          " passed to globalToLocal() in MPIPatternP2P is"
          " neither present in its locally owned ranges nor in its "
          " ghost indices.";
        throwException<InvalidArgument>(found, msg);

        return returnValue;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      std::pair<size_type, size_type>
      MPIPatternP2P<memorySpace>::globalToLocalAndRangeId(
        const global_size_type globalId) const
      {
        std::pair<size_type, size_type> returnValue;
        bool                            found = false;
        size_type                       rangeId, rangeIdSorted;

        findRange(d_locallyOwnedRangesSorted,
                                         globalId,
                                         found,
                                         rangeIdSorted);
        if (found)
          {
            rangeId = d_locallyOwnedRangesIdPermutation[rangeIdSorted];
            returnValue.first =
              d_locallyOwnedRangesCumulativePairs[rangeId].first +
              (globalId - d_locallyOwnedRanges[rangeId].first);
            returnValue.second = rangeId;
          }

        else
          {
            size_type localGhostIndex;
            d_ghostIndicesOptimizedIndexSet.getPosition(globalId,
                                                        localGhostIndex,
                                                        found);
            if (found)
              {
                returnValue.first  = localGhostIndex + d_numLocallyOwnedIndices;
                returnValue.second = d_ghostIndicesRangeId[localGhostIndex];
              }
          }

        std::string msg =
          "In processor " + std::to_string(d_myRank) + ", the global index " +
          std::to_string(globalId) +
          " passed to globalToLocal() in MPIPatternP2P is"
          " neither present in its locally owned ranges nor in its "
          " ghost indices.";
        throwException<InvalidArgument>(found, msg);

        return returnValue;
      }


      template <dftefe::utils::MemorySpace memorySpace>
      std::pair<bool, size_type>
      MPIPatternP2P<memorySpace>::inLocallyOwnedRanges(
        const global_size_type globalId) const
      {
        bool      found = false;
        size_type rangeIdSorted;
        findRange(d_locallyOwnedRangesSorted,
                                         globalId,
                                         found,
                                         rangeIdSorted);
        size_type rangeId = 0;
        if (found)
          rangeId = d_locallyOwnedRangesIdPermutation[rangeIdSorted];

        std::pair<bool, size_type> returnValue = std::make_pair(found, rangeId);
        return returnValue;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      std::pair<bool, size_type>
      MPIPatternP2P<memorySpace>::isGhostEntry(
        const global_size_type globalId) const
      {
        bool      found = false;
        size_type localId;
        d_ghostIndicesOptimizedIndexSet.getPosition(globalId, localId, found);
        size_type rangeId = 0;
        if (found)
          rangeId = d_ghostIndicesRangeId[localId];

        std::pair<bool, size_type> returnValue = std::make_pair(found, rangeId);
        return returnValue;
      }

      template <dftefe::utils::MemorySpace memorySpace>
      bool
      MPIPatternP2P<memorySpace>::isCompatible(
        const MPIPatternP2P<memorySpace> &rhs) const
      {
        if (d_nprocs != rhs.d_nprocs)
          return false;

        else if (d_nGlobalIndices != rhs.d_nGlobalIndices)
          return false;

        else if (d_locallyOwnedRanges != rhs.d_locallyOwnedRanges)
          return false;

        else if (d_numGhostIndices != rhs.d_numGhostIndices)
          return false;

        else
          return (d_ghostIndicesOptimizedIndexSet ==
                  rhs.d_ghostIndicesOptimizedIndexSet);
      }
    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
