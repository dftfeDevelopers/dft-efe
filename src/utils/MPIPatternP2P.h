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
 * @author Sambit Das, Bikash Kanungo
 */

#ifndef dftefeMPIPatternP2P_h
#define dftefeMPIPatternP2P_h

#ifdef DFTEFE_WITH_MPI
#  include <mpi.h>
#endif

#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <utils/OptimizedIndexSet.h>
#include <vector>
namespace dftefe
{
  namespace utils
  {
    /** @brief A class template to store the communication pattern
     * (i.e., which entries/nodes to receive from which processor and
     * which entries/nodes to send to which processor).
     *
     *
     * + <b>Assumptions</b>
     *    1. It assumes that a a sparse communication pattern. That is,
     *       a given processor only communicates with a few processors.
     *       This object should be avoided if the communication pattern
     *       is dense (e.g., all-to-all communication)
     *    2. It assumes that the each processor owns a set of \em continuous
     *       integers (indices). Further, the ownership is exclusive (i.e.,
     *       no index is owned by more than one processor). In other words,
     *       the different sets of owning indices across all the processors
     *       are disjoint.
     *
     * @tparam memorySpace Defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the various data members of this object must reside.
     */
    template <dftefe::utils::MemorySpace memorySpace>
    class MPIPatternP2P
    {
      ///
      /// typedefs
      ///
    public:
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;
      using GlobalSizeTypeVector =
        utils::MemoryStorage<global_size_type, memorySpace>;

    public:
      virtual ~MPIPatternP2P() = default;
#ifdef DFTEFE_WITH_MPI
      /**
       * @brief Constructor. This constructor is the typical way of
       * creation of an MPI pattern.
       *
       * @param[in] locallyOwnedRange A pair of non-negtive integers
       * \f$(a,b)\f$ which defines a range of indices (continuous)
       * that are owned by the current processor.
       * @note It is an open interval where \f$a\f$ is included,
       * but \f$b\f$ is not included.
       *
       * @param[in] ghostIndices An ordered set of non-negtive indices specifyin
       * the ghost indices for the current processor.
       * @note the vector must be ordered
       * (i.e., ordered in increasing order and non-repeating)
       *
       * @param[in] mpiComm The MPI communicator object which defines the
       * set of processors for which the MPI pattern needs to be created.
       *
       * @throw Throws exception if \p mpiComm is in an invalid state, if
       * the \p locallyOwnedRange across all the processors are not disjoint,
       * if \p ghostIndices are not ordered (if it is not strictly increasing),
       * or if some sanity checks with respect to MPI sends and receives fail.
       */
      MPIPatternP2P(
        const std::pair<global_size_type, global_size_type> &locallyOwnedRange,
        const std::vector<dftefe::global_size_type> &        ghostIndices,
        const MPI_Comm &                                     mpiComm);

      // void
      // reinit(
      //  const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      //  const std::vector<dftefe::global_size_type> &       ghostIndices,
      //  MPI_Comm &                                          mpiComm);
#else
      /**
       * @brief Constructor. This constructor is provided to create a dummy MPI
       * pattern while not using MPI. This is provided for applications to
       * interface with this class even while not using MPI. As a result,
       * a piece of application code that is written to be used with MPI
       * can seamlessly be used without MPI as well.
       *
       * @param[in] locallyOwnedRange A pair of non-negtive integers
       * \f$(a,b)\f$ which defines a range of indices (continuous)
       * that are owned by the current processor.
       * @note It is an open interval where \f$a\f$ is included,
       * but \f$b\f$ is not included.
       */
      MPIPatternP2P(
        const std::pair<global_size_type, global_size_type> &locallyOwnedRange);

      // void
      // reinit(){};
#endif

      size_type
      localOwnedSize() const;

      size_type
      localGhostSize() const;

      bool
      inLocallyOwnedRange(const global_size_type globalId) const;

      bool
      isGhostEntry(const global_size_type globalId) const;

      size_type
      globalToLocal(const global_size_type globalId) const;

      global_size_type
      localToGlobal(const size_type localId) const;

      const GlobalSizeTypeVector &
      getGhostIndices() const;

      const SizeTypeVector &
      getGhostProcIds() const;

      const SizeTypeVector &
      getNumGhostIndicesInProcs() const;

      size_type
      getNumGhostIndicesInProc(const size_type procId) const;

      SizeTypeVector
      getGhostLocalIndices(const size_type procId) const;

      const SizeTypeVector &
      getGhostLocalIndicesRanges() const;

      const SizeTypeVector &
      getTargetProcIds() const;

      const SizeTypeVector &
      getNumOwnedIndicesForTargetProcs() const;

      size_type
      getNumOwnedIndicesForTargetProc(const size_type procId) const;

      const SizeTypeVector &
      getOwnedLocalIndicesForTargetProcs() const;

      SizeTypeVector
      getOwnedLocalIndices(const size_type procId) const;

      size_type
      nmpiProcesses() const;

      size_type
      thisProcessId() const;

      global_size_type
      nGlobalIndices() const;

#ifdef DFTEFE_WITH_MPI
      const MPI_Comm &
      mpiCommunicator() const;
#endif

    private:
      /**
       * A pair \f$(a,b)\f$ which defines a range of indices (continuous)
       * that are owned by the current processor.
       *
       * @note It is an open interval where \f$a\f$ is included,
       * but \f$b\f$ is not included.
       */
      std::pair<global_size_type, global_size_type> d_locallyOwnedRange;

      /**
       * A vector of size 2 times number of processors to store the
       * locallyOwnedRange of each processor. That is it store the list
       * \f$[a_0,b_0,a_1,b_1,\ldots,a_{P-1},b_{P-1}]\f$, where the pair
       * \f$(a_i,b_i)\f$ defines a range of indices (continuous) that are owned
       * by the \f$i-\f$th processor.
       *
       * @note \f$a\f$ is included but \f$b\f$ is not included.
       */
      GlobalSizeTypeVector d_allOwnedRanges;

      /**
       * Number of locally owned indices in the current processor
       */
      size_type d_numLocallyOwnedIndices;

      /**
       * Number of ghost indices in the current processor
       */
      size_type d_numGhostIndices;

      /**
       * Vector to store an ordered set of ghost indices
       * (ordered in increasing order and non-repeating)
       */
      GlobalSizeTypeVector d_ghostIndices;

      /**
       * A copy of the above d_ghostIndices stored as an STL set
       */
      std::set<global_size_type> d_ghostIndicesSetSTL;

      /**
       * An OptimizedIndexSet object to store the ghost indices for
       * efficient operations. The OptimizedIndexSet internally creates
       * contiguous sub-ranges within the set of indices and hence can
       * optimize the finding of an index
       */
      OptimizedIndexSet<global_size_type> d_ghostIndicesOptimizedIndexSet;

      /**
       * Number of ghost processors for the current processor. A ghost processor
       * is one which owns at least one of the ghost indices of this processor.
       */
      size_type d_numGhostProcs;

      /**
       * Vector to store the ghost processor Ids. A ghost processor is
       * one which owns at least one of the ghost indices of this processor.
       */
      SizeTypeVector d_ghostProcIds;

      /** Vector of size number of ghost processors to store how many ghost
       * indices
       *  of this current processor are owned by a ghost processor.
       */
      SizeTypeVector d_numGhostIndicesInGhostProcs;

      /**
       * A flattened vector of size number of ghosts containing the ghost
       * indices ordered as per the list of ghost processor Ids in
       * d_ghostProcIds In other words it stores a concatentaion of the lists
       * \f$L_i = \{g^{(k_i)}_1,g^{(k_i)}_2,\ldots,g^{(k_i)}_{N_i}\}\f$, where
       * \f$g\f$'s are the ghost indices, \f$k_i\f$ is the rank of the
       * \f$i\f$-th ghost processor (i.e., d_ghostProcIds[i]) and \f$N_i\f$
       * is the number of ghost indices owned by the \f$i\f$-th
       * ghost processor (i.e., d_numGhostIndicesInGhostProcs[i]).

       * @note \f$L_i\f$ has to be an increasing set.

       * @note We store only the ghost index local to this processor, i.e.,
       * position of the ghost index in d_ghostIndicesSetSTL or d_ghostIndices.
       * This is done to use size_type which is unsigned int instead of
       * global_size_type which is long unsigned it. This helps in reducing the
       * volume of data transfered during MPI calls.

       * @note In the case that the locally owned ranges across all the
       * processors are ordered as per the processor Id, this vector is
       * redundant and one can only work with d_ghostIndices and
       * d_numGhostIndicesInGhostProcs. By locally owned range being ordered as
       * per the processor Id, means that the ranges for processor
       * \f$0, 1,\ldots,P-1\f$ are
       * \f$[N_0,N_1), [N_1, N_2), [N_2, N_3), ..., [N_{P-1},N_P)\f$ with
       * \f$N_0, N_1,\ldots, N_P\f$ beign non-decreasing. But in a more general
       * case, where the locally owned ranges are not ordered as per the
       processor
       * Id, this following array is useful.
       */
      SizeTypeVector d_flattenedLocalGhostIndices;

      /**
       * @brief A vector of size 2 times the number of ghost processors
       * to store the range of local ghost indices that are owned by the
       * ghost processors. In other words, it stores the list
       * \f$L=\{a_1,b_1, a_2, b_2, \ldots, a_G, b_G\}\f$, where
       * \f$a_i\f$ and \f$b_i\f$is are the start local ghost index
       * and one-past-the-last local ghost index of the current processor
       * that is owned by the \f$i\f$-th ghost processor
       * (i.e., d_ghostProcIds[i]). Put it differently, \f$[a_i,b_i)\f$ form an
       * open interval, where \f$a_i\f$ is included but \f$b_i\f$ is not
       * included.
       *
       * @note Given the fact that the locally owned indices of each processor
       * are contiguous and the global ghost indices (i.e., d_ghostIndices) is
       * ordered, it is sufficient to just store the range of local ghost
       * indices for each ghost procId. The actual global ghost indices
       * belonging to the \f$i\f$-th ghost processor can be fetched from
       * d_ghostIndices (i.e., it is the subset of d_ghostIndices lying bewteen
       *  d_ghostIndices[a_i] and d_ghostIndices[b_i].
       */
      SizeTypeVector d_localGhostIndicesRanges;

      /**
       * Number of target processors for the current processor. A
       * target processor is one which owns at least one of the locally owned
       * indices of this processor as its ghost index.
       */
      size_type d_numTargetProcs;

      /**
       * Vector to store the target processor Ids. A target processor is
       * one which contains at least one of the locally owned indices of this
       * processor as its ghost index.
       */
      SizeTypeVector d_targetProcIds;

      /**
       * Vector of size number of target processors to store how many locally
       * owned indices
       * of this current processor are need ghost in each of the target
       *  processors.
       */
      SizeTypeVector d_numOwnedIndicesForTargetProcs;

      /** Vector of size \f$\sum_i\f$ d_numOwnedIndicesForTargetProcs[i]
       * to store all thelocally owned indices
       * which other processors need (i.e., which are ghost indices in other
       * processors). It is stored as a concatentation of lists where the
       * \f$i\f$-th list indices
       * \f$L_i = \{o^{(k_i)}_1,o^{(k_i)}_2,\ldots,o^{(k_i)}_{N_i}\}\f$, where
       * where \f$o\f$'s are indices target to other processors,
       * \f$k_i\f$ is the rank of the \f$i\f$-th target processor
       * (i.e., d_targetProcIds[i]) and N_i is the number of
       * indices to be sent to i-th target processor (i.e.,
       * d_numOwnedIndicesForTargetProcs[i]).
       *
       * @note We store only the indices local to this processor, i.e.,
       * the relative position of the index in the locally owned range of this
       * processor This is done to use size_type which is unsigned int instead
       * of global_size_type which is long unsigned it. This helps in reducing
       * the volume of data transfered during MPI calls.
       *
       *  @note The list \f$L_i\f$ must be ordered.
       */
      SizeTypeVector d_flattenedLocalTargetIndices;

      /// Number of processors in the MPI Communicator.
      int d_nprocs;

      /// Rank of the current processor.
      int d_myRank;

      /**
       * Total number of unique indices across all processors
       */
      global_size_type d_nGlobalIndices;

#ifdef DFTEFE_WITH_MPI
      /// MPI Communicator object.
      MPI_Comm d_mpiComm;
#endif
    };

  } // end of namespace utils

} // end of namespace dftefe

#include <utils/MPIPatternP2P.t.cpp>
#endif // dftefeMPIPatternP2P_h
