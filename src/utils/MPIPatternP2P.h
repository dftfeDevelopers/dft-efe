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

#include <vector>
namespace dftefe
{
  namespace utils
  {
    /** @brief A class template to store the communication pattern
     * (i.e., which entries/nodes to receive from which processor and 
     * which entries/nodes to send to which processor). 
     *
     * The template parameter memorySpace defines the MemorySpace (i.e., HOST or DEVICE)
     * in which the various data members of this object must reside.
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
     */
    template <dftefe::utils::MemorySpace memorySpace>
    class MPIPatternP2P 
    {
      //
      // typedefs
      //
    public:
      template <dftefe::utils::MemorySpace memorySpace>
      using SizeTypeVector =
        utils::VectorStorage<size_type, dftefe::utils::MemorySpace memorySpace>;

      template <dftefe::utils::MemorySpace memorySpace>
      using GlobalSizeTypeVector =
        utils::VectorStorage<global_size_type,
                             dftefe::utils::MemorySpace memorySpace>;

    public:
      virtual ~MPIPatternP2P() = default;
#ifdef DFTEFE_WITH_MPI
      MPIPatternP2P(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const std::vector<dftefe::global_size_type> &          ghostIndices,
        MPI_Comm &                                          mpiComm);

      void
      reinit(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const std::vector<dftefe::global_size_type> &          ghostIndices,
        MPI_Comm &                                          mpiComm);
#else
      MPIPatternP2P() = default;

      void
      reinit(){};
#endif

      size_type
      localOwnedSize() const;

      size_type
      localGhostSize() const;

      size_type
      localSize() const;

      bool
      inLocallyOwnedRange(const global_size_type) const;

      bool
      isGhostEntry(const global_size_type) const;

      size_type
      globalToLocal(const global_size_type) const;

      global_size_type
      localToGlobal(const size_type) const;


      const std::vector<global_size_type> &
      getGhostIndicesVec() const;

      const std::vector<global_size_type> &
      getImportIndicesVec() const;


      const std::vector<size_type> &
      getGhostProcIdToNumGhostMap() const;

      const std::vector<size_type> &
      getImportProcIdToNumLocallyOwnedMap() const;

      size_type
      nmpiProcesses() const override;
      size_type
      thisProcessId() const override;

    private:
      std::pair<global_size_type, global_size_type> d_locallyOwnedRange;


      /// Vector to store an ordered set of ghost indices
      /// (ordered in increasing order and non-repeating)
      GlobalSizeTypeVector<memorySpace> d_ghostIndices;

      ///
      // Number of ghost processors for the current processor. A ghost processor
      // is one which owns at least of the ghost indices of this processor
      ///
      size_type d_numGhostProcs;

      ///
      // Vector to store the ghost processor Ids. A ghost processor is
      // one which owns at least of the ghost indices of this processor.
      ///
      SizeTypeVector<memorySpace> d_ghostProcIds;

      /// Vector of size number of ghost processors to store how many ghost
      /// indices
      //  of this current processor are owned by a ghost processor.
      SizeTypeVector<memorySpace> d_numGhostIndicesInGhostProcs;

      ///
      //  A flattened vector of size number of ghosts containing the ghost
      //  indices ordered as per the list of ghost processor Ids in
      //  d_ghostProcIds In other words it stores a concatentaion of the lists
      //  L_i = {g_{k_i,1},, g_{k_i,2}, ..., g_{k_i,N_i}}, where g's are the
      //  ghost indices, k_i is the rank of the i-th ghost processor (i.e.,
      //  d_ghostProcIds[i]) and N_i is the number of ghost indices owned by the
      //  i-th ghost processor (i.e., d_numGhostIndicesInGhostProcs[i]).
      //
      //  NOTE: L_i has to be an increasing set.
      //
      //  NOTE: We store only the ghost index local to this processor, i.e.,
      //  position of the ghost index in d_ghostIndicesSet or d_ghostIndices.
      //  This is done to use size_type which is unsigned int instead of
      //  global_size_type which is long unsigned it. This helps in reducing the
      //  volume of data transfered during MPI calls.
      //
      //  NOTE: In the case that the locally owned ranges across all the
      //  processors are ordered as per the processor Id, this vector is
      //  redundant and one can only work with d_ghostIndices and
      //  d_numGhostIndicesInGhostProcs. By locally owned range being ordered as
      //  per the processor Id, means that the ranges for processor 0, 1, ....,
      //  P-1 are [N_0,N_1), [N_1, N_2), [N_2, N_3), ..., [N_{P-1},N_P) where
      //  N_0, N_1, ..., N_P are non-decreasing. But a more general case,
      //  the locally owned ranges are not ordered as per the processor Id, this
      //  following array is useful
      ///
      SizeTypeVector<memorySpace> d_flattenedLocalGhostIndices;

      ///
      // Vector to store the to-send processor Ids. A to-send processor is
      // one which owns at least of one of the locally owned indices of this
      // processor as its ghost index
      ///
      SizeTypeVector<memorySpace> d_toSendProcIds;

      /// Vector of size number of to-send processors to store how many locally
      /// owned indices
      //  of this current processor are need ghost in each of the to-send
      //  processors.
      SizeTypeVector<memorySpace> d_numOwnedIndicesToSendToProcs;

      /// Vector of size \sum_i d_numOwnedForToSendProcs[i] to store all the
      /// locally owned indices
      //  which other processors need (i.e., which are ghost indices in other
      //  processors). It is stored as a concatentation of lists where the i-th
      //  list indices L_i = {o_{k_i,1}, o_{k_i,2}, ..., o_{k_i,N_i}}, where o's
      //  are indices to-send to other processors, k_i is the rank of the i-th
      //  to-send processor (i.e., d_toSendProcIds[i]) and N_i is the number of
      //  indices to be sent to i-th to-send processor (i.e.,
      //  d_numOwnedIndicesToSendProcs[i])
      //
      //  NOTE: We store only the indices local to this processor, i.e.,
      //  the relative position of the index in the locally owned range of this
      //  processor This is done to use size_type which is unsigned int instead
      //  of global_size_type which is long unsigned it. This helps in reducing
      //  the volume of data transfered during MPI calls.
      //
      //  NOTE: The list L_i must be ordered.
      SizeTypeVector<memorySpace> d_flattenedLocalToSendIndices;

      MPI_Comm  d_mpiComm;
      size_type d_nprocs;
      size_type d_myRank;
    };

  } // end of namespace utils

} // end of namespace dftefe

#endif // dftefeMPIPatternP2P_h
