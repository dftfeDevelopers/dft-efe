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

#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <utils/MPITypes.h>
#include <utils/OptimizedIndexSet.h>
#include <vector>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      /** @brief A class template to store the communication pattern
       * (i.e., which entries/nodes to receive from which processor and
       * which entries/nodes to send to which processor).
       *
       * + <b>Assumptions</b>
       *    1. It assumes that a a sparse communication pattern. That is,
       *       a given processor only communicates with a few processors.
       *       This object should be avoided if the communication pattern
       *       is dense (e.g., all-to-all communication)
       *    2. It assumes that the each processor owns multiple sets of
       *       \em continuous integers (indices) that are disjoint to each
       *       other. This means the following. Let there be \f$N$\f unique
       *       integers (indices) across \f$p$\f processors. Let each
       *       processor own \f$k$\f different (\em possibly empty) ranges of
       *       integers. Let us denote the \f$l$\f-th range
       * (\f$l=1,2,\ldots,k$\f) in the \f$i$\f-th processor as \f$R_l^i =
       * [a_l^i, b_l^i)$\f, where the open interval \f$[a,b)$\f is the list of
       * integers \f$[a,b) = {a,a+1,a+2,\ldots,b-1}$\f (i.e., \f$a$\f is
       * included but \f$b$\f is not). Then \f$R_l^i$\f mus satisfy the
       * following: a. \f$R_l^i \cap R_m^j = \emptyset$\f, if either \f$l \neq
       * m$\f or \f$i \neq j$\f. That is no index can be owned by two or more
       * processors. Further, within a processor, no index can belong to two or
       * more ranges owned by that processor. b.
       * \f$\bigcup\limits{l=1,i=1}^{l=k,i=p} R_l^i=[n_0,n_0+N)$\f. That is, the
       * union of all the ranges across all the ranges must form a set of
       * \f$N$\f contiguous integers \f$\{n_0, \n_0+1, n_0+2, \ldots,
       * n_0+N-1\}$\f, where \f$n_0$\f can be any non-negative integer. The
       * usual case is \f$n_0=0$\f.
       *    3. The \f$i$\f-processor has a set of \f$G_i$\f integers that are
       *       ordered in increasing manner and are not owned by it. We term
       *       this set as the ghost set. To elaborate, the ghost set for the
       *       \f$i$\f-th processor is given by \f$U^i=\{u_1^i,u_2^i,
       * u_{G_i\}^i}$\f, where \f$u_1^i < u_2^i < u_3^i < \ldots <  u_{G_i}^i$\f
       * and \f$U^i \cap R_l^i$=\emptyset\f. The latter condition implies that
       * no integer (index) in the ghost set in a processor should belong to any
       * of its owned ranges of integers (indices) (i.e., each integer (index)
       * in the ghost set  of a processor must be owned by some other processor
       * and not the same processor). Note that there is no requirement for the
       * ghost set of integers to be contiguous.
       *
       *   @note The reason we allow for \f$k$\f owned ranges in each processor
       *    is to seamlessly handle cases where the \f$N$\f indices from across
       *    all the processors is a concatenation of $k$ different sets of
       *    contiguous integers. For example, let there be a two vectors
       *    \$\mathbf{v}_1$\f and  \$\mathbf{v}_2$\f of sizes \f$N_1$\f and
       *    \f$N_2$\f, respectively, that are partitioned across the same set
       *    of processors. Let \f$r_1^i=[n_1^i, m_1^i)$\f and \f$$r_2^i=[n_2^i,
       * m_2^i)$\f be the locally owned range for \f$\mathbf{v}_1$\f and
       * \f$\mathbf{v}_2$\f in the \f$i$\f-th processor, respectively.
       * Similarly, let \f$X^i=\{x_1^i,x_2^i,\ldots,x_{nx_i}^i\}$\f and
       *    \f$Y^i=\{y_1^i,y_2^i,\ldots,y_{ny_i}^i\}$\f be two ordered
       *    (increasing and non-repeating) sets that define the ghost set of
       * indices in the \f$i$\f-th processor for \$\mathbf{v}_1$\f and
       * \$\mathbf{v}_2$\f, respectively. Then, we can construct a composite
       * vector \f$\mathbf{w}$\f of size \f$N=N_1+N_2$\f by concatenating
       *    \f$\mathbf{v}_1$\f and \f$\mathbf{v}_2$\f. If we are to partition
       * across the same set of processors in a manner that preserves the
       * partitioning of the \f$\mathbf{v}_1$\f and \f$\mathbf{v}_2$\f parts of
       * it, then for a given processor id (say \f$i$\f) we need to provide two
       * owned ranges: \f$R_1^i=[n_1^i,m_1^i)$\f and \f$R_2^i=[N_1 + n_2^i, N_1
       * + m_2^i)$\f (note that the second range is offset by the total size of
       * the first vector(\f$N_1$\f)). Further, the ghost set for
       * \f$\mathbf{w}$\f in the \f$i$\f-th processor (say \f$U^i$\f) becomes
       * the concatenation of the ghost sets of \f$\mathbf{v}_1$\f and
       * \f$\mathbf{v}_2$\f. That is \f$U_i=\{x_1^i,x_2^i,\ldots,x_{nx_i}^i\}
       * \cup \{N_1 + y_1^i, N_1 + y_2^i, \ldots, y_{ny_i}^i\}$\f (note that the
       * second ghost set is offset by the total size of the first vector
       * (\f$N_1$\f)). The above process can be extended to a composition of
       * \f$k$\f vectors instead of 2 vectors.
       *
       *    A typical scenario where such a composite vector arises while
       * dealing with direct sums of two or more vector spaces. For instance,
       * let there be a function expressed as a linear combination of two
       * mutually orthogonal basis sets, where each basis set is partitioned
       * across the same set of processors. Then, instead of paritioning two
       * vectors each containing the linear coefficients of one of the basis
       * sets, it is more logistically simpler to construct a composite vector
       * that concatenates the two vectors and partition it in a way that
       * preserves the original partitioning of the individual vectors.
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

        /**
         * @brief Constructor. This constructor is the typical way of
         * creation of an MPI pattern.
         *
         * @param[in] locallyOwnedRanges A vector containing different
         * non-overlapping ranges of non-negative integers that are owned by
         * the current processor. If the current processor id is \f$i$\f then
         * the \f$l$\f-the entry in \p locallyOwnedRanges denotes the
         * pair \f$a_l^i$\f and \f$b_l^i$\f that define the range \f$R_l^i$\f
         * defined above (see top of this page).
         * @note The pair \f$a_l^i$\f and \f$b_l^i$\f define an open interval,
         * where \f$a_l^i\f$ is included, but \f$b_l^i\f$ is not included.
         *
         * @param[in] ghostIndices An ordered (in increasing manner) set of
         * non-negtive indices specifying the ghost indices for the current
         * processor. If the current processor id is \f$i$\f then
         * \p ghostIndices is the set \f$U^i$\f defined above
         * (see top of this page)
         * @note the vector must be ordered
         * (i.e., ordered in increasing order and non-repeating)
         *
         * @param[in] mpiComm The MPI communicator object which defines the
         * set of processors for which the MPI pattern needs to be created.
         *
         * @throw Throws exception if:
         * 1. \p mpiComm is in an invalid state, or
         * 2. the ranges within \p locallyOwnedRanges across all the processors
         *    are not disjoint, or
         * 3. \p ghostIndices are not ordered (if it is not strictly
         *    increasing), or
         * 4. \p ghostIndices have any overlap with locallyOwnedRanges
         *    (i.e., an index is simultaneously owned and ghost for a processor)
         * 5. some sanity checks with respect to MPI sends and
         *    receives fail.
         *
         * @note Care is taken to create a dummy MPIPatternP2P while not linking
         * to an MPI library. This allows the user code to seamlessly link and
         * delink an MPI library.
         */
        MPIPatternP2P(const std::pair<global_size_type, global_size_type>
                        &locallyOwnedRange,
                      const std::vector<dftefe::global_size_type> &ghostIndices,
                      const MPIComm &                              mpiComm);
        /**
         * @brief Constructor. This constructor is to create an MPI Pattern for
         * a serial case. This is provided so that one can seamlessly use
         * has to be used even for a serial case. In this case, all the indices
         * are owned by the current processor.
         *
         * @param[in] size Total number of indices.
         * @note This is an explicitly serial construction (i.e., it uses
         * MPI_COMM_SELF), which is different from the dummy MPIPatternP2P
         * created while not linking to an MPI library. For examples,
         * within a parallel run, one might have the need to create a serial
         * MPIPatternP2P. A typical case is creation of a serial vector as a
         * special case of distributed vector.
         * @note Similar to the previous
         * constructor, care is taken to create a dummy MPIPatternP2P while not
         * linking to an MPI library.
         */
        MPIPatternP2P(const size_type size);

        size_type
        nLocallyOwnedRanges() const;

        std::vector<std::pair<global_size_type, global_size_type>>
        getLocallyOwnedRanges() const;

        size_type
        localOwnedSize(size_type rangeId) const;

        size_type
        localOwnedSize() const;

        size_type
        localGhostSize() const;

        bool
        inLocallyOwnedRanges(const global_size_type globalId) const;

        bool
        isGhostEntry(const global_size_type globalId) const;

        size_type
        globalToLocal(const global_size_type globalId) const;

        global_size_type
        localToGlobal(const size_type localId) const;

        const std::vector<global_size_type> &
        getGhostIndices() const;

        const std::vector<size_type> &
        getGhostProcIds() const;

        const std::vector<size_type> &
        getNumGhostIndicesInProcs() const;

        size_type
        getNumGhostIndicesInProc(const size_type procId) const;

        SizeTypeVector
        getGhostLocalIndices(const size_type procId) const;

        const std::vector<size_type> &
        getGhostLocalIndicesRanges() const;

        const std::vector<size_type> &
        getTargetProcIds() const;

        const std::vector<size_type> &
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

        const MPIComm &
        mpiCommunicator() const;

        bool
        isCompatible(const MPIPatternP2P<memorySpace> &rhs) const;

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
         * \f$(a_i,b_i)\f$ defines a range of indices (continuous) that are
         * owned by the \f$i-\f$th processor.
         *
         * @note \f$a\f$ is included but \f$b\f$ is not included.
         */
        std::vector<global_size_type> d_allOwnedRanges;

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
        std::vector<global_size_type> d_ghostIndices;

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
         * Number of ghost processors for the current processor. A ghost
         * processor is one which owns at least one of the ghost indices of this
         * processor.
         */
        size_type d_numGhostProcs;

        /**
         * Vector to store the ghost processor Ids. A ghost processor is
         * one which owns at least one of the ghost indices of this processor.
         */
        std::vector<size_type> d_ghostProcIds;

        /** Vector of size number of ghost processors to store how many ghost
         * indices
         *  of this current processor are owned by a ghost processor.
         */
        std::vector<size_type> d_numGhostIndicesInGhostProcs;

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
         * position of the ghost index in d_ghostIndicesSetSTL or
         d_ghostIndices.
         * This is done to use size_type which is unsigned int instead of
         * global_size_type which is long unsigned it. This helps in reducing
         * the volume of data transfered during MPI calls.

         * @note In the case that the locally owned ranges across all the
         * processors are ordered as per the processor Id, this vector is
         * redundant and one can only work with d_ghostIndices and
         * d_numGhostIndicesInGhostProcs. By locally owned range being ordered
         as
         * per the processor Id, means that the ranges for processor
         * \f$0, 1,\ldots,P-1\f$ are
         * \f$[N_0,N_1), [N_1, N_2), [N_2, N_3), ..., [N_{P-1},N_P)\f$ with
         * \f$N_0, N_1,\ldots, N_P\f$ beign non-decreasing. But in a more
         general
         * case, where the locally owned ranges are not ordered as per the
         processor
         * Id, this following array is useful.
         */
        SizeTypeVector d_flattenedLocalGhostIndices;

        /**
         * @brief A vector of size 2 times the number of ghost processors
         * to store the range of indices in the above
         * \p d_flattenedLocalGhostIndices array that contain the local ghost
         * indices owned by the ghost processors. To elaborate, it stores the
         * list \f$L=\{a_1,b_1, a_2, b_2, \ldots, a_G, b_G\}\f$, such that the
         * part of \p d_flattenedLocalGhostIndices lying between the indices
         * [\f$a_i\f$, \f$b_i\f$) defines the local ghost indices that are owned
         * by the \f$i\f$-th ghost processor (i.e., d_ghostProcIds[i]).
         * In other words, the set {\p d_flattenedLocalGhostIndices[\f$a_i$\f],
         * d_flattenedLocalGhostIndices[\f$a_{i+1}$\f], ..., \p
         * d_flattenedLocalGhostIndices[\f$b_{i-1}$\f]} defines the local ghost
         * indices that are owned by the \f$i\f$-th ghost processor (i.e.,
         * d_ghostProcIds[i]).
         *
         * @note \f$[a_i,b_i)\f$ forms an open interval, where \f$a_i\f$ is
         * included but \f$b_i\f$ is not included.
         */
        std::vector<size_type> d_localGhostIndicesRanges;

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
        std::vector<size_type> d_targetProcIds;

        /**
         * Vector of size number of target processors to store how many locally
         * owned indices
         * of this current processor are need ghost in each of the target
         *  processors.
         */
        std::vector<size_type> d_numOwnedIndicesForTargetProcs;

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

        /// MPI Communicator object.
        MPIComm d_mpiComm;
      };

    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe

#include <utils/MPIPatternP2P.t.cpp>
#endif // dftefeMPIPatternP2P_h
