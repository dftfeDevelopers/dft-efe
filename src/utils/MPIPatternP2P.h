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
       * + <b>Problem Setup </b> \n
       *    Let there be \f$K\f$ non-overlapping
       *    intervals of non-negative integers given as
       * \f$[N_l^{start},N_l^{end})\f$, \f$l=0,1,2,\ldots,K-1\f$. We term these
       * intervals as <b>\p global-ranges</b> and the index \f$l\f$ as <b>\p
       * rangeId </b>. Here, \f$[a,b)\f$ denotes a half-open interval where
       * \f$a\f$ is included, but \f$b\f$ is not included. Instead of
       * partitioning each of the global interval separately, we are interested
       * in partitioning all of them simultaneously across the the same set of
       * \f$p\f$ processors. Had there been just one global interval, say
       * \f$[N_0^{start},N_0^{end}]\f$, the paritioning would result in each
       * processor having a <b> \p locally-owned-range</b> defined as contiguous
       * sub-range \f$[a,b) \in [N_0^{start},N_0^{end}]\f$), such that it has no
       * overlap with the \p locally-owned-range in other processors.
       * Additionally, each processor will have a set of indices (not
       * necessarily contiguous), called <b>\p ghost-set</b> that are not part
       * of its \p locally-owned-range (i.e., they are pwned by some other
       * processor). If we extend the above logic of partitioning to the case
       * where there are \f$K\f$ different \p global-ranges, then each processor
       *    will have \f$K\f$ different \p locally-owned-ranges. For \p
       * ghost-set, although \f$K\f$ \p global-ranges will lead to \f$K\f$
       * different sets of \p ghost-sets, for simplicity we can concatenate them
       * into one set of indices. We can do this concatenation because the
       * individual \p ghost-sets are just sets of indices. Once again, for
       * simplicity, we term the concatenated \p ghost-set as just \p ghost-set.
       *    For the \f$i^{th}\f$ processor, we denote the \f$K\f$ \p
       * locally-owned-ranges as \f$R_l^i=[a_l^i, b_l^i)\f$, where
       * \f$l=0,1,\ldots,K-1\f$ and \f$i=0,1,2,\ldots,p-1\f$. Further, for
       * \f$i^{th}\f$ processor, the \p ghost-set is given by an
       * strictly-ordered set of non-negative integers \f$U^i=\{u_1^i,u_2^i,
       * u_{G_i}^i\}\f$. By strictly ordered we mean \f$u_1^i < u_2^i < u_3^i <
       * \ldots <  u_{G_i}^i\f$. Thus, given the \f$R_l^i\f$'s and \f$U^i\f$'s,
       *    this class figures out which processor needs to communicate with
       * which other processors.
       *
       *    We list down the definitions that will
       *    be handy to understand the implementation and usage of this class.
       *    Some of the definitions are already mentioned in the description
       * above
       *    + <b> \p global-ranges </b>: \f$K\f$ non-overlapping intervals of
       * non-negative integers given as \f$[N_l^{start},N_l^{end})\f$, where
       * \f$l=0,1,2,\ldots,K-1\f$. Each of the interval is partitioned across
       * the same set of \f$p\f$ processors.
       *    + <b> \p rangeId </b>: An index \f$l=0,1,\ldots,K-1\f$, which
       * indexes the \p global-ranges.
       *    + <b> \p locally-owned-ranges </b>: For a given processor (say with
       * rank \f$i\f$), \p locally-owned-ranges define \f$K\f$ intervals of
       * non-negative integers given as \f$R_l^i=[a_l^i,b_l^i)\f$,
       * \f$l=0,1,2,\ldots,K-1\f$, that are owned by the processor
       *    + <b> \p ghost-set </b>: For a given processor (say with rank
       * \f$i\f$), the \p ghost-set is an ordered (strictly increasing) set of
       * non-negative integers given as \f$U^i=\{u_1^i,u_2^i, u_{G_i}^i\}\f$.
       *    + <b> \p numLocallyOwnedIndices </b>: For a given processor (say
       * with rank \f$i\f$), the \p numLocallyOwnedIndices is the number of
       * indices that it owns. That is, \p numLocallyOwnedIndices =
       * \f$\sum_{i=0}^{K-1} |R_l^i| = b_l^i - a_l^i\f$, where \f$|.|\f$ denotes
       * the number of elements in a set (cardinality of a set).
       *    + <b> numGhostIndices </b>: For a given processor (say with rank
       * \f$i\f$), is the size of its \p ghost-set. That is, \p numGhostIndices
       * = \f$|U^i|\f$.
       *    + <b> numLocalIndices </b>: For a given processor (say with rank
       * \f$i\f$), it is the sum of the \p numLocallyOwnedIndices and \p
       * numGhostIndices.
       *    + <b> localId </b>: In a processor (say with rank \f$i\f$), given an
       * integer (say \f$x\f$) that belongs either to the \p
       * locally-owned-ranges or the \b ghost-set, we assign it a unique index
       * between \f$[0,numLocalIndices)\f$ called the \p localId. We follow the
       * simple approach of using the position that \f$x\f$ will have if we
       * concatenate the \p locally-owned-ranges and \p ghost-set as its \p
       * localId. That is, if \f$V=R_0^i \oplus R_1^i \oplus \ldots R_{K-1}^i
       * \oplus U^i\f$, where \f$\oplus\f$ denotes concatenation of two sets,
       * then the \p localId of \f$x\f$ is its position (starting from 0 for the
       * first entry) in \f$V\f$.
       *
       *
       * + <b>Assumptions</b>
       *    1. It assumes that a a sparse communication pattern. That is,
       *       a given processor only communicates with a few processors.
       *       This object should be avoided if the communication pattern
       *       is dense (e.g., all-to-all communication)
       *    2. The \f$R_l^i\f$ must satisfy the following
       *       -# \f$R_l^i = [a_l^i, b_l^i) \in [N_l^{start},N_l^{end})\f$. That
       * is the \f$l^{th}\f$ \p locally-owned-range in a processor must a
       * sub-set of the \f$l^{th}\f$ \p global-interval.
       *       -# \f$\bigcup_{i=0}^{p-1}R_l^i=[N_l^{start}, N_l^{end}]\f$. That
       * is, for a given \p rangeId \f$l\f$, the union of the \f$l^{th}\f$ \p
       * locally-owned-range from each processor should equate the \f$l^{th}\f$
       * \p global-range.
       *       -# \f$R_l^i \cap R_m^j = \emptyset\f$, if either \f$l \neq
       *          m\f$ or \f$i \neq j\f$. That is no index can be owned by two
       * or more processors. Further, within a processor, no index can belong to
       * two or more \p locally-owned-ranges.
       *       -# \f$U^i \cap R_l^i=\emptyset\f$. That is no index in the \p
       * ghost-set of a processor should belong to any of its \p
       * locally-owned-ranges. In other words, index in the \p ghost-set of a
       * processor must be owned by some other processor and not the same
       * processor.
       *
       *    A typical example which illustrates the use of \f$K\f$ \p
       * global-ranges is the following. Let there be a two vectors
       *    \f$\mathbf{v}_1\f$ and  \f$\mathbf{v}_2\f$ of sizes \f$N_1\f$ and
       *    \f$N_2\f$, respectively, that are partitioned across the same set
       *    of processors. Let \f$r_1^i=[n_1^i, m_1^i)\f$ and \f$r_2^i=[n_2^i,
       *    m_2^i)\f$ be the \p locally-owned-range for \f$\mathbf{v}_1\f$ and
       *    \f$\mathbf{v}_2\f$ in the \f$i^{th}\f$ processor, respectively.
       *    Similarly, let \f$X^i=\{x_1^i,x_2^i,\ldots,x_{nx_i}^i\}\f$ and
       *    \f$Y^i=\{y_1^i,y_2^i,\ldots,y_{ny_i}^i\}\f$ be two strictly-ordered
       *    sets that define the \p ghost-set in the \f$i^{th}\f$ processor for
       * \f$\mathbf{v}_1\f$ and \f$\mathbf{v}_2\f$, respectively. Then, we can
       * construct a composite vector \f$\mathbf{w}\f$ of size \f$N=N_1+N_2\f$
       * by concatenating \f$\mathbf{v}_1\f$ and \f$\mathbf{v}_2\f$. We now want
       * to partition \f$\mathbf{w}\f$ across the same set of processors in a
       * manner that preserves the partitioning of the \f$\mathbf{v}_1\f$ and
       * \f$\mathbf{v}_2\f$ parts of it. To do so, we define two \p
       * global-ranges \f$[A_1, A_1 + N_1)\f$ and \f$[A_1 + N_1 + A_2, A_1 + N_1
       * + A_2 + N_2)\f$, where \f$A_1\f$ and  \f$A_2\f$ are any non-negative
       * integers, to index the \f$\mathbf{v}_1\f$ and \f$\mathbf{v}_2\f$ parts
       * of \f$\mathbf{w}\f$. In usual cases, both \f$A_1\f$ and  \f$A_2\f$ are
       * zero. However, one can use non-zero values for \f$A_1\f$ and \f$A_2\f$,
       * as that will not violate the non-overlapping condition on the \p
       * global-ranges. Now, if we are to partition \f$\mathbf{w}\f$ such that
       *    it preserves the individual partitiioning of \f$\mathbf{v}_1\f$ and
       *    \f$\mathbf{v}_2\f$ across the same set of processors,
       *    then for a given processor id (say \f$i\f$) we need to provide two
       *    owned ranges: \f$R_1^i=[A_1 + n_1^i, A_1 + m_1^i)\f$ and
       * \f$R_2^i=[A_1 + N_1 + A_2 + n_2^i, A_1 + N_1 + A_2
       *    + m_2^i)\f$. Further, the ghost set for
       *    \f$\mathbf{w}\f$ in the \f$i^{th}\f$ processor (say \f$U^i\f$)
       * becomes the concatenation of the ghost sets of \f$\mathbf{v}_1\f$ and
       *    \f$\mathbf{v}_2\f$. That is \f$U_i=\{A_1 + x_1^i,A_1 + x_2^i,\ldots,
       * A_1 + x_{nx_i}^i\} \cup \{A_1 + N_1 + A_2 + y_1^i, A_1 + N_1 + A_2 +
       * y_2^i, \ldots, A_1 + N_1 + A_2 + y_{ny_i}^i\}\f$ The above process can
       * be extended to a composition of \f$K\f$ vectors instead of two vectors.
       * \n \n
       *
       *    A typical scenario where such a composite vector arises is while
       *    dealing with direct sums of two or more vector spaces. For instance,
       *    let there be a function expressed as a linear combination of two
       *    mutually orthogonal basis sets, where each basis set is partitioned
       *    across the same set of processors. Then, instead of paritioning two
       *    vectors each containing the linear coefficients of one of the basis
       *    sets, it is more logistically simpler to construct a composite
       * vector that concatenates the two vectors and partition it in a way that
       *    preserves the original partitioning of the individual vectors.
       *
       *
       *
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
         * creation of an MPI pattern for multiple \p global-ranges.
         *
         * @param[in] locallyOwnedRanges A vector containing different
         * non-overlapping ranges of non-negative integers that are owned by
         * the current processor. If the current processor id is \f$i\f$ then
         * the \f$l^{th}\f$e entry in \p locallyOwnedRanges denotes the
         * pair \f$a_l^i\f$ and \f$b_l^i\f$ that define the range \f$R_l^i\f$
         * defined above (see top of this page).
         *
         * @param[in] ghostIndices An ordered (in increasing manner) set of
         *  non-negtive indices specifying the \p ghost-set for the current
         *  processor (see above for definition).
         *
         * @param[in] mpiComm The MPI communicator object which defines the
         * set of processors for which the MPI pattern needs to be created.
         *
         * @throw Throws exception if:
         * 1. \p mpiComm is in an invalid state, or
         *    (i.e., an index is simultaneously owned and ghost for a processor)
         * 2. Some sanity checks with respect to MPI sends and
         *    receives fail.
         * 3. Any of the assumptions listed above fails
         *
         * @note
         * -# The pair \f$a_l^i\f$ and \f$b_l^i\f$ in \p locallyOwnedRanges must
         *  define an open interval, where \f$a_l^i\f$ is included, but
         * \f$b_l^i\f$ is not included.
         * -# The vector \p ghostIndices must be ordered
         *  (i.e., increasing and non-repeating)
         * -# Care is taken to create a dummy MPIPatternP2P while not linking
         *  to an MPI library. This allows the user code to seamlessly link and
         *  delink an MPI library.
         */
        MPIPatternP2P(
          const std::vector<std::pair<global_size_type, global_size_type>>
            &                                          locallyOwnedRanges,
          const std::vector<dftefe::global_size_type> &ghostIndices,
          const MPIComm &                              mpiComm);

        /**
         * @brief Constructor. This constructor is the typical way of
         * creation of an MPI pattern for a single \p global-range.
         *
         * @param[in] locallyOwnedRange A pair of non-negative integers defining
         * the range of indices that are owned by
         * the current processor.
         *
         * @param[in] ghostIndices An ordered (in increasing manner) set of
         *  non-negtive indices specifying the \p ghost-set for the current
         *  processor (see above for definition).
         *
         * @param[in] mpiComm The MPI communicator object which defines the
         * set of processors for which the MPI pattern needs to be created.
         *
         * @throw Throws exception if:
         * 1. \p mpiComm is in an invalid state, or
         *    (i.e., an index is simultaneously owned and ghost for a processor)
         * 2. Some sanity checks with respect to MPI sends and
         *    receives fail.
         * 3. Any of the assumptions listed above fails
         *
         * @note
         * -# The pair \f$a\f$ and \f$b\f$ in \p locallyOwnedRange must
         *  define an open interval, where \f$a\f$ is included, but \f$b\f$
         *  is not included.
         * -# The vector \p ghostIndices must be ordered
         *  (i.e., increasing and non-repeating)
         * -# Care is taken to create a dummy MPIPatternP2P while not linking
         *  to an MPI library. This allows the user code to seamlessly link and
         *  delink an MPI library.
         */
        MPIPatternP2P(const std::pair<global_size_type, global_size_type>
                        &locallyOwnedRange,
                      const std::vector<dftefe::global_size_type> &ghostIndices,
                      const MPIComm &                              mpiComm);

        /**
         * @brief Constructor. This constructor is to create an MPI Pattern for
         * a serial case with multiple \p global-ranges.
         * This is provided so that one can seamlessly use
         * this class even for a serial case. In this case, all the indices
         * are owned by the current processor.
         *
         * @param[in] sizes Vector containig the sizes of each global range
         * @note
         * -# The \p global-ranges will be defined in a cumulative manner based
         *  on the input \p sizes. That is, the \f$i^{th}\f$ \p global-range
         *  is defined by the half-open interval \f$[C_i,sizes[i])\f$, where
         *  \f$C_i=\sum_{j=0}^{i-1}sizes[j]\f$ is the cumulative number of
         *  indices preceding the \f$i^{th}\f$ \p global-range.
         * -#This is an explicitly serial construction (i.e., it uses
         * MPI_COMM_SELF), which is different from the dummy MPIPatternP2P
         * created while not linking to an MPI library. For examples,
         * within a parallel run, one might have the need to create a serial
         * MPIPatternP2P. A typical case is creation of a serial vector as a
         * special case of distributed vector.
         * @note Similar to the previous
         * constructor, care is taken to create a dummy MPIPatternP2P while not
         * linking to an MPI library.
         */
        MPIPatternP2P(const std::vector<size_type> &sizes);

        /**
         * @brief Constructor. This constructor is to create an MPI Pattern for
         * a serial case with a single \p global-range . This is provided so
         * that one can seamlessly use this class even for a serial case. In
         * this case, all the indices are owned by the current processor.
         *
         * @param[in] size Total number of indices.
         * @note
         * -# The \p global-range will defined as the half-open interval
         * [0,size)
         * -# This is an explicitly serial construction (i.e., it uses
         * MPI_COMM_SELF), which is different from the dummy MPIPatternP2P
         * created while not linking to an MPI library. For examples,
         * within a parallel run, one might have the need to create a serial
         * MPIPatternP2P. A typical case is creation of a serial vector as a
         * special case of distributed vector.
         * @note Similar to the previous
         * constructor, care is taken to create a dummy MPIPatternP2P while not
         * linking to an MPI library.
         */
        MPIPatternP2P(const size_type &size);

        void
        reinit(const std::vector<std::pair<global_size_type, global_size_type>>
                 &                                          locallyOwnedRanges,
               const std::vector<dftefe::global_size_type> &ghostIndices,
               const MPIComm &                              mpiComm);

        void
        reinit(const std::vector<size_type> &sizes);

        size_type
        nGlobalRanges() const;

        std::vector<std::pair<global_size_type, global_size_type>>
        getGlobalRanges() const;

        std::vector<std::pair<global_size_type, global_size_type>>
        getLocallyOwnedRanges() const;

        std::pair<global_size_type, global_size_type>
        getLocallyOwnedRange(size_type rangeId) const;

        size_type
        localOwnedSize(size_type rangeId) const;

        size_type
        localOwnedSize() const;

        size_type
        localGhostSize() const;

        /**
         * @brief For a given globalId, returns whether it lies in any of the \p locally-owned-ranges
         * and if true the index of the \p global-range it belongs to
         * @param[in] globalId The input global index
         * @returns A pair where:
         * (a) First entry contains a boolean which is true if the \p globalId
         * belongs to any of the \p locally-owned-ranges, or else is false.
         * (b) Second entry contains the index of the \p global-range to which
         * \p globaId belongs. This value is meaningful only if the first entry
         * is true, or else its value is undefined.
         */
        std::pair<bool, size_type>
        inLocallyOwnedRanges(const global_size_type globalId) const;

        /**
         * @brief For a given globalId, returns whether it belongs to the current processor's
         * \p ghost-set and if true the index of the \p global-range it belongs
         * to
         * @param[in] globalId The input global index
         * @returns A pair where:
         * (a) First entry contains a boolean which is true if the \p globalId
         * belongs to the \p ghost-set, or else is false.
         * (b) Second entry contains the index of the \p global-range to which
         * \p globaId belongs. This value is meaningful only if the first entry
         * is true, or else its value is undefined.
         */
        std::pair<bool, size_type>
        isGhostEntry(const global_size_type globalId) const;

        size_type
        globalToLocal(const global_size_type globalId) const;

        global_size_type
        localToGlobal(const size_type localId) const;

        /**
         * @brief For a given global index, returns a pair containing the local index in the procesor
         * and the index of the \p global-range it belongs to
         * @param[in] globalId The input global index
         * @returns A pair where the first entry contains the local index in the procesor for \p globalId
         * and second entry contains the index of the \p global-range to which
         * it belongs.
         */
        std::pair<size_type, size_type>
        globalToLocalAndRangeId(const global_size_type globalId) const;

        /**
         * @brief For a given local index, returns a pair containing its global index
         * and the index of the \p global-range it belongs to
         * param[in] localId The input local index
         * @returns A pair where the first entry contains the global index for \p localId
         * and second entry contains the index of the \p global-range to which
         * it belongs.
         */
        std::pair<global_size_type, size_type>
        localToGlobalAndRangeId(const size_type localId) const;

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
         * A vector containing the pairs of locally owned ranges for the current
         * processor. If the current processor id is \f$i\f$, then the
         * \f$l^{th}\f$ entry in \p d_locallyOwnedRange is the pair \f$a_l^i\f$
         * and \f$b_l^i\f$ that define the range \f$R_l^i\f$ (see top of this
         * page)
         *
         * @note Any pair \f$a\f$ and \f$b\f$ define an open interval
         * where \f$a\f$ is included but \f$b\f$ is not included.
         */
        std::vector<std::pair<global_size_type, global_size_type>>
          d_locallyOwnedRanges;

        /**
         * A non-negative integer storing the number of locally owned ranges in
         * each processor (must be same for all processor). In other words, it
         * stores \p d_locallyOwnedRanges.size()
         */
        size_type d_nGlobalRanges;


        /**
         * A sorted (lower to higher) version of the ranges of the above \p
         * d_locallyOwnedRanges. The sorting is done as per the start point of
         * each non-empty range. If two ranges have the same start point, then
         * they are sorted as per the end point.
         */
        std::vector<std::pair<global_size_type, global_size_type>>
          d_locallyOwnedRangesSorted;

        /**
         * A vector storing the index permutation obtained while sorting the
         * ranges in \p d_locallyOwnedRanges. That is, \p
         * d_locallyOwnedRangesIdPermutation[i] tells where the i-th range in \p
         * d_locallyOwnedRangesSorted lies in the original \p
         * d_locallyOwnedRanges
         */
        std::vector<size_type> d_locallyOwnedRangesIdPermutation;

        /**
         * A 2D vector to store the locally owned ranges for each processor.
         * The first index is range id (i.e., ranges from 0 \p d_nGlobalRanges).
         * For range id \f$l\f$ , it stores pairs defining the \f$l^{th}\f$
         * locally owned range in each processor. That is, \p
         * d_allOwnedRanges[l] = \f$\{\{a_l^0,b_l^0\}, \{a_l^1,b_l^1\}, \ldots,
         * \{a_l^{p-1},b_l^{p-1}\}\}\f$, where \f$p\f$ is the number of
         * processors and the pair \f$(a_l^i,b_l^i)\f$ defines the \f$l^{th}\f$
         * locally owned range for the \f$i^{th}\f$ processor.
         *
         * @note Any pair \f$a\f$ and \f$n\f$ define an open interval,
         * where \f$a\f$ is included but \f$b\f$ is not included.
         */
        std::vector<std::vector<std::pair<global_size_type, global_size_type>>>
          d_allOwnedRanges;

        /**
         * A vector of size \p d_nGlobalRanges that stores the \p global-ranges
         * That is, d_globalRanges[i] = \f$\{N_l^{start}, N_l^{end}\}\f$,
         * such that the half-open interval \f$[N_l^{start}, N_l^{end})\f$
         * defines the \f$l^{th}\f$ \p global-range (see top of the page fr
         * details)
         */
        std::vector<std::pair<global_size_type, global_size_type>>
          d_globalRanges;

        /**
         * Number of locally owned indices in the current processor.
         * See \p numLocallyOwnedIndices at the top of the page for description
         */
        size_type d_numLocallyOwnedIndices;

        /**
         * A vector of size \p d_nGlobalRanges storing the cumulative start and
         * end point of each \p locally-owned-ranges. That is
         * d_locallyOwnedRangesCumulativePairs[l].first = \sum^{j=0}^{l-1}
         * (b_j-a_j)\f$, and d_locallyOwnedRangesCumulativePairs[l].second =
         * \sum_{j=0}^{l} (b_j-a_j)\f$ where \f$a_j\f$ and \f$b_j\f$ are
         * d_locallyOwnedRanges[j].first and d_locallyOwnedRanges[j].second In
         * other words, if we concatenate the indices defined by all the \p
         * d_locallyOwnedRanges in sequence,
         * d_nLocallyOwnedRangesCumulativeEndIds[l] tells us where the start and
         * end point of the \f$l^{th}\f$ locally-owned-range will lie in the
         * concatenated list
         */
        std::vector<std::pair<size_type, size_type>>
          d_locallyOwnedRangesCumulativePairs;

        /**
         * Vector to store \p ghost-set (see top of the page for description)
         * This is an ordered set (strictly ncreasing) of non-negative integers
         */
        std::vector<global_size_type> d_ghostIndices;

        /**
         * Number of ghost indices in the current processor,
         * i.e., the size of \p d_ghostIndices
         */
        size_type d_numGhostIndices;


        /**
         * An OptimizedIndexSet object to store the ghost indices for
         * efficient operations. The OptimizedIndexSet internally creates
         * contiguous sub-ranges within the set of indices and hence can
         * optimize the finding of an index
         */
        OptimizedIndexSet<global_size_type> d_ghostIndicesOptimizedIndexSet;

        /**
         * Vector of size \p d_numGhostIndices to store \p rangeId of each ghost
         * index. That is d_ghostIndicesRangeId[i] tells to which of the \f$K\f$
         * \p global-ranges the i-th ghost index in \p d_ghostIndices belongs to
         */
        std::vector<size_type> d_ghostIndicesRangeId;

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
         * indices of the current processor are owned by a ghost processor.
         * That is \p d_numGhostIndicesInGhostProcs[i] stores the number of
         * ghost indices owned by the processor id given by \p d_ghostProcIds[i]
         */
        std::vector<size_type> d_numGhostIndicesInGhostProcs;

        /**
         * A flattened vector of size number of ghosts containing the ghost
         * indices ordered as per the list of ghost processor Ids in
         * d_ghostProcIds. To elaborate, let  \f$M_i=\f$ \p
         * d_numGhostIndicesInGhostProcs[i] be the number of ghost indices owned
         * by the \f$i^{th}\f$ ghost processor (i.e., d_ghostProcIds[i]). Let
         * \f$S_i = \{x_1^i,x_2^i,\ldots,x_{M_i}^i\}\f$, be an ordered set
         * containing the ghost indices owned by the \f$i^{th}\f$ ghost
         * processor (i.e., d_ghostProcIds[i]). Then we can define \f$s_i =
         * \{z_1^i,z_2^i,\ldots,z_{M_i}^i\}\f$ to be set defining the positions
         * of the \f$x_j^i\f$ in \p d_ghostIndices, i.e., \f$x_j^i=\f$ \p
         * d_ghostIndices[\f$z_j^i\f$]. The indices \f$x_j^i\f$ are called local
         * ghost indices as they store the relative position of a ghost index in
         * \p d_ghostIndices. Given that \f$S_i\f$ and \p d_ghostIndices are
         * both ordered sets, \f$s_i\f$ will also be ordered. The vector
         * d_flattenedLocalGhostIndices stores the concatenation of \f$s_i\f$'s.
         *
         * @note We store only the local ghost idnex index local to this processor, i.e.,
         * position of the ghost index in d_ghostIndices.
         * This is done to use size_type which is unsigned int instead of
         * global_size_type which is long unsigned it. This helps in reducing
         * the volume of data transfered during MPI calls.
         */
        SizeTypeVector d_flattenedLocalGhostIndices;

        /**
         * A vector of size 2 times the number of ghost processors
         * to store the start and end positions in the above
         * \p d_flattenedLocalGhostIndices that define the local ghost
         * indices owned by each of the ghost processors. To elaborate, for the
         * \f$i^{th}\f$ ghost processor (i.e., d_ghostProcIds[i]), the two
         * integers \f$n=\f$ \p d_localGhostIndicesRanges[2*i] and \f$m=\f$ \p
         * d_localGhostIndicesRanges[2*i+1] define the start and end positions
         * in \p d_flattenedLocalGhostIndices that belong to the \f$i^{th}\f$
         * ghost processor. In other words, the set \f$s_i\f$ (defined in \p
         * d_flattenedLocalGhostIndices above) containing the local ghost
         * indices owned by the \f$i^{th}\f$ ghost processor is given by:
         * \f$s_i=\f$ {\p d_flattenedLocalGhostIndices[\f$n\f$],
         * d_flattenedLocalGhostIndices[\f$n+1\f$], ..., \p
         * d_flattenedLocalGhostIndices[\f$m-1\f$].
         */
        std::vector<size_type> d_localGhostIndicesRanges;

        /**
         * A 2D vector containing the cumulative start point of each
         * locally-owned-ranges for each ghost processors for the current
         * processor. For the \f$i^{th}\f$ ghost processor (i.e., the one whose
         * rank/id is given by d_ghostProcIds[i]) and \f$l^{th}\f$ \p
         * locally-owned-range, d_ghostProcLocallyOwnedRangesCumulative[i][l] =
         * \f$\sum_{j=0}^{l-1} (b_j^i - a_j^i)\f$, where \f$a_j^i\f$ and
         * \f$b_j^i\f$ define the \f$j^{th}\f$ \p locally-owned-range in the
         * \f$i^{th}\f$ ghost processor (i.e., processor with rank/id given by
         * d_ghostProcIds[i]) In other words, if we concatenate the indices
         * defined by all the \p locally-owned-ranges of the \f$i^{th}\f$ ghost
         * processor in sequence,
         * d_ghostProcLocallyOwnedRangesCumulativePairs[i][l] tells us where the
         * start point of the \f$l^{th}\f$ locally-owned-range for the
         * \f$i^{th}\f$ ghost processor will lie in the concatenated list
         */
        std::vector<std::vector<size_type>>
          d_ghostProcLocallyOwnedRangesCumulative;

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

        /** Vector of size \f$\sum_i\f$ \p
         * d_numOwnedIndicesForTargetProcs[\f$i\f$] to store all the locally
         * owned indices which other processors need (i.e., which are ghost
         * indices in other processors). It is stored as a concatentation of
         * lists \f$L_i = \{o_1^i,o_2^i,\ldots,o_{M_i}^i\}\f$, where where
         * \f$o_j^i\f$'s are locally owned indices that are needed by the
         * \f$i^{th}\f$ target processor (i.e., d_targetProcIds[\f$i\f$]) and
         * \f$M_i=\f$ \p d_numOwnedIndicesForTargetProcs[\f$i\f$] is the number
         * of indices to be sent to \f$i^{th}\f$ target processor. The list
         * \f$L_i\f$ must be ordered.
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
