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
 * @author Sambit Das
 */


#ifndef dftefeDistributedMultiVector_h
#define dftefeDistributedMultiVector_h

#include <linearAlgebra/Vector.h>
#include <linearAlgebra/VectorAttributes.h>
#include <utils/MemoryStorage.h>
#include <utils/MPICommunicatorP2P.h>
#include <utils/MPIPatternP2P.h>
#include <utils/TypeConfig.h>
#include <memory>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief A derived class of MultiVector for a distributed multi vector
     * (i.e., a multi vector that is distributed across a set of processors)
     *
     * @tparam template parameter ValueType defines underlying datatype being stored
     *  in the multi vector (i.e., int, double, complex<double>, etc.)
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the multi vector must reside.
     *
     * @note Broadly, there are two ways of constructing a DistributedMultiVector.
     *  1. [<b>Prefered and efficient approach</b>] The first approach takes a
     * pointer to an MPICommunicatorP2P as an input argument. The
     * MPICommunicatorP2P, in turn, contains all the information reagrding the
     * locally owned and ghost part of the DistributedMultiVector as well as the
     * number of vectors. This is the most efficient way of constructing a
     * DistributedMultiVector as it allows for reusing of an already constructed
     * MPICommunicator.
     *  2. [<b> Expensive approach</b>] The second approach takes in the locally
     * owned, ghost indices, and number of vectors and internally creates an
     * MPIPatternP2P object which, in turn, is used to create an
     * MPICommunicatorP2P object. Given that the creation of an MPIPatternP2P is
     * expensive, this route of constructing a DistributedMultiVector
     * <b>should</b> be avoided.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class DistributedMultiVector : public MultiVector<ValueType, memorySpace>
    {
    public:
      //
      // Pulling base class (MultiVector) protected names here so to avoid full
      // name scoping inside the source file. The other work around is to use
      // this->d_m (where d_m is a protected data member of base class). This is
      // something which is peculiar to inheritance using class templates. The
      // reason why this is so is the fact that C++ does not consider base class
      // templates for name resolution (i.e., they are dependent names and
      // dependent names are not considered)
      //
      using Vector<ValueType, memorySpace>::d_storage;
      using Vector<ValueType, memorySpace>::d_vectorAttributes;
      using Vector<ValueType, memorySpace>::d_globalSize;
      using Vector<ValueType, memorySpace>::d_locallyOwnedSize;
      using Vector<ValueType, memorySpace>::d_ghostSize;
      using Vector<ValueType, memorySpace>::d_localSize;
      using Vector<ValueType, memorySpace>::d_numVectors;

    public:
      /**
       * @brief Default Destructor
       */
      ~DistributedMultiVector() = default;

#ifdef DFTEFE_WITH_MPI
      /**
       * @brief Constructor based on an input mpiCommunicatorP2P.
       *
       * @param[in] mpiCommunicatorP2P A shared_ptr to const MPICommunicatorP2P
       * based on which the DistributedMultiVector will be created.
       * @param[in] initVal value with which the DistributedMultiVector shoud be
       * initialized
       *
       */
      DistributedMultiVector(
        std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
                        mpiCommunicatorP2P,
        const ValueType initVal = ValueType());

      /**
       * @brief Constructor with predefined Vector::Storage (i.e., utils::MemoryStorage) and mpiCommunicatorP2P.
       * This allows the DistributedMultiVector to take ownership of input
       * MultiVector::Storage (i.e., utils::MemoryStorage) This is useful when
       * one does not want to allocate new memory and instead use memory
       * allocated in the MultiVector::Storage (i.e., MemoryStorage).
       *
       * @param[in] storage unique_ptr to MultiVector::Storage whose ownership
       * is to be transfered to the DistributedMultiVector
       * @param[in] mpiCommunicatorP2P A shared_ptr to const MPICommunicatorP2P
       * based on which the DistributedMultiVector will be created.
       *
       * @note This Constructor transfers the ownership from the input unique_ptr \p storage to the internal data member of the DistributedMultiVector.
       * Thus, after the function call \p storage will point to NULL and any
       * access through \p storage will lead to <b>undefined behavior</b>.
       *
       */
      DistributedMultiVector(
        std::unique_ptr<typename Vector<ValueType, memorySpace>::Storage>
          &storage,
        std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
          mpiCommunicatorP2P);

      /**
       * @brief Constructor based on locally owned and ghost indices.
       * @note This way of construction is expensive. One should use the other
       * constructor based on an input MPICommunicatorP2P as far as possible.
       *
       * @param locallyOwnedRange a pair \f$(a,b)\f$ which defines a range of indices (continuous)
       * that are owned by the current processor.
       * @param ghostIndices vector containing an ordered set of ghost indices
       * (ordered in increasing order and non-repeating)
       *
       * @note The locallyOwnedRange should be an open interval where the start index included,
       * but the end index is not included.
       */
      DistributedMultiVector(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const std::vector<dftefe::global_size_type> &       ghostIndices,
        const MPI_Comm &                                    mpiComm,
        size_type                                           numVectors,
        ValueType initVal = ValueType());
#endif // DFTEFE_WITH_MPI

      /**
       * @brief Copy constructor
       * @param[in] u DistributedMultiVector object to copy from
       */
      DistributedMultiVector(const DistributedMultiVector &u);

      /**
       * @brief Move constructor
       * @param[in] u DistributedMultiVector object to move from
       */
      DistributedMultiVector(DistributedMultiVector &&u) noexcept;

      /**
       * @brief Copy assignment operator
       * @param[in] u const reference to DistributedMultiVector object to copy
       * from
       * @return reference to this object after copying data from u
       */
      DistributedMultiVector &
      operator=(const DistributedMultiVector &u);

      /**
       * @brief Move assignment operator
       * @param[in] u const reference to DistributedMultiVector object to move
       * from
       * @return reference to this object after moving data from u
       */
      DistributedMultiVector &
      operator=(DistributedMultiVector &&u);

      /**
       * @brief Default constructor
       */
      DistributedMultiVector() = default;

      /**
       * @brief Returns \f$ l_2 \f$ norms of all the \f$N\f$ vectors in the  MultiVector
       * @return \f$ l_2 \f$  norms of the various vectors as std::vector<double> type
       */
      std::vector<double>
      l2Norms() const override;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norms of all the \f$N\f$ vectors in the  MultiVector
       * @return \f$ l_{\inf} \f$  norms of the various vectors as std::vector<double> type
       */
      std::vector<double>
      lInfNorms() const override;

      void
      updateGhostValues(const size_type communicationChannel = 0) override;

      void
      accumulateAddLocallyOwned(
        const size_type communicationChannel = 0) override;

      void
      updateGhostValuesBegin(const size_type communicationChannel = 0) override;

      void
      updateGhostValuesEnd() override;

      void
      accumulateAddLocallyOwnedBegin(
        const size_type communicationChannel = 0) override;

      void
      accumulateAddLocallyOwnedEnd() override;

    private:
      std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
        d_mpiCommunicatorP2P;
      std::shared_ptr<const utils::MPIPatternP2P<memorySpace>> d_mpiPatternP2P;
    };

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/DistributedMultiVector.t.cpp>
#endif // dftefeDistributedMultiVector_h
