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


#ifndef dftefeDistributedVector_h
#define dftefeDistributedVector_h

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
     * @brief A derived class of VectorBase for a distributed vector
     * (i.e., a vector that is distributed across a set of processors)
     *
     * @tparam template parameter ValueType defines underlying datatype being stored
     *  in the vector (i.e., int, double, complex<double>, etc.)
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the vector must reside.
     *
     * @note Broadly, there are two ways of constructing a DistributedVector.
     *  1. [<b>Prefered and efficient approach</b>] The first approach takes a
     * pointer to an MPICommunicatorP2P as an input argument. The
     * MPICommunicatorP2P, in turn, contains all the information reagrding the
     * locally owned and ghost part of the DistributedVector. This is the most
     * efficient way of constructing a DistributedVector as it allows for
     * reusing of an already constructed MPICommunicator.
     *  2. [<b> Expensive approach</b>] The second approach takes in the locally
     * owned and ghost indices and internally creates an MPIPatternP2P object
     * which, in turn, is used to create an MPICommunicatorP2P object. Given
     * that the creation of an MPIPatternP2P is expensive, this route of
     *     constructing a DistributedVector <b>should</b> be avoided.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class DistributedVector : public Vector<ValueType, memorySpace>
    {
    public:
      //
      // Pulling base class (Vector) protected names here so to avoid full name
      // scoping inside the source file. The other work around is to use
      // this->d_m (where d_m is a protected data member of base class). This is
      // something which is peculiar to inheritance using class templates. The
      // reason why this is so is the fact that C++ does not consider base class
      // templates for name resolution (i.e., they are dependent names and
      // dependent names are not considered)
      //
      using Vector<ValueType, memorySpace>::d_storage;
      using Vector<ValueType, memorySpace>::d_BlasQueue;
      using Vector<ValueType, memorySpace>::d_vectorAttributes;
      using Vector<ValueType, memorySpace>::d_globalSize;
      using Vector<ValueType, memorySpace>::d_locallyOwnedSize;
      using Vector<ValueType, memorySpace>::d_ghostSize;
      using Vector<ValueType, memorySpace>::d_localSize;

    public:
      /**
       * @brief Default Destructor
       */
      ~DistributedVector() = default;

#ifdef DFTEFE_WITH_MPI
      /**
       * @brief Constructor based on an input mpiCommunicatorP2P.
       *
       * @param[in] mpiCommunicatorP2P A shared_ptr to const MPICommunicatorP2P
       * based on which the DistributedVector will be created.
       * @param[in] initVal value with which the DistributedVector shoud be
       * initialized
       * @param[in] BlasQueue handle for linear algebra operations on
       * HOST/DEVICE.
       *
       */
      DistributedVector(
        std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
                        mpiCommunicatorP2P,
        const ValueType initVal,
        std::shared_ptr<blasLapack::BlasQueueType<memorySpace>> BlasQueue);

      /**
       * @brief Constructor with predefined Vector::Storage (i.e., utils::MemoryStorage) and mpiCommunicatorP2P.
       * This allows the DistributedVector to take ownership of input
       * Vector::Storage (i.e., utils::MemoryStorage) This is useful when one
       * does not want to allocate new memory and instead use memory allocated
       * in the Vector::Storage (i.e., MemoryStorage).
       *
       * @param[in] storage unique_ptr to Vector::Storage whose ownership
       * is to be transfered to the DistributedVector
       * @param[in] mpiCommunicatorP2P A shared_ptr to const MPICommunicatorP2P
       * based on which the DistributedVector will be created.
       * @param[in] BlasQueue handle for linear algebra operations on
       * HOST/DEVICE.
       *
       * @note This Constructor transfers the ownership from the input unique_ptr \p storage to the internal data member of the DistributedVector.
       * Thus, after the function call \p storage will point to NULL and any
       * access through \p storage will lead to <b>undefined behavior</b>.
       *
       */
      DistributedVector(
        std::unique_ptr<typename Vector<ValueType, memorySpace>::Storage>
          &storage,
        std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
          mpiCommunicatorP2P,
        std::shared_ptr<blasLapack::BlasQueueType<memorySpace>> BlasQueue);

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
      DistributedVector(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const std::vector<dftefe::global_size_type> &       ghostIndices,
        const MPI_Comm &                                    mpiComm,
        const ValueType                                     initVal,
        std::shared_ptr<blasLapack::BlasQueueType<memorySpace>> BlasQueue);

      /**
       * @brief Constructor based on locally owned indices. This does not contain
       * any ghost indices. This is to store the locally owned part of a
       * distributed Vector
       * @note This way of construction is expensive. One should use the other
       * constructor based on an input MPICommunicatorP2P as far as possible.
       *
       * @param locallyOwnedRange a pair \f$(a,b)\f$ which defines a range of indices (continuous)
       * that are owned by the current processor.
       *
       * @note The locallyOwnedRange should be an open interval where the start index included,
       * but the end index is not included.
       */
      DistributedVector(
        const std::pair<global_size_type, global_size_type> locallyOwnedRange,
        const MPI_Comm &                                    mpiComm,
        const ValueType                                     initVal,
        std::shared_ptr<blasLapack::BlasQueueType<memorySpace>> BlasQueue);


      /**
       * @brief Constructor based on total number of global indices.
       * This does not contain any ghost indices. This is to store the locally
       * owned part of a distributed Vector. The vector is divided to ensure
       * equitability as much as possible.
       * @note This way of construction is expensive. One should use the other
       * constructor based on an input MPICommunicatorP2P as far as possible.
       * Further, the decomposotion is not compatible with other ways of
       * distributed vector construction.
       *
       * @param totalGlobalDofs Total number of global indices that is distributed
       * over the processors.
       *
       */
      DistributedVector(
        const global_size_type                                  totalGlobalDofs,
        const MPI_Comm &                                        mpiComm,
        const ValueType                                         initVal,
        std::shared_ptr<blasLapack::BlasQueueType<memorySpace>> BlasQueue);

#endif // DFTEFE_WITH_MPI

      /**
       * @brief Copy constructor
       * @param[in] u DistributedVector object to copy from
       */
      DistributedVector(const DistributedVector &u);

      /**
       * @brief Copy constructor with reinitialisation
       * @param[in] u DistributedVector object to copy from
       * @param[in] initVal Initial value of the vector
       */
      DistributedVector(const DistributedVector &u, ValueType initVal);

      /**
       * @brief Move constructor
       * @param[in] u DistributedVector object to move from
       */
      DistributedVector(DistributedVector &&u) noexcept;

      /**
       * @brief Copy assignment operator
       * @param[in] u const reference to DistributedVector object to copy from
       * @return reference to this object after copying data from u
       */
      DistributedVector &
      operator=(const DistributedVector &u);

      /**
       * @brief Move assignment operator
       * @param[in] u const reference to DistributedVector object to move from
       * @return reference to this object after moving data from u
       */
      DistributedVector &
      operator=(DistributedVector &&u);

      /**
       * @brief Default constructor
       */
      DistributedVector() = default;
      /**
       * @brief Returns \f$ l_2 \f$ norm of the DistributedVector
       * @return \f$ l_2 \f$  norm of the vector
       */
      double
      l2Norm() const override;

      /**
       * @brief Returns \f$ l_{\inf} \f$ norm of the DistributedVector
       * @return \f$ l_{\inf} \f$  norm of the vector
       */
      double
      lInfNorm() const override;

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
#include <linearAlgebra/DistributedVector.t.cpp>
#endif // dftefeDistributedVector_h
