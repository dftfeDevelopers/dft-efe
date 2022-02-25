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

#include <linearAlgebra/VectorBase.h>
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
    class DistributedVector : public VectorBase<ValueType, memorySpace>
    {
    public:
      /**
       * @brief Default Destructor
       */
      ~DistributedVector() = default;

#ifdef DFTEFE_WITH_MPI
      /**
       * @brief Constructor based on an input mpiCommunicatorP2P.
       *
       * @param mpiCommunicatorP2P A shared_ptr to const MPICommunicatorP2P
       * based on which the DistributedVector will be created.
       * @param initVal value with which the DistributedVector shoud be
       * initialized
       *
       */
      DistributedVector(
        std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
                        mpiCommunicatorP2P,
        const ValueType initVal = ValueType());

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
        const ValueType initVal = ValueType());
#endif // DFTEFE_WITH_MPI

      /**
       * @brief Copy constructor
       * @param[in] u DistributedVector object to copy from
       */
      DistributedVector(const DistributedVector &u);

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
       * @brief Compound addition for elementwise addition lhs += rhs
       * @param[in] rhs the vector to add
       * @return the original vector
       */
      virtual VectorBase<ValueType, memorySpace> &
      operator+=(const VectorBase<ValueType, memorySpace> &rhs) = 0;

      /**
       * @brief Compound subtraction for elementwise addition lhs -= rhs
       * @param[in] rhs the vector to subtract
       * @return the original vector
       */
      virtual VectorBase<ValueType, memorySpace> &
      operator-=(const VectorBase<ValueType, memorySpace> &rhs) = 0;

      /**
       * @brief Return iterator pointing to the beginning of DistributedVector data.
       *
       * @returns Iterator pointing to the beginning of DistributedVector.
       */
      typename VectorBase<ValueType, memorySpace>::iterator
      begin() override;

      /**
       * @brief Return iterator pointing to the beginning of DistributedVector
       * data.
       *
       * @returns Constant iterator pointing to the beginning of
       * DistributedVector.
       */
      typename VectorBase<ValueType, memorySpace>::const_iterator
      begin() const override;

      /**
       * @brief Return iterator pointing to the end of DistributedVector data.
       *
       * @returns Iterator pointing to the end of DistributedVector.
       */
      typename VectorBase<ValueType, memorySpace>::iterator
      end() override;

      /**
       * @brief Return iterator pointing to the end of DistributedVector data.
       *
       * @returns Constant iterator pointing to the end of
       * DistributedVector.
       */
      typename VectorBase<ValueType, memorySpace>::const_iterator
      end() const override;

      /**
       * @brief Returns the size of the DistributedVector
       * @returns size of the DistributedVector
       */
      size_type
      size() const override;

      /**
       * @brief Return the raw pointer to the DistributedVector data
       * @return pointer to data
       */
      ValueType *
      data() override;

      /**
       * @brief Return the constant raw pointer to the DistributedVector data
       * @return pointer to const data
       */
      const ValueType *
      data() const override;

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

      /**
       * @brief Returns a const reference to the underlying storage
       * of the DistributedVector.
       *
       * @return const reference to the underlying MemoryStorage.
       */
      const typename VectorBase<ValueType, memorySpace>::Storage &
      getStorage() const override;

      /**
       * @brief Returns a VectorAttributes object that stores various attributes
       * (e.g., Serial or Distributed, number of components, etc)
       *
       * @return const reference to the VectorAttributes
       */
      const VectorAttributes &
      getVectorAttributes() const = 0;

      void
      scatterToGhost(const size_type communicationChannel = 0) override;

      void
      gatherFromGhost(const size_type communicationChannel = 0) override;

      void
      scatterToGhostBegin(const size_type communicationChannel = 0) override;

      void
      scatterToGhostEnd() override;

      void
      gatherFromGhostBegin(const size_type communicationChannel = 0) override;

      void
      gatherFromGhostEnd() override;

    private:
      std::shared_ptr<typename VectorBase<ValueType, memorySpace>::Storage>
                       d_storage;
      VectorAttributes d_vectorAttributes;
      size_type        d_localSize;
      size_type        d_localOwnedSize;
      size_type        d_localGhostSize;
      size_type        d_globalSize;
      std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
        d_mpiCommunicatorP2P;
      std::shared_ptr<const utils::MPIPatternP2P<memorySpace>> d_mpiPatternP2P;
    };

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/DistributedVector.t.cpp>
#endif // dftefeDistributedVector_h
