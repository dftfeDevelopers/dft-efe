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
#include <cmath>
#include <linearAlgebra/BlasLapack.h>

namespace dftefe
{
  namespace linearAlgebra
  {
#ifdef DFTEFE_WITH_MPI
    //
    // Constructor using an existing MPICommunicatorP2P object
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
                      mpiCommunicatorP2P,
      const ValueType initVal,
      std::shared_ptr<blasLapack::blasQueueType<memorySpace>> blasQueue)
      : d_mpiCommunicatorP2P(mpiCommunicatorP2P)
      , d_mpiPatternP2P(mpiCommunicatorP2P.getMPIPatternP2P())
    {
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::DISTRIBUTED);
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          d_localSize, initVal);
      d_blasQueue = blasQueue;
    }

    //
    // Constructor using user provided Vector::Storage (i.e.,
    // utils::MemoryStorage)
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      std::unique_ptr<typename Vector<ValueType, memorySpace>::Storage>
        &storage,
      std::shared_ptr<const utils::MPICommunicatorP2P<ValueType, memorySpace>>
        mpiCommunicatorP2P,
      std::shared_ptr<blasLapack::blasQueueType<memorySpace>> blasQueue)
      : d_mpiCommunicatorP2P(mpiCommunicatorP2P)
      , d_mpiPatternP2P(mpiCommunicatorP2P.getMPIPatternP2P())
    {
      d_storage   = std::move(storage);
      d_blasQueue = std::move(blasQueue);
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::DISTRIBUTED);
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
    }

    /**
     * @brief Constructor using locally owned range and ghost indices
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPICommunicatorP2P as far as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      const std::pair<global_size_type, global_size_type>     locallyOwnedRange,
      const std::vector<dftefe::global_size_type> &           ghostIndices,
      const MPI_Comm &                                        mpiComm,
      const ValueType                                         initVal,
      std::shared_ptr<blasLapack::blasQueueType<memorySpace>> blasQueue)
    {
      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = MPI_Comm_rank(mpiComm, &mpiRank);
      std::string msg = "Error occured while using MPI_Comm_rank. "
                        "Error code: " +
                        std::to_string(err);
      utils::throwException(err == MPI_SUCCESS, msg);
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a DistributedVector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the other constructor based on an input MPICommunicatorP2P.";
          std::cout << msg << std::endl;
        }
      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_shared<
        const utils::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, blockSize);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          d_localSize, initVal);
      d_blasQueue = blasQueue;
    }

    /**
     * @brief Constructor using locally owned range. This vector does not contain any
     * ghost indices
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPICommunicatorP2P or another vector as far
     * as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      const std::pair<global_size_type, global_size_type>     locallyOwnedRange,
      const MPI_Comm &                                        mpiComm,
      const ValueType                                         initVal,
      std::shared_ptr<blasLapack::blasQueueType<memorySpace>> blasQueue)
    {
      std::vector<dftefe::global_size_type> ghostIndices;
      ghostIndices.resize(0);
      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = MPI_Comm_rank(mpiComm, &mpiRank);
      std::string msg = "Error occured while using MPI_Comm_rank. "
                        "Error code: " +
                        std::to_string(err);
      utils::throwException(err == MPI_SUCCESS, msg);
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a DistributedVector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the other constructor based on an input MPICommunicatorP2P.";
          std::cout << msg << std::endl;
        }
      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_shared<
        const utils::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, blockSize);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_storage =
        std::make_shared<typename Vector<ValueType, memorySpace>::Storage>(
          d_localSize, initVal);
      d_blasQueue = blasQueue;
    }


    /**
     * @brief This constructor takes the total indices and divides them
     * up equitably across all processors. This decompositon is not compatible
     * with other constructors.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPICommunicatorP2P or another vector as far
     * as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      const global_size_type                                  totalGlobalDofs,
      const MPI_Comm &                                        mpiComm,
      const ValueType                                         initVal,
      std::shared_ptr<blasLapack::blasQueueType<memorySpace>> blasQueue)
    {
      std::vector<dftefe::global_size_type> ghostIndices;
      ghostIndices.resize(0);

      std::pair<global_size_type, global_size_type> locallyOwnedRange;

      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = MPI_Comm_rank(mpiComm, &mpiRank);
      std::string msg = "Error occured while using MPI_Comm_rank. "
                        "Error code: " +
                        std::to_string(err);
      utils::throwException(err == MPI_SUCCESS, msg);
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a DistributedVector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the other constructor based on an input MPICommunicatorP2P.";
          std::cout << msg << std::endl;
        }

      int mpiProcess;

      int         errProc = MPI_Comm_size(mpiComm, &mpiProcess);
      std::string msgProc = "Error occured while using MPI_Comm_size. "
                            "Error code: " +
                            std::to_string(errProc);
      utils::throwException(errProc == MPI_SUCCESS, msgProc);

      dftefe::global_size_type locallyOwnedSize = totalGlobalDofs / mpiProcess;
      if (mpiRank < totalGlobalDofs % mpiProcess)
        locallyOwnedSize++;

      dftefe::global_size_type startIndex =
        mpiRank * (totalGlobalDofs / mpiProcess);
      if (mpiRank < totalGlobalDofs % mpiProcess)
        startIndex += mpiRank;
      else
        startIndex += totalGlobalDofs % mpiProcess;

      locallyOwnedRange.first  = startIndex;
      locallyOwnedRange.second = startIndex + locallyOwnedSize;


      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_shared<
        const utils::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, blockSize);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_storage =
        std::make_shared<typename Vector<ValueType, memorySpace>::Storage>(
          d_localSize, initVal);
      d_blasQueue = blasQueue;
    }

#endif // DFTEFE_WITH_MPI

    //
    // Copy Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      const DistributedVector &u)
    {
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage           = *(u.d_storage);
      d_blasQueue          = u.d_blasQueue;
      d_vectorAttributes   = u.d_vectorAttributes;
      d_localSize          = u.d_localSize;
      d_locallyOwnedSize   = u.d_locallyOwnedSize;
      d_ghostSize          = u.d_ghostSize;
      d_globalSize         = u.d_globalSize;
      d_mpiCommunicatorP2P = u.d_mpiCommunicatorP2P;
      d_mpiPatternP2P      = u.d_mpiPatternP2P;
      VectorAttributes vectorAttributesDistributed(
        VectorAttributes::Distribution::DISTRIBUTED);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      const DistributedVector &u,
      ValueType                initVal)
    {
      d_storage =
        std::make_shared<typename Vector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size(), initVal);
      d_blasQueue          = u.d_blasQueue;
      d_vectorAttributes   = u.d_vectorAttributes;
      d_localSize          = u.d_localSize;
      d_locallyOwnedSize   = u.d_locallyOwnedSize;
      d_ghostSize          = u.d_ghostSize;
      d_globalSize         = u.d_globalSize;
      d_mpiCommunicatorP2P = u.d_mpiCommunicatorP2P;
      d_mpiPatternP2P      = u.d_mpiPatternP2P;
      VectorAttributes vectorAttributesDistributed(
        VectorAttributes::Distribution::DISTRIBUTED);
    }

    //
    // Move Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      DistributedVector &&u) noexcept
    {
      d_storage            = std::move(u.d_storage);
      d_blasQueue          = std::move(u.d_blasQueue);
      d_vectorAttributes   = std::move(u.d_vectorAttributes);
      d_localSize          = std::move(u.d_localSize);
      d_locallyOwnedSize   = std::move(u.d_locallyOwnedSize);
      d_ghostSize          = std::move(u.d_ghostSize);
      d_globalSize         = std::move(u.d_globalSize);
      d_mpiCommunicatorP2P = std::move(u.d_mpiCommunicatorP2P);
      d_mpiPatternP2P      = std::move(u.d_mpiPatternP2P);
      VectorAttributes vectorAttributesDistributed(
        VectorAttributes::Distribution::DISTRIBUTED);
    }

    //
    // Copy Assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace> &
    DistributedVector<ValueType, memorySpace>::operator=(
      const DistributedVector &u)
    {
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage           = *(u.d_storage);
      d_blasQueue          = u.d_blasQueue;
      d_vectorAttributes   = u.d_vectorAttributes;
      d_localSize          = u.d_localSize;
      d_locallyOwnedSize   = u.d_locallyOwnedSize;
      d_ghostSize          = u.d_ghostSize;
      d_globalSize         = u.d_globalSize;
      d_mpiCommunicatorP2P = u.d_mpiCommunicatorP2P;
      d_mpiPatternP2P      = u.d_mpiPatternP2P;
      VectorAttributes vectorAttributesDistributed(
        VectorAttributes::Distribution::DISTRIBUTED);
      return *this;
    }

    //
    // Move Assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace> &
    DistributedVector<ValueType, memorySpace>::operator=(DistributedVector &&u)
    {
      d_storage            = std::move(u.d_storage);
      d_blasQueue          = std::move(u.d_blasQueue);
      d_vectorAttributes   = std::move(u.d_vectorAttributes);
      d_localSize          = std::move(u.d_localSize);
      d_locallyOwnedSize   = std::move(u.d_locallyOwnedSize);
      d_ghostSize          = std::move(u.d_ghostSize);
      d_globalSize         = std::move(u.d_globalSize);
      d_mpiCommunicatorP2P = std::move(u.d_mpiCommunicatorP2P);
      d_mpiPatternP2P      = std::move(u.d_mpiPatternP2P);
      VectorAttributes vectorAttributesDistributed(
        VectorAttributes::Distribution::DISTRIBUTED);

      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    DistributedVector<ValueType, memorySpace>::l2Norm() const
    {
      const double l2NormLocallyOwned =
        blasLapack::nrm2(d_locallyOwnedSize, this->data(), 1, *d_blasQueue);
      const double l2NormLocallyOwnedSquare =
        l2NormLocallyOwned * l2NormLocallyOwned;
      double returnValue = 0.0;
#ifdef DFTEFE_WITH_MPI
      MPI_Allreduce(&l2NormLocallyOwnedSquare,
                    &returnValue,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    d_mpiPatternP2P->mpiCommunicator());
#else
      returnValue = l2NormLocallyOwnedSquare;
#endif // DFTEFE_WITH_MPI
      returnValue = std::sqrt(returnValue);
      return returnValue;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    DistributedVector<ValueType, memorySpace>::lInfNorm() const
    {
      const double lInfNormLocallyOwned =
        blasLapack::amax(d_locallyOwnedSize, this->data(), 1, *d_blasQueue);
      double returnValue = lInfNormLocallyOwned;
#ifdef DFTEFE_WITH_MPI
      MPI_Allreduce(&lInfNormLocallyOwned,
                    &returnValue,
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    d_mpiPatternP2P->mpiCommunicator());
#endif // DFTEFE_WITH_MPI
      return returnValue;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::updateGhostValues(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->updateGhostValues(*d_storage, communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::accumulateAddLocallyOwned(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->gatherFromGhost(*d_storage, communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::updateGhostValuesBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->scatterToGhostBegin(*d_storage,
                                                communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::updateGhostValuesEnd()
    {
      d_mpiCommunicatorP2P->scatterToGhostEnd();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::accumulateAddLocallyOwnedBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->gatherFromGhostBegin(*d_storage,
                                                 communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd()
    {
      d_mpiCommunicatorP2P->gatherFromGhostEnd(*d_storage);
    }
  } // end of namespace linearAlgebra
} // namespace dftefe
