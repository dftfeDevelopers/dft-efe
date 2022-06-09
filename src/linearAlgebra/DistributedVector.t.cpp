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
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <linearAlgebra/BlasLapack.h>
#include <cmath>
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
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                    mpiPatternP2P,
      LinAlgOpContext<memorySpace> *linAlgOpContext,
      const ValueType               initVal)
      : d_mpiPatternP2P(mpiPatternP2P)
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
      d_linAlgOpContext = linAlgOpContext;

      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(mpiPatternP2P,
                                                                blockSize);
    }

    //
    // Constructor using user provided Vector::Storage (i.e.,
    // utils::MemoryStorage)
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      std::unique_ptr<typename Vector<ValueType, memorySpace>::Storage>
        &storage,
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                    mpiPatternP2P,
      LinAlgOpContext<memorySpace> *linAlgOpContext)
      : d_mpiPatternP2P(mpiPatternP2P)
    {
      d_storage         = std::move(storage);
      d_linAlgOpContext = std::move(linAlgOpContext);
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::DISTRIBUTED);
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(mpiPatternP2P,
                                                                blockSize);
    }

    /**
     * @brief Constructor using locally owned range and ghost indices
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const std::vector<dftefe::global_size_type> &       ghostIndices,
      const utils::mpi::MPIComm &                         mpiComm,
      LinAlgOpContext<memorySpace> *                      linAlgOpContext,
      const ValueType                                     initVal)
    {
      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = utils::mpi::MPICommRank(mpiComm, &mpiRank);
      std::string msg = "Error occured while using MPI_Comm_rank. "
                        "Error code: " +
                        std::to_string(err);
      utils::throwException(err == utils::mpi::MPISuccess, msg);
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a DistributedVector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the other constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }
      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                blockSize);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          d_localSize, initVal);
      d_linAlgOpContext = linAlgOpContext;
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
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const utils::mpi::MPIComm &                         mpiComm,
      LinAlgOpContext<memorySpace> *                      linAlgOpContext,
      const ValueType                                     initVal)
    {
      std::vector<dftefe::global_size_type> ghostIndices;
      ghostIndices.resize(0);
      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = utils::mpi::MPICommRank(mpiComm, &mpiRank);
      std::string msg = "Error occured while using MPI_Comm_rank. "
                        "Error code: " +
                        std::to_string(err);
      utils::throwException(err == utils::mpi::MPISuccess, msg);
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a DistributedVector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the other constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }
      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                blockSize);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          d_localSize, initVal);
      d_linAlgOpContext = linAlgOpContext;
    }


    /**
     * @brief This constructor takes the total indices and divides them
     * up equitably across all processors. This decompositon is not compatible
     * with other constructors.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P or another vector as far
     * as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      const global_size_type        totalGlobalDofs,
      const utils::mpi::MPIComm &   mpiComm,
      LinAlgOpContext<memorySpace> *linAlgOpContext,
      const ValueType               initVal)
    {
      std::vector<dftefe::global_size_type> ghostIndices;
      ghostIndices.resize(0);

      std::pair<global_size_type, global_size_type> locallyOwnedRange;

      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = utils::mpi::MPICommRank(mpiComm, &mpiRank);
      std::string msg = "Error occured while using MPI_Comm_rank. "
                        "Error code: " +
                        std::to_string(err);
      utils::throwException(err == utils::mpi::MPISuccess, msg);
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a DistributedVector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the other constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }

      int mpiProcess;

      int         errProc = utils::mpi::MPICommSize(mpiComm, &mpiProcess);
      std::string msgProc = "Error occured while using MPI_Comm_size. "
                            "Error code: " +
                            std::to_string(errProc);
      utils::throwException(errProc == utils::mpi::MPISuccess, msgProc);

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
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                blockSize);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          d_localSize, initVal);
      d_linAlgOpContext = linAlgOpContext;
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
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, (u.d_mpiCommunicatorP2P).getBlockSize());
      d_linAlgOpContext  = u.d_linAlgOpContext;
      d_vectorAttributes = u.d_vectorAttributes;
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_mpiPatternP2P    = u.d_mpiPatternP2P;
      VectorAttributes vectorAttributesDistributed(
        VectorAttributes::Distribution::DISTRIBUTED);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      const DistributedVector &u,
      ValueType                initVal)
    {
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size(), initVal);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, (u.d_mpiCommunicatorP2P).getBlockSize());
      d_linAlgOpContext  = u.d_linAlgOpContext;
      d_vectorAttributes = u.d_vectorAttributes;
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_mpiPatternP2P    = u.d_mpiPatternP2P;
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
      d_linAlgOpContext    = std::move(u.d_linAlgOpContext);
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
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, (u.d_mpiCommunicatorP2P).getBlockSize());
      d_linAlgOpContext  = u.d_linAlgOpContext;
      d_vectorAttributes = u.d_vectorAttributes;
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_mpiPatternP2P    = u.d_mpiPatternP2P;
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
      d_linAlgOpContext    = std::move(u.d_linAlgOpContext);
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
        blasLapack::nrm2<ValueType, memorySpace>(d_locallyOwnedSize,
                                                 this->data(),
                                                 1,
                                                 *d_linAlgOpContext);
      const double l2NormLocallyOwnedSquare =
        l2NormLocallyOwned * l2NormLocallyOwned;
      double returnValue = 0.0;
      utils::mpi::MPIAllreduce(&l2NormLocallyOwnedSquare,
                               &returnValue,
                               1,
                               utils::mpi::MPIDouble,
                               utils::mpi::MPISum,
                               d_mpiPatternP2P->mpiCommunicator());
      returnValue = std::sqrt(returnValue);
      return returnValue;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    DistributedVector<ValueType, memorySpace>::lInfNorm() const
    {
      const double lInfNormLocallyOwned =
        blasLapack::amax<ValueType, memorySpace>(d_locallyOwnedSize,
                                                 this->data(),
                                                 1,
                                                 *d_linAlgOpContext);
      double returnValue = lInfNormLocallyOwned;
      utils::mpi::MPIAllreduce(&lInfNormLocallyOwned,
                               &returnValue,
                               1,
                               utils::mpi::MPIDouble,
                               utils::mpi::MPIMax,
                               d_mpiPatternP2P->mpiCommunicator());
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
      d_mpiCommunicatorP2P->accumulateAddLocallyOwned(*d_storage,
                                                      communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::updateGhostValuesBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->updateGhostValuesBegin(*d_storage,
                                                   communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::updateGhostValuesEnd()
    {
      d_mpiCommunicatorP2P->updateGhostValuesEnd();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::accumulateAddLocallyOwnedBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwnedBegin(
        *d_storage, communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DistributedVector<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd()
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwnedEnd(*d_storage);
    }
  } // end of namespace linearAlgebra
} // namespace dftefe
