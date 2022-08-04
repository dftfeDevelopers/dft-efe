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
#include <utils/Exceptions.h>
#include <cmath>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief Constructor for a \b serial MultiVector using size, numVectors and
     * init value
     **/
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const size_type               size,
      const size_type               numVectors,
      LinAlgOpContext<memorySpace> *linAlgOpContext,
      const ValueType               initVal)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          size * numVectors, initVal);
      d_linAlgOpContext = linAlgOpContext;
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::SERIAL);
      d_globalSize       = size;
      d_locallyOwnedSize = size;
      d_ghostSize        = 0;
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(size);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                numVectors);
    }

    /**
     * @brief Constructor for a \serial MultiVector with a predefined
     * MultiVector::Storage (i.e., utils::MemoryStorage).
     * This constructor transfers the ownership of the input Storage to the
     * MultiVector. This is useful when one does not want to allocate new
     * memory and instead use memory allocated in the MultiVector::Storage
     * (i.e., utils::MemoryStorage).
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
                                    storage,
      const size_type               numVectors,
      LinAlgOpContext<memorySpace> *linAlgOpContext)
    {
      d_storage         = std::move(storage);
      d_linAlgOpContext = linAlgOpContext;
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::SERIAL);
      d_globalSize       = d_storage.size();
      d_locallyOwnedSize = d_storage.size();
      d_ghostSize        = 0;
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          d_localSize);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                numVectors);
    }

    //
    // Constructor for \distributed MultiVector using an existing
    // MPIPatternP2P object
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                    mpiPatternP2P,
      LinAlgOpContext<memorySpace> *linAlgOpContext,
      const size_type               numVectors,
      const ValueType               initVal)
      : d_mpiPatternP2P(mpiPatternP2P)
    {
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::DISTRIBUTED);
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * d_numVectors, initVal);
      d_linAlgOpContext    = linAlgOpContext;
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(mpiPatternP2P,
                                                                numVectors);
    }

    /**
     * @brief Constructor for a \b distributed MultiVector with a predefined
     * MultiVector::Storage (i.e., utils::MemoryStorage) and MPIPatternP2P.
     * This constructor transfers the ownership of the input Storage to the
     * MultiVector. This is useful when one does not want to allocate new
     * memory and instead use memory allocated in the input MultiVector::Storage
     * (i.e., utils::MemoryStorage).
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
        &storage,
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                    mpiPatternP2P,
      LinAlgOpContext<memorySpace> *linAlgOpContext,
      const size_type               numVectors)
      : d_mpiPatternP2P(mpiPatternP2P)
    {
      d_storage         = std::move(storage);
      d_linAlgOpContext = std::move(linAlgOpContext);
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::DISTRIBUTED);
      d_globalSize         = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize   = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize          = d_mpiPatternP2P->localGhostSize();
      d_localSize          = d_locallyOwnedSize + d_ghostSize;
      d_numVectors         = numVectors;
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(mpiPatternP2P,
                                                                numVectors);
    }

    /**
     * @brief Constructor for a \distributed MultiVector based on locally
     * owned and ghost indices.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const std::vector<global_size_type> &               ghostIndices,
      const utils::mpi::MPIComm &                         mpiComm,
      LinAlgOpContext<memorySpace> *                      linAlgOpContext,
      const size_type                                     numVectors,
      const ValueType                                     initVal)
    {
      //
      // TODO Move the warning message to a Logger class
      //
      int mpiRank;
      int err = utils::mpi::MPICommRank(mpiComm, &mpiRank);
      std::pair<bool, std::string> errIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(errIsSuccessAndMsg.first,
                            errIsSuccessAndMsg.second);
      std::string msg;
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a distributed MultiVector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }
      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      d_mpiCommunicatorP2P = std::make_unique<
        const utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, numVectors);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * numVectors, initVal);
      d_linAlgOpContext = linAlgOpContext;
    }

    /**
     * @brief Constructor for a special case of \b distributed MultiVector where none
     * none of the processors have any ghost indices.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const utils::mpi::MPIComm &                         mpiComm,
      LinAlgOpContext<memorySpace> *                      linAlgOpContext,
      const size_type                                     numVectors,
      const ValueType                                     initVal)
    {
      //
      // TODO Move the warning message to a Logger class
      //
      int mpiRank;
      int err = utils::mpi::MPICommRank(mpiComm, &mpiRank);
      std::pair<bool, std::string> errIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(errIsSuccessAndMsg.first,
                            errIsSuccessAndMsg.second);
      std::string msg;
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a distributed MultiVector using only locally owned "
            "range is expensive. As far as possible, one should "
            " use the constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }
      ////////////
      std::vector<dftefe::global_size_type> ghostIndices;
      ghostIndices.resize(0);
      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      d_mpiCommunicatorP2P = std::make_unique<
        const utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, numVectors);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * numVectors, initVal);
      d_linAlgOpContext = linAlgOpContext;
    }


    /**
     * @brief Constructor for a \b distributed MultiVector based on total number of global indices.
     * The resulting Vector will not contain any ghost indices on any of the
     * processors. Internally, the vector is divided to ensure as much
     * equitable distribution across all the processors much as possible.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far as possible.
     * Further, the decomposition is not compatible with other ways of
     * distributed MultiVector construction.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const global_size_type        globalSize,
      const utils::mpi::MPIComm &   mpiComm,
      LinAlgOpContext<memorySpace> *linAlgOpContext,
      const size_type               numVectors,
      const ValueType               initVal)
    {
      std::vector<dftefe::global_size_type> ghostIndices;
      ghostIndices.resize(0);

      std::pair<global_size_type, global_size_type> locallyOwnedRange;

      //
      // TODO Move the warning message to a Logger class
      //
      int mpiRank;
      int err = utils::mpi::MPICommRank(mpiComm, &mpiRank);
      std::pair<bool, std::string> errIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(errIsSuccessAndMsg.first,
                            errIsSuccessAndMsg.second);
      std::string msg;
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a MultiVector using total number of indices across all processors "
            "is expensive. As far as possible, one should "
            " use the constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }

      int mpiProcess;
      int errProc        = utils::mpi::MPICommSize(mpiComm, &mpiProcess);
      errIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(errProc);
      utils::throwException(errIsSuccessAndMsg.first,
                            errIsSuccessAndMsg.second);

      dftefe::global_size_type locallyOwnedSize = globalSize / mpiProcess;
      if (mpiRank < globalSize % mpiProcess)
        locallyOwnedSize++;

      dftefe::global_size_type startIndex = mpiRank * (globalSize / mpiProcess);
      if (mpiRank < globalSize % mpiProcess)
        startIndex += mpiRank;
      else
        startIndex += globalSize % mpiProcess;

      locallyOwnedRange.first  = startIndex;
      locallyOwnedRange.second = startIndex + locallyOwnedSize;


      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      d_mpiCommunicatorP2P = std::make_unique<
        const utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, numVectors);

      d_vectorAttributes = VectorAttributes::Distribution::DISTRIBUTED;
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * numVectors, initVal);
      d_linAlgOpContext = linAlgOpContext;
    }



    //
    // Copy Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(const MultiVector &u)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, u.d_numVectors);
      d_linAlgOpContext  = u.d_linAlgOpContext;
      *d_storage         = *(u.d_storage);
      d_vectorAttributes = u.d_vectorAttributes;
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_numVectors       = u.d_numVectors;
      d_mpiPatternP2P    = u.d_mpiPatternP2P;
    }

    //
    // Copy Constructor with reinitialization
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(const MultiVector &u,
                                                     const ValueType    initVal)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size(), initVal);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, u.d_numVectors);
      d_linAlgOpContext  = u.d_linAlgOpContext;
      d_vectorAttributes = u.d_vectorAttributes;
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_numVectors       = u.d_numVectors;
      d_mpiPatternP2P    = u.d_mpiPatternP2P;
    }

    //
    // Move Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(MultiVector &&u) noexcept
    {
      d_storage            = std::move(u.d_storage);
      d_linAlgOpContext    = std::move(u.d_linAlgOpContext);
      d_vectorAttributes   = std::move(u.d_vectorAttributes);
      d_localSize          = std::move(u.d_localSize);
      d_locallyOwnedSize   = std::move(u.d_locallyOwnedSize);
      d_ghostSize          = std::move(u.d_ghostSize);
      d_globalSize         = std::move(u.d_globalSize);
      d_numVectors         = std::move(u.d_numVectors);
      d_mpiCommunicatorP2P = std::move(u.d_mpiCommunicatorP2P);
      d_mpiPatternP2P      = std::move(u.d_mpiPatternP2P);
    }

    //
    // Copy Assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace> &
    MultiVector<ValueType, memorySpace>::operator=(const MultiVector &u)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage           = *(u.d_storage);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, u.d_numVectors);
      d_linAlgOpContext  = u.d_linAlgOpContext;
      d_vectorAttributes = u.d_vectorAttributes;
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_numVectors       = u.d_numVectors;
      d_mpiPatternP2P    = u.d_mpiPatternP2P;
      return *this;
    }

    //
    // Move Assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace> &
    MultiVector<ValueType, memorySpace>::operator=(MultiVector &&u)
    {
      d_storage            = std::move(u.d_storage);
      d_linAlgOpContext    = std::move(u.d_linAlgOpContext);
      d_vectorAttributes   = std::move(u.d_vectorAttributes);
      d_localSize          = std::move(u.d_localSize);
      d_locallyOwnedSize   = std::move(u.d_locallyOwnedSize);
      d_ghostSize          = std::move(u.d_ghostSize);
      d_globalSize         = std::move(u.d_globalSize);
      d_numVectors         = std::move(u.d_numVectors);
      d_mpiCommunicatorP2P = std::move(u.d_mpiCommunicatorP2P);
      d_mpiPatternP2P      = std::move(u.d_mpiPatternP2P);
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::setValue(const ValueType val)
    {
      d_storage->setValue(val);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    MultiVector<ValueType, memorySpace>::l2Norms() const
    {
      const std::vector<double> l2NormsLocallyOwned =
        blasLapack::nrms2MultiVector<ValueType, memorySpace>(
          this->locallyOwnedSize(),
          this->numVectors(),
          this->data(),
          *d_linAlgOpContext);

      std::vector<double> l2NormsLocallyOwnedSquare(d_numVectors, 0.0);
      for (size_type i = 0; i < d_numVectors; ++i)
        l2NormsLocallyOwnedSquare[i] =
          l2NormsLocallyOwned[i] * l2NormsLocallyOwned[i];

      std::vector<double> returnValues(d_numVectors, 0.0);
      utils::mpi::MPIAllreduce<memorySpace>(&l2NormsLocallyOwnedSquare,
                                            &returnValues[0],
                                            d_numVectors,
                                            utils::mpi::MPIDouble,
                                            utils::mpi::MPIMax,
                                            d_mpiPatternP2P->mpiCommunicator());
      for (size_type i = 0; i < d_numVectors; ++i)
        returnValues[i] = std::sqrt(returnValues[i]);
      return returnValues;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    MultiVector<ValueType, memorySpace>::lInfNorms() const
    {
      const std::vector<double> lInfNormsLocallyOwned =
        blasLapack::amaxsMultiVector<ValueType, memorySpace>(
          this->locallyOwnedSize(),
          this->numVectors(),
          this->data(),
          *d_linAlgOpContext);

      std::vector<double> returnValues(d_numVectors, 0.0);
      utils::mpi::MPIAllreduce<memorySpace>(&lInfNormsLocallyOwned,
                                            &returnValues[0],
                                            d_numVectors,
                                            utils::mpi::MPIDouble,
                                            utils::mpi::MPIMax,
                                            d_mpiPatternP2P->mpiCommunicator());
      return returnValues;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::updateGhostValues(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->updateGhostValues(*d_storage, communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::accumulateAddLocallyOwned(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwned(*d_storage,
                                                      communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::updateGhostValuesBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->updateGhostValuesBegin(*d_storage,
                                                   communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::updateGhostValuesEnd()
    {
      d_mpiCommunicatorP2P->updateGhostValuesEnd(*d_storage);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::accumulateAddLocallyOwnedBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwnedBegin(
        *d_storage, communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd()
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwnedEnd(*d_storage);
    }
  } // end of namespace linearAlgebra
} // namespace dftefe
