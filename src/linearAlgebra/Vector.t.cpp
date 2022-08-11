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
#include <utils/MPIWrapper.h>
#include <linearAlgebra/BlasLapack.h>
#include <cmath>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief Constructor for a \b serial Vector using size and init value
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      const size_type               size,
      LinAlgOpContext<memorySpace> *linAlgOpContext,
      const ValueType               initVal)
    {
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          size, initVal);
      d_linAlgOpContext = linAlgOpContext;
      d_vectorAttributes =
        VectorAttributes(VectorAttributes::Distribution::SERIAL);
      d_globalSize       = size;
      d_locallyOwnedSize = size;
      d_ghostSize        = 0;
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(size);
      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                blockSize);
    }

    /**
     * @brief Constructor for a \b serial Vector using user provided Vector::Storage (i.e.,
     * utils::MemoryStorage)
     *
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      std::unique_ptr<typename Vector<ValueType, memorySpace>::Storage> storage,
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
      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          d_globalSize);
      // block size set to 1 as it is a single vector
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P      = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                blockSize);
    }

    /**
     * @brief Constructor for a \b distributed Vector using an existing MPIPatternP2P object
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
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

    /**
     * @brief Constructorfor a \b distributed Vector using user provided Vector::Storage (i.e.,
     * utils::MemoryStorage) and an existing MPIPatternP2P
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
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
     * @brief Constructor for a \b distributed Vector
     * using locally owned range and ghost indices
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const std::vector<dftefe::global_size_type> &       ghostIndices,
      const utils::mpi::MPIComm &                         mpiComm,
      LinAlgOpContext<memorySpace> *                      linAlgOpContext,
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
            "WARNING: Constructing a Vector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the constructor based on an input MPIPatternP2P.";
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
     * @brief Constructor for a special case of \b distributed Vector where none
     * none of the processors have any ghost indices.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far
     * as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
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
      int mpiRank;
      int err = utils::mpi::MPICommRank(mpiComm, &mpiRank);
      std::pair<bool, std::string> errIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(errIsSuccessAndMsg.first,
                            errIsSuccessAndMsg.second);
      std::string msg;
      if (mpiRank == 0)
        {
          msg = "WARNING: Constructing a Vector using only locally owned "
                "range is expensive. As far as possible, one should "
                " use the constructor based on an input MPIPatternP2P.";
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
     * @brief Constructor for a \b distributed Vector that takes the total number indices across all processors
     * and divides them up equitably (as far as possible) across all processors.
     * The resulting Vector will not contain any ghost indices on any of the
     * processors. This decompositon is not compatible with other constructors.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P or another vector as far
     * as possible.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(
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
            "WARNING: Constructing a Vector using total number of indices across all processors "
            "is expensive. As far as possible, one should "
            " use the constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }

      int mpiProcess;

      int errProc        = utils::mpi::MPICommSize(mpiComm, &mpiProcess);
      errIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(errProc);
      utils::throwException(errIsSuccessAndMsg.first,
                            errIsSuccessAndMsg.second);

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
    }


    //
    // Copy Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(const Vector &u)
    {
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage           = *(u.d_storage);
      d_mpiPatternP2P      = u.d_mpiPatternP2P;
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, (u.d_mpiCommunicatorP2P).getBlockSize());
      d_linAlgOpContext  = u.d_linAlgOpContext;
      d_vectorAttributes = u.d_vectorAttributes;
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(const Vector &u, ValueType initVal)
    {
      d_storage =
        std::make_unique<typename Vector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size(), initVal);
      d_mpiPatternP2P      = u.d_mpiPatternP2P;
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, (u.d_mpiCommunicatorP2P).getBlockSize());
      d_linAlgOpContext  = u.d_linAlgOpContext;
      d_vectorAttributes = u.d_vectorAttributes;
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
    }

    //
    // Move Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace>::Vector(Vector &&u) noexcept
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
    }

    //
    // Copy Assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator=(const Vector &u)
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
      return *this;
    }

    //
    // Move Assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    Vector<ValueType, memorySpace> &
    Vector<ValueType, memorySpace>::operator=(Vector &&u)
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
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::iterator
    Vector<ValueType, memorySpace>::begin()
    {
      return d_storage->begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::const_iterator
    Vector<ValueType, memorySpace>::begin() const
    {
      return d_storage->begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::iterator
    Vector<ValueType, memorySpace>::end()
    {
      return d_storage->end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<ValueType, memorySpace>::const_iterator
    Vector<ValueType, memorySpace>::end() const
    {
      return d_storage->end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType *
    Vector<ValueType, memorySpace>::data()
    {
      return d_storage->data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType *
    Vector<ValueType, memorySpace>::data() const
    {
      return d_storage->data();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::setValue(const ValueType val)
    {
      d_storage->setValue(val);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    Vector<ValueType, memorySpace>::l2Norm() const
    {
      const double l2NormLocallyOwned =
        blasLapack::nrm2<ValueType, memorySpace>(d_locallyOwnedSize,
                                                 this->data(),
                                                 1,
                                                 *d_linAlgOpContext);
      const double l2NormLocallyOwnedSquare =
        l2NormLocallyOwned * l2NormLocallyOwned;
      double returnValue = 0.0;
      utils::mpi::MPIAllreduce<memorySpace>(&l2NormLocallyOwnedSquare,
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
    Vector<ValueType, memorySpace>::lInfNorm() const
    {
      const double lInfNormLocallyOwned =
        blasLapack::amax<ValueType, memorySpace>(d_locallyOwnedSize,
                                                 this->data(),
                                                 1,
                                                 *d_linAlgOpContext);
      double returnValue = lInfNormLocallyOwned;
      utils::mpi::MPIAllreduce<memorySpace>(&lInfNormLocallyOwned,
                                            &returnValue,
                                            1,
                                            utils::mpi::MPIDouble,
                                            utils::mpi::MPIMax,
                                            d_mpiPatternP2P->mpiCommunicator());
      return returnValue;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::updateGhostValues(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->updateGhostValues(*d_storage, communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::accumulateAddLocallyOwned(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwned(*d_storage,
                                                      communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::updateGhostValuesBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->updateGhostValuesBegin(*d_storage,
                                                   communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::updateGhostValuesEnd()
    {
      d_mpiCommunicatorP2P->updateGhostValuesEnd(*d_storage);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::accumulateAddLocallyOwnedBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwnedBegin(
        *d_storage, communicationChannel);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd()
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwnedEnd(*d_storage);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    bool
    Vector<ValueType, memorySpace>::isCompatible(
      const Vector<ValueType, memorySpace> &rhs) const
    {
      if (d_vectorAttributes.areDistributionCompatible(
            rhs.d_vectorAttributes) == false)
        return false;
      else if (d_globalSize != rhs.d_globalSize)
        return false;
      else if (d_localSize != rhs.d_localSize)
        return false;
      else if (d_locallyOwnedSize != rhs.d_locallyOwnedSize)
        return false;
      else if (d_ghostSize != rhs.d_ghostSize)
        return false;
      else
        return (d_mpiPatternP2P->isCompatible(*(rhs.d_mpiPatternP2P)));
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
    Vector<ValueType, memorySpace>::getMPIPatternP2P() const
    {
      return d_mpiPatternP2P;
    }

    //
    // Helper functions
    //


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                             a,
        const Vector<ValueType, memorySpace> &u,
        ValueType                             b,
        const Vector<ValueType, memorySpace> &v,
        Vector<ValueType, memorySpace> &      w)
    {
      const VectorAttributes &uVectorAttributes = u.getVectorAttributes();
      const VectorAttributes &vVectorAttributes = v.getVectorAttributes();
      const VectorAttributes &wVectorAttributes = w.getVectorAttributes();
      bool                    areCompatible =
        uVectorAttributes.areDistributionCompatible(vVectorAttributes);
      utils::throwException(
        areCompatible,
        "Trying to add incompatible Vectors. One is a serial Vector and the other a distributed Vector.");
      areCompatible =
        vVectorAttributes.areDistributionCompatible(wVectorAttributes);
      utils::throwException(
        areCompatible,
        "Trying to add incompatible Vectors. One is a serial Vector and the other a distributed Vector.");
      utils::throwException<utils::LengthError>(
        (u.size() == v.size()) && (v.size() == w.size()) &&
          (u.localSize() == v.localSize()) && (v.localSize() == w.localSize()),
        "Mismatch of sizes of the Vectors that are added.");
      const size_type uStorageSize = (u.getValues()).size();
      const size_type vStorageSize = (v.getValues()).size();
      const size_type wStorageSize = (w.getValues()).size();
      utils::throwException<utils::LengthError>(
        (uStorageSize == vStorageSize) && (vStorageSize == wStorageSize),
        "Mismatch of sizes of the underlying storages"
        "of the Vectors that are added.");

      blasLapack::axpby<ValueType, memorySpace>(
        u.localSize(),
        a,
        u.data(),
        b,
        v.data(),
        w.data(),
        (w.getLinAlgOpContext())->getBlasQueue());
    }

  } // end of namespace linearAlgebra
} // namespace dftefe
