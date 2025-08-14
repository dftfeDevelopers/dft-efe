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
 * @author Ian C. Lin, Sambit Das.
 */

#include <utils/MemoryManager.h>
#include <utils/Exceptions.h>
#include <utils/MemoryTransfer.h>

namespace dftefe
{
  namespace utils
  {
    //
    // Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace>::MemoryStorage(
      const size_type size,
      const ValueType initVal)
      : d_size(size)
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::allocate(size,
                                                                     &d_data);
      dftefe::utils::MemoryManager<ValueType, memorySpace>::set(size,
                                                                d_data,
                                                                initVal);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::resize(const size_type size,
                                                  const ValueType initVal)
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::deallocate(d_data);
      d_size = size;
      if (size > 0)
        {
          dftefe::utils::MemoryManager<ValueType, memorySpace>::allocate(
            size, &d_data);
          dftefe::utils::MemoryManager<ValueType, memorySpace>::set(size,
                                                                    d_data,
                                                                    initVal);
        }
      else
        d_data = nullptr;
    }

    //
    // Destructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace>::~MemoryStorage()
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::deallocate(d_data);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace>::MemoryStorage(
      const MemoryStorage<ValueType, memorySpace> &u)
      : d_size(u.d_size)
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::allocate(d_size,
                                                                     &d_data);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(d_size,
                                                            d_data,
                                                            u.d_data);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::setValue(const ValueType val)
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::set(d_size,
                                                                d_data,
                                                                val);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::setZero(size_type       size,
                                                   const size_type offset)
    {
      dftefe::utils::MemoryManager<ValueType, memorySpace>::setZero(d_size,
                                                                    d_data +
                                                                      offset);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace>::MemoryStorage(
      MemoryStorage<ValueType, memorySpace> &&u) noexcept
      : d_size(u.d_size)
      , d_data(nullptr)
    {
      *this = std::move(u);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    MemoryStorage<ValueType, memorySpace>::size() const
    {
      return d_size;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::iterator
    MemoryStorage<ValueType, memorySpace>::begin()
    {
      return d_data;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::const_iterator
    MemoryStorage<ValueType, memorySpace>::begin() const
    {
      return d_data;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::iterator
    MemoryStorage<ValueType, memorySpace>::end()
    {
      return (d_data + d_size);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::const_iterator
    MemoryStorage<ValueType, memorySpace>::end() const
    {
      return (d_data + d_size);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace> &
    MemoryStorage<ValueType, memorySpace>::operator=(
      const MemoryStorage<ValueType, memorySpace> &rhs)
    {
      if (&rhs != this)
        {
          if (rhs.d_size != d_size)
            {
              this->resize(rhs.d_size);
            }
          utils::MemoryTransfer<memorySpace, memorySpace>::copy(rhs.d_size,
                                                                this->d_data,
                                                                rhs.d_data);
        }
      return (*this);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace> &
    MemoryStorage<ValueType, memorySpace>::operator=(
      MemoryStorage<ValueType, memorySpace> &&rhs) noexcept
    {
      if (&rhs != this)
        {
          delete[] d_data;
          d_data     = rhs.d_data;
          d_size     = rhs.d_size;
          rhs.d_size = 0;
          rhs.d_data = nullptr;
        }
      return (*this);
    }

    //    // This part does not work for GPU version, will work on this until
    //    // having cleaner solution.
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //    typename MemoryStorage<ValueType, memorySpace>::reference
    //    MemoryStorage<ValueType, memorySpace>::operator[](const size_type i)
    //    {
    //
    //      return d_data[i];
    //    }
    //
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //    typename MemoryStorage<ValueType, memorySpace>::const_reference
    //    MemoryStorage<ValueType, memorySpace>::operator[](const size_type i)
    //    const
    //    {
    //      return d_data[i];
    //    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType *
    MemoryStorage<ValueType, memorySpace>::data() noexcept
    {
      return d_data;
    }
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType *
    MemoryStorage<ValueType, memorySpace>::data() const noexcept
    {
      return d_data;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceDst>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      MemoryStorage<ValueType, memorySpaceDst> &dstMemoryStorage) const
    {
      DFTEFE_AssertWithMsg(
        d_size <= dstMemoryStorage.size(),
        "The allocated size of destination MemoryStorage is insufficient to "
        "copy from the the MemoryStorage.");
      MemoryTransfer<memorySpaceDst, memorySpace>::copy(
        d_size, dstMemoryStorage.begin(), this->begin());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceDst>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      MemoryStorage<ValueType, memorySpaceDst> &dstMemoryStorage,
      const size_type                           N,
      const size_type                           srcOffset,
      const size_type                           dstOffset) const
    {
      DFTEFE_AssertWithMsg(
        srcOffset + N <= d_size,
        "The offset and copy size specified for the source MemoryStorage"
        " is out of range for it.");

      DFTEFE_AssertWithMsg(
        dstOffset + N <= dstMemoryStorage.size(),
        "The offset and size specified for the destination MemoryStorage"
        " is out of range for it.");

      MemoryTransfer<memorySpaceDst, memorySpace>::copy(
        N, dstMemoryStorage.begin() + dstOffset, this->begin() + srcOffset);
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceSrc>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(
      const MemoryStorage<ValueType, memorySpaceSrc> &srcMemoryStorage)
    {
      DFTEFE_AssertWithMsg(
        srcMemoryStorage.size() <= d_size,
        "The allocated size of the MemoryStorage is insufficient to "
        " copy from the source MemoryStorage.");
      MemoryTransfer<memorySpace, memorySpaceSrc>::copy(
        srcMemoryStorage.size(), this->begin(), srcMemoryStorage.begin());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceSrc>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(
      MemoryStorage<ValueType, memorySpaceSrc> &srcMemoryStorage,
      const size_type                           N,
      const size_type                           srcOffset,
      const size_type                           dstOffset)
    {
      DFTEFE_AssertWithMsg(
        srcOffset + N <= srcMemoryStorage.size(),
        "The offset and copy size specified for the source MemoryStorage"
        " is out of range for it.");

      DFTEFE_AssertWithMsg(
        dstOffset + N <= d_size,
        "The offset and size specified for the destination MemoryStorage"
        " is out of range for it.");

      MemoryTransfer<memorySpace, memorySpaceSrc>::copy(
        N, this->begin() + dstOffset, srcMemoryStorage.begin() + srcOffset);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceDst>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(ValueType *dst) const
    {
      MemoryTransfer<memorySpaceDst, memorySpace>::copy(d_size,
                                                        dst,
                                                        this->begin());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceDst>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      ValueType *     dst,
      const size_type N,
      const size_type srcOffset,
      const size_type dstOffset) const
    {
      DFTEFE_AssertWithMsg(
        srcOffset + N <= d_size,
        "The offset and copy size specified for the source MemoryStorage"
        " is out of range for it.");
      MemoryTransfer<memorySpaceDst, memorySpace>::copy(
        N, dst + dstOffset, this->begin() + srcOffset);
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceSrc>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(const ValueType *src)
    {
      MemoryTransfer<memorySpace, memorySpaceSrc>::copy(d_size,
                                                        this->begin(),
                                                        src);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceSrc>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(const ValueType *src,
                                                    const size_type  N,
                                                    const size_type  srcOffset,
                                                    const size_type  dstOffset)
    {
      DFTEFE_AssertWithMsg(
        dstOffset + N <= d_size,
        "The offset and size specified for the destination MemoryStorage"
        " is out of range for it.");

      MemoryTransfer<memorySpace, memorySpaceSrc>::copy(
        N, this->begin() + dstOffset, src + srcOffset);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      std::vector<ValueType> &dst) const
    {
      if (dst.size() < d_size)
        dst.resize(d_size);

      MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        d_size, dst.data(), this->begin());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      std::vector<ValueType> &dst,
      const size_type         N,
      const size_type         srcOffset,
      const size_type         dstOffset) const
    {
      DFTEFE_AssertWithMsg(
        srcOffset + N <= d_size,
        "The offset and copy size specified for the source MemoryStorage"
        " is out of range for it.");
      if (dst.size() < N + dstOffset)
        dst.resize(N + dstOffset);

      MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        N, dst.data() + dstOffset, this->begin() + srcOffset);
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(
      const std::vector<ValueType> &src)
    {
      DFTEFE_AssertWithMsg(
        src.size() <= d_size,
        "The allocated size of the MemoryStorage is insufficient to copy from "
        "the source STL vector");
      MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(src.size(),
                                                                  this->begin(),
                                                                  src.data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(
      const std::vector<ValueType> &src,
      const size_type               N,
      const size_type               srcOffset,
      const size_type               dstOffset)
    {
      DFTEFE_AssertWithMsg(
        dstOffset + N <= d_size,
        "The offset and size specified for the destination MemoryStorage"
        " is out of range for it.");

      DFTEFE_AssertWithMsg(
        srcOffset + N <= src.size(),
        "The offset and size specified for the source STL vector "
        " is out of range for it.");

      MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        N, this->begin() + dstOffset, src.data() + srcOffset);
    }

    template <typename ValueType, utils::MemorySpace memorySpaceDst>
    MemoryStorage<ValueType, memorySpaceDst>
    memoryStorageFromSTL(const std::vector<ValueType> &src)
    {
      MemoryStorage<ValueType, memorySpaceDst> returnValue(src.size());
      MemoryTransfer<memorySpaceDst, utils::MemorySpace::HOST>::copy(
        src.size(), returnValue.begin(), src.data());
      return returnValue;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    swap(MemoryStorage<ValueType, memorySpace> &X,
         MemoryStorage<ValueType, memorySpace> &Y)
    {
      MemoryStorage<ValueType, memorySpace> tmp(std::move(X));
      X = std::move(Y);
      Y = std::move(tmp);
    }

  } // namespace utils
} // namespace dftefe
