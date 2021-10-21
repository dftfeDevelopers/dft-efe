#include <complex>
#include "MemoryManager.h"
#include "Vector.h"
#include "VectorKernels.h"
#include "Exceptions.h"
#include "MemoryTransfer.h"
#include <iostream>

namespace dftefe
{
  namespace linearAlgebra
  {
    //
    // Constructor
    //
    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    Vector<NumberType, memorySpace>::Vector(const size_type  size,
                                            const NumberType initVal)
      : d_size(size)
    {
      dftefe::utils::MemoryManager<NumberType, memorySpace>::allocate(size,
                                                                      &d_data);
      dftefe::utils::MemoryManager<NumberType, memorySpace>::set(size,
                                                                 d_data,
                                                                 initVal);
    }

    // todo
    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<NumberType, memorySpace>::resize(const size_type  size,
                                            const NumberType initVal)
    {
      dftefe::utils::MemoryManager<NumberType, memorySpace>::deallocate(d_data);
      d_size = size;
      if (size > 0)
        {
          d_data =
            dftefe::utils::MemoryManager<NumberType, memorySpace>::allocate(
              size, nullptr);
          dftefe::utils::MemoryManager<NumberType, memorySpace>::set(size,
                                                                     d_data,
                                                                     initVal);
        }
    }

    //
    // Destructor
    //
    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    Vector<NumberType, memorySpace>::~Vector()
    {
      dftefe::utils::MemoryManager<NumberType, memorySpace>::deallocate(d_data);
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    Vector<NumberType, memorySpace>::Vector(
      const Vector<NumberType, memorySpace> &u)
      : d_size(u.d_size)
    {
      utils::MemoryTransfer<NumberType, memorySpace, memorySpace>::copy(
        d_size, this->d_data, u.d_data);
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Vector<NumberType, memorySpace>::size() const
    {
      return d_size;
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<NumberType, memorySpace>::iterator
    Vector<NumberType, memorySpace>::begin()
    {
      return d_data;
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<NumberType, memorySpace>::const_iterator
    Vector<NumberType, memorySpace>::begin() const
    {
      return d_data;
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<NumberType, memorySpace>::iterator
    Vector<NumberType, memorySpace>::end()
    {
      return (d_data + d_size);
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<NumberType, memorySpace>::const_iterator
    Vector<NumberType, memorySpace>::end() const
    {
      return (d_data + d_size);
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    Vector<NumberType, memorySpace> &
    Vector<NumberType, memorySpace>::operator=(
      const Vector<NumberType, memorySpace> &rhs)
    {
      if (&rhs != this)
        {
          if (rhs.d_size != d_size)
            {
              this->resize(rhs.d_size);
            }
          utils::MemoryTransfer<NumberType, memorySpace, memorySpace>::copy(
            rhs.d_size, this->d_data, rhs.d_data);
        }
      return (*this);
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<NumberType, memorySpace>::reference
    Vector<NumberType, memorySpace>::operator[](const size_type i)
    {
      return d_data[i];
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<NumberType, memorySpace>::const_reference
    Vector<NumberType, memorySpace>::operator[](const size_type i) const
    {
      return d_data[i];
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    Vector<NumberType, memorySpace> &
    Vector<NumberType, memorySpace>::operator+=(const Vector &rhs)
    {
      // todo add assertion to check size of the two vectors
      VectorKernels<NumberType, memorySpace>::add(d_size, rhs.d_data, d_data);
      return *this;
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    NumberType *
    Vector<NumberType, memorySpace>::data() noexcept
    {
      return d_data;
    }
    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    const NumberType *
    Vector<NumberType, memorySpace>::data() const noexcept
    {
      return d_data;
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    void
    add(NumberType                             a,
        const Vector<NumberType, memorySpace> &u,
        NumberType                             b,
        const Vector<NumberType, memorySpace> &v,
        Vector<NumberType, memorySpace>       &w)
    {
      // todo add assertion to check sizes of the three vectors are consistent
      VectorKernels<NumberType, memorySpace>::add(
        u.size(), a, u.data(), b, v.data(), w.data());
    }

  } // namespace linearAlgebra
} // namespace dftefe
