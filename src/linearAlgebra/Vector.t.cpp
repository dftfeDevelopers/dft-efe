#include <complex>
#include "MemoryManager.h"
#include "Vector.h"
#include "VectorKernels.h"

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
      , d_data(
          dftefe::utils::MemoryManager<NumberType, memorySpace>::allocate(size))
    {}
    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    void
    Vector<NumberType, memorySpace>::resize(const size_type  size,
                                            const NumberType initVal)
    {
      dftefe::utils::MemoryManager<NumberType, memorySpace>::deallocate(d_data);
      d_size = size;
      if (size > 0)
        d_data =
          dftefe::utils::MemoryManager<NumberType, memorySpace>::allocate(size);
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
    {}

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
      const Vector<NumberType, memorySpace> &u)
    {
      return (*this);
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<NumberType, memorySpace>::reference
    Vector<NumberType, memorySpace>::operator[](size_type i)
    {
      return d_data[i];
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    typename Vector<NumberType, memorySpace>::const_reference
    Vector<NumberType, memorySpace>::operator[](size_type i) const
    {
      return d_data[i];
    }

    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    Vector<NumberType, memorySpace> &
    operator+(Vector<NumberType, memorySpace> &      v,
              const Vector<NumberType, memorySpace> &u)
    {
      VectorKernels<NumberType, memorySpace>::add(u.size(), u, v);
    }

  } // namespace linearAlgebra
} // namespace dftefe
