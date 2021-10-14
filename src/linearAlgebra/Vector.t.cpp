#include <complex>
#include "MemoryManager.h"
#include "Vector.h"


namespace dftefe
{
  namespace utils
  {
    //
    // Constructor
    //
    template <typename NumberType, MemorySpace memorySpace>
    Vector<NumberType, memorySpace>::Vector(const size_type  size,
                                            const NumberType initVal)
      : d_size(size)
      , d_data(MemoryManager<NumberType, memorySpace>::allocate(size))
    {}
    template <typename NumberType, MemorySpace memorySpace>
    void
    Vector<NumberType, memorySpace>::resize(const size_type  size,
                                            const NumberType initVal)
    {
      MemoryManager<NumberType, memorySpace>::deallocate(d_data);
      d_size = size;
      if (size > 0)
        d_data = MemoryManager<NumberType, memorySpace>::allocate(size);
    }

    //
    // Destructor
    //
    template <typename NumberType, MemorySpace memorySpace>
    Vector<NumberType, memorySpace>::~Vector()
    {
      MemoryManager<NumberType, memorySpace>::deallocate(d_data);
    }

    template <typename NumberType, MemorySpace memorySpace>
    Vector<NumberType, memorySpace>::Vector(const Vector &v)
    {}

    template <typename NumberType, MemorySpace memorySpace>
    size_type
    Vector<NumberType, memorySpace>::size() const
    {
      return d_size;
    }
    template <typename NumberType, MemorySpace memorySpace>
    void
    Vector<NumberType, memorySpace>::add(
      const NumberType                       a,
      const Vector<NumberType, memorySpace> &V)
    {}
    template <typename NumberType, MemorySpace memorySpace>
    NumberType
    Vector<NumberType, memorySpace>::operator[](size_type i) const
    {
      return d_data[i];
    }
    template <typename NumberType, MemorySpace memorySpace>
    NumberType &
    Vector<NumberType, memorySpace>::operator[](size_type i)
    {
      return d_data[i];
    }


  } // namespace utils
} // namespace dftefe
