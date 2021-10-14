#include <complex>
#include "MemoryManager.h"

namespace dftefe
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
    else
      d_data = nullptr;
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

} // namespace dftefe
