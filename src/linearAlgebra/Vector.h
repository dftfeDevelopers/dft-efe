#ifndef dftefeVector_h
#define dftefeVector_h

#include "MemoryManager.h"
#include "TypeConfig.h"
#include <vector>
namespace dftefe
{
  namespace utils
  {
    template <typename NumberType, MemorySpace memorySpace>
    class Vector
    {
    public:
      Vector(const size_type size, const NumberType initVal = 0);

      ~Vector();

      Vector() = default;

      Vector(const Vector &vector);

      void
      add(const NumberType a, const Vector<NumberType, memorySpace> &V);

      NumberType
      operator[](size_type i) const;

      NumberType &
      operator[](size_type i);

      // Will overwrite old data
      void
      resize(const size_type size, const NumberType initVal = 0);

      size_type
      size() const;

    private:
      NumberType *d_data = nullptr;
      size_type   d_size = 0;
    };
  } // namespace utils
} // end of namespace dftefe

#include "Vector.t.cpp"


#endif
