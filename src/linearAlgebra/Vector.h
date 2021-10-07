#ifndef dftefeVector_h
#define dftefeVector_h

#include <MemoryManager.h>
#include <typeConfig.h>
#include <vector>
namespace dftefe
{
  template <typename NumberType, MemorySpace memorySpace>
  class Vector
  {
  public:
    Vector(const size_type size, const NumberType initVal = 0);

    ~Vector();

    Vector() = default;

    Vector(const Vector &vector);

    // Will overwrite old data
    void
    resize(const size_type size, const NumberType initVal = 0);

    size_type
    size() const;

  private:
    NumberType *d_data = nullptr;
    size_type   d_size = 0;
  };
} // end of namespace dftefe



#endif
