#ifndef dftefeVector_h
#define dftefeVector_h

#include <MemoryManager.h>
#include <vector>
namespace dftefe
{
  template <typename NumberType, MemorySpace memorySpace>
  class Vector
  {
  public:
    using size_type = unsigned int;

    Vector() = default;

    Vector(const size_type size, const NumberType initVal = 0);

    void
    testDgemv();

  private:
    NumberType *d_data;
  };
} // end of namespace dftefe



#endif
