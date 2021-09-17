#ifndef dftefeVector_h
#define dftefeVector_h

// FIXME fix relative path
#include "./../MemorySpace.h"

namespace dftefe
{
  template <typename NumberType, typename MemorySpace>
  class Vector
  {
  public:
    using size_type = unsigned int;

    Vector() = default;

    Vector(const size_type size, const NumberType initVal = 0);

  private:
    NumberType *d_data;
  };

} // end of namespace dftefe

class Vector
{
public:
  using size_type = unsigned int;
};

class ParallelVector : public Vector
{
public:
  using size_type = unsigned long;
};

#endif
