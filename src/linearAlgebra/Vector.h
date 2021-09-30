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

   // FIXME initVal not used
    Vector(const size_type size, const NumberType initVal = 0);
    ~Vector();

    Vector() = default;

    // FIXME: does it even make sense to have a copy constructor for gpu vector
    Vector(const Vector & vector);

// Will overwrite old data
    void resize(const size_type size, const NumberType initVal = 0);

    size_type size() const;


    //
    // temporary function to test dgemv
    //
    void
    testDgemv();


  private:
    // FIXME does nullptr exist for cuda 
    NumberType *d_data = nullptr;
    size_type d_size = 0; 
  };
} // end of namespace dftefe



#endif
