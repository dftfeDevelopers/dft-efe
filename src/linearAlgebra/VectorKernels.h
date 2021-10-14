#ifndef dftefeVectorKernels_h
#define dftefeVectorKernels_h

#include "MemoryManager.h"
#include "Vector.h"

namespace dftefe
{
  namespace utils
  {
    /**
     * @tparam NumberType
     * @tparam memorySpace
     * @param a
     * @param V1
     * @param V2
     */
    template <typename NumberType, MemorySpace memorySpace>
    void
    addKernel(const NumberType                       a,
              const Vector<NumberType, memorySpace> &V1,
              Vector<NumberType, memorySpace>       &V2);

  } // namespace utils
} // namespace dftefe



#endif // dftefeVectorKernels_h
