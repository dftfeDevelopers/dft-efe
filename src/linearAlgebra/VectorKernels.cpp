#include "VectorKernels.h"

namespace dftefe
{
  namespace utils
  {
    template <typename NumberType>
    void
    addKernel(const NumberType                             a,
              const Vector<NumberType, MemorySpace::HOST> &V1,
              Vector<NumberType, MemorySpace::HOST>       &V2)
    {
      // todo: add assert condition: V1.size() == V2.size();
      for (size_type i = 0; i < V1.size(); ++i)
        {
          V2[i] += a * V1[i];
        }
    }
  } // namespace utils
} // namespace dftefe
