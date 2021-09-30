#ifndef dftefeCUDAUtils_h
#define dftefeCUDAUtils_h


#ifdef DFTEFE_WITH_CUDA

namespace dftefe
{
  class CUDAUtils
  {
  public:
    static void
    initialize(const int world_rank);
  };

} // namespace dftefe

#endif


#endif
