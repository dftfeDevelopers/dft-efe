#ifdef DFTEFE_WITH_DEVICE

#  ifndef dftefeDeviceComplexUtils_cuh
#    define dftefeDeviceComplexUtils_cuh

#    include <complex>
#    include <cuComplex.h>
#    include "TypeConfig.h"


namespace dftefe
{
  namespace utils
  {
    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyComplexArrToRealArrsGPU(const size_type          size,
                                const NumberTypeComplex *complexArr,
                                NumberTypeReal *         realArr,
                                NumberTypeReal *         imagArr);

    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyRealArrsToComplexArrGPU(const size_type       size,
                                const NumberTypeReal *realArr,
                                const NumberTypeReal *imagArr,
                                NumberTypeComplex *   complexArr);

  } // end of namespace utils
} // end of namespace dftefe



#  endif // dftefeDeviceComplexUtils_cuh

#endif // DFTEFE_WITH_DEVICE
