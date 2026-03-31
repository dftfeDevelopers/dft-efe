#ifndef dftefe_SmoothCutoffFunctions_h
#define dftefe_SmoothCutoffFunctions_h

#include <vector>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/DeviceTypeConfig.h>
#ifdef DFTEFE_WITH_DEVICE
  #  include <utils/DeviceKernelLauncherHelpers.h>
#endif

namespace dftefe
{
  namespace atoms
  {
    ///////////////////////////////////////////////////////////////////////////
    ///////////// START OF SMOOTH CUTOFF FUNCTION RELATED FUNCTIONS ///////////
    ///////////////////////////////////////////////////////////////////////////
    double
    f1(const double x);

    double
    f1Der(const double x);

    double
    f2(const double x);

    double
    f2Der(const double x, const double tolerance);

    double
    Y(const double x, const double r, const double d);

    double
    YDer(const double x, const double r, const double d);

    double
    smoothCutoffValue(const double x, const double r, const double d);

    double
    smoothCutoffDerivative(const double x,
                           const double r,
                           const double d,
                           const double tolerance);

    template <dftefe::utils::MemorySpace memorySpace>
    void
    smoothCutoffValue(size_type             numPoints,
                      const double *        x,
                      const double          r,
                      const double          d,
                      double *              out,
                      utils::deviceStream_t streamId = utils::defaultStream);

    template <dftefe::utils::MemorySpace memorySpace>
    void
    smoothCutoffDerivative(size_type             numPoints,
                           const double *        x,
                           const double          r,
                           const double          d,
                           const double          tolerance,
                           double *              out,
                           utils::deviceStream_t streamId = utils::defaultStream);

    ///////////////////////////////////////////////////////////////////////////
    ///////////// END OF SMOOTH CUTOFF FUNCTION RELATED FUNCTIONS ///////////
    ///////////////////////////////////////////////////////////////////////////
  } // namespace atoms
} // namespace dftefe

#ifdef DFTEFE_WITH_DEVICE
#include <atoms/SmoothCutoffFunctionsDeviceKernels.h>
#endif

#endif // dftefe_SmoothCutoffFunctions_h
