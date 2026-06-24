#ifndef dftefe_SmoothCutoffFunctions_h
#define dftefe_SmoothCutoffFunctions_h

#include <vector>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/DeviceTypeConfig.h>
#include <utils/DeviceKernelLauncherHelpers.h>
#include <cmath>

namespace dftefe
{
  namespace atoms
  {
    ///////////////////////////////////////////////////////////////////////////
    ///////////// START OF SMOOTH CUTOFF FUNCTION RELATED FUNCTIONS ///////////
    ///////////////////////////////////////////////////////////////////////////

    // scalar single-point evaluation — inline so every TU (CPU or GPU) that
    // includes this header gets its own inline copy callable from host and
    // device.
    DFTEFE_HOST_DEVICE_FUNC double
    smoothCutoffValue(const double x, const double r, const double d)
    {
      const double y      = 1.0 - d * (x - r) / r;
      const double f1_y   = (y <= 0.0) ? 0.0 : exp(-1.0 / y);
      const double omy    = 1.0 - y;
      const double f1_1my = (omy <= 0.0) ? 0.0 : exp(-1.0 / omy);
      return f1_y / (f1_y + f1_1my);
    }

    DFTEFE_HOST_DEVICE_FUNC double
    smoothCutoffDerivative(const double x,
                           const double r,
                           const double d,
                           const double tolerance)
    {
      const double y = 1.0 - d * (x - r) / r;
      if (fabs(y) < tolerance || fabs(1.0 - y) < tolerance)
        return 0.0;
      const double f1_y      = (y <= 0.0) ? 0.0 : exp(-1.0 / y);
      const double omy       = 1.0 - y;
      const double f1_1my    = (omy <= 0.0) ? 0.0 : exp(-1.0 / omy);
      const double f1Der_y   = f1_y / (y * y);
      const double f1Der_1my = f1_1my / (omy * omy);
      const double denom     = f1_y + f1_1my;
      const double f2Der =
        (f1Der_y * f1_1my + f1_y * f1Der_1my) / (denom * denom);
      return f2Der * (-d / r);
    }

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
    smoothCutoffDerivative(
      size_type             numPoints,
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

#endif // dftefe_SmoothCutoffFunctions_h
