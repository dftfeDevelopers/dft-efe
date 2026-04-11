#ifndef linearSolverCGDeviceKernels_H
#define linearSolverCGDeviceKernels_H
#include <utils/DeviceAPICalls.h>
#include <utils/DeviceDataTypeOverloads.h>
#include <utils/DeviceKernelLauncherHelpers.h>

namespace dftefe
{
  namespace electrostatics
  {  
  /**
   * @brief Combines precondition and dot product
   *
   */
  void
  applyPreconditionAndComputeDotProductDevice(double          *d_dvec,
                                              double          *d_devSum,
                                              const double    *d_rvec,
                                              const double    *d_jacobi,
                                              const size_type N);

  /**
   * @brief Combines precondition, sadd and dot product
   *
   */
  void
  applyPreconditionComputeDotProductAndSaddDevice(double          *d_qvec,
                                                  double          *d_devSum,
                                                  const double    *d_rvec,
                                                  const double    *d_jacobi,
                                                  const size_type N);

  /**
   * @brief Combines scaling and norm
   *
   */
  void
  scaleXRandComputeNormDevice(double          *x,
                              double          *d_rvec,
                              double          *d_devSum,
                              const double    *d_qvec,
                              const double    *d_dvec,
                              const double     alpha,
                              const size_type N);

  void
  saddDevice(double *y, double *x, const double beta, const size_type size);
  }
} // namespace dftefe
#endif