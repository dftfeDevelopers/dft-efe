#ifndef linearSolverCGDeviceKernels_H
#define linearSolverCGDeviceKernels_H
#include <utils/DeviceAPICalls.h>
#include <utils/DeviceDataTypeOverloads.h>
#include <utils/DeviceKernelLauncherHelpers.h>
#include <linearAlgebra/LinAlgOpContext.h>

namespace dftefe
{
  namespace electrostatics
  {
    /**
     * @brief Combines precondition and dot product
     *
     */
    void
    applyPreconditionAndComputeDotProductDevice(
      double *        d_dvec,
      double *        d_devSum,
      const double *  d_rvec,
      const double *  d_jacobi,
      const size_type N,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
        &linAlgOpContext);

    /**
     * @brief Combines precondition, sadd and dot product
     *
     */
    void
    applyPreconditionComputeDotProductAndSaddDevice(
      double *        d_qvec,
      double *        d_devSum,
      const double *  d_rvec,
      const double *  d_jacobi,
      const size_type N,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
        &linAlgOpContext);

    /**
     * @brief Combines scaling and norm
     *
     */
    void
    scaleXRandComputeNormDevice(
      double *        x,
      double *        d_rvec,
      double *        d_devSum,
      const double *  d_qvec,
      const double *  d_dvec,
      const double    alpha,
      const size_type N,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
        &linAlgOpContext);

    void
    saddDevice(double *        y,
               double *        x,
               const double    beta,
               const size_type size,
               linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
                 &linAlgOpContext);
  } // namespace electrostatics
} // namespace dftefe
#endif
