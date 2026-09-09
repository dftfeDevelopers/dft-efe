#ifdef DFTEFE_WITH_DEVICE
#  include "PoissonSolverDealiiMatrixFreeFEDeviceKernels.h"
#  include <utils/DeviceUtils.h>
#  include <utils/DeviceTypeConfig.h>
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceAPICalls.h>
#  include <utils/DeviceDataTypeOverloads.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  include <linearAlgebra/BlasLapackKernels.h>
#  include <linearAlgebra/BlasLapack.h>
namespace dftefe
{
  namespace electrostatics
  {
    namespace
    {
      template <typename Type, size_type blockSize>
      DFTEFE_CREATE_KERNEL_SMEM_S(
        Type,
        blockSize,
        void,
        applyPreconditionAndComputeDotProductKernel,
        DFTEFE_KERNEL_ARGUMENT({
          Type      localSum;
          size_type idx = threadId + blockId * (blockSize * 2);

          if (idx < N)
            {
              Type jacobi = d_jacobi[idx];
              Type r      = d_rvec[idx];

              localSum    = jacobi * r * r;
              d_dvec[idx] = jacobi * r;
            }
          else
            localSum = 0;

          if (idx + blockSize < N)
            {
              Type jacobi = d_jacobi[idx + blockSize];
              Type r      = d_rvec[idx + blockSize];
              localSum += jacobi * r * r;
              d_dvec[idx + blockSize] = jacobi * r;
            }

          smem[threadId] = localSum;
          SYNCTHREADS;

          _Pragma("unroll") for (size_type size =
                                   dftefe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
                                 size >= 4 * dftefe::utils::DEVICE_WARP_SIZE;
                                 size /= 2)
          {
            if ((blockSize >= size) && (threadId < size / 2))
              smem[threadId] = localSum = localSum + smem[threadId + size / 2];

#  if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
    defined(DFTEFE_WITH_DEVICE_LANG_HIP)
            __syncthreads();
#  elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
            sycl::group_barrier(ind.get_group());
#  endif
          }

          if (threadId < dftefe::utils::DEVICE_WARP_SIZE)
            {
              if (blockSize >= 2 * dftefe::utils::DEVICE_WARP_SIZE)
                localSum += smem[threadId + dftefe::utils::DEVICE_WARP_SIZE];

              _Pragma("unroll") for (size_type offset =
                                       dftefe::utils::DEVICE_WARP_SIZE / 2;
                                     offset > 0;
                                     offset /= 2)
              {
#  ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
                unsigned mask = 0xffffffff;
                localSum += __shfl_down_sync(mask, localSum, offset);
#  elif DFTEFE_WITH_DEVICE_LANG_HIP
                localSum += __shfl_down(localSum,
                                        offset,
                                        dftefe::utils::DEVICE_WARP_SIZE);
#  elif DFTEFE_WITH_DEVICE_LANG_SYCL
                localSum +=
                  sycl::shift_group_left(ind.get_sub_group(), localSum, offset);
#  endif
              }
            }

          if (threadId == 0)
            dftefe::utils::atomicAddWrapper(&d_devSum[0], localSum);
        }),
        Type *          d_dvec,
        Type *          d_devSum,
        const Type *    d_rvec,
        const Type *    d_jacobi,
        const size_type N);


      template <typename Type, size_type blockSize>
      DFTEFE_CREATE_KERNEL_SMEM_S(
        Type,
        blockSize,
        void,
        applyPreconditionComputeDotProductAndSaddKernel,
        DFTEFE_KERNEL_ARGUMENT({
          size_type idx = threadId + blockId * (blockSize * 2);

          Type localSum;

          if (idx < N)
            {
              Type jacobi = d_jacobi[idx];
              Type r      = d_rvec[idx];

              localSum    = jacobi * r * r;
              d_qvec[idx] = -1 * jacobi * r;
            }
          else
            localSum = 0;

          if (idx + blockSize < N)
            {
              Type jacobi = d_jacobi[idx + blockSize];
              Type r      = d_rvec[idx + blockSize];
              localSum += jacobi * r * r;
              d_qvec[idx + blockSize] = -1 * jacobi * r;
            }

          smem[threadId] = localSum;
          SYNCTHREADS;

          _Pragma("unroll") for (size_type size =
                                   dftefe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
                                 size >= 4 * dftefe::utils::DEVICE_WARP_SIZE;
                                 size /= 2)
          {
            if ((blockSize >= size) && (threadId < size / 2))
              smem[threadId] = localSum = localSum + smem[threadId + size / 2];
#  if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
    defined(DFTEFE_WITH_DEVICE_LANG_HIP)
            __syncthreads();
#  elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
            sycl::group_barrier(ind.get_group());
#  endif
          }

          if (threadId < dftefe::utils::DEVICE_WARP_SIZE)
            {
              if (blockSize >= 2 * dftefe::utils::DEVICE_WARP_SIZE)
                localSum += smem[threadId + dftefe::utils::DEVICE_WARP_SIZE];

              _Pragma("unroll") for (size_type offset =
                                       dftefe::utils::DEVICE_WARP_SIZE / 2;
                                     offset > 0;
                                     offset /= 2)
              {
#  ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
                unsigned mask = 0xffffffff;
                localSum += __shfl_down_sync(mask, localSum, offset);
#  elif DFTEFE_WITH_DEVICE_LANG_HIP
                localSum += __shfl_down(localSum,
                                        offset,
                                        dftefe::utils::DEVICE_WARP_SIZE);
#  elif DFTEFE_WITH_DEVICE_LANG_SYCL
                localSum +=
                  sycl::shift_group_left(ind.get_sub_group(), localSum, offset);
#  endif
              }
            }

          if (threadId == 0)
            dftefe::utils::atomicAddWrapper(&d_devSum[0], localSum);
        }),
        Type *          d_qvec,
        Type *          d_devSum,
        const Type *    d_rvec,
        const Type *    d_jacobi,
        const size_type N);


      template <typename Type, size_type blockSize>
      DFTEFE_CREATE_KERNEL_SMEM_S(
        Type,
        blockSize,
        void,
        scaleXRandComputeNormKernel,
        DFTEFE_KERNEL_ARGUMENT({
          size_type idx = threadId + blockId * (blockSize * 2);

          Type localSum;

          if (idx < N)
            {
              Type rNew;
              Type rOld = d_rvec[idx];
              x[idx] += alpha * d_qvec[idx];
              rNew        = rOld + alpha * d_dvec[idx];
              localSum    = rNew * rNew;
              d_rvec[idx] = rNew;
            }
          else
            localSum = 0;

          if (idx + blockSize < N)
            {
              Type rNew;
              Type rOld = d_rvec[idx + blockSize];
              x[idx + blockSize] += alpha * d_qvec[idx + blockSize];
              rNew = rOld + alpha * d_dvec[idx + blockSize];
              localSum += rNew * rNew;
              d_rvec[idx + blockSize] = rNew;
            }

          smem[threadId] = localSum;
          SYNCTHREADS;

          _Pragma("unroll") for (size_type size =
                                   dftefe::utils::DEVICE_MAX_BLOCK_SIZE / 2;
                                 size >= 4 * dftefe::utils::DEVICE_WARP_SIZE;
                                 size /= 2)
          {
            if ((blockSize >= size) && (threadId < size / 2))
              smem[threadId] = localSum = localSum + smem[threadId + size / 2];

#  if defined(DFTEFE_WITH_DEVICE_LANG_CUDA) || \
    defined(DFTEFE_WITH_DEVICE_LANG_HIP)
            __syncthreads();
#  elif defined(DFTEFE_WITH_DEVICE_LANG_SYCL)
            sycl::group_barrier(ind.get_group());
#  endif
          }

          if (threadId < dftefe::utils::DEVICE_WARP_SIZE)
            {
              if (blockSize >= 2 * dftefe::utils::DEVICE_WARP_SIZE)
                localSum += smem[threadId + dftefe::utils::DEVICE_WARP_SIZE];

              _Pragma("unroll") for (size_type offset =
                                       dftefe::utils::DEVICE_WARP_SIZE / 2;
                                     offset > 0;
                                     offset /= 2)
              {
#  ifdef DFTEFE_WITH_DEVICE_LANG_CUDA
                unsigned mask = 0xffffffff;
                localSum += __shfl_down_sync(mask, localSum, offset);
#  elif DFTEFE_WITH_DEVICE_LANG_HIP
                localSum += __shfl_down(localSum,
                                        offset,
                                        dftefe::utils::DEVICE_WARP_SIZE);
#  elif DFTEFE_WITH_DEVICE_LANG_SYCL
                localSum +=
                  sycl::shift_group_left(ind.get_sub_group(), localSum, offset);
#  endif
              }
            }

          if (threadId == 0)
            dftefe::utils::atomicAddWrapper(&d_devSum[0], localSum);
        }),
        Type *          x,
        Type *          d_rvec,
        Type *          d_devSum,
        const Type *    d_qvec,
        const Type *    d_dvec,
        const Type      alpha,
        const size_type N);

      template <typename Type>
      DFTEFE_CREATE_KERNEL(
        void,
        saddKernel,
        {
          for (size_type idx = globalThreadId; idx < size;
               idx += nThreadsPerBlock * nThreadBlock)
            {
              y[idx] = beta * y[idx] - x[idx];
              x[idx] = 0;
            }
        },
        Type *          y,
        Type *          x,
        const Type      beta,
        const size_type size);
    } // namespace

    void
    applyPreconditionAndComputeDotProductDevice(
      double *        d_dvec,
      double *        d_devSum,
      const double *  d_rvec,
      const double *  d_jacobi,
      const size_type N,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
        &linAlgOpContext)
    {
      const size_type blocks =
        (N + (dftefe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
        (dftefe::utils::DEVICE_BLOCK_SIZE * 2);
      DFTEFE_LAUNCH_KERNEL_SMEM_S(DFTEFE_KERNEL_ARGUMENT(
                                    applyPreconditionAndComputeDotProductKernel<
                                      double,
                                      dftefe::utils::DEVICE_BLOCK_SIZE>),
                                  blocks,
                                  dftefe::utils::DEVICE_BLOCK_SIZE,
                                  double,
                                  dftefe::utils::DEVICE_BLOCK_SIZE,
                                  linAlgOpContext.getBlasStream(),
                                  d_dvec,
                                  d_devSum,
                                  d_rvec,
                                  d_jacobi,
                                  N);
    }


    void
    applyPreconditionComputeDotProductAndSaddDevice(
      double *        d_qvec,
      double *        d_devSum,
      const double *  d_rvec,
      const double *  d_jacobi,
      const size_type N,
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
        &linAlgOpContext)
    {
      const size_type blocks =
        (N + (dftefe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
        (dftefe::utils::DEVICE_BLOCK_SIZE * 2);
      DFTEFE_LAUNCH_KERNEL_SMEM_S(
        DFTEFE_KERNEL_ARGUMENT(applyPreconditionComputeDotProductAndSaddKernel<
                               double,
                               dftefe::utils::DEVICE_BLOCK_SIZE>),
        blocks,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        double,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        linAlgOpContext.getBlasStream(),
        d_qvec,
        d_devSum,
        d_rvec,
        d_jacobi,
        N);
    }


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
        &linAlgOpContext)
    {
      const size_type blocks =
        (N + (dftefe::utils::DEVICE_BLOCK_SIZE * 2 - 1)) /
        (dftefe::utils::DEVICE_BLOCK_SIZE * 2);
      DFTEFE_LAUNCH_KERNEL_SMEM_S(
        DFTEFE_KERNEL_ARGUMENT(
          scaleXRandComputeNormKernel<double,
                                      dftefe::utils::DEVICE_BLOCK_SIZE>),
        blocks,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        double,
        dftefe::utils::DEVICE_BLOCK_SIZE,
        linAlgOpContext.getBlasStream(),
        x,
        d_rvec,
        d_devSum,
        d_qvec,
        d_dvec,
        alpha,
        N);
    }

    void
    saddDevice(double *        y,
               double *        x,
               const double    beta,
               const size_type size,
               linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
                 &linAlgOpContext)
    {
      const size_type gridSize =
        (size / dftefe::utils::DEVICE_BLOCK_SIZE) +
        (size % dftefe::utils::DEVICE_BLOCK_SIZE == 0 ? 0 : 1);
      DFTEFE_LAUNCH_KERNEL(saddKernel,
                           gridSize,
                           dftefe::utils::DEVICE_BLOCK_SIZE,
                           linAlgOpContext.getBlasStream(),
                           y,
                           x,
                           beta,
                           size);
    }
  } // namespace electrostatics
} // namespace dftefe
#endif
