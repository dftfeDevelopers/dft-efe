#ifdef DFTEFE_WITH_DEVICE

#  include <utils/DeviceComplexUtils.cuh>
#  include <utils/DeviceKernelLauncher.h>
namespace dftefe
{
  namespace utils
  {
    namespace
    {
      template <typename NumberTypeComplex, typename NumberTypeReal>
      __global__ void
      copyComplexArrToRealArrsCUDAKernel(const size_type          size,
                                         const NumberTypeComplex *complexArr,
                                         NumberTypeReal *         realArr,
                                         NumberTypeReal *         imagArr)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

        for (size_type index = globalThreadId; index < size;
             index += blockDim.x * gridDim.x)
          {
            realArr[index] = complexArr[index].x;
            imagArr[index] = complexArr[index].y;
          }
      }


      template <typename NumberTypeComplex, typename NumberTypeReal>
      __global__ void
      copyRealArrsToComplexArrCUDAKernel(const size_type       size,
                                         const NumberTypeReal *realArr,
                                         const NumberTypeReal *imagArr,
                                         NumberTypeComplex *   complexArr)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

        for (size_type index = globalThreadId; index < size;
             index += blockDim.x * gridDim.x)
          {
            complexArr[index].x = realArr[index];
            complexArr[index].y = imagArr[index];
          }
      }


    } // end of namespace

    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyComplexArrToRealArrsGPU(const size_type          size,
                                const NumberTypeComplex *complexArr,
                                NumberTypeReal *         realArr,
                                NumberTypeReal *         imagArr)
    {
      dftefe::utils::copyComplexArrToRealArrsCUDAKernel<NumberTypeComplex,
                                                        NumberTypeReal>
        <<<size / dftefe::utils::BLOCK_SIZE + 1, dftefe::utils::BLOCK_SIZE>>>(
          size, complexArr, realArr, imagArr);
    }


    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyRealArrsToComplexArrGPU(const size_type       size,
                                const NumberTypeReal *realArr,
                                const NumberTypeReal *imagArr,
                                NumberTypeComplex *   complexArr)
    {
      dftefe::utils::copyRealArrsToComplexArrCUDAKernel<NumberTypeComplex,
                                                        NumberTypeReal>
        <<<size / dftefe::utils::BLOCK_SIZE + 1, dftefe::utils::BLOCK_SIZE>>>(
          size, realArr, imagArr, complexArr);
    }


  } // end of namespace utils


} // end of namespace dftefe



#endif
