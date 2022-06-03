#ifdef DFTEFE_WITH_DEVICE_CUDA

#  include "ConstraintsInternal.h"
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/DataTypeOverloads.h>
#  include <utils/DeviceKernelLauncher.h>
namespace dftefe
{
  namespace basis
  {
    namespace constraintsInternal
    {
      template <typename ValueType>
      __global__ void
      setZeroKernel(ValueType *      xVec,
                    const size_type *constraintLocalRowIds,
                    const size_type  numConstraints,
                    const size_type  blockSize)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        const size_type numberEntries  = numConstraints * blockSize;

        for (size_type index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const size_type blockIndex      = index / blockSize;
            const size_type intraBlockIndex = index - blockIndex * blockSize;
            dftefe::utils::setRealValue(
              &(xVec[constraintLocalRowIds[blockIndex] * blockSize +
                     intraBlockIndex]),
              0.0);
          }
      }

      /*
                template <typename ValueType>
                __global__ void
                distributeParentToChildKernel(
                  const size_type  contiguousBlockSize,
                  ValueType *    xVec,
                  const size_type *constraintLocalRowIds,
                  const size_type  numConstraints,
                  const size_type *constraintRowSizes,
                  const size_type *constraintRowSizesAccumulated,
                  const size_type *constraintLocalColumnIds,
                  const double *      constraintColumnValues,
                  const ValueType *      inhomogenities)
                {
                  const size_type globalThreadId =
                    blockIdx.x * blockDim.x + threadIdx.x;
                  const size_type numberEntries =
                    numConstraints * contiguousBlockSize;

                  for (size_type index = globalThreadId;
                       index < numberEntries;
                       index += blockDim.x * gridDim.x)
                    {
                      const size_type blockIndex      = index /
         contiguousBlockSize; const size_type intraBlockIndex = index -
         blockIndex*contiguousBlockSize; const size_type constrainedRowId =
                        constraintLocalRowIds[blockIndex];
                      const size_type numberColumns =
         constraintRowSizes[blockIndex]; const size_type startingColumnNumber =
                        constraintRowSizesAccumulated[blockIndex];
                      const size_type xVecStartingIdRow =
                        localIndexMapUnflattenedToFlattened[constrainedRowId];
                      xVec[xVecStartingIdRow + intraBlockIndex] =
                        make_cuFloatComplex(inhomogenities[blockIndex], 0.0);
                      for (size_type i = 0; i < numberColumns; ++i)
                        {
                          const size_type constrainedColumnId =
                            constraintLocalColumnIdsAllRowsUnflattened
                              [startingColumnNumber + i];
                          const dealii::types::global_dof_index
         xVecStartingIdColumn =
                            localIndexMapUnflattenedToFlattened[constrainedColumnId];
                          xVec[xVecStartingIdRow + intraBlockIndex] =
                            cuCaddf(xVec[xVecStartingIdRow + intraBlockIndex],
                                    make_cuFloatComplex(
                                      xVec[xVecStartingIdColumn +
         intraBlockIndex].x * constraintColumnValuesAllRowsUnflattened
                                          [startingColumnNumber + i],
                                      xVec[xVecStartingIdColumn +
         intraBlockIndex].y * constraintColumnValuesAllRowsUnflattened
                                          [startingColumnNumber + i]));
                        }
                    }
                }

      */

    } // end of namespace constraintsInternal


    template <typename ValueType>
    void
    ConstraintsInternal<ValueType, dftefe::utils::MemorySpace::DEVICE>::
      constraintsSetConstrainedNodesToZero(
        linearAlgebra::Vector<ValueType, dftefe::utils::MemorySpace::DEVICE>
          &             vectorData,
        const size_type blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;

      constraintsInternal::setZeroKernel<<<
        numConstrainedDofs * blockSize / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        dftefe::utils::makeDataTypeDeviceCompatible(vectorData.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        blockSize);
    }


    template class ConstraintsInternal<double,
                                       dftefe::utils::MemorySpace::DEVICE>;
    template class ConstraintsInternal<float,
                                       dftefe::utils::MemorySpace::DEVICE>;
    template class ConstraintsInternal<std::complex<double>,
                                       dftefe::utils::MemorySpace::DEVICE>;
    template class ConstraintsInternal<std::complex<float>,
                                       dftefe::utils::MemorySpace::DEVICE>;


  } // namespace basis
} // namespace dftefe

#endif
