#ifdef DFTEFE_WITH_DEVICE_CUDA

#  include "ConstraintsInternal.h"
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/DeviceDataTypeOverloads.cuh>
#  include <utils/DataTypeOverloads.h>
#  include <utils/DeviceKernelLauncher.h>
#  include <utils/DeviceComplexUtils.cuh>

namespace dftefe
{
  namespace basis
  {
    namespace constraintsInternal
    {
      template <typename ValueTypeBasisCoeff>
      __global__ void
      setZeroKernel(ValueTypeBasisCoeff *xVec,
                    const size_type *    constraintLocalRowIds,
                    const size_type      numConstraints,
                    const size_type      blockSize)
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

      template <typename ValueTypeBasisCoeff>
      __global__ void
      distributeParentToChildKernel(
        const size_type            contiguousBlockSize,
        ValueTypeBasisCoeff *      xVec,
        const size_type *          constraintLocalRowIds,
        const size_type            numConstraints,
        const size_type *          constraintRowSizes,
        const size_type *          constraintRowSizesAccumulated,
        const size_type *          constraintLocalColumnIds,
        const double *             constraintColumnValues,
        const ValueTypeBasisCoeff *inhomogenities)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        const size_type numberEntries  = numConstraints * contiguousBlockSize;
        for (size_type index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const size_type blockIndex = index / contiguousBlockSize;
            const size_type intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            const size_type constrainedRowId =
              constraintLocalRowIds[blockIndex];
            const size_type numberColumns = constraintRowSizes[blockIndex];
            const size_type startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const size_type xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize + intraBlockIndex;
            xVec[xVecStartingIdRow + intraBlockIndex] =
              inhomogenities[blockIndex];
            for (size_type i = 0; i < numberColumns; ++i)
              {
                const global_size_type xVecStartingIdColumn =
                  constraintLocalColumnIds[startingColumnNumber + i];
                const global_size_type xVecColumnId =
                  xVecStartingIdColumn * contiguousBlockSize + intraBlockIndex;
                xVec[xVecStartingIdRow] = dftefe::utils::add(
                  xVec[xVecStartingIdRow],
                  dftefe::utils::mult(
                    constraintColumnValues[startingColumnNumber + i],
                    xVec[xVecColumnId]));
              }
          }
      }


      template <typename ValueTypeBasisCoeff>
      __global__ void
      distributeChildToParentKernel(
        const size_type      contiguousBlockSize,
        ValueTypeBasisCoeff *xVec,
        const size_type *    constraintLocalRowIds,
        const size_type      numConstraints,
        const size_type *    constraintRowSizes,
        const size_type *    constraintRowSizesAccumulated,
        const size_type *    constraintLocalColumnIds,
        const double *       constraintColumnValues)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        const size_type numberEntries  = numConstraints * contiguousBlockSize;
        for (size_type index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const size_type blockIndex = index / contiguousBlockSize;
            const size_type intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            const size_type constrainedRowId =
              constraintLocalRowIds[blockIndex];
            const size_type numberColumns = constraintRowSizes[blockIndex];
            const size_type startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const size_type xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize + intraBlockIndex;
            for (size_type i = 0; i < numberColumns; ++i)
              {
                const global_size_type xVecStartingIdColumn =
                  constraintLocalColumnIds[startingColumnNumber + i];
                const global_size_type xVecColumnId =
                  xVecStartingIdColumn * contiguousBlockSize + intraBlockIndex;
                ValueTypeBasisCoeff tempVal = dftefe::utils::mult(
                  constraintColumnValues[startingColumnNumber + i],
                  xVec[xVecStartingIdRow]);
                atomicAdd(&xVec[xVecColumnId], tempVal);
              }
            dftefe::utils::setRealValue(&(xVec[xVecStartingIdRow]), 0.0);
          }
      }

    } // end of namespace constraintsInternal


    template <typename ValueTypeBasisCoeff>
    void
    ConstraintsInternal<ValueTypeBasisCoeff,
                        dftefe::utils::MemorySpace::DEVICE>::
      constraintsSetConstrainedNodesToZero(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                   dftefe::utils::MemorySpace::DEVICE>
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

    template <typename ValueTypeBasisCoeff>
    void
    ConstraintsInternal<ValueTypeBasisCoeff,
                        dftefe::utils::MemorySpace::DEVICE>::
      constraintsDistributeParentToChild(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                   dftefe::utils::MemorySpace::DEVICE>
          &             vectorData,
        const size_type blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsSizes,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsValues,
        const utils::MemoryStorage<ValueTypeBasisCoeff,
                                   dftefe::utils::MemorySpace::DEVICE>
          &constraintsInhomogenities)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;

      constraintsInternal::distributeParentToChildKernel<<<
        numConstrainedDofs * blockSize / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(vectorData.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        rowConstraintsSizes.data(),
        columnConstraintsAccumulated.data(),
        columnConstraintsIdsLocal.data(),
        columnConstraintsValues.data(),
        dftefe::utils::makeDataTypeDeviceCompatible(
          constraintsInhomogenities.data()));
    }

    template <typename ValueTypeBasisCoeff>
    void
    ConstraintsInternal<ValueTypeBasisCoeff,
                        dftefe::utils::MemorySpace::DEVICE>::
      constraintsDistributeChildToParent(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                   dftefe::utils::MemorySpace::DEVICE>
          &             vectorData,
        const size_type blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsSizes,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsValues)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;

      constraintsInternal::distributeChildToParentKernel<<<
        numConstrainedDofs * blockSize / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(vectorData.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        rowConstraintsSizes.data(),
        columnConstraintsAccumulated.data(),
        columnConstraintsIdsLocal.data(),
        columnConstraintsValues.data());
    }


    template <>
    void
    ConstraintsInternal<std::complex<float>,
                        dftefe::utils::MemorySpace::DEVICE>::
      constraintsDistributeChildToParent(
        linearAlgebra::MultiVector<std::complex<float>,
                                   dftefe::utils::MemorySpace::DEVICE>
          &             vectorData,
        const size_type blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsSizes,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsValues)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;
      size_type sizeOfVector = vectorData.size();
      utils::MemoryStorage<float, dftefe::utils::MemorySpace::DEVICE> tempReal,
        tempImag;
      tempReal.resize(sizeOfVector);
      tempImag.resize(sizeOfVector);
      dftefe::utils::copyComplexArrToRealArrsGPU<std::complex<float>, float>(
        vectorData.size(),
        vectorData.begin(),
        tempReal.begin(),
        tempImag.begin());
      constraintsInternal::distributeChildToParentKernel<<<
        numConstrainedDofs * blockSize / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(tempReal.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        rowConstraintsSizes.data(),
        columnConstraintsAccumulated.data(),
        columnConstraintsIdsLocal.data(),
        columnConstraintsValues.data());

      constraintsInternal::distributeChildToParentKernel<<<
        numConstrainedDofs * blockSize / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(tempImag.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        rowConstraintsSizes.data(),
        columnConstraintsAccumulated.data(),
        columnConstraintsIdsLocal.data(),
        columnConstraintsValues.data());

      dftefe::utils::copyRealArrsToComplexArrGPU(vectorData.size(),
                                                 tempReal.begin(),
                                                 tempImag.begin(),
                                                 vectorData.begin());
    }

    template <>
    void
    ConstraintsInternal<std::complex<double>,
                        dftefe::utils::MemorySpace::DEVICE>::
      constraintsDistributeChildToParent(
        linearAlgebra::MultiVector<std::complex<double>,
                                   dftefe::utils::MemorySpace::DEVICE>
          &             vectorData,
        const size_type blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &rowConstraintsSizes,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsIdsLocal,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsAccumulated,
        const utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &columnConstraintsValues)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;
      size_type sizeOfVector = vectorData.size();
      utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE> tempReal,
        tempImag;
      tempReal.resize(sizeOfVector);
      tempImag.resize(sizeOfVector);
      dftefe::utils::copyComplexArrToRealArrsGPU<std::complex<double>, double>(
        vectorData.size(),
        vectorData.begin(),
        tempReal.begin(),
        tempImag.begin());
      constraintsInternal::distributeChildToParentKernel<<<
        numConstrainedDofs * blockSize / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(tempReal.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        rowConstraintsSizes.data(),
        columnConstraintsAccumulated.data(),
        columnConstraintsIdsLocal.data(),
        columnConstraintsValues.data());

      constraintsInternal::distributeChildToParentKernel<<<
        numConstrainedDofs * blockSize / dftefe::utils::BLOCK_SIZE + 1,
        dftefe::utils::BLOCK_SIZE>>>(
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(tempImag.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        rowConstraintsSizes.data(),
        columnConstraintsAccumulated.data(),
        columnConstraintsIdsLocal.data(),
        columnConstraintsValues.data());

      dftefe::utils::copyRealArrsToComplexArrGPU<std::complex<double>, double>(
        vectorData.size(),
        tempReal.begin(),
        tempImag.begin(),
        vectorData.begin());
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
