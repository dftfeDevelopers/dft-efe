#ifdef DFTEFE_WITH_DEVICE

#  include "ConstraintsInternal.h"
#  include <utils/DeviceUtils.h>
#  include <utils/DeviceTypeConfig.h>
#  include <utils/DeviceKernelLauncherHelpers.h>
#  include <utils/DeviceAPICalls.h>
#  include <utils/DeviceDataTypeOverloads.h>
#  include <utils/DeviceTypeConfigHalfPrec.h>
#  include <linearAlgebra/BlasLapackKernels.h>
#  include <linearAlgebra/BlasLapack.h>
#  include<utils/Exceptions.h>
#  include <complex>
namespace dftefe
{
  namespace basis
  {
    namespace constraintsInternal
    {
      template <typename ValueTypeBasisCoeff>
      DFTEFE_CREATE_KERNEL(
        void,
        setValueKernel,
        {
          assert(false && "setValueKernel() is not implemented for utils::MemorySpace::DEVICE");
        },
        ValueTypeBasisCoeff *     xVec,
        const size_type *         constraintLocalRowIds,
        const size_type           numConstraints,
        const size_type           blockSize,
        const ValueTypeBasisCoeff alpha);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        setValueKernel,
        {
          const size_type numberEntries = numConstraints * blockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockIndex      = index / blockSize;
              const size_type intraBlockIndex = index - blockIndex * blockSize;
              utils::copyValue((xVec +
                                constraintLocalRowIds[blockIndex] * blockSize +
                                intraBlockIndex),
                               alpha);
            }
        },
        float *          xVec,
        const size_type *constraintLocalRowIds,
        const size_type  numConstraints,
        const size_type  blockSize,
        const float      alpha);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        setValueKernel,
        {
          const size_type numberEntries = numConstraints * blockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockIndex      = index / blockSize;
              const size_type intraBlockIndex = index - blockIndex * blockSize;
              utils::copyValue((xVec +
                                constraintLocalRowIds[blockIndex] * blockSize +
                                intraBlockIndex),
                               alpha);
            }
        },
        double *         xVec,
        const size_type *constraintLocalRowIds,
        const size_type  numConstraints,
        const size_type  blockSize,
        const double     alpha);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        setValueKernel,
        {
          const size_type numberEntries = numConstraints * blockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockIndex      = index / blockSize;
              const size_type intraBlockIndex = index - blockIndex * blockSize;
              utils::copyValue((xVec +
                                constraintLocalRowIds[blockIndex] * blockSize +
                                intraBlockIndex),
                               alpha);
            }
        },
        utils::deviceFloatComplex *     xVec,
        const size_type *               constraintLocalRowIds,
        const size_type                 numConstraints,
        const size_type                 blockSize,
        const utils::deviceFloatComplex alpha);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        setValueKernel,
        {
          const size_type numberEntries = numConstraints * blockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const size_type blockIndex      = index / blockSize;
              const size_type intraBlockIndex = index - blockIndex * blockSize;
              utils::copyValue((xVec +
                                constraintLocalRowIds[blockIndex] * blockSize +
                                intraBlockIndex),
                               alpha);
            }
        },
        utils::deviceDoubleComplex *     xVec,
        const size_type *                constraintLocalRowIds,
        const size_type                  numConstraints,
        const size_type                  blockSize,
        const utils::deviceDoubleComplex alpha);

      template <typename ValueTypeBasisCoeff>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeParentToChildKernel,
        {
          assert(false && "distributeParentToChildKernel() is not implemented for utils::MemorySpace::DEVICE");
        },
        const size_type      contiguousBlockSize,
        ValueTypeBasisCoeff *xVec,
        const size_type *    constraintLocalRowIds,
        const size_type      numConstraints,
        const size_type *    constraintRowSizes,
        const size_type *    constraintRowSizesAccumulated,
        const size_type *    constraintLocalColumnIds,
        const double *       constraintColumnValues,
        const double *       inhomogenities);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeParentToChildKernel,
        {
          const size_type numberEntries = numConstraints * contiguousBlockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
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
              xVec[xVecStartingIdRow] =
                inhomogenities[blockIndex];
              for (size_type i = 0; i < numberColumns; ++i)
                {
                  const global_size_type xVecStartingIdColumn =
                    constraintLocalColumnIds[startingColumnNumber + i];
                  const global_size_type xVecColumnId =
                    xVecStartingIdColumn * contiguousBlockSize +
                    intraBlockIndex;
                  xVec[xVecStartingIdRow] = dftefe::utils::add(
                    xVec[xVecStartingIdRow],
                    dftefe::utils::mult(
                      constraintColumnValues[startingColumnNumber + i],
                      xVec[xVecColumnId]));
                }
            }
        },
        const size_type  contiguousBlockSize,
        float *          xVec,
        const size_type *constraintLocalRowIds,
        const size_type  numConstraints,
        const size_type *constraintRowSizes,
        const size_type *constraintRowSizesAccumulated,
        const size_type *constraintLocalColumnIds,
        const double *   constraintColumnValues,
        const double *   inhomogenities);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeParentToChildKernel,
        {
          const size_type numberEntries = numConstraints * contiguousBlockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
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
              xVec[xVecStartingIdRow] =
                inhomogenities[blockIndex];
              for (size_type i = 0; i < numberColumns; ++i)
                {
                  const global_size_type xVecStartingIdColumn =
                    constraintLocalColumnIds[startingColumnNumber + i];
                  const global_size_type xVecColumnId =
                    xVecStartingIdColumn * contiguousBlockSize +
                    intraBlockIndex;
                  xVec[xVecStartingIdRow] = dftefe::utils::add(
                    xVec[xVecStartingIdRow],
                    dftefe::utils::mult(
                      constraintColumnValues[startingColumnNumber + i],
                      xVec[xVecColumnId]));
                }
            }
        },
        const size_type  contiguousBlockSize,
        double *         xVec,
        const size_type *constraintLocalRowIds, // rowConstraintsIdsLocal.data(),
        const size_type  numConstraints, // rowConstraintsIdsLocal.size()
        const size_type *constraintRowSizes, // rowConstraintsSizes.data()
        const size_type *constraintRowSizesAccumulated, //  columnConstraintsAccumulated.data()
        const size_type *constraintLocalColumnIds, // columnConstraintsIdsLocal.data(),
        const double *   constraintColumnValues, // columnConstraintsValues.data(),
        const double *   inhomogenities); // constraintsInhomogenities.data()

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeParentToChildKernel,
        {
          const size_type numberEntries = numConstraints * contiguousBlockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
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
              dftefe::utils::copyValue(xVec + xVecStartingIdRow,
                                       inhomogenities[blockIndex]);
              for (size_type i = 0; i < numberColumns; ++i)
                {
                  const global_size_type xVecStartingIdColumn =
                    constraintLocalColumnIds[startingColumnNumber + i];
                  const global_size_type xVecColumnId =
                    xVecStartingIdColumn * contiguousBlockSize +
                    intraBlockIndex;
                  dftefe::utils::copyValue(
                    xVec + xVecStartingIdRow,
                    dftefe::utils::add(
                      xVec[xVecStartingIdRow],
                      dftefe::utils::makeComplex(
                        dftefe::utils::realPartDevice(dftefe::utils::mult(
                          constraintColumnValues[startingColumnNumber + i],
                          xVec[xVecColumnId])),
                        dftefe::utils::imagPartDevice(dftefe::utils::mult(
                          constraintColumnValues[startingColumnNumber + i],
                          xVec[xVecColumnId])))));
                }
            }
        },
        const size_type                    contiguousBlockSize,
        dftefe::utils::deviceFloatComplex *xVec,
        const size_type *                  constraintLocalRowIds,
        const size_type                    numConstraints,
        const size_type *                  constraintRowSizes,
        const size_type *                  constraintRowSizesAccumulated,
        const size_type *                  constraintLocalColumnIds,
        const double *                     constraintColumnValues,
        const double *                     inhomogenities);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeParentToChildKernel,
        {
          const size_type numberEntries = numConstraints * contiguousBlockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
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
              dftefe::utils::copyValue(xVec + xVecStartingIdRow,
                                       inhomogenities[blockIndex]);
              for (size_type i = 0; i < numberColumns; ++i)
                {
                  const global_size_type xVecStartingIdColumn =
                    constraintLocalColumnIds[startingColumnNumber + i];
                  const global_size_type xVecColumnId =
                    xVecStartingIdColumn * contiguousBlockSize +
                    intraBlockIndex;
                  dftefe::utils::copyValue(
                    xVec + xVecStartingIdRow,
                    dftefe::utils::add(
                      xVec[xVecStartingIdRow],
                      dftefe::utils::makeComplex(
                        dftefe::utils::realPartDevice(dftefe::utils::mult(
                          constraintColumnValues[startingColumnNumber + i],
                          xVec[xVecColumnId])),
                        dftefe::utils::imagPartDevice(dftefe::utils::mult(
                          constraintColumnValues[startingColumnNumber + i],
                          xVec[xVecColumnId])))));
                }
            }
        },
        const size_type                     contiguousBlockSize,
        dftefe::utils::deviceDoubleComplex *xVec,
        const size_type *                   constraintLocalRowIds,
        const size_type                     numConstraints,
        const size_type *                   constraintRowSizes,
        const size_type *                   constraintRowSizesAccumulated,
        const size_type *                   constraintLocalColumnIds,
        const double *                      constraintColumnValues,
        const double *                      inhomogenities);

      template <typename ValueTypeBasisCoeff>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeChildToParentKernel,
        {
          assert(false && "distributeChildToParentKernel() is not implemented for utils::MemorySpace::DEVICE");
        },
        const size_type      contiguousBlockSize,
        ValueTypeBasisCoeff *xVec,
        const size_type *    constraintLocalRowIds,
        const size_type      numConstraints,
        const size_type *    constraintRowSizes,
        const size_type *    constraintRowSizesAccumulated,
        const size_type *    constraintLocalColumnIds,
        const double *       constraintColumnValues);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeChildToParentKernel,
        {
          const size_type numberEntries = numConstraints * contiguousBlockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
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
                    xVecStartingIdColumn * contiguousBlockSize +
                    intraBlockIndex;
                  float tempVal = dftefe::utils::mult(
                    constraintColumnValues[startingColumnNumber + i],
                    xVec[xVecStartingIdRow]);
                  dftefe::utils::atomicAddWrapper(&xVec[xVecColumnId], tempVal);
                }
              xVec[xVecStartingIdRow] = 0.0;
            }
        },
        const size_type  contiguousBlockSize,
        float *          xVec,
        const size_type *constraintLocalRowIds,
        const size_type  numConstraints,
        const size_type *constraintRowSizes,
        const size_type *constraintRowSizesAccumulated,
        const size_type *constraintLocalColumnIds,
        const double *   constraintColumnValues);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeChildToParentKernel,
        {
          const size_type numberEntries = numConstraints * contiguousBlockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
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
                    xVecStartingIdColumn * contiguousBlockSize +
                    intraBlockIndex;
                  double tempVal = dftefe::utils::mult(
                    constraintColumnValues[startingColumnNumber + i],
                    xVec[xVecStartingIdRow]);
                  dftefe::utils::atomicAddWrapper(&xVec[xVecColumnId], tempVal);
                }
              xVec[xVecStartingIdRow] = 0.0;
            }
        },
        const size_type  contiguousBlockSize,
        double *         xVec,
        const size_type *constraintLocalRowIds,
        const size_type  numConstraints,
        const size_type *constraintRowSizes,
        const size_type *constraintRowSizesAccumulated,
        const size_type *constraintLocalColumnIds,
        const double *   constraintColumnValues);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeChildToParentKernel,
        {
          const size_type numberEntries = numConstraints * contiguousBlockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
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
                    xVecStartingIdColumn * contiguousBlockSize +
                    intraBlockIndex;
                  dftefe::utils::deviceDoubleComplex tempComplval =
                    dftefe::utils::mult(
                      constraintColumnValues[startingColumnNumber + i],
                      xVec[xVecStartingIdRow]);
                  auto *add_real =
                    reinterpret_cast<float *>(&xVec[xVecColumnId]);
                  auto *add_imag = add_real + 1;
                  dftefe::utils::atomicAddWrapper(
                    add_real, dftefe::utils::realPartDevice(tempComplval));
                  dftefe::utils::atomicAddWrapper(
                    add_imag, dftefe::utils::imagPartDevice(tempComplval));
                }
              xVec[xVecStartingIdRow] =
                dftefe::utils::makeComplex((float)0.0, (float)0.0);
            }
        },
        const size_type                    contiguousBlockSize,
        dftefe::utils::deviceFloatComplex *xVec,
        const size_type *                  constraintLocalRowIds,
        const size_type                    numConstraints,
        const size_type *                  constraintRowSizes,
        const size_type *                  constraintRowSizesAccumulated,
        const size_type *                  constraintLocalColumnIds,
        const double *                     constraintColumnValues);

      template <>
      DFTEFE_CREATE_KERNEL(
        void,
        distributeChildToParentKernel,
        {
          const size_type numberEntries = numConstraints * contiguousBlockSize;
          for (size_type index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
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
                    xVecStartingIdColumn * contiguousBlockSize +
                    intraBlockIndex;
                  dftefe::utils::deviceDoubleComplex tempComplval =
                    dftefe::utils::mult(
                      constraintColumnValues[startingColumnNumber + i],
                      xVec[xVecStartingIdRow]);
                  auto *add_real =
                    reinterpret_cast<double *>(&xVec[xVecColumnId]);
                  auto *add_imag = add_real + 1;
                  dftefe::utils::atomicAddWrapper(
                    add_real, dftefe::utils::realPartDevice(tempComplval));
                  dftefe::utils::atomicAddWrapper(
                    add_imag, dftefe::utils::imagPartDevice(tempComplval));
                }
              xVec[xVecStartingIdRow] = dftefe::utils::makeComplex(0.0, 0.0);
            }
        },
        const size_type                     contiguousBlockSize,
        dftefe::utils::deviceDoubleComplex *xVec,
        const size_type *                   constraintLocalRowIds,
        const size_type                     numConstraints,
        const size_type *                   constraintRowSizes,
        const size_type *                   constraintRowSizesAccumulated,
        const size_type *                   constraintLocalColumnIds,
        const double *                      constraintColumnValues);

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

      DFTEFE_LAUNCH_KERNEL(
        constraintsInternal::setValueKernel,
        numConstrainedDofs * blockSize / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        utils::DEVICE_BLOCK_SIZE,
        dftefe::utils::defaultStream,
        dftefe::utils::makeDataTypeDeviceCompatible(vectorData.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible((ValueTypeBasisCoeff)0.0));
    }

    template <typename ValueTypeBasisCoeff>
    void
    ConstraintsInternal<ValueTypeBasisCoeff,
                        dftefe::utils::MemorySpace::DEVICE>::
      constraintsSetConstrainedNodes(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                   dftefe::utils::MemorySpace::DEVICE>
          &             vectorData,
        const size_type blockSize,
        const utils::MemoryStorage<size_type,
                                   dftefe::utils::MemorySpace::DEVICE>
          &                       rowConstraintsIdsLocal,
        const ValueTypeBasisCoeff alpha)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;

      DFTEFE_LAUNCH_KERNEL(
        constraintsInternal::setValueKernel,
        numConstrainedDofs * blockSize / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        utils::DEVICE_BLOCK_SIZE,
        dftefe::utils::defaultStream,
        dftefe::utils::makeDataTypeDeviceCompatible(vectorData.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(alpha));
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
        const utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>
          &constraintsInhomogenities,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;

      DFTEFE_LAUNCH_KERNEL(
        constraintsInternal::distributeParentToChildKernel,
        numConstrainedDofs * blockSize / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        utils::DEVICE_BLOCK_SIZE,
        linAlgOpContext.getBlasStream(),
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(vectorData.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        rowConstraintsSizes.data(),
        columnConstraintsAccumulated.data(),
        columnConstraintsIdsLocal.data(),
        columnConstraintsValues.data(),
        constraintsInhomogenities.data());
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
          &columnConstraintsValues,
        linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::DEVICE>
          &linAlgOpContext)
    {
      const size_type numConstrainedDofs = rowConstraintsIdsLocal.size();

      if (numConstrainedDofs == 0)
        return;

      DFTEFE_LAUNCH_KERNEL(
        constraintsInternal::distributeChildToParentKernel,
        numConstrainedDofs * blockSize / dftefe::utils::DEVICE_BLOCK_SIZE + 1,
        utils::DEVICE_BLOCK_SIZE,
        linAlgOpContext.getBlasStream(),
        blockSize,
        dftefe::utils::makeDataTypeDeviceCompatible(vectorData.data()),
        rowConstraintsIdsLocal.data(),
        numConstrainedDofs,
        rowConstraintsSizes.data(),
        columnConstraintsAccumulated.data(),
        columnConstraintsIdsLocal.data(),
        columnConstraintsValues.data());
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
