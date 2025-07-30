/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

#ifndef elpaScalapackOperations_h
#define elpaScalapackOperations_h

#include "ProcessGrid.h"
#include "ScalapackWrapper.h"
#include <elpa/elpa.h>
#include <unordered_map>
#include <iostream>
namespace dftefe
{
  namespace linearAlgebra
  {
    namespace elpaScalaOpInternal
    {
      /**
       *  @brief Contains internal functions used in linearAlgebra
       *
       *  @author Sambit Das, Aviup Sircar
       */
      /** @brief setup ELPA parameters.
       *
       */
      void
      setupELPAHandleParameters(
        const utils::mpi::MPIComm &               mpi_communicator,
        utils::mpi::MPIComm &                     processGridCommunicatorActive,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const size_type                           na,
        const size_type                           nev,
        const size_type                           blockSize,
        elpa_t &                                  elpaHandle,
        const bool                                useELPADeviceKernel = false);

      /** @brief Wrapper function to create a two dimensional processor grid for a square matrix in
       * ScaLAPACKMatrix storage format.
       *
       */
      void
      createProcessGridSquareMatrix(
        const utils::mpi::MPIComm &         mpi_communicator,
        const size_type                     size,
        std::shared_ptr<const ProcessGrid> &processGrid,
        const size_type                     scalapackParalProcs,
        const bool                          useELPA,
        const bool                          useOnlyThumbRule = false);

      /** @brief Wrapper function to create a two dimensional processor grid for a rectangular matrix in
       * ScaLAPACKMatrix storage format.
       *
       */
      void
      createProcessGridRectangularMatrix(
        const utils::mpi::MPIComm &         mpi_communicator,
        const size_type                     sizeRows,
        const size_type                     sizeColumns,
        std::shared_ptr<const ProcessGrid> &processGrid);


      /** @brief Creates global row/column id to local row/column ids for ScaLAPACKMatrix
       *
       */
      template <typename T>
      void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const ScaLAPACKMatrix<T> &                mat,
        std::unordered_map<size_type, size_type> &globalToLocalRowIdMap,
        std::unordered_map<size_type, size_type> &globalToLocalColumnIdMap);

      /** @brief scale a ScaLAPACKMat with a scalar
       *
       *
       */
      template <typename T>
      void
      scaleScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        /*const std::shared_ptr<
          linearAlgebra::BLASWrapper<utils::MemorySpace::HOST>>
          &                        BLASWrapperPtr,*/
        ScaLAPACKMatrix<T> &mat,
        const T             scalar);

      /** @brief Computes Sc=X^{T}*Xc and stores in a parallel ScaLAPACK matrix.
       * X^{T} is the subspaceVectorsArray stored in the column major format (N
       * x M). Sc is the overlapMatPar.
       *
       * The overlap matrix computation and filling is done in a blocked
       * approach which avoids creation of full serial overlap matrix memory,
       * and also avoids creation of another full X memory.
       *
       */
      template <typename T>
      void
      fillParallelOverlapMatrix(
        const T *X,
        /*const std::shared_ptr<
          linearAlgebra::BLASWrapper<utils::MemorySpace::HOST>>
          &                                              BLASWrapperPtr,*/
        const size_type                           XLocalSize,
        const size_type                           numberVectors,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const utils::mpi::MPIComm &               mpiComm,
        const size_type                           wfcBlockSize,
        ScaLAPACKMatrix<T> &                      overlapMatPar);

      /** @brief Computes X^{T}=Q*X^{T} inplace. X^{T} is the subspaceVectorsArray
       * stored in the column major format (N x M). Q is rotationMatPar (N x N).
       *
       * The subspace rotation inside this function is done in a blocked
       * approach which avoids creation of full serial rotation matrix memory,
       * and also avoids creation of another full subspaceVectorsArray memory.
       * subspaceVectorsArrayLocalSize=N*M
       *
       */
      template <typename T>
      void
      subspaceRotation(
      ValueType *X,
      const size_type  M,
      const size_type  N,
      /*std::shared_ptr<
        linearAlgebra::BLASWrapper<memorySpace>> &BLASWrapperPtr,*/
      const std::shared_ptr<const ProcessGrid> &processGrid,
      const MPI_Comm                                  &mpiCommDomain,
      const ScaLAPACKMatrix<ValueType> &rotationMatPar,
      const size_type                   subspaceRotDofsBlockSize,
      const size_type                   wfcBlockSize,
      const bool                       rotationMatTranspose = false,
      const bool                       isRotationMatLowerTria = false)

    } // namespace elpaScalaOpInternal
  }   // namespace linearAlgebra
} // namespace dftefe
#endif
