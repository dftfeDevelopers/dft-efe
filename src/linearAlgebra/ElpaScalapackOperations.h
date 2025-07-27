// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------

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
    /**
     *  @brief Contains internal functions used in linearAlgebra
     *
     *  @author Sambit Das
     */
    /** @brief setup ELPA parameters.
     *
     */
    void
    setupELPAHandleParameters(
      const utils::mpi::MPIComm &mpi_communicator,
      utils::mpi::MPIComm &      processGridCommunicatorActive,
      const std::shared_ptr<const ProcessGrid> &processGrid,
      const size_type                                na,
      const size_type                                nev,
      const size_type                                blockSize,
      elpa_t &                                         elpaHandle,
      const dftParameters &                            dftParams);

    /** @brief Wrapper function to create a two dimensional processor grid for a square matrix in
     * ScaLAPACKMatrix storage format.
     *
     */
    void
    createProcessGridSquareMatrix(
      const utils::mpi::MPIComm &                mpi_communicator,
      const size_type                          size,
      std::shared_ptr<const ProcessGrid> &processGrid,
      const dftParameters &                      dftParams,
      const bool                                 useOnlyThumbRule = false);

    /** @brief Wrapper function to create a two dimensional processor grid for a rectangular matrix in
     * ScaLAPACKMatrix storage format.
     *
     */
    void
    createProcessGridRectangularMatrix(
      const utils::mpi::MPIComm &                mpi_communicator,
      const size_type                          sizeRows,
      const size_type                          sizeColumns,
      std::shared_ptr<const ProcessGrid> &processGrid,
      const dftParameters &                      dftParams);


    /** @brief Creates global row/column id to local row/column ids for ScaLAPACKMatrix
     *
     */
    template <typename T>
    void
    createGlobalToLocalIdMapsScaLAPACKMat(
      const std::shared_ptr<const ProcessGrid> &processGrid,
      const ScaLAPACKMatrix<T> &                mat,
      std::unordered_map<size_type, size_type> &   globalToLocalRowIdMap,
      std::unordered_map<size_type, size_type> &globalToLocalColumnIdMap);


    /** @brief Mpi all reduce of ScaLAPACKMat across a given inter communicator.
     * Used for band parallelization.
     *
     */
    template <typename T>
    void
    sumAcrossInterCommScaLAPACKMat(
      const std::shared_ptr<const ProcessGrid> &processGrid,
      ScaLAPACKMatrix<T> &                      mat,
      const utils::mpi::MPIComm &                      interComm);


    /** @brief scale a ScaLAPACKMat with a scalar
     *
     *
     */
    template <typename T>
    void
    scaleScaLAPACKMat(
      const std::shared_ptr<const ProcessGrid> &processGrid,
      /*const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &                        BLASWrapperPtr,*/
      ScaLAPACKMatrix<T> &mat,
      const T                    scalar);


    /** @brief MPI_Bcast of ScaLAPACKMat across a given inter communicator from a given broadcast root.
     * Used for band parallelization.
     *
     */
    template <typename T>
    void
    broadcastAcrossInterCommScaLAPACKMat(
      const std::shared_ptr<const ProcessGrid> &processGrid,
      ScaLAPACKMatrix<T> &                      mat,
      const utils::mpi::MPIComm &                      interComm,
      const size_type                                broadcastRoot);

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
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &                                              BLASWrapperPtr,*/
      const size_type                                XLocalSize,
      const size_type                                numberVectors,
      const std::shared_ptr<const ProcessGrid> &processGrid,
      const utils::mpi::MPIComm &                      mpiComm,
      ScaLAPACKMatrix<T> &                      overlapMatPar,
      const dftParameters &                            dftParams);


    /** @brief Computes Sc=X^{T}*Xc and stores in a parallel ScaLAPACK matrix.
     * X^{T} is the subspaceVectorsArray stored in the column major format (N
     * x M). Sc is the overlapMatPar.
     *
     * The overlap matrix computation and filling is done in a blocked
     * approach which avoids creation of full serial overlap matrix memory,
     * and also avoids creation of another full X memory.
     *
     */
    template <typename T, typename TLowPrec>
    void
    fillParallelOverlapMatrixMixedPrec(
      const T *X,
      /*const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &                                              BLASWrapperPtr,*/
      const size_type                                XLocalSize,
      const size_type                                numberVectors,
      const std::shared_ptr<const ProcessGrid> &processGrid,
      const utils::mpi::MPIComm &                      mpiComm,
      ScaLAPACKMatrix<T> &                      overlapMatPar,
      const dftParameters &                            dftParams);
  } // namespace linearAlgebra
} // namespace dftefe
#endif
