
//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// --------------------------------------------------------------------------------------
//
// @author Sambit Das
//
#include <elpaScalaManager.h>
#include <linearAlgebraOperationsInternal.h>

//
// Constructor.
//
namespace dftfe
{
  elpaScalaManager::elpaScalaManager(const utils::mpi::MPIComm &mpi_comm_replica)
    : d_mpi_communicator(mpi_comm_replica)
    , d_processGridCommunicatorActive(MPI_COMM_NULL)
    , d_processGridCommunicatorActivePartial(MPI_COMM_NULL)
  {}


  //
  // Destructor.
  //
  elpaScalaManager::~elpaScalaManager()
  {
    if (d_processGridCommunicatorActive != MPI_COMM_NULL)
      MPI_Comm_free(&d_processGridCommunicatorActive);

    if (d_processGridCommunicatorActivePartial != MPI_COMM_NULL)
      MPI_Comm_free(&d_processGridCommunicatorActivePartial);
    //
    //
    //
    return;
  }
  //
  // Get relevant mpi communicator
  //
  const utils::mpi::MPIComm &
  elpaScalaManager::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }


  void
  elpaScalaManager::processGridELPASetup(const unsigned int   na)
  {
    linearAlgebraOperations::internal::createProcessGridSquareMatrix(
      getMPICommunicator(), na, d_processGridDftfeWrapper, dftParams);


    d_scalapackBlockSize =
      std::min(dftParams.scalapackBlockSize,
               dftfe::uInt(
                 (na + d_processGridDftfeWrapper->get_process_grid_rows() - 1) /
                 d_processGridDftfeWrapper->get_process_grid_rows()));

    if (dftParams.useELPA)
      {
        linearAlgebraOperations::internal::setupELPAHandleParameters(
          getMPICommunicator(),
          d_processGridCommunicatorActive,
          d_processGridDftfeWrapper,
          na,
          na,
          d_scalapackBlockSize,
          d_elpaHandle,
          dftParams);
      }

    // std::cout<<"nblk: "<<d_scalapackBlockSize<<std::endl;
  }

  void
  elpaScalaManager::elpaDeallocateHandles()
  {
    if (dftParams.useELPA)
      {
        int error;
        if (d_processGridCommunicatorActive != MPI_COMM_NULL)
          {
            elpa_deallocate(d_elpaHandle, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: elpa error."));
          }

        if (d_processGridCommunicatorActivePartial != MPI_COMM_NULL)
          {
            elpa_deallocate(d_elpaHandlePartialEigenVec, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: elpa error."));
          }
      }
  }
} // namespace dftfe