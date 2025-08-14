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

// @author Sambit Das, Aviup Sircar
//
#include "ElpaScalapackManager.h"
//
// Constructor.
//
namespace dftefe
{
  namespace linearAlgebra
  {
    ElpaScalapackManager::ElpaScalapackManager(
      const utils::mpi::MPIComm &mpi_comm_replica,
      const size_type            scalapackParalProcs,
      const bool                 useELPA,
      const size_type            scalapackBlockSize,
      const bool                 useELPADeviceKernel)
      : d_mpi_communicator(mpi_comm_replica)
      , d_processGridCommunicatorActive(utils::mpi::MPICommNull)
      , d_processGridCommunicatorActivePartial(utils::mpi::MPICommNull)
      , d_scalapackBlockSizeInp(scalapackBlockSize)
      , d_useELPA(useELPA)
      , d_scalapackParalProcs(scalapackParalProcs)
      , d_useELPADeviceKernel(useELPADeviceKernel)
    {}


    //
    // Destructor.
    //
    ElpaScalapackManager::~ElpaScalapackManager()
    {
      int mpi_initialized = 0, mpi_finalized = 0;
      utils::mpi::MPIInitialized(&mpi_initialized);
      utils::mpi::MPIFinalized(&mpi_finalized);

      if (mpi_initialized && !mpi_finalized)
        {
          if (d_processGridCommunicatorActive != utils::mpi::MPICommNull)
            utils::mpi::MPICommFree(&d_processGridCommunicatorActive);
          d_processGridCommunicatorActive = utils::mpi::MPICommNull;

          if (d_processGridCommunicatorActivePartial != utils::mpi::MPICommNull)
            utils::mpi::MPICommFree(&d_processGridCommunicatorActivePartial);
          d_processGridCommunicatorActivePartial = utils::mpi::MPICommNull;
        }

      return;
    }
    //
    // Get relevant mpi communicator
    //
    const utils::mpi::MPIComm &
    ElpaScalapackManager::getMPICommunicator() const
    {
      return d_mpi_communicator;
    }


    void
    ElpaScalapackManager::processGridELPASetup(const unsigned int na)
    {
      linearAlgebra::elpaScalaOpInternal::createProcessGridSquareMatrix(
        getMPICommunicator(),
        na,
        d_processGridDftefeWrapper,
        d_scalapackParalProcs,
        d_useELPA);


      d_scalapackBlockSize =
        std::min(d_scalapackBlockSizeInp,
                 size_type(
                   (na + d_processGridDftefeWrapper->get_process_grid_rows() -
                    1) /
                   d_processGridDftefeWrapper->get_process_grid_rows()));

      if (d_useELPA)
        {
          linearAlgebra::elpaScalaOpInternal::setupELPAHandleParameters(
            getMPICommunicator(),
            d_processGridCommunicatorActive,
            d_processGridDftefeWrapper,
            na,
            na,
            d_scalapackBlockSize,
            d_elpaHandle,
            d_useELPADeviceKernel);
        }

      // std::cout<<"nblk: "<<d_scalapackBlockSize<<std::endl;
    }

    void
    ElpaScalapackManager::elpaDeallocateHandles()
    {
      if (d_useELPA)
        {
          int error;
          if (d_processGridCommunicatorActive != utils::mpi::MPICommNull)
            {
              elpa_deallocate(d_elpaHandle, &error);
              DFTEFE_AssertWithMsg(error == ELPA_OK,
                                   "DFT-EFE Error: elpa error.");
            }

          if (d_processGridCommunicatorActivePartial != utils::mpi::MPICommNull)
            {
              elpa_deallocate(d_elpaHandlePartialEigenVec, &error);
              DFTEFE_AssertWithMsg(error == ELPA_OK,
                                   "DFT-EFE Error: elpa error.");
            }
        }
    }
  } // namespace linearAlgebra
} // namespace dftefe
