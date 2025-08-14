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
#include "ElpaScalapackOperations.h"
#include <utils/ConditionalOStream.h>
#include "BlasLapack.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace elpaScalaOpInternal
    {
      void
      setupELPAHandleParameters(
        const utils::mpi::MPIComm &               mpi_communicator,
        utils::mpi::MPIComm &                     processGridCommunicatorActive,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const size_type                           na,
        const size_type                           nev,
        const size_type                           blockSize,
        elpa_t &                                  elpaHandle,
        const bool                                useELPADeviceKernel)
      {
        int error;

        if (processGrid->is_process_active())
          {
            elpaHandle = elpa_allocate(&error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));
          }

        // Get the group of processes in mpi_communicator
        int                  ierr = 0;
        utils::mpi::MPIGroup all_group;
        ierr = utils::mpi::MPICommGroup(mpi_communicator, &all_group);
        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(ierr);
        DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                             "MPI Error:" + mpiIsSuccessAndMsg.second);

        // Construct the group containing all ranks we need:
        const size_type n_active_mpi_processes =
          processGrid->get_process_grid_rows() *
          processGrid->get_process_grid_columns();
        std::vector<int> active_ranks(0);
        for (size_type i = 0; i < n_active_mpi_processes; ++i)
          active_ranks.push_back(i);

        utils::mpi::MPIGroup active_group;
        const int            n = active_ranks.size();
        ierr                   = utils::mpi::MPIGroupIncl(all_group,
                                        n,
                                        active_ranks.data(),
                                        &active_group);
        mpiIsSuccessAndMsg     = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
        DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                             "MPI Error:" + mpiIsSuccessAndMsg.second);

        // Create the communicator based on active_group.
        // Note that on all the inactive processs the resulting
        // utils::mpi::MPIComm processGridCommunicatorActive will be
        // utils::mpi::MPICommNull. utils::mpi::MPIComm
        // processGridCommunicatorActive;
        ierr               = utils::mpi::MPICommCreateGroup(mpi_communicator,
                                              active_group,
                                              50,
                                              &processGridCommunicatorActive);
        mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
        DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                             "MPI Error:" + mpiIsSuccessAndMsg.second);

        ierr               = utils::mpi::MPIGroupFree(&all_group);
        mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
        DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                             "MPI Error:" + mpiIsSuccessAndMsg.second);
        ierr               = utils::mpi::MPIGroupFree(&active_group);
        mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(ierr);
        DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                             "MPI Error:" + mpiIsSuccessAndMsg.second);


        ScaLAPACKMatrix<double> tempMat(na, processGrid, blockSize);
        if (processGrid->is_process_active())
          {
            /* Set parameters the matrix and it's MPI distribution */
            elpa_set_integer(elpaHandle, "na", na, &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));


            elpa_set_integer(elpaHandle, "nev", nev, &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

            elpa_set_integer(elpaHandle, "nblk", blockSize, &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "mpi_comm_parent",
                             MPI_Comm_c2f(processGridCommunicatorActive),
                             &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));


            // std::cout<<"local_nrows: "<<tempMat.local_m() <<std::endl;
            // std::cout<<"local_ncols: "<<tempMat.local_n() <<std::endl;
            // std::cout<<"process_row:"<<processGrid->get_this_process_row()<<std::endl;
            // std::cout<<"process_col:"<<processGrid->get_this_process_column()<<std::endl;

            elpa_set_integer(elpaHandle,
                             "local_nrows",
                             tempMat.local_m(),
                             &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "local_ncols",
                             tempMat.local_n(),
                             &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "process_row",
                             processGrid->get_this_process_row(),
                             &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "process_col",
                             processGrid->get_this_process_column(),
                             &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

            elpa_set(elpaHandle,
                     "blacs_context",
                     processGrid->get_blacs_context(),
                     &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

            elpa_set(elpaHandle, "cannon_for_generalized", 0, &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

            /* Setup */
            if (elpa_setup(elpaHandle) != ELPA_OK)
              {
                utils::throwException(false, ("DFT-EFE Error: ELPA Error."));
              }

            // #ifdef DFTFE_WITH_DEVICE

            //             if (useELPADeviceKernel)
            //               {
            // #  ifdef DFTFE_WITH_DEVICE_NVIDIA
            //                 elpa_set_integer(elpaHandle, "nvidia-gpu", 1,
            //                 &error); DFTEFE_AssertWithMsg(error == ELPA_OK,
            //                             ("DFT-EFE Error: ELPA Error."));
            // #  elif DFTFE_WITH_DEVICE_AMD
            //                 elpa_set_integer(elpaHandle, "amd-gpu", 1,
            //                 &error); DFTEFE_AssertWithMsg(error == ELPA_OK,
            //                             ("DFT-EFE Error: ELPA Error."));
            // #  endif
            //                 elpa_set_integer(elpaHandle,
            //                                  "solver",
            //                                  ELPA_SOLVER_1STAGE,
            //                                  &error);
            //                 DFTEFE_AssertWithMsg(error == ELPA_OK,
            //                             ("DFT-EFE Error: ELPA Error."));

            //                 int gpuID = 0;
            //                 utils::getDevice(&gpuID);

            //                 elpa_set_integer(elpaHandle, "use_gpu_id", gpuID,
            //                 &error); DFTEFE_AssertWithMsg(error == ELPA_OK,
            //                             ("DFT-EFE Error: ELPA Error."));

            //                 error = elpa_setup_gpu(elpaHandle);
            //                 DFTEFE_AssertWithMsg(error == ELPA_OK,
            //                             ("DFT-EFE Error: ELPA Error."));
            //               }
            //             else
            //               {
            //                 elpa_set_integer(elpaHandle,
            //                                  "solver",
            //                                  ELPA_SOLVER_2STAGE,
            //                                  &error);
            //                 DFTEFE_AssertWithMsg(error == ELPA_OK,
            //                             ("DFT-EFE Error: ELPA Error."));
            //               }
            // #else
            elpa_set_integer(elpaHandle, "solver", ELPA_SOLVER_2STAGE, &error);
            DFTEFE_AssertWithMsg(error == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));
            // #endif

            // elpa_set_integer(elpaHandle,
            // "real_kernel",ELPA_2STAGE_REAL_AVX512_BLOCK6, &error);
            // DFTEFE_AssertWithMsg(error==ELPA_OK,
            //   ("DFT-EFE Error: ELPA Error."));

            // #ifdef DEBUG
            //             elpa_set_integer(elpaHandle, "debug", 1, &error);
            //             DFTEFE_AssertWithMsg(error == ELPA_OK,
            //                         ("DFT-EFE Error: ELPA Error."));
            // #endif
          }

        // d_elpaAutoTuneHandle = elpa_autotune_setup(d_elpaHandle,
        // ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, &error);   // create
        // autotune object
      }

      void
      createProcessGridSquareMatrix(
        const utils::mpi::MPIComm &         mpi_communicator,
        const size_type                     size,
        std::shared_ptr<const ProcessGrid> &processGrid,
        const size_type                     scalapackParalProcs,
        const bool                          useELPA,
        const bool                          useOnlyThumbRule)
      {
        const size_type numberProcs =
          utils::mpi::numMPIProcesses(mpi_communicator);

        // Rule of thumb from
        // http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
        size_type rowProcs =
          (scalapackParalProcs == 0 || useOnlyThumbRule) ?
            std::min(std::floor(std::sqrt(numberProcs)),
                     std::ceil((double)size / (double)(1000))) :
            std::min((size_type)std::floor(std::sqrt(numberProcs)),
                     scalapackParalProcs);

        rowProcs = ((scalapackParalProcs == 0 || useOnlyThumbRule) && useELPA) ?
                     std::min((size_type)std::floor(std::sqrt(numberProcs)),
                              (size_type)std::floor(rowProcs * 3.0)) :
                     rowProcs;

        rowProcs = std::min(rowProcs,
                            (size_type)std::ceil((double)size / (double)(100)));

        utils::ConditionalOStream rootCout(std::cout);
        rootCout.setCondition(utils::mpi::thisMPIProcess(mpi_communicator) ==
                              0);
        rootCout << "Scalapack Matrix created, row procs: " << rowProcs
                 << std::endl;

        processGrid = std::make_shared<const ProcessGrid>(mpi_communicator,
                                                          rowProcs,
                                                          rowProcs);
      }


      void
      createProcessGridRectangularMatrix(
        const utils::mpi::MPIComm &         mpi_communicator,
        const size_type                     sizeRows,
        const size_type                     sizeColumns,
        std::shared_ptr<const ProcessGrid> &processGrid)
      {
        const size_type numberProcs =
          utils::mpi::numMPIProcesses(mpi_communicator);

        // Rule of thumb from
        // http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
        const size_type rowProcs =
          std::min(std::floor(std::sqrt(numberProcs)),
                   std::ceil((double)sizeRows / (double)(1000)));
        const size_type columnProcs =
          std::min(std::floor(std::sqrt(numberProcs)),
                   std::ceil((double)sizeColumns / (double)(1000)));

        utils::ConditionalOStream rootCout(std::cout);
        rootCout.setCondition(utils::mpi::thisMPIProcess(mpi_communicator) ==
                              0);
        rootCout << "Scalapack Matrix created, row procs x column procs: "
                 << rowProcs << " x " << columnProcs << std::endl;

        processGrid = std::make_shared<const ProcessGrid>(mpi_communicator,
                                                          rowProcs,
                                                          columnProcs);
      }
    } // namespace elpaScalaOpInternal
  }   // namespace linearAlgebra
} // namespace dftefe
