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
        std::vector<int> active_ranks;
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
            // std::cout<<"process_row:
            // "<<processGrid->get_this_process_row()<<std::endl;
            // std::cout<<"process_col:
            // "<<processGrid->get_this_process_column()<<std::endl;

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
            DFTEFE_AssertWithMsg(elpa_setup(elpaHandle) == ELPA_OK,
                                 ("DFT-EFE Error: ELPA Error."));

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
            //                 dftfe::utils::getDevice(&gpuID);

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


      template <typename T>
      void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const ScaLAPACKMatrix<T> &                mat,
        std::unordered_map<size_type, size_type> &globalToLocalRowIdMap,
        std::unordered_map<size_type, size_type> &globalToLocalColumnIdMap)
      {
        globalToLocalRowIdMap.clear();
        globalToLocalColumnIdMap.clear();
        if (processGrid->is_process_active())
          {
            for (size_type i = 0; i < mat.local_m(); ++i)
              globalToLocalRowIdMap[mat.global_row(i)] = i;

            for (size_type j = 0; j < mat.local_n(); ++j)
              globalToLocalColumnIdMap[mat.global_column(j)] = j;
          }
      }


      template <typename T>
      void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        ScaLAPACKMatrix<T> &                      mat,
        const utils::mpi::MPIComm &               interComm)
      {
        // sum across all inter communicator groups
        if (processGrid->is_process_active() &&
            utils::mpi::numMPIProcesses(interComm) > 1)
          {
            utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              utils::mpi::MPIInPlace,
              &mat.local_el(0, 0),
              mat.local_m() * mat.local_n(),
              utils::mpi::Types<T>::getMPIDatatype(),
              utils::mpi::MPISum,
              interComm);
          }
      }

      template <typename T>
      void
      scaleScaLAPACKMat(const std::shared_ptr<const ProcessGrid> &processGrid,
                        ScaLAPACKMatrix<T> &                      mat,
                        const T                                   scalar)
      {
        // if (processGrid->is_process_active())
        //   {
        //     const size_type numberComponents = mat.local_m() *
        //     mat.local_n(); const size_type inc              = 1;
        //     xscal(&numberComponents, &scalar, &mat.local_el(0, 0), &inc);
        //   }
      }



      template <typename T>
      void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        ScaLAPACKMatrix<T> &                      mat,
        const utils::mpi::MPIComm &               interComm,
        const size_type                           broadcastRoot)
      {
        // sum across all inter communicator groups
        if (processGrid->is_process_active() &&
            utils::mpi::numMPIProcesses(interComm) > 1)
          {
            utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
              &mat.local_el(0, 0),
              mat.local_m() * mat.local_n(),
              utils::mpi::Types<T>::getMPIDatatype(),
              broadcastRoot,
              interComm);
          }
      }

      template <typename T, typename TLowPrec>
      void
      fillParallelOverlapMatrixMixedPrec(
        const T *                                 subspaceVectorsArray,
        const size_type                           subspaceVectorsArrayLocalSize,
        const size_type                           N,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const utils::mpi::MPIComm &               mpiComm,
        const size_type                           wfcBlockSize,
        ScaLAPACKMatrix<T> &                      overlapMatPar)
      {
        const size_type numLocalDofs = subspaceVectorsArrayLocalSize / N;

        // // band group parallelization data structures
        // const size_type numberBandGroups =
        //   utils::mpi::numMPIProcesses(interBandGroupComm);
        // const size_type bandGroupTaskId =
        //   utils::mpi::thisMPIProcess(interBandGroupComm);
        // std::vector<size_type> bandGroupLowHighPlusOneIndices;
        // dftUtils::createBandParallelizationIndices(
        //   interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        // get global to local index maps for Scalapack matrix
        std::unordered_map<size_type, size_type> globalToLocalColumnIdMap;
        std::unordered_map<size_type, size_type> globalToLocalRowIdMap;
        elpaScalaOpInternal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          overlapMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);


        /*
         * Sc=X^{T}*Xc is done in a blocked approach for memory optimization:
         * Sum_{blocks} X^{T}*XcBlock. The result of each X^{T}*XBlock
         * has a much smaller memory compared to X^{T}*Xc.
         * X^{T} is a matrix with size number of wavefunctions times
         * number of local degrees of freedom (N x MLoc).
         * MLoc is denoted by numLocalDofs.
         * Xc denotes complex conjugate of X.
         * XcBlock is a matrix with size (MLoc x B). B is the block size.
         * A further optimization is done to reduce floating point operations:
         * As X^{T}*Xc is a Hermitian matrix, it suffices to compute only the
         * lower triangular part. To exploit this, we do X^{T}*Xc=Sum_{blocks}
         * XTrunc^{T}*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of
         * X^{T} with the row indices ranging fromt the lowest global index of
         * XcBlock (denoted by ivec in the code) to N. D=N-ivec. The parallel
         * ScaLapack overlap matrix is directly filled from the
         * XTrunc^{T}*XcBlock result
         */
        const size_type vectorsBlockSize =
          wfcBlockSize; //, bandGroupLowHighPlusOneIndices[1]);

        std::vector<T>        overlapMatrixBlock(N * vectorsBlockSize, T(0.0));
        std::vector<TLowPrec> overlapMatrixBlockLowPrec(N * vectorsBlockSize,
                                                        TLowPrec(0.0));
        std::vector<T>        overlapMatrixBlockDoublePrec(vectorsBlockSize *
                                                      vectorsBlockSize,
                                                    T(0.0));

        std::vector<TLowPrec> subspaceVectorsArrayLowPrec(
          subspaceVectorsArray,
          subspaceVectorsArray + subspaceVectorsArrayLocalSize);
        for (size_type ivec = 0; ivec < N; ivec += vectorsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            const size_type B = std::min(vectorsBlockSize, N - ivec);

            // // If one plus the ending index of a block lies within a band
            // // parallelization group do computations for that block within
            // the
            // // band group, otherwise skip that block. This is only activated
            // if
            // // NPBAND>1
            // if ((ivec + B) <=
            //       bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            //     (ivec + B) >
            //       bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            //   {
            blasLapack::Op
              transA = blasLapack::Op::NoTrans /*'N'*/,
              transB = std::is_same<T, std::complex<double>>::value ?
                         linearAlgebra::blasLapack::Op::ConjTrans /*'C'*/ :
                         linearAlgebra::blasLapack::Op::Trans /*'T'*/;
            const T        scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
            const TLowPrec scalarCoeffAlphaLowPrec = 1.0,
                           scalarCoeffBetaLowPrec  = 0.0;

            std::fill(overlapMatrixBlock.begin(), overlapMatrixBlock.end(), 0.);
            std::fill(overlapMatrixBlockLowPrec.begin(),
                      overlapMatrixBlockLowPrec.end(),
                      0.);

            const size_type D = N - ivec;

            blasLapack::gemm<T, T, utils::MemorySpace::HOST>(
              blasLapack::Layout::ColMajor,
              transA,
              transB,
              B,
              B,
              numLocalDofs,
              scalarCoeffAlpha,
              subspaceVectorsArray + ivec,
              N,
              subspaceVectorsArray + ivec,
              N,
              scalarCoeffBeta,
              &overlapMatrixBlockDoublePrec[0],
              B,
              *LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);

            const size_type DRem = D - B;
            if (DRem != 0)
              {
                blasLapack::gemm<TLowPrec, TLowPrec, utils::MemorySpace::HOST>(
                  blasLapack::Layout::ColMajor,
                  transA,
                  transB,
                  DRem,
                  B,
                  numLocalDofs,
                  scalarCoeffAlphaLowPrec,
                  &subspaceVectorsArrayLowPrec[0] + ivec + B,
                  N,
                  &subspaceVectorsArrayLowPrec[0] + ivec,
                  N,
                  scalarCoeffBetaLowPrec,
                  &overlapMatrixBlockLowPrec[0],
                  DRem,
                  *LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);
              }

            utils::mpi::MPIBarrier(mpiComm);
            // Sum local XTrunc^{T}*XcBlock for double precision across
            // domain decomposition processors
            utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              utils::mpi::MPIInPlace,
              &overlapMatrixBlockDoublePrec[0],
              B * B,
              utils::mpi::Types<T>::getMPIDatatype(),
              utils::mpi::MPISum,
              mpiComm);

            utils::mpi::MPIBarrier(mpiComm);
            // Sum local XTrunc^{T}*XcBlock for single precision across
            // domain decomposition processors
            utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              utils::mpi::MPIInPlace,
              &overlapMatrixBlockLowPrec[0],
              DRem * B,
              utils::mpi::Types<T>::getMPIDatatype(),
              utils::mpi::MPISum,
              mpiComm);

            for (size_type i = 0; i < B; ++i)
              {
                for (size_type j = 0; j < B; ++j)
                  overlapMatrixBlock[i * D + j] =
                    overlapMatrixBlockDoublePrec[i * B + j];

                for (size_type j = 0; j < DRem; ++j)
                  overlapMatrixBlock[i * D + j + B] =
                    overlapMatrixBlockLowPrec[i * DRem + j];
              }

            // Copying only the lower triangular part to the ScaLAPACK
            // overlap matrix
            if (processGrid->is_process_active())
              for (size_type i = 0; i < B; ++i)
                if (globalToLocalColumnIdMap.find(i + ivec) !=
                    globalToLocalColumnIdMap.end())
                  {
                    const size_type localColumnId =
                      globalToLocalColumnIdMap[i + ivec];
                    for (size_type j = ivec + i; j < N; ++j)
                      {
                        std::unordered_map<size_type, size_type>::iterator it =
                          globalToLocalRowIdMap.find(j);
                        if (it != globalToLocalRowIdMap.end())
                          overlapMatPar.local_el(it->second, localColumnId) =
                            overlapMatrixBlock[i * D + j - ivec];
                      }
                  }
            // } // band parallelization
          } // block loop


        // accumulate contribution from all band parallelization groups
        // linearAlgebra::elpaScalaOpInternal::sumAcrossInterCommScaLAPACKMat(
        //   processGrid, overlapMatPar, interBandGroupComm);
      }


      template <typename T>
      void
      fillParallelOverlapMatrix(
        const T *                                 subspaceVectorsArray,
        const size_type                           subspaceVectorsArrayLocalSize,
        const size_type                           N,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const utils::mpi::MPIComm &               mpiComm,
        const size_type                           wfcBlockSize,
        ScaLAPACKMatrix<T> &                      overlapMatPar)
      {
        const size_type numLocalDofs = subspaceVectorsArrayLocalSize / N;

        // // band group parallelization data structures
        // const size_type numberBandGroups =
        //   utils::mpi::numMPIProcesses(interBandGroupComm);
        // const size_type bandGroupTaskId =
        //   utils::mpi::thisMPIProcess(interBandGroupComm);
        // std::vector<size_type> bandGroupLowHighPlusOneIndices;
        // dftUtils::createBandParallelizationIndices(
        //   interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        // get global to local index maps for Scalapack matrix
        std::unordered_map<size_type, size_type> globalToLocalColumnIdMap;
        std::unordered_map<size_type, size_type> globalToLocalRowIdMap;
        elpaScalaOpInternal::createGlobalToLocalIdMapsScaLAPACKMat(
          processGrid,
          overlapMatPar,
          globalToLocalRowIdMap,
          globalToLocalColumnIdMap);


        /*
         * Sc=X^{T}*Xc is done in a blocked approach for memory optimization:
         * Sum_{blocks} X^{T}*XcBlock. The result of each X^{T}*XBlock
         * has a much smaller memory compared to X^{T}*Xc.
         * X^{T} is a matrix with size number of wavefunctions times
         * number of local degrees of freedom (N x MLoc).
         * MLoc is denoted by numLocalDofs.
         * Xc denotes complex conjugate of X.
         * XcBlock is a matrix with size (MLoc x B). B is the block size.
         * A further optimization is done to reduce floating point operations:
         * As X^{T}*Xc is a Hermitian matrix, it suffices to compute only the
         * lower triangular part. To exploit this, we do X^{T}*Xc=Sum_{blocks}
         * XTrunc^{T}*XcBlock where XTrunc^{T} is a (D x MLoc) sub matrix of
         * X^{T} with the row indices ranging fromt the lowest global index of
         * XcBlock (denoted by ivec in the code) to N. D=N-ivec. The parallel
         * ScaLapack overlap matrix is directly filled from the
         * XTrunc^{T}*XcBlock result
         */
        const size_type vectorsBlockSize =
          wfcBlockSize; //, bandGroupLowHighPlusOneIndices[1]);

        std::vector<T> overlapMatrixBlock(N * vectorsBlockSize, 0.0);

        for (size_type ivec = 0; ivec < N; ivec += vectorsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            const size_type B = std::min(vectorsBlockSize, N - ivec);

            // // If one plus the ending index of a block lies within a band
            // // parallelization group do computations for that block within
            // the
            // // band group, otherwise skip that block. This is only activated
            // if
            // // NPBAND>1
            // if ((ivec + B) <=
            //       bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            //     (ivec + B) >
            //       bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            //   {
            blasLapack::Op
              transA = blasLapack::Op::NoTrans /*'N'*/,
              transB = std::is_same<T, std::complex<double>>::value ?
                         linearAlgebra::blasLapack::Op::ConjTrans /*'C'*/ :
                         linearAlgebra::blasLapack::Op::Trans /*'T'*/;
            const T scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;

            std::fill(overlapMatrixBlock.begin(), overlapMatrixBlock.end(), 0.);

            const size_type D = N - ivec;

            // Comptute local XTrunc^{T}*XcBlock.
            blasLapack::gemm<T, T, utils::MemorySpace::HOST>(
              blasLapack::Layout::ColMajor,
              transA,
              transB,
              D,
              B,
              numLocalDofs,
              scalarCoeffAlpha,
              subspaceVectorsArray + ivec,
              N,
              subspaceVectorsArray + ivec,
              N,
              scalarCoeffBeta,
              &overlapMatrixBlock[0],
              D,
              *LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST);

            utils::mpi::MPIBarrier(mpiComm);
            // Sum local XTrunc^{T}*XcBlock across domain decomposition
            // processors
            utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              utils::mpi::MPIInPlace,
              &overlapMatrixBlock[0],
              D * B,
              utils::mpi::Types<T>::getMPIDatatype(),
              utils::mpi::MPISum,
              mpiComm);

            // Copying only the lower triangular part to the ScaLAPACK
            // overlap matrix
            if (processGrid->is_process_active())
              for (size_type i = 0; i < B; ++i)
                if (globalToLocalColumnIdMap.find(i + ivec) !=
                    globalToLocalColumnIdMap.end())
                  {
                    const size_type localColumnId =
                      globalToLocalColumnIdMap[i + ivec];
                    for (size_type j = ivec + i; j < N; ++j)
                      {
                        std::unordered_map<size_type, size_type>::iterator it =
                          globalToLocalRowIdMap.find(j);
                        if (it != globalToLocalRowIdMap.end())
                          overlapMatPar.local_el(it->second, localColumnId) =
                            overlapMatrixBlock[i * D + j - ivec];
                      }
                  }
            // } // band parallelization
          } // block loop


        // accumulate contribution from all band parallelization groups
        // linearAlgebra::elpaScalaOpInternal::sumAcrossInterCommScaLAPACKMat(
        //   processGrid, overlapMatPar, interBandGroupComm);
      }

      template void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const ScaLAPACKMatrix<double> &           mat,
        std::unordered_map<size_type, size_type> &globalToLocalRowIdMap,
        std::unordered_map<size_type, size_type> &globalToLocalColumnIdMap);

      template void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &   processGrid,
        const ScaLAPACKMatrix<std::complex<double>> &mat,
        std::unordered_map<size_type, size_type> &   globalToLocalRowIdMap,
        std::unordered_map<size_type, size_type> &   globalToLocalColumnIdMap);

      template void
      fillParallelOverlapMatrix(
        const double *                            X,
        const size_type                           XLocalSize,
        const size_type                           numberVectors,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const utils::mpi::MPIComm &               mpiComm,
        const size_type                           wfcBlockSize,
        ScaLAPACKMatrix<double> &                 overlapMatPar);

      template void
      fillParallelOverlapMatrix(
        const std::complex<double> *              X,
        const size_type                           XLocalSize,
        const size_type                           numberVectors,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const utils::mpi::MPIComm &               mpiComm,
        const size_type                           wfcBlockSize,
        ScaLAPACKMatrix<std::complex<double>> &   overlapMatPar);


      template void
      fillParallelOverlapMatrixMixedPrec<double, float>(
        const double *                            X,
        const size_type                           XLocalSize,
        const size_type                           numberVectors,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const utils::mpi::MPIComm &               mpiComm,
        const size_type                           wfcBlockSize,
        ScaLAPACKMatrix<double> &                 overlapMatPar);

      template void
      fillParallelOverlapMatrixMixedPrec<std::complex<double>,
                                         std::complex<float>>(
        const std::complex<double> *              X,
        const size_type                           XLocalSize,
        const size_type                           numberVectors,
        const std::shared_ptr<const ProcessGrid> &processGrid,
        const utils::mpi::MPIComm &               mpiComm,
        const size_type                           wfcBlockSize,
        ScaLAPACKMatrix<std::complex<double>> &   overlapMatPar);

      template void
      scaleScaLAPACKMat(const std::shared_ptr<const ProcessGrid> &processGrid,
                        ScaLAPACKMatrix<double> &                 mat,
                        const double                              scalar);

      template void
      scaleScaLAPACKMat(const std::shared_ptr<const ProcessGrid> &processGrid,
                        ScaLAPACKMatrix<std::complex<double>> &   mat,
                        const std::complex<double>                scalar);

      template void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        ScaLAPACKMatrix<double> &                 mat,
        const utils::mpi::MPIComm &               interComm);

      template void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        ScaLAPACKMatrix<std::complex<double>> &   mat,
        const utils::mpi::MPIComm &               interComm);

      template void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        ScaLAPACKMatrix<double> &                 mat,
        const utils::mpi::MPIComm &               interComm,
        const size_type                           broadcastRoot);

      template void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const ProcessGrid> &processGrid,
        ScaLAPACKMatrix<std::complex<double>> &   mat,
        const utils::mpi::MPIComm &               interComm,
        const size_type                           broadcastRoot);
    } // namespace elpaScalaOpInternal
  }   // namespace linearAlgebra
} // namespace dftefe
