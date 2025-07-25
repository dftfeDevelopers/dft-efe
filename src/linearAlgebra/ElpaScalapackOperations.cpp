// ---------------------------------------------------------------------
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
// ---------------------------------------------------------------------
//
// @author Sambit Das
//


#include <dftUtils.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <BLASWrapper.h>
namespace dftfe
{
  namespace linearAlgebraOperations
  {
    namespace internal
    {
      void
      setupELPAHandleParameters(
        const utils::mpi::MPIComm &mpi_communicator,
        utils::mpi::MPIComm       &processGridCommunicatorActive,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::uInt                                na,
        const dftfe::uInt                                nev,
        const dftfe::uInt                                blockSize,
        elpa_t                                          &elpaHandle,
        const dftParameters                             &dftParams)
      {
        int error;

        if (processGrid->is_process_active())
          {
            elpaHandle = elpa_allocate(&error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));
          }

        // Get the group of processes in mpi_communicator
        int       ierr = 0;
        MPI_Group all_group;
        ierr = MPI_Comm_group(mpi_communicator, &all_group);
        AssertThrowMPI(ierr);

        // Construct the group containing all ranks we need:
        const dftfe::uInt n_active_mpi_processes =
          processGrid->get_process_grid_rows() *
          processGrid->get_process_grid_columns();
        std::vector<int> active_ranks;
        for (dftfe::uInt i = 0; i < n_active_mpi_processes; ++i)
          active_ranks.push_back(i);

        MPI_Group active_group;
        const int n = active_ranks.size();
        ierr = MPI_Group_incl(all_group, n, active_ranks.data(), &active_group);
        AssertThrowMPI(ierr);

        // Create the communicator based on active_group.
        // Note that on all the inactive processs the resulting utils::mpi::MPIComm
        // processGridCommunicatorActive will be MPI_COMM_NULL.
        // utils::mpi::MPIComm processGridCommunicatorActive;
        ierr = dealii::Utilities::MPI::create_group(
          mpi_communicator, active_group, 50, &processGridCommunicatorActive);
        AssertThrowMPI(ierr);

        ierr = MPI_Group_free(&all_group);
        AssertThrowMPI(ierr);
        ierr = MPI_Group_free(&active_group);
        AssertThrowMPI(ierr);


        dftfe::ScaLAPACKMatrix<double> tempMat(na, processGrid, blockSize);
        if (processGrid->is_process_active())
          {
            /* Set parameters the matrix and it's MPI distribution */
            elpa_set_integer(elpaHandle, "na", na, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));


            elpa_set_integer(elpaHandle, "nev", nev, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle, "nblk", blockSize, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "mpi_comm_parent",
                             MPI_Comm_c2f(processGridCommunicatorActive),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));


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
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "local_ncols",
                             tempMat.local_n(),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "process_row",
                             processGrid->get_this_process_row(),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set_integer(elpaHandle,
                             "process_col",
                             processGrid->get_this_process_column(),
                             &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set(elpaHandle,
                     "blacs_context",
                     processGrid->get_blacs_context(),
                     &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            elpa_set(elpaHandle, "cannon_for_generalized", 0, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

            /* Setup */
            AssertThrow(elpa_setup(elpaHandle) == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));

#ifdef DFTFE_WITH_DEVICE

            if (dftParams.useELPADeviceKernel)
              {
#  ifdef DFTFE_WITH_DEVICE_NVIDIA
                elpa_set_integer(elpaHandle, "nvidia-gpu", 1, &error);
                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#  elif DFTFE_WITH_DEVICE_AMD
                elpa_set_integer(elpaHandle, "amd-gpu", 1, &error);
                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#  endif
                elpa_set_integer(elpaHandle,
                                 "solver",
                                 ELPA_SOLVER_1STAGE,
                                 &error);
                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));

                int gpuID = 0;
                dftfe::utils::getDevice(&gpuID);

                elpa_set_integer(elpaHandle, "use_gpu_id", gpuID, &error);
                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));

                error = elpa_setup_gpu(elpaHandle);
                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));
              }
            else
              {
                elpa_set_integer(elpaHandle,
                                 "solver",
                                 ELPA_SOLVER_2STAGE,
                                 &error);
                AssertThrow(error == ELPA_OK,
                            dealii::ExcMessage("DFT-FE Error: ELPA Error."));
              }
#else
            elpa_set_integer(elpaHandle, "solver", ELPA_SOLVER_2STAGE, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif

              // elpa_set_integer(elpaHandle,
              // "real_kernel",ELPA_2STAGE_REAL_AVX512_BLOCK6, &error);
              // AssertThrow(error==ELPA_OK,
              //   dealii::ExcMessage("DFT-FE Error: ELPA Error."));

#ifdef DEBUG
            elpa_set_integer(elpaHandle, "debug", 1, &error);
            AssertThrow(error == ELPA_OK,
                        dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
          }

        // d_elpaAutoTuneHandle = elpa_autotune_setup(d_elpaHandle,
        // ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, &error);   // create
        // autotune object
      }

      void
      createProcessGridSquareMatrix(
        const utils::mpi::MPIComm                            &mpi_communicator,
        const dftfe::uInt                          size,
        std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftParameters                       &dftParams,
        const bool                                 useOnlyThumbRule)
      {
        const dftfe::uInt numberProcs =
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

        // Rule of thumb from
        // http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
        dftfe::uInt rowProcs =
          (dftParams.scalapackParalProcs == 0 || useOnlyThumbRule) ?
            std::min(std::floor(std::sqrt(numberProcs)),
                     std::ceil((double)size / (double)(1000))) :
            std::min((dftfe::uInt)std::floor(std::sqrt(numberProcs)),
                     dftParams.scalapackParalProcs);

        rowProcs = ((dftParams.scalapackParalProcs == 0 || useOnlyThumbRule) &&
                    dftParams.useELPA) ?
                     std::min((dftfe::uInt)std::floor(std::sqrt(numberProcs)),
                              (dftfe::uInt)std::floor(rowProcs * 3.0)) :
                     rowProcs;
        if (!dftParams.useDevice)
          {
            if (!dftParams.reproducible_output)
              rowProcs =
                std::min(rowProcs,
                         (dftfe::uInt)std::ceil((double)size / (double)(100)));

            else
              rowProcs =
                std::min(rowProcs,
                         (dftfe::uInt)std::ceil((double)size / (double)(10)));
          }
        else
          {
            rowProcs =
              std::min(rowProcs,
                       (dftfe::uInt)std::ceil((double)size / (double)(100)));
          }


        if (dftParams.verbosity >= 4)
          {
            dealii::ConditionalOStream pcout(
              std::cout,
              (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
               0));
            pcout << "Scalapack Matrix created, row procs: " << rowProcs
                  << std::endl;
          }

        processGrid =
          std::make_shared<const dftfe::ProcessGrid>(mpi_communicator,
                                                     rowProcs,
                                                     rowProcs);
      }


      void
      createProcessGridRectangularMatrix(
        const utils::mpi::MPIComm                            &mpi_communicator,
        const dftfe::uInt                          sizeRows,
        const dftfe::uInt                          sizeColumns,
        std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftParameters                       &dftParams)
      {
        const dftfe::uInt numberProcs =
          dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

        // Rule of thumb from
        // http://netlib.org/scalapack/slug/node106.html#SECTION04511000000000000000
        const dftfe::uInt rowProcs =
          std::min(std::floor(std::sqrt(numberProcs)),
                   std::ceil((double)sizeRows / (double)(1000)));
        const dftfe::uInt columnProcs =
          std::min(std::floor(std::sqrt(numberProcs)),
                   std::ceil((double)sizeColumns / (double)(1000)));

        if (dftParams.verbosity >= 4)
          {
            dealii::ConditionalOStream pcout(
              std::cout,
              (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
               0));
            pcout << "Scalapack Matrix created, row procs x column procs: "
                  << rowProcs << " x " << columnProcs << std::endl;
          }

        processGrid =
          std::make_shared<const dftfe::ProcessGrid>(mpi_communicator,
                                                     rowProcs,
                                                     columnProcs);
      }


      template <typename T>
      void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::ScaLAPACKMatrix<T>                 &mat,
        std::unordered_map<dftfe::uInt, dftfe::uInt>    &globalToLocalRowIdMap,
        std::unordered_map<dftfe::uInt, dftfe::uInt> &globalToLocalColumnIdMap)
      {
        globalToLocalRowIdMap.clear();
        globalToLocalColumnIdMap.clear();
        if (processGrid->is_process_active())
          {
            for (dftfe::uInt i = 0; i < mat.local_m(); ++i)
              globalToLocalRowIdMap[mat.global_row(i)] = i;

            for (dftfe::uInt j = 0; j < mat.local_n(); ++j)
              globalToLocalColumnIdMap[mat.global_column(j)] = j;
          }
      }


      template <typename T>
      void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T>                       &mat,
        const utils::mpi::MPIComm                                  &interComm)
      {
        // sum across all inter communicator groups
        if (processGrid->is_process_active() &&
            dealii::Utilities::MPI::n_mpi_processes(interComm) > 1)
          {
            MPI_Allreduce(MPI_IN_PLACE,
                          &mat.local_el(0, 0),
                          mat.local_m() * mat.local_n(),
                          dataTypes::mpi_type_id(&mat.local_el(0, 0)),
                          MPI_SUM,
                          interComm);
          }
      }

      template <typename T>
      void
      scaleScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                  &BLASWrapperPtr,
        dftfe::ScaLAPACKMatrix<T> &mat,
        const T                    scalar)
      {
        // if (processGrid->is_process_active())
        //   {
        //     const dftfe::uInt numberComponents = mat.local_m() *
        //     mat.local_n(); const dftfe::uInt inc              = 1;
        //     xscal(&numberComponents, &scalar, &mat.local_el(0, 0), &inc);
        //   }
      }



      template <typename T>
      void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<T>                       &mat,
        const utils::mpi::MPIComm                                  &interComm,
        const dftfe::uInt                                broadcastRoot)
      {
        // sum across all inter communicator groups
        if (processGrid->is_process_active() &&
            dealii::Utilities::MPI::n_mpi_processes(interComm) > 1)
          {
            MPI_Bcast(&mat.local_el(0, 0),
                      mat.local_m() * mat.local_n(),
                      dataTypes::mpi_type_id(&mat.local_el(0, 0)),
                      broadcastRoot,
                      interComm);
          }
      }

      template <typename T, typename TLowPrec>
      void
      fillParallelOverlapMatrixMixedPrec(
        const T *subspaceVectorsArray,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                         &BLASWrapperPtr,
        const dftfe::uInt subspaceVectorsArrayLocalSize,
        const dftfe::uInt N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const utils::mpi::MPIComm                                  &interBandGroupComm,
        const utils::mpi::MPIComm                                  &mpiComm,
        dftfe::ScaLAPACKMatrix<T>                       &overlapMatPar,
        const dftParameters                             &dftParams)
      {
        const dftfe::uInt numLocalDofs = subspaceVectorsArrayLocalSize / N;

        // band group parallelization data structures
        const dftfe::uInt numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const dftfe::uInt bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        // get global to local index maps for Scalapack matrix
        std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
        std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
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
        const dftfe::uInt vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

        std::vector<T>        overlapMatrixBlock(N * vectorsBlockSize, T(0.0));
        std::vector<TLowPrec> overlapMatrixBlockLowPrec(N * vectorsBlockSize,
                                                        TLowPrec(0.0));
        std::vector<T>        overlapMatrixBlockDoublePrec(vectorsBlockSize *
                                                      vectorsBlockSize,
                                                    T(0.0));

        std::vector<TLowPrec> subspaceVectorsArrayLowPrec(
          subspaceVectorsArray,
          subspaceVectorsArray + subspaceVectorsArrayLocalSize);
        for (dftfe::uInt ivec = 0; ivec < N; ivec += vectorsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            const dftfe::uInt B = std::min(vectorsBlockSize, N - ivec);

            // If one plus the ending index of a block lies within a band
            // parallelization group do computations for that block within the
            // band group, otherwise skip that block. This is only activated if
            // NPBAND>1
            if ((ivec + B) <=
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                (ivec + B) >
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                const char transA = 'N',
                           transB =
                             std::is_same<T, std::complex<double>>::value ?
                               'C' :
                               'T';
                const T        scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
                const TLowPrec scalarCoeffAlphaLowPrec = 1.0,
                               scalarCoeffBetaLowPrec  = 0.0;

                std::fill(overlapMatrixBlock.begin(),
                          overlapMatrixBlock.end(),
                          0.);
                std::fill(overlapMatrixBlockLowPrec.begin(),
                          overlapMatrixBlockLowPrec.end(),
                          0.);

                const dftfe::uInt D = N - ivec;

                BLASWrapperPtr->xgemm(transA,
                                      transB,
                                      B,
                                      B,
                                      numLocalDofs,
                                      &scalarCoeffAlpha,
                                      subspaceVectorsArray + ivec,
                                      N,
                                      subspaceVectorsArray + ivec,
                                      N,
                                      &scalarCoeffBeta,
                                      &overlapMatrixBlockDoublePrec[0],
                                      B);

                const dftfe::uInt DRem = D - B;
                if (DRem != 0)
                  {
                    BLASWrapperPtr->xgemm(transA,
                                          transB,
                                          DRem,
                                          B,
                                          numLocalDofs,
                                          &scalarCoeffAlphaLowPrec,
                                          &subspaceVectorsArrayLowPrec[0] +
                                            ivec + B,
                                          N,
                                          &subspaceVectorsArrayLowPrec[0] +
                                            ivec,
                                          N,
                                          &scalarCoeffBetaLowPrec,
                                          &overlapMatrixBlockLowPrec[0],
                                          DRem);
                  }

                MPI_Barrier(mpiComm);
                // Sum local XTrunc^{T}*XcBlock for double precision across
                // domain decomposition processors
                MPI_Allreduce(MPI_IN_PLACE,
                              &overlapMatrixBlockDoublePrec[0],
                              B * B,
                              dataTypes::mpi_type_id(
                                &overlapMatrixBlockDoublePrec[0]),
                              MPI_SUM,
                              mpiComm);

                MPI_Barrier(mpiComm);
                // Sum local XTrunc^{T}*XcBlock for single precision across
                // domain decomposition processors
                MPI_Allreduce(MPI_IN_PLACE,
                              &overlapMatrixBlockLowPrec[0],
                              DRem * B,
                              dataTypes::mpi_type_id(
                                &overlapMatrixBlockLowPrec[0]),
                              MPI_SUM,
                              mpiComm);

                for (dftfe::uInt i = 0; i < B; ++i)
                  {
                    for (dftfe::uInt j = 0; j < B; ++j)
                      overlapMatrixBlock[i * D + j] =
                        overlapMatrixBlockDoublePrec[i * B + j];

                    for (dftfe::uInt j = 0; j < DRem; ++j)
                      overlapMatrixBlock[i * D + j + B] =
                        overlapMatrixBlockLowPrec[i * DRem + j];
                  }

                // Copying only the lower triangular part to the ScaLAPACK
                // overlap matrix
                if (processGrid->is_process_active())
                  for (dftfe::uInt i = 0; i < B; ++i)
                    if (globalToLocalColumnIdMap.find(i + ivec) !=
                        globalToLocalColumnIdMap.end())
                      {
                        const dftfe::uInt localColumnId =
                          globalToLocalColumnIdMap[i + ivec];
                        for (dftfe::uInt j = ivec + i; j < N; ++j)
                          {
                            std::unordered_map<dftfe::uInt,
                                               dftfe::uInt>::iterator it =
                              globalToLocalRowIdMap.find(j);
                            if (it != globalToLocalRowIdMap.end())
                              overlapMatPar.local_el(it->second,
                                                     localColumnId) =
                                overlapMatrixBlock[i * D + j - ivec];
                          }
                      }
              } // band parallelization
          }     // block loop


        // accumulate contribution from all band parallelization groups
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, overlapMatPar, interBandGroupComm);
      }


      template <typename T>
      void
      fillParallelOverlapMatrix(
        const T *subspaceVectorsArray,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                         &BLASWrapperPtr,
        const dftfe::uInt subspaceVectorsArrayLocalSize,
        const dftfe::uInt N,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const utils::mpi::MPIComm                                  &interBandGroupComm,
        const utils::mpi::MPIComm                                  &mpiComm,
        dftfe::ScaLAPACKMatrix<T>                       &overlapMatPar,
        const dftParameters                             &dftParams)
      {
        const dftfe::uInt numLocalDofs = subspaceVectorsArrayLocalSize / N;

        // band group parallelization data structures
        const dftfe::uInt numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
        const dftfe::uInt bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

        // get global to local index maps for Scalapack matrix
        std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalColumnIdMap;
        std::unordered_map<dftfe::uInt, dftfe::uInt> globalToLocalRowIdMap;
        internal::createGlobalToLocalIdMapsScaLAPACKMat(
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
        const dftfe::uInt vectorsBlockSize =
          std::min(dftParams.wfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

        std::vector<T> overlapMatrixBlock(N * vectorsBlockSize, 0.0);

        for (dftfe::uInt ivec = 0; ivec < N; ivec += vectorsBlockSize)
          {
            // Correct block dimensions if block "goes off edge of" the matrix
            const dftfe::uInt B = std::min(vectorsBlockSize, N - ivec);

            // If one plus the ending index of a block lies within a band
            // parallelization group do computations for that block within the
            // band group, otherwise skip that block. This is only activated if
            // NPBAND>1
            if ((ivec + B) <=
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                (ivec + B) >
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                const char transA = 'N',
                           transB =
                             std::is_same<T, std::complex<double>>::value ?
                               'C' :
                               'T';
                const T scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;

                std::fill(overlapMatrixBlock.begin(),
                          overlapMatrixBlock.end(),
                          0.);

                const dftfe::uInt D = N - ivec;

                // Comptute local XTrunc^{T}*XcBlock.
                BLASWrapperPtr->xgemm(transA,
                                      transB,
                                      D,
                                      B,
                                      numLocalDofs,
                                      &scalarCoeffAlpha,
                                      subspaceVectorsArray + ivec,
                                      N,
                                      subspaceVectorsArray + ivec,
                                      N,
                                      &scalarCoeffBeta,
                                      &overlapMatrixBlock[0],
                                      D);

                MPI_Barrier(mpiComm);
                // Sum local XTrunc^{T}*XcBlock across domain decomposition
                // processors
                MPI_Allreduce(MPI_IN_PLACE,
                              &overlapMatrixBlock[0],
                              D * B,
                              dataTypes::mpi_type_id(&overlapMatrixBlock[0]),
                              MPI_SUM,
                              mpiComm);

                // Copying only the lower triangular part to the ScaLAPACK
                // overlap matrix
                if (processGrid->is_process_active())
                  for (dftfe::uInt i = 0; i < B; ++i)
                    if (globalToLocalColumnIdMap.find(i + ivec) !=
                        globalToLocalColumnIdMap.end())
                      {
                        const dftfe::uInt localColumnId =
                          globalToLocalColumnIdMap[i + ivec];
                        for (dftfe::uInt j = ivec + i; j < N; ++j)
                          {
                            std::unordered_map<dftfe::uInt,
                                               dftfe::uInt>::iterator it =
                              globalToLocalRowIdMap.find(j);
                            if (it != globalToLocalRowIdMap.end())
                              overlapMatPar.local_el(it->second,
                                                     localColumnId) =
                                overlapMatrixBlock[i * D + j - ivec];
                          }
                      }
              } // band parallelization
          }     // block loop


        // accumulate contribution from all band parallelization groups
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, overlapMatPar, interBandGroupComm);
      }

      template void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const dftfe::ScaLAPACKMatrix<double>            &mat,
        std::unordered_map<dftfe::uInt, dftfe::uInt>    &globalToLocalRowIdMap,
        std::unordered_map<dftfe::uInt, dftfe::uInt> &globalToLocalColumnIdMap);

      template void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid>    &processGrid,
        const dftfe::ScaLAPACKMatrix<std::complex<double>> &mat,
        std::unordered_map<dftfe::uInt, dftfe::uInt> &globalToLocalRowIdMap,
        std::unordered_map<dftfe::uInt, dftfe::uInt> &globalToLocalColumnIdMap);

      template void
      fillParallelOverlapMatrix(
        const double *X,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
        const dftfe::uInt                                XLocalSize,
        const dftfe::uInt                                numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const utils::mpi::MPIComm                                  &interBandGroupComm,
        const utils::mpi::MPIComm                                  &mpiComm,
        dftfe::ScaLAPACKMatrix<double>                  &overlapMatPar,
        const dftParameters                             &dftParams);

      template void
      fillParallelOverlapMatrix(
        const std::complex<double> *X,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
        const dftfe::uInt                                XLocalSize,
        const dftfe::uInt                                numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const utils::mpi::MPIComm                                  &interBandGroupComm,
        const utils::mpi::MPIComm                                  &mpiComm,
        dftfe::ScaLAPACKMatrix<std::complex<double>>    &overlapMatPar,
        const dftParameters                             &dftParams);


      template void
      fillParallelOverlapMatrixMixedPrec<double, float>(
        const double *X,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
        const dftfe::uInt                                XLocalSize,
        const dftfe::uInt                                numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const utils::mpi::MPIComm                                  &interBandGroupComm,
        const utils::mpi::MPIComm                                  &mpiComm,
        dftfe::ScaLAPACKMatrix<double>                  &overlapMatPar,
        const dftParameters                             &dftParams);

      template void
      fillParallelOverlapMatrixMixedPrec<std::complex<double>,
                                         std::complex<float>>(
        const std::complex<double> *X,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                        &BLASWrapperPtr,
        const dftfe::uInt                                XLocalSize,
        const dftfe::uInt                                numberVectors,
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const utils::mpi::MPIComm                                  &interBandGroupComm,
        const utils::mpi::MPIComm                                  &mpiComm,
        dftfe::ScaLAPACKMatrix<std::complex<double>>    &overlapMatPar,
        const dftParameters                             &dftParams);

      template void
      scaleScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                       &BLASWrapperPtr,
        dftfe::ScaLAPACKMatrix<double> &mat,
        const double                    scalar);

      template void
      scaleScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                                     &BLASWrapperPtr,
        dftfe::ScaLAPACKMatrix<std::complex<double>> &mat,
        const std::complex<double>                    scalar);

      template void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<double>                  &mat,
        const utils::mpi::MPIComm                                  &interComm);

      template void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<std::complex<double>>    &mat,
        const utils::mpi::MPIComm                                  &interComm);

      template void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<double>                  &mat,
        const utils::mpi::MPIComm                                  &interComm,
        const dftfe::uInt                                broadcastRoot);

      template void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
        dftfe::ScaLAPACKMatrix<std::complex<double>>    &mat,
        const utils::mpi::MPIComm                                  &interComm,
        const dftfe::uInt                                broadcastRoot);
    } // namespace internal
  }   // namespace linearAlgebraOperations
} // namespace dftfe