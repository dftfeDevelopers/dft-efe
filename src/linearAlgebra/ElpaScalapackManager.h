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
#ifndef elpaScalaManager_h
#define elpaScalaManager_h

#include "ProcessGrid.h"
#include <vector>
#include <elpa/elpa.h>

namespace dftfe
{
  namespace linearAlgebra
  {
    /**
     * @brief Manager class for ELPA and ScaLAPACK
     *
     * @author Sambit Das
     */
    class elpaScalaManager
    {
      //
      // methods
      //
    public:
      unsigned int
      getScalapackBlockSize() const;

      std::shared_ptr<const ProcessGrid>
      getProcessGridDftfeScalaWrapper() const;

      void
      processGridELPASetup(const unsigned int na);
      void
      elpaDeallocateHandles();

      elpa_t &
      getElpaHandle();

      elpa_t &
      getElpaHandlePartialEigenVec();

      elpa_autotune_t &
      getElpaAutoTuneHandle();


      /**
       * @brief Get relevant mpi communicator
       *
       * @return mpi communicator
       */
      const utils::mpi::MPIComm &
      getMPICommunicator() const;


      /**
       * @brief Constructor.
       */
      elpaScalaManager(const utils::mpi::MPIComm &mpi_comm_replica);

      /**
       * @brief Destructor.
       */
      ~elpaScalaManager();

      //
      // mpi communicator
      //
      utils::mpi::MPIComm d_mpi_communicator;

      /// ELPA handle
      elpa_t d_elpaHandle;

      /// ELPA handle for partial eigenvectors of full proj ham
      elpa_t d_elpaHandlePartialEigenVec;

      /// ELPA autotune handle
      elpa_autotune_t d_elpaAutoTuneHandle;

      /// processGrid mpi communicator
      utils::mpi::MPIComm d_processGridCommunicatorActive;

      utils::mpi::MPIComm d_processGridCommunicatorActivePartial;


      /// ScaLAPACK distributed format block size
      unsigned int d_scalapackBlockSize;

      std::shared_ptr<const ProcessGrid> d_processGridDftfeWrapper;
    };

    /*--------------------- Inline functions --------------------------------*/

    inline unsigned int
    elpaScalaManager::getScalapackBlockSize() const
    {
      return d_scalapackBlockSize;
    }

    inline std::shared_ptr<const ProcessGrid>
    elpaScalaManager::getProcessGridDftfeScalaWrapper() const
    {
      return d_processGridDftfeWrapper;
    }

    inline elpa_t &
    elpaScalaManager::getElpaHandle()
    {
      return d_elpaHandle;
    }

    inline elpa_t &
    elpaScalaManager::getElpaHandlePartialEigenVec()
    {
      return d_elpaHandlePartialEigenVec;
    }


    inline elpa_autotune_t &
    elpaScalaManager::getElpaAutoTuneHandle()
    {
      return d_elpaAutoTuneHandle;
    }
  } // namespace linearAlgebra
} // namespace dftfe
#endif
