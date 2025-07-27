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

// @author Sambit Das , Avirup Sircar
//
#ifndef elpaScalaManager_h
#define elpaScalaManager_h

#include "ProcessGrid.h"
#include <vector>
#include <elpa/elpa.h>
#include "ScalapackWrapper.h"
#include "ElpaScalapackOperations.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     * @brief Manager class for ELPA and ScaLAPACK
     *
     * @author Sambit Das, Aviup Sircar
     */
    class ElpaScalapackManager
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
      ElpaScalapackManager(const utils::mpi::MPIComm &mpi_comm_replica);

      /**
       * @brief Destructor.
       */
      ~ElpaScalapackManager();

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
    ElpaScalapackManager::getScalapackBlockSize() const
    {
      return d_scalapackBlockSize;
    }

    inline std::shared_ptr<const ProcessGrid>
    ElpaScalapackManager::getProcessGridDftfeScalaWrapper() const
    {
      return d_processGridDftfeWrapper;
    }

    inline elpa_t &
    ElpaScalapackManager::getElpaHandle()
    {
      return d_elpaHandle;
    }

    inline elpa_t &
    ElpaScalapackManager::getElpaHandlePartialEigenVec()
    {
      return d_elpaHandlePartialEigenVec;
    }


    inline elpa_autotune_t &
    ElpaScalapackManager::getElpaAutoTuneHandle()
    {
      return d_elpaAutoTuneHandle;
    }
  } // namespace linearAlgebra
} // namespace dftefe
#endif
