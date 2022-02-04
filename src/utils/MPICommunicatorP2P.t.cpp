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

/*
 * @author Sambit Das.
 */

#include <utils/Exceptions.h>


namespace dftefe
{
  namespace utils
  {

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::MPICommunicatorP2P(const std::shared_ptr< const MPIPatternP2P > & mpiPatternP2P,
                                                                   const size_type blockSize):
    d_mpiPatternP2P(mpiPatternP2P),
    d_blockSize(blockSize)
    {
#ifdef DFTEFE_WITH_MPI        
      d_mpiCommunicator=d_mpiPatternP2P->mpiCommunicator();
      d_sendRecvBuffer.resize(d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()*blockSize);
      d_recvRequestsScatterToGhost.resize(d_mpiPatternP2P->getGhostProcIds());
      d_sendRequestsScatterToGhost.resize(d_mpiPatternP2P->getTargetProcIds());   
      d_recvRequestsGatherFromGhost.resize(d_mpiPatternP2P->getTargetProcIds());
      d_sendRequestsGatherFromGhost.resize(d_mpiPatternP2P->getGhostProcIds());  
#endif      
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhost(VectorStorage<ValueType,memorySpace> & dataArray)
    {
#ifdef DFTEFE_WITH_MPI      
      scatterToGhostBegin(dataArray);
      scatterToGhostEnd(dataArray);
#endif      
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhostBegin(VectorStorage<ValueType,memorySpace> & dataArray)
    {
#ifdef DFTEFE_WITH_MPI
      for (unsigned int i = 0; i < n_ghost_targets; ++i)
        {
          const int ierr =
            MPI_Irecv(ghost_array_ptr,
                      ghost_targets_data[i].second * sizeof(Number),
                      MPI_BYTE,
                      ghost_targets_data[i].first,
                      mpi_tag,
                      communicator,
                      &requests[i]);
          AssertThrowMPI(ierr);
          ghost_array_ptr += ghost_targets_data[i].second;
        }      
#endif      
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::scatterToGhostEnd(VectorStorage<ValueType,memorySpace> & dataArray)
    {
#ifdef DFTEFE_WITH_MPI
#endif      
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::gatherFromGhost(VectorStorage<ValueType,memorySpace> & dataArray)
    {
      gatherFromGhostBegin(dataArray);
      gatherFromGhostEnd(dataArray);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::gatherFromGhostBegin(VectorStorage<ValueType,memorySpace> & dataArray)
    {
#ifdef DFTEFE_WITH_MPI
#endif           
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2P<ValueType, memorySpace>::gatherFromGhostEnd(VectorStorage<ValueType,memorySpace> & dataArray)
    {
#ifdef DFTEFE_WITH_MPI
#endif         
    }

  } // namespace utils
} // namespace dftefe
