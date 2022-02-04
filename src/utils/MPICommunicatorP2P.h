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
 * @author Sambit Das
 */

#ifndef dftefeMPICommunicatorP2P_h
#define dftefeMPICommunicatorP2P_h

#include <utils/MemorySpaceType.h>
#include <utils/MPIPatternP2P.h>
#include <utils/TypeConfig.h>
#include <utils/VectorStorage.h>


namespace dftefe
{
  namespace utils
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>      
    class MPICommunicatorP2P
    {

      void MPICommunicatorP2P(const std::shared_ptr< const MPIPatternP2P > & mpiPatternP2P,
                              const size_type blockSize);

      void
      scatterToGhost(VectorStorage<ValueType,memorySpace> & dataArray);

      void
      gatherFromGhost(VectorStorage<ValueType,memorySpace> & dataArray);


      void
      scatterToGhostBegin(VectorStorage<ValueType,memorySpace> & dataArray);

      void
      scatterToGhostEnd(VectorStorage<ValueType,memorySpace> & dataArray);

      void
      gatherFromGhostBegin(VectorStorage<ValueType,memorySpace> & dataArray);

      void
      gatherFromGhostEnd(VectorStorage<ValueType,memorySpace> & dataArray);

    private:

      std::shared_ptr< const MPIPatternP2P > 	d_mpiPatternP2P;

      size_type d_blockSize;

      VectorStorage<ValueType,memorySpace> d_sendRecvBuffer;

      std::vector<MPI_Request> d_sendRequestsScatterToGhost;

      std::vector<MPI_Request> d_recvRequestsScatterToGhost;   

      std::vector<MPI_Request> d_sendRequestsGatherFromGhost;

      std::vector<MPI_Request> d_recvRequestsGatherFromGhost;  

      MPI_Comm  d_mpiCommunicator;
    };

  } // namespace utils
} // namespace dftefe

#include "MPICommunicatorP2P.t.cpp"

#endif // dftefeMPICommunicatorBase_h
