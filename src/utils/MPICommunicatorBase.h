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

#ifndef dftefeMPICommunicatorBase_h
#define dftefeMPICommunicatorBase_h

#include <mpi.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpace.h>

namespace dftefe
{
  namespace utils
  {
    class MPICommunicatorBase
    {
    public:
 
       virtual ~MPICommunicatorBase() = default;


       virtual MPICommunicatorBase(const  std::vector<dftefe::global_size_type> & locallyOwnedIndices,
                                   const  std::vector<dftefe::global_size_type> & ghostIndices,
                                   MPI_Comm & d_mpiComm)=0;


      template <typename ValueType, MemorySpace memorySpace>
      void
      scatterToGhost(dftefe::utils::DistributedVectorStorage<ValueType,memorySpace> & distributedVectorStorage) =0;

      template <typename ValueType, MemorySpace memorySpace>
      void
      gatherFromGhost(dftefe::utils::DistributedVectorStorag<ValueType,memorySpace> & distributedVectorStorage) =0;      


      template <typename ValueType, MemorySpace memorySpace>
      void
      scatterToGhostBegin(dftefe::utils::DistributedVectorStorage<ValueType,memorySpace> & distributedVectorStorage) =0;

      template <typename ValueType, MemorySpace memorySpace>
      void
      scatterToGhostEnd(dftefe::utils::DistributedVectorStorage<ValueType,memorySpace> & distributedVectorStorage) =0;

      template <typename ValueType, MemorySpace memorySpace>
      void
      gatherFromGhostBegin(dftefe::utils::DistributedVectorStorag<ValueType,memorySpace> & distributedVectorStorage) =0;

      template <typename ValueType, MemorySpace memorySpace>
      void
      gatherFromGhostEnd(dftefe::utils::DistributedVectorStorag<ValueType,memorySpace> & distributedVectorStorage) =0
      ;

       


    };
  }    // namespace utils
} // namespace dftefe

#endif // dftefeMPICommunicatorBase_h
