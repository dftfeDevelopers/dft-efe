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
 * @author Sambit Das, Bikash Kanungo
 */

#ifndef dftefeMPIPatternHost_h
#define dftefeMPIPatternHost_h

#include <mpi.h>
#include <utils/MPIPatternBase.h>
#include <set>
namespace dftefe {

  namespace utils {

    class MPIPatternHost  : public MPIPatternBase
    {

      public:
    
       virtual ~MPIPatternHost() = default;

       MPIPatternHost(const std::pair<global_size_type, global_size_type> locallyOwnedRange,
           const std::set<dftefe::global_size_type> & ghostIndices,
           MPI_Comm & d_mpiComm);

       reinit(const std::pair<global_size_type, global_size_type> locallyOwnedRange,
           const std::set<dftefe::global_size_type> & ghostIndices,
           MPI_Comm & d_mpiComm);      

      size_type 	localOwnedSize() const;

      size_type 	localGhostSize() const;

      size_type 	localSize() const;      

      bool 	inLocallyOwnedRange(const global_size_type) const;

      bool 	isGhostEntry(const global_size_type) const;

      size_type 	globalToLocal(const global_size_type) const;
 
      global_size_type 	localToGlobal(const size_type) const;     


      const std::vector<global_size_type> &	getGhostIndicesVec() const;

      const std::vector<global_size_type> &	getImportIndicesVec() const;   


      const std::vector<size_type> &	getGhostProcIdToNumGhostMap() const;

      const std::vector<size_type> &	getImportProcIdToNumLocallyOwnedMap() const;        

      size_type nmpiProcesses() const override;
      size_type thisProcessId() const override;

      private:
      
      /// Vector to store an ordered set of ghost indices 
      /// (ordered in increasing order and non-repeating)
      std::vector<global_size_type> d_ghostIndices;
     
      /// Copy of the above but stored as a set
      std::vector<global_size_tye> d_ghostIndicesSet;
      
      /// Vector of locally owned indices which other processors need to imported
      /// Non-decreasing, but can have repeated entries
      std::vector<global_size_tye> d_importIndices;

      /// Vector of size 2*(number of ghost procs), where a ghost proc is one that owns at least 
      /// one of the ghost indices of this processor. The entries as arranged as:
      /// <ghost proc1 ID> <number of ghost indices owned> <ghost proc2 ID> <number of ghost indices owned> ...
      std::vector<size_type> d_ghostProcIdToNumGhostMap;

      /// Vector of size 2*(number of import procs), where an import proc is one which imports at least
      /// one of the locally owned indices of this processor. The entries are arranged as:
      /// <import proc1 ID> <number of import indices> <import proc2 ID> <number of import indices> ...
      std::vector<size_type> d_importProcIdToNumLocallyOwnedMap;

      MPI_Comm d_communicator;
      size_type d_nprocs;
      size_type thisProcId;
      global_size_type d_globalSize;
      std::vector<

       
    };

} // end of namespace utils

} // end of namespace dftefe

#endif // dftefeMPIPatternHost_h
