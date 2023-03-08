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
 * @author Avirup Sircar
 */

#include <utils/TypeConfig.h>
#include <string>
#include <set>
#include <vector>
#include <atoms/AtomIdsPartition.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
namespace dftefe
{
  namespace atoms
  {
    template <unsigned int dim>
    AtomIdsPartition::AtomIdsPartition( const std::vector<utils::Point> &                atomCoordinates,                    
                                        const std::vector<double> &                      minbound,  
                                        const std::vector<double> &                      maxbound,
                                        const std::vector<std::vector<utils::Point>> &   cellVerticesVector,
                                        const double                                     tolerance,
                                        const MPIComm &                                  comm,
                                        const size_type                                  nProcs)
      : d_atomCoordinates(atomCoordinates)
      , d_minbound(minbound)
      , d_maxbound(maxbound)
      , d_cellVerticesVector(cellVerticesVector)
      , d_tol(tolerance)
    {}

    // get atom ids overlapping with a box according to a tolerance , eg:- the processor ;  atom ids start from 0
    // do mpiallreduce from the collected atomids to assign repeating atomids in processors to the highest processor
    // rank. 

    template <unsigned int dim>
    void
    AtomIdsPartition::getOverlappingAtomIdsInBox(std::vector<size_type> & atomIds) const
    {
      atomIds.resize(0);
      size_type              Id = 0;
      boolean                flag;

      for (auto it : d_atomCoordinates)
        {
          flag = true;
          for (unsigned int i = 0; i < dim; i++)
          {
            double a = d_minbound[i];
            double b = d_maxbound[i];
            double c = it[i]; 
            DFTEFE_Assert(b>=a);
            if ((a-d_tol >= c) || (c >= b+d_tol))
              {flag = false;break;}
          }
          if (flag)
            atomIds.push_back(Id);
          Id++;
        }

      size_type rank;
      int err = utils::mpi::MPICommRank(comm, &rank);
      std::pair<bool, std::string> mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first, "MPI Error:" + mpiIsSuccessAndMsg.second);

      size_type nAtoms = atomCoordinates.size();
      std::vector<int> processorIdTmp;
      processorIdTmp.resize(nAtoms,-1);
      for ( Id : atomIds )
      {
          processorIdTmp[Id] = rank;
      }
      err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(&processorIdTmp[0], &processorId[0], processorIdTmp.size(),
      utils::mpi::MPIUnsigned, utils::mpi::MPIMax, comm);
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first, "MPI Error:" + mpiIsSuccessAndMsg.second);

      atomIds.resize(0);
      auto iter = processorIds.begin();
      for( ; iter!=processorIds.end() ; iter++)
        if ( *(iter) == rank )
          atomIds.pushback(iter - processorIds.begin());
    }

    // get the vector of atom ids overlapping with a cell given the cell vertices, 
    // these vectors are theselves stored as a vector eg. {{10,19,100},{50,150},...}
    // where each integer is an atom id.
   
    template<unsigned int dim>
    void
    AtomIdsPartition::getOverlappingAtomIdsInCells(std::vector<std::vector<size_type>> & overlappingAtomIdsInCells) const
    {
      overlappingAtomIdsInCells.resize(0);
      std::vector<double> minCellBound;
      std::vector<double> maxCellBound;
      std::vector<size_type> atomIdVector;
      std::vector<size_type> atomIds;
      bool                   flag;

      atomIds.resize(0);
      getOverlappingAtomIdsInBox(atomIds);
      auto cellIter = d_cellVerticesVector.begin();
      for ( ; cellIter != d_cellVerticesVector.end(); ++cellIter)
      {
        maxCellBound.resize(dim,0);
        minCellBound.resize(dim,0);
        for( unsigned int k=0;k<dim;k++)
        {
          auto cellVertices = cellIter->begin();
          double maxtmp = *(cellVertices->begin()+k),mintmp = *(cellVertices->begin()+k);
          for( ; cellVertices != cellIter->end(); ++cellVertices)
          {
            if(maxtmp<*(cellVertices->begin()+k)) maxtmp = *(cellVertices->begin()+k);
            if(mintmp>*(cellVertices->begin()+k)) mintmp = *(cellVertices->begin()+k);
          }
          maxCellBound[k]=maxtmp;
          minCellBound[k]=mintmp;
        }

        atomIdVector.resize(0,0);


        for (auto i : atomIds) 
        {
          auto it = d_atomCoordinates.begin();
          it = it+i;
          flag = true;
          for (unsigned int k = 0; k < dim; i++)
          {
            // assert for the cell and processor bounds
            DFTEFE_AssertWithMsg(minCellBound[k]>=d_minbound[k] && maxCellBound[k]>=d_minbound[k]
              && minCellBound[k]<=d_maxbound[k] && maxCellBound[k]<=d_maxbound[k]
              ,"Cell Vertices are outside the processor maximum and minimum bounds");
            double a = minCellBound[k];
            double b = maxCellBound[k];
            double c = (*it)[k];

            DFTEFE_Assert(b>=a);
            if ((a-d_tol >= c) || (c >= b+d_tol))
              {flag = false;break;}
          }
          if (flag)
            atomIdVector.push_back(i);
        }

      overlappingAtomIdsInCells.push_back(atomIdVector);
      }
    }

    // merge the repeated indices in the cells to get the atomIds within a processor.

    template<unsigned int dim>
    void
    AtomIdsPartition::getLocalAtomIds() 
    {
      std::set<size_type> atomIdsInProcessorTmp; 
      std::vector<std::vector<size_type>> overlappingAtomIdsInCells;
      getOverlappingAtomIdsInCells(overlappingAtomIdsInCells);
      auto iter = overlappingAtomIdsInCells.begin();
      for( ; iter != overlappingAtomIdsInCells.end() ; iter++)
      {
        auto it = iter->begin();
        for ( ; it != iter->end() ; it++)
        {
          atomIdsInProcessorTmp.insert(*(it));
        }
      }
      for(auto i:atomIdsInProcessorTmp)
        d_atomIdsInProcessor.push_back(i);
      atomIdsInProcessorTmp.clear();
    }

    template<unsigned int dim>
    void
    AtomIdsPartition::getNAtomIdsInProcessor() 
    {
      // get the number of atom ids in each proc.  
      // also get the cumulative number of atomids in a processor.
      // take a vector = size of no procs init to 0
      // each proc set it as no of atom ids.
      // do mpiallreduce

      size_type rank;
      int err = utils::mpi::MPICommRank(comm, &rank);
      std::pair<bool, std::string> mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first, "MPI Error:" + mpiIsSuccessAndMsg.second);

      std::vector<size_type> nAtomIdsInProcessorTmp;
      nAtomIdsInProcessorTmp.resize(nProcs , 0);
      d_nAtomIdsInProcessor.resize(nProcs , 0);
      nAtomIdsInProcessorTmp[rank] = d_atomIdsInProcessor.size();
      err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(&nAtomIdsInProcessorTmp[0], &d_nAtomIdsInProcessor[0], nAtomIdsInProcessorTmp.size(),
       utils::mpi::MPIUnsigned, utils::mpi::MPIMax, comm);
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first, "MPI Error:" + mpiIsSuccessAndMsg.second);
      nAtomIdsInProcessorTmp.clear();

      size_type count = 1;
      d_nAtomIdsInProcessorCumulative.resize(nProcs,0);
      auto iter = d_nAtomIdsInProcessor.begin();
      d_nAtomIdsInProcessorCumulative[0] = d_nAtomIdsInProcessor[0];
      for( ; iter != d_nAtomIdsInProcessor.end() ; iter++)
      {
        d_nAtomIdsInProcessorCumulative[count] = d_nAtomIdsInProcessorCumulative[count-1] + d_nAtomIdsInProcessor[count];
        count = count+1;
      }
    }


    template<unsigned int dim>
    void
    AtomIdsPartition::renumberAtomIds() 
    {
      /*Here implemet the vector of renumbered atom ids
       * MPI. This is done by asigning a vector of 1...number_atomids to each proc with init -1 and init the 
       * proclocalatomid th position of vector to 0,1 in proc 0. Then do natom0+0 .. .do mpi_min at last and return 
       * the current proc vector.*/
      //get the set of local atom ids
      //store a vector of size (nAtomIds, -1)

      size_type rank;
      int err = utils::mpi::MPICommRank(comm, &rank);
      std::pair<bool, std::string> mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first, "MPI Error:" + mpiIsSuccessAndMsg.second);

      std::vector<int> newAtomIdsTmp;
      size_type nAtomIds = d_atomCoordinates.size();
      newAtomIdsTmp.resize(nAtomIds,-1);
      d_oldAtomIds.resize(nAtomIds,0);
      d_newAtomIds.resize(nAtomIds,0);
      size_type newIds = 0;
      for( auto i:d_atomIdsInProcessor )
      {
        if(rank == 0)
          newAtomIdsTmp[i] = newIds;
        else
          newAtomIdsTmp[i] = newIds + d_nAtomIdsInProcessorCumulative[rank-1];
        newIds = newIds + 1;
      }
      
      //store the vector of new atom ids
      err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(&newAtomIdsTmp[0], &d_newAtomIds[0], newAtomIdsTmp.size(),
       utils::mpi::MPIUnsigned, utils::mpi::MPIMax, comm);
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first, "MPI Error:" + mpiIsSuccessAndMsg.second);
      newAtomIdsTmp.clear(); 

      // create a vector of old atom ids based on the new atom ids vector
      size_type oldIds = 0;
      for ( auto i:d_newAtomIds)
      {
        d_oldAtomIds[i] = oldIds;
        oldIds = oldIds + 1;
      }
    }

    template<unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition::nAtomIdsInProcessor() const
    {
      return d_nAtomIdsInProcessor;
    }

    template<unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition::nAtomIdsInProcessorCumulative() const
    {
      return d_nAtomIdsInProcessorCumulative;
    }

    template<unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition::oldAtomIds() const
    {
      return d_oldAtomIds;
    }

    template<unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition::newAtomIds() const
    {
      return d_newAtomIds;
    }

    template<unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition::locallyOwnedAtomIds() const
    {
      return d_atomIdsInProcessor;
    }


  } // end of namespace atoms
} // end of namespace dftefe