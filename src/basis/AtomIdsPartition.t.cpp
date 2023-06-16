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

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <string>
#include <set>
#include <climits>
#include <cfloat>
#include <vector>
#include <basis/AtomIdsPartition.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
namespace dftefe
{
  namespace basis
  {
    namespace AtomIdsPartitionInternal
    {
      // get atom ids overlapping with a box according to a tolerance , eg:- the
      // processor ;  atom ids start from 0 do mpiallreduce from the collected
      // atomids to assign repeating atomids in processors to the highest
      // processor rank.
      template <unsigned int dim>
      void
      getOverlappingAtomIdsInBox(
        std::vector<size_type> &         atomIds,
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      minbound,
        const std::vector<double> &      maxbound,
        const double                     tolerance,
        const utils::mpi::MPIComm &      comm)
      {
        atomIds.resize(0);
        size_type Id = 0;
        bool      flag;

        for (auto it : atomCoordinates)
          {
            flag = true;
            for (unsigned int i = 0; i < dim; i++)
              {
                double a = minbound[i];
                double b = maxbound[i];
                double c = it[i];
                DFTEFE_Assert(b >= a);
                if ((a - tolerance >= c) || (c >= b + tolerance))
                  {
                    flag = false;
                    break;
                  }
              }
            if (flag)
              atomIds.push_back(Id);
            Id++;
          }
      }

      // get the vector of atom ids overlapping with a cell given the cell
      // vertices, these vectors are theselves stored as a vector eg.
      // {{10,19,100},{50,150},...} where each integer is an atom id.

      template <unsigned int dim>
      void
      getOverlappingAtomIdsInCells(
        std::vector<std::vector<size_type>> &         overlappingAtomIdsInCells,
        std::vector<size_type> &                      atomIds,
        const std::vector<utils::Point> &             atomCoordinates,
        const std::vector<double> &                   minbound,
        const std::vector<double> &                   maxbound,
        const std::vector<std::vector<utils::Point>> &cellVerticesVector,
        const double                                  tolerance,
        const utils::mpi::MPIComm &                   comm)
      {
        overlappingAtomIdsInCells.resize(0);
        std::vector<double>    minCellBound;
        std::vector<double>    maxCellBound;
        std::vector<size_type> atomIdVector;
        bool                   flag;

        auto cellIter = cellVerticesVector.begin();
        for (; cellIter != cellVerticesVector.end(); ++cellIter)
          {
            maxCellBound.resize(dim, 0);
            minCellBound.resize(dim, 0);
            for (unsigned int k = 0; k < dim; k++)
              {
                auto cellVertices = cellIter->begin();
                // double maxtmp = *(cellVertices->begin()+k),mintmp =
                // *(cellVertices->begin()+k);
                double maxtmp = -DBL_MAX, mintmp = DBL_MAX;
                for (; cellVertices != cellIter->end(); ++cellVertices)
                  {
                    if (maxtmp <= *(cellVertices->begin() + k))
                      maxtmp = *(cellVertices->begin() + k);
                    if (mintmp >= *(cellVertices->begin() + k))
                      mintmp = *(cellVertices->begin() + k);
                  }
                maxCellBound[k] = maxtmp;
                minCellBound[k] = mintmp;
              }

            atomIdVector.resize(0);

            for (auto i : atomIds)
              {
                auto it = atomCoordinates.begin();
                it      = it + i;
                flag    = true;
                for (unsigned int k = 0; k < dim; k++)
                  {
                    // assert for the cell and processor bounds
                    DFTEFE_AssertWithMsg(
                      minCellBound[k] >= minbound[k] &&
                        maxCellBound[k] >= minbound[k] &&
                        minCellBound[k] <= maxbound[k] &&
                        maxCellBound[k] <= maxbound[k],
                      "Cell Vertices are outside the processor maximum and minimum bounds");
                    double a = minCellBound[k];
                    double b = maxCellBound[k];
                    double c = (*it)[k];

                    DFTEFE_Assert(b >= a);
                    if ((a - tolerance >= c) || (c >= b + tolerance))
                      {
                        flag = false;
                        break;
                      }
                  }
                if (flag)
                  atomIdVector.push_back(i);
              }
            overlappingAtomIdsInCells.push_back(atomIdVector);
          }

        /*  std::cout<<"\n"<<"atoms in cells: "<<"\n";

          std::cout<<"\n";
          auto iter3 = overlappingAtomIdsInCells.begin();
          for( ; iter3 != overlappingAtomIdsInCells.end() ; iter3++)
          {
          std::cout<<"{";
              auto iter2 = iter3->begin();
              for( ; iter2 != iter3->end() ; iter2++)
              {
                  std::cout<<*(iter2)<<",";
              }
              std::cout<<"}, ";
          }*/
      }

      // merge the repeated indices in the cells to get the atomIds within a
      // processor.
      // Function to populate the vector of atom ids in a processor. This gives
      // the number of atomids actually in a processor in a SORTED order.

      template <unsigned int dim>
      void
      getLocalAtomIds(
        std::vector<size_type> &                      atomIdsInProcessor,
        std::vector<std::vector<size_type>> &         overlappingAtomIdsInCells,
        const std::vector<utils::Point> &             atomCoordinates,
        const std::vector<double> &                   minbound,
        const std::vector<double> &                   maxbound,
        const std::vector<std::vector<utils::Point>> &cellVerticesVector,
        const double                                  tolerance,
        const utils::mpi::MPIComm &                   comm)
      {
        std::set<size_type> atomIdsInProcessorTmp;
        auto                iter = overlappingAtomIdsInCells.begin();
        for (; iter != overlappingAtomIdsInCells.end(); iter++)
          {
            auto it = iter->begin();
            for (; it != iter->end(); it++)
              {
                atomIdsInProcessorTmp.insert(*(it));
              }
          }
        for (auto i : atomIdsInProcessorTmp)
          atomIdsInProcessor.push_back(i);
        atomIdsInProcessorTmp.clear();

        int                          rank;
        int                          err = utils::mpi::MPICommRank(comm, &rank);
        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        size_type              nAtoms = atomCoordinates.size();
        std::vector<size_type> processorIdTmp;
        std::vector<size_type> processorIds;
        processorIdTmp.resize(nAtoms, UINT_MAX);
        processorIds.resize(nAtoms, UINT_MAX);
        for (auto Id : atomIdsInProcessor)
          {
            processorIdTmp[Id] = rank;
          }

        err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          processorIdTmp.data(),
          processorIds.data(),
          processorIdTmp.size(),
          utils::mpi::MPIUnsigned,
          utils::mpi::MPIMin,
          comm);
        mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        atomIdsInProcessor.resize(0);
        auto prociter = processorIds.begin();
        for (; prociter != processorIds.end(); prociter++)
          if (*(prociter) == rank)
            atomIdsInProcessor.push_back(prociter - processorIds.begin());
      }

      // Function to renumber the atom ids and populate the old and new atomids
      // vector.
      template <unsigned int dim>
      void
      getNAtomIdsInProcessor(
        std::vector<size_type> &   atomIdsInProcessor,
        std::vector<size_type> &   nAtomIdsInProcessor,
        std::vector<size_type> &   nAtomIdsInProcessorCumulative,
        const utils::mpi::MPIComm &comm,
        const int            nProcs)
      {
        // get the number of atom ids in each proc.
        // also get the cumulative number of atomids in a processor.
        // take a vector = size of no procs init to 0
        // each proc set it as no of atom ids.
        // do mpiallreduce

        int                          rank;
        int                          err = utils::mpi::MPICommRank(comm, &rank);
        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        std::vector<size_type> nAtomIdsInProcessorTmp;
        nAtomIdsInProcessorTmp.resize(nProcs, 0);
        nAtomIdsInProcessor.resize(nProcs, 0);
        nAtomIdsInProcessorTmp[rank] = atomIdsInProcessor.size();

        err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          nAtomIdsInProcessorTmp.data(),
          nAtomIdsInProcessor.data(),
          nAtomIdsInProcessorTmp.size(),
          utils::mpi::MPIUnsigned,
          utils::mpi::MPIMax,
          comm);
        mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);
        nAtomIdsInProcessorTmp.clear();

        size_type sum = 0;
        nAtomIdsInProcessorCumulative.resize(0);
        auto iter = nAtomIdsInProcessor.begin();
        for (; iter != nAtomIdsInProcessor.end(); iter++)
          {
            sum = sum + *(iter);
            nAtomIdsInProcessorCumulative.push_back(sum);
          }
      }

      // Function to renumber the atom ids and populate the old and new atomids
      // vector.
      template <unsigned int dim>
      void
      renumberAtomIds(std::vector<size_type> &oldAtomIds,
                      std::vector<size_type> &newAtomIds,
                      std::vector<size_type> &nAtomIdsInProcessorCumulative,
                      std::vector<size_type> &atomIdsInProcessor,
                      const std::vector<utils::Point> &atomCoordinates,
                      const utils::mpi::MPIComm &      comm,
                      const int                  nProcs)
      {
        /*Here implemet the vector of renumbered atom ids
         * MPI. This is done by asigning a vector of 1...number_atomids to each
         * proc with init maximum_uint and init the proclocalatomid th position
         * of vector to 0,1 in proc 0. Then do natom0+0 .. .do mpi_max at last
         * and return the current proc vector.*/
        // get the set of local atom ids
        // store a vector of size (nAtomIds, UINT_MAX)

        int                          rank;
        int                          err = utils::mpi::MPICommRank(comm, &rank);
        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);

        std::vector<size_type> newAtomIdsTmp;
        size_type              nAtomIds = atomCoordinates.size();
        newAtomIdsTmp.resize(nAtomIds, UINT_MAX);
        oldAtomIds.resize(nAtomIds, UINT_MAX);
        newAtomIds.resize(nAtomIds, UINT_MAX);
        size_type newIds = 0;
        for (auto i : atomIdsInProcessor)
          {
            if (rank == 0)
              newAtomIdsTmp[i] = newIds;
            else
              newAtomIdsTmp[i] =
                newIds + nAtomIdsInProcessorCumulative[rank - 1];
            newIds = newIds + 1;
          }

        // store the vector of new atom ids
        err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          newAtomIdsTmp.data(),
          newAtomIds.data(),
          newAtomIdsTmp.size(),
          utils::mpi::MPIUnsigned,
          utils::mpi::MPIMin,
          comm);
        mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);
        newAtomIdsTmp.clear();

        // create a vector of old atom ids based on the new atom ids vector
        size_type oldIds = 0;
        for (auto i : newAtomIds)
          {
            oldAtomIds[i] = oldIds;
            oldIds        = oldIds + 1;
          }
      }
    } // namespace AtomIdsPartitionInternal

    template <unsigned int dim>
    AtomIdsPartition<dim>::AtomIdsPartition(
      const std::vector<utils::Point> &             atomCoordinates,
      const std::vector<double> &                   minbound,
      const std::vector<double> &                   maxbound,
      const std::vector<std::vector<utils::Point>> &cellVerticesVector,
      const double                                  tolerance,
      const utils::mpi::MPIComm &                   comm)
    {
      // Get the number of processes
      int nProcs;
      utils::mpi::MPICommSize(comm, &nProcs);

      std::vector<std::vector<size_type>> overlappingAtomIdsInCells;
      std::vector<size_type>              atomIds;
      AtomIdsPartitionInternal::getOverlappingAtomIdsInBox<dim>(
        atomIds, atomCoordinates, minbound, maxbound, tolerance, comm);
      AtomIdsPartitionInternal::getOverlappingAtomIdsInCells<dim>(
        overlappingAtomIdsInCells,
        atomIds,
        atomCoordinates,
        minbound,
        maxbound,
        cellVerticesVector,
        tolerance,
        comm);
      AtomIdsPartitionInternal::getLocalAtomIds<dim>(d_atomIdsInProcessor,
                                                     overlappingAtomIdsInCells,
                                                     atomCoordinates,
                                                     minbound,
                                                     maxbound,
                                                     cellVerticesVector,
                                                     tolerance,
                                                     comm);
      AtomIdsPartitionInternal::getNAtomIdsInProcessor<dim>(
        d_atomIdsInProcessor,
        d_nAtomIdsInProcessor,
        d_nAtomIdsInProcessorCumulative,
        comm,
        nProcs);
      AtomIdsPartitionInternal::renumberAtomIds<dim>(
        d_oldAtomIds,
        d_newAtomIds,
        d_nAtomIdsInProcessorCumulative,
        d_atomIdsInProcessor,
        atomCoordinates,
        comm,
        nProcs);
    }

    template <unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition<dim>::nAtomIdsInProcessor() const
    {
      return d_nAtomIdsInProcessor;
    }

    template <unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition<dim>::nAtomIdsInProcessorCumulative() const
    {
      return d_nAtomIdsInProcessorCumulative;
    }

    template <unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition<dim>::oldAtomIds() const
    {
      return d_oldAtomIds;
    }

    template <unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition<dim>::newAtomIds() const
    {
      return d_newAtomIds;
    }

    template <unsigned int dim>
    std::vector<size_type>
    AtomIdsPartition<dim>::locallyOwnedAtomIds() const
    {
      return d_atomIdsInProcessor;
    }
  } // end of namespace basis
} // end of namespace dftefe
