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

#ifndef dftefeAtomIdsPartition_h
#define dftefeAtomIdsPartition_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <vector>
#include <string>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Class to get the renumbered Ids of the locally owned Atom ids
     * returns the vector of no of atoms in each processor and the vector of
     * old atom ids. so oldatomid(0) = 2 and so on. So also newatomid(2) = 0;
     * but you do not need to store as oldatomid vector is same as newatomid
     * i.e. memory layout should be  'locally owned enriched ids' should be
     * consecutive integers.
     */
    template <unsigned int dim>
    class AtomIdsPartition
    {
    public:
      /**
       * @brief Constructor takes as coordinates of the atomids from the input file with the processor maximum and
       * minimum bounds. It also takes the cell vertices vector.
       * @param[in] atomCoordinates Vector of Coordinates of the atoms
       * @param[in] minbound Minimum boundary of the processor
       * @param[in] maxbound Maximum boundary of the processor
       * @param[in] cellVerticesVector vector of vectors of all the coordinates
       * of the locally owned cells in the processor
       * @param[in] tolerance set the tolerance for partitioning the atomids
       * @param[in] comm MPI_Comm object if defined with MPI
       * @param[in] nProcs Number of processors if defined with MPI
       */
      AtomIdsPartition(
        const std::vector<utils::Point> &             atomCoordinates,
        const std::vector<double> &                   minbound,
        const std::vector<double> &                   maxbound,
        const std::vector<std::vector<utils::Point>> &cellVerticesVector,
        const double                                  tolerance,
        const utils::mpi::MPIComm &                   comm);

      /**
       * @brief Destructor for the class
       */
      ~AtomIdsPartition() = default;

      /**
       * @brief Function to return the vector of number of atoms in each processor
       */
      std::vector<size_type>
      nAtomIdsInProcessor() const;

      /**
       * @brief Function to return the vector of cumulative number of atoms in each processor
       */
      std::vector<size_type>
      nAtomIdsInProcessorCumulative() const;

      /**
       * @brief Function to return the vector of old atom ids i.e. oldAtomIds[newatomid] = oldatomid
       */
      std::vector<size_type>
      oldAtomIds() const;

      /**
       * @brief Function to return the vector of new atom ids i.e. newAtomIds[oldatomid] = newatomid
       */
      std::vector<size_type>
      newAtomIds() const;

      /**
       * @brief Function to return the vector of locally owned atom is in each processor
       */
      std::vector<size_type>
      locallyOwnedAtomIds() const;

      size_type
      nTotalAtomIds() const;

    private:
      std::vector<size_type> d_nAtomIdsInProcessor;
      std::vector<size_type> d_nAtomIdsInProcessorCumulative;
      std::vector<size_type> d_oldAtomIds;
      std::vector<size_type> d_newAtomIds;
      std::vector<size_type> d_atomIdsInProcessor;
    }; // end of class AtomIdsPartition
  }    // end of namespace basis
} // end of namespace dftefe
#include "AtomIdsPartition.t.cpp"
#endif // dftefeAtomIdsPartition_h
