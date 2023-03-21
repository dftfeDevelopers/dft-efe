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

#ifndef dftefeEnrichmentIdsPartition_h
#define dftefeEnrichmentIdsPartition_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <set>
#include <string>
#include <vector>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <map>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Class to get the gost and locally owned enriched ids from the renumbered atom ids in Atom Partition 
     * i.e. memory layout should be 'locally owned enriched ids which would be contiguous' -> 'ghost enriched ids'
     * The class gives us the vector of cell enriched Ids, locallyowned enriched ids range, ghost enriched ids.
     */
    template <unsigned int dim>
    class EnrichmentIdsPartition
    {
    public:
      /**
       * @brief Constructor takes as coordinates of the atomids , vector of aomsymbol from the input file with the
       * processor maximum and minimum bounds. It also takes the cell vertices vector and the fieldname.
       * @param[in] atomIdsPartition Object of class AtomIdsPartition
       * @param[in] atomSphericalDataContainer Object of class AtomSphericalDataContainer
       * @param[in] atomCoordinates Vector of Coordinates of the atoms
       * @param[in] fieldName Fieldname wanted
       * @param[in] minbound Minimum boundary of the processor
       * @param[in] maxbound Maximum boundary of the processor
       * @param[in] cellVerticesVector vector of vectors of all the coordinates of the locally owned cells in the processor
       * @param[in] comm MPI_Comm object if defined with MPI
       * @return 
       */
      EnrichmentIdsPartition( const atoms::AtomSphericalDataContainer &        atomSphericalDataContainer,
                              const AtomIdsPartition<dim> &                    atomIdsPartition,
                              const std::vector<std::string> &                 atomSymbol,
                              const std::vector<utils::Point> &                atomCoordinates,
                              const std::string                                fieldName,                   
                              const std::vector<double> &                      minbound,  
                              const std::vector<double> &                      maxbound,
                              const std::vector<std::vector<utils::Point>> &   cellVerticesVector,
                              const utils::mpi::MPIComm &                      comm); 

      /**
       * @brief Destructor
       */
      ~EnrichmentIdsPartition() = default;

      /**
       * @brief Function to populate the vector of offset . It considers the newAtomIds.
       * For example getNewAtomIdToEnrichedIdOffset(0) = the no of enrichment fns in new atom id 0...
       * getNewAtomIdToEnrichedIdOffset(1) = the no of enrichment fns in new atom id 0 + enrichment fns in new atom id 1...
       * and so on.
       */
      void
      getNewAtomIdToEnrichedIdOffset() const;

      /**
       * @brief Function to populate the pair local enrichment ids in the processor. It returns the pair [a,b) where all
       * the enriched ids in a to b-1 are there in that processor.
       */
      void
      getLocalEnrichedIds() const;

      /**
       * @brief Function to populate the vector of overlapping atom ids based on the maximum cutoff of each atoms enriched 
       * id in a field.
       */
      void
      getOverlappingAtomIdsInBox(std::vector<size_type> & atomIds) const;

      /**
       * @brief Function to populate the vector of overlapping enriched ids in cells.
       */
      void
      getOverlappingEnrichedIdsInCells() const;

      /**
       * @brief Function to return the ghost enriched ids in the processor.
       */
      void
      getGhostEnrichedIds() const;

      std::vector<size_type>
      newAtomIdToEnrichedIdOffset() const;

      std::vector<std::vector<size_type>>
      overlappingEnrichedIdsInCells() const;

      std::pair<size_type,size_type> 
      locallyOwnedEnrichedIds() const;

      std::vector<size_type> 
      ghostEnrichedIds() const;

      std::map<size_type,size_type>
      enrichedIdToNewAtomIdMap() const;

      std::map<size_type,size_type>
      enrichedIdToQuantumIdMap() const;

      /** The data members are as follows.
      */

    private:
      const std::vector<std::string>                      d_atomSymbol;
      const std::vector<utils::Point>                     d_atomCoordinates;
      const std::string                                   d_fieldName;
      const std::vector<double>                           d_minbound;
      const std::vector<double>                           d_maxbound;
      const std::vector<std::vector<utils::Point>>        d_cellVerticesVector;
      std::vector<size_type>                              d_newAtomIdToEnrichedIdOffset;
      std::vector<std::vector<size_type>>                 d_overlappingEnrichedIdsInCells;
      std::vector<double>                                 d_rCutoffMax;
      std::vector<size_type>                              d_enrichedIdsInProcessor;
      std::pair<size_type,size_type>                      d_locallyOwnedEnrichedIds;
      std::vector<size_type>                              d_ghostEnrichedIds;
      std::map<size_type,size_type>                       d_enrichedIdToNewAtomIdMap;
      std::map<size_type,size_type>                       d_enrichedIdToQuantumIdMap;

    }; // end of class EnrichmentIdsPartition
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeEnrichement_h