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
    struct EnrichmentIdAttribute
    {
      size_type atomId;
      size_type localIdInAtom;
    };
    /**
     * @brief Class to get the gost and locally owned enrichment ids from the renumbered atom ids in Atom Partition
     * i.e. memory layout should be 'locally owned enrichment ids which would be
     * contiguous' -> 'ghost enrichment ids' The class gives us the vector of
     * cell enrichment Ids, locallyowned enrichment ids range, ghost enrichment
     * ids.
     */
    template <unsigned int dim>
    class EnrichmentIdsPartition
    {
    public:
      /**
       * @brief Constructor takes as coordinates of the atomids , vector of aomsymbol from the input file with the
       * processor maximum and minimum bounds. It also takes the cell vertices
       * vector and the fieldname.
       * @param[in] atomIdsPartition Object of class AtomIdsPartition
       * @param[in] atomSphericalDataContainer Object of class
       * AtomSphericalDataContainer
       * @param[in] atomCoordinates Vector of Coordinates of the atoms
       * @param[in] fieldName Fieldname wanted
       * @param[in] minbound Minimum boundary of the processor
       * @param[in] maxbound Maximum boundary of the processor
       * @param[in] cellVerticesVector vector of vectors of all the coordinates
       * of the locally owned cells in the processor
       * @param[in] comm MPI_Comm object if defined with MPI
       * @return
       */
      EnrichmentIdsPartition(
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                                     atomSphericalDataContainer,
        std::shared_ptr<const AtomIdsPartition<dim>> atomIdsPartition,
        const std::vector<std::string> &             atomSymbol,
        const std::vector<utils::Point> &            atomCoordinates,
        const std::string                            fieldName,
        const std::vector<double> &                  minbound,
        const std::vector<double> &                  maxbound,
        const std::vector<std::vector<utils::Point>> &cellVerticesVector,
        const utils::mpi::MPIComm &                  comm);

      /**
       * @brief Destructor
       */
      ~EnrichmentIdsPartition() = default;

      std::vector<size_type>
      newAtomIdToEnrichmentIdOffset() const;

      std::vector<std::vector<size_type>>
      overlappingEnrichmentIdsInCells() const;

      std::pair<size_type, size_type>
      locallyOwnedEnrichmentIds() const;

      std::vector<size_type>
      ghostEnrichmentIds() const;

      size_type
      getAtomId(const size_type enrichmentId) const;

      EnrichmentIdAttribute
      getEnrichmentIdAttribute(const size_type enrichmentId) const; 

      // std::map<size_type, size_type>
      // enrichmentIdToNewAtomIdMap() const;

      // std::map<size_type, size_type>
      // enrichmentIdToQuantumIdMap() const;

      /** The data members are as follows.
       */

    private:
      std::vector<size_type>              d_newAtomIdToEnrichmentIdOffset;
      std::vector<std::vector<size_type>> d_overlappingEnrichmentIdsInCells;
      std::vector<size_type>              d_enrichmentIdsInProcessor;
      std::pair<size_type, size_type>     d_locallyOwnedEnrichmentIds;
      std::vector<size_type>              d_ghostEnrichmentIds;
      std::map<size_type, size_type>      d_enrichmentIdToOldAtomIdMap;
      std::map<size_type, size_type>      d_enrichmentIdToQuantumIdMap;

    }; // end of class EnrichmentIdsPartition
  }    // end of namespace basis
} // end of namespace dftefe
#include "EnrichmentIdsPartition.t.cpp"
#endif // dftefeEnrichement_h
