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
        double                                       additionalCutoff,
        const std::vector<utils::Point> &            globalDomainBoundVec,
        const std::vector<bool> &                    isPeriodicFlags,
        const std::vector<std::vector<utils::Point>> &cellVerticesVector,
        const utils::mpi::MPIComm &                   comm);

      /**
       * @brief Destructor
       */
      ~EnrichmentIdsPartition() = default;

      std::vector<global_size_type>
      newAtomIdToEnrichmentIdOffset() const;

      std::vector<std::vector<global_size_type>>
      overlappingEnrichmentIdsInCells() const;

      std::pair<global_size_type, global_size_type>
      locallyOwnedEnrichmentIds() const;

      std::vector<global_size_type>
      ghostEnrichmentIds() const;

      // Works with the global enrichmentid

      size_type
      getAtomId(const global_size_type enrichmentId) const;

      std::vector<size_type>
      getAtomIdsForLocalEnrichments() const;

      std::vector<std::string>
      getAtomSymbolsForLocalEnrichments() const;

      EnrichmentIdAttribute
      getEnrichmentIdAttribute(const global_size_type enrichmentId) const;

      size_type
      nEnrichmentIds(const size_type atomId) const;

      size_type
      nLocallyOwnedEnrichmentIds() const;

      size_type
      nLocalEnrichmentIds() const;

      global_size_type
      nTotalEnrichmentIds() const;

      std::shared_ptr<const AtomIdsPartition<dim>>
      getAtomIdsPartition() const;

      void
      modifyNumCellsOverlapWithEnrichments(
        const std::vector<std::vector<global_size_type>>
          &overlappingEnrichmentIdsInCells);

      // std::map<size_type, size_type>
      // enrichmentIdToNewAtomIdMap() const;

      // std::map<size_type, size_type>
      // enrichmentIdToQuantumIdMap() const;

      /** The data members are as follows.
       */

    private:
      std::vector<global_size_type> d_newAtomIdToEnrichmentIdOffset;
      std::vector<std::vector<global_size_type>>
                                    d_overlappingEnrichmentIdsInCells;
      std::vector<global_size_type> d_enrichmentIdsInProcessor;
      std::pair<global_size_type, global_size_type> d_locallyOwnedEnrichmentIds;
      std::vector<global_size_type>                 d_ghostEnrichmentIds;
      std::unordered_map<global_size_type, size_type>
        d_enrichmentIdToOldAtomIdMap;
      std::unordered_map<global_size_type, size_type>
                             d_enrichmentIdToQuantumIdMap;
      std::vector<size_type> d_oldAtomIdsVec;
      const std::shared_ptr<const AtomIdsPartition<dim>> d_atomIdsPartition;
      const std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                      d_atomSphericalDataContainer;
      std::string                     d_fieldName;
      const std::vector<std::string> &d_atomSymbol;

      std::vector<size_type>   d_atomIdsForLocalEnrichments;
      std::vector<std::string> d_atomSymbolsForLocalEnrichments;

      // std::vector<global_size_type> d_enrichmentIdsVec;
      // std::vector<size_type> d_oldAtomIdsFromEnrichIdsVec;
      // std::vector<size_type> d_quantumIdsFromEnrichIdsVec;

    }; // end of class EnrichmentIdsPartition
  }    // end of namespace basis
} // end of namespace dftefe
#include "EnrichmentIdsPartition.t.cpp"
#endif // dftefeEnrichement_h
