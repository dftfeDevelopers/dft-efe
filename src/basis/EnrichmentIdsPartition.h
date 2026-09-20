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
#include <basis/Defaults.h>
#include <string>
#include <vector>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <map>
#include <basis/PeriodicImageAtomGenerator.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief What an enrichment id is: species + orbital. @p atomId is always
     * the master, never an image. Which images reach an enrichment is a
     * property of the (cell, enrichment) pair, so it is not carried here but
     * in getExtendedAtomIdsForCellEnrich(cellId, enrichIdInCell).
     */
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
     *
     * @section atomidspaces Two atom id spaces
     *
     * A plain atomId is a real atom, always in [0, d_nMasterAtoms), indexing
     * atomCoordinates and atomSymbol. An extendedAtomId additionally denotes a
     * periodic image, pivoting on d_nMasterAtoms: below it a master, at or
     * above it image (value - d_nMasterAtoms) of the truncated list.
     *
     * Both are size_type and look interchangeable, so the naming keeps them
     * apart throughout. Indexing atomSymbol or the coordinates with an
     * extendedAtomId picks the wrong atom, or runs off the end. Use
     * getPositionOfExtendedAtomId instead.
     */
    template <size_type dim>
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
        const utils::mpi::MPIComm &                   comm,
        std::shared_ptr<const PeriodicImageAtomGenerator> imageAtomGenerator =
          nullptr);

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

      global_size_type
      nTotalEnrichmentIds() const;

      std::shared_ptr<const AtomIdsPartition<dim>>
      getAtomIdsPartition() const;

      void
      modifyNumCellsOverlapWithEnrichments(
        const std::vector<std::vector<global_size_type>>
          &overlappingEnrichmentIdsInCells);

      size_type
      nLocalEnrichmentIds() const;

      std::vector<size_type>
      overlappingCellsWithLocalEnrichmentIds() const;

      std::vector<size_type>
      localToCellLocalEIdsVec() const;

      std::vector<global_size_type>
      localToGlobalEnrichmentIds() const;

      std::vector<size_type>
      cellsInLocalEIdVec() const;

      /**
       * @brief The extended atom ids whose ball reaches @p cellIdx for the
       * enrichment at position @p enrichIdInCell, decoded with
       * getPositionOfExtendedAtomId. Without periodicity, just the master.
       */
      std::vector<size_type>
      getExtendedAtomIdsForCellEnrich(const size_type cellIdx,
                              const size_type enrichIdInCell) const;

      /**
       * @brief The same ids as getExtendedAtomIdsForCellEnrich but for every
       * enrichment of @p cellIdx at once, delimited by
       * getExtendedAtomIdOffsetsForAllEnrichInCell. An atom repeats once per
       * enrichment it feeds, so this is not "the atoms in this cell".
       */
      const std::vector<size_type> &
      getExtendedAtomIdsForAllEnrichInCell(const size_type cellIdx) const;

      /**
       * @brief Start offsets into
       * getExtendedAtomIdsForAllEnrichInCell(cellIdx), of size
       * numEnrichInCell + 1: enrichIdInCell owns the entries in
       * [offset[enrichIdInCell], offset[enrichIdInCell + 1]).
       */
      const std::vector<size_type> &
      getExtendedAtomIdOffsetsForAllEnrichInCell(const size_type cellIdx) const;

      /**
       * @brief Position of an extended atom id, hiding the master versus image
       * branch from callers.
       */
      utils::Point
      getPositionOfExtendedAtomId(const size_type extendedAtomId) const;

    private:
      std::vector<global_size_type> d_newAtomIdToEnrichmentIdOffset;
      std::vector<std::vector<global_size_type>>
        d_overlappingEnrichmentIdsInCells;
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

      std::vector<size_type>        d_overlappingCellsWithLocalEnrichmentIds;
      std::vector<size_type>        d_localToCellLocalEIdsVec;
      std::vector<global_size_type> d_localToGlobalEnrichmentIds;
      std::vector<size_type>        d_cellsInLocalEIdVec;

      // Null when the system is non-periodic, in which case no image is ever
      // considered and every list below holds just the master atom.
      std::shared_ptr<const PeriodicImageAtomGenerator> d_imageAtomGenerator;
      size_type                                         d_nMasterAtoms;
      std::vector<utils::Point> d_masterAtomCoordinates;

      // [cellIdx][enrichIdInCell] -> start of that enrichment's extended atom
      // ids within d_cellEnrichIdToExtendedAtomId[cellIdx]. Inner size is
      // d_overlappingEnrichmentIdsInCells[cellIdx].size() + 1.
      std::vector<std::vector<size_type>> d_cellEnrichIdToExtendedAtomIdOffset;
      // [cellIdx] -> the extended atom ids of every enrichment in that cell,
      // concatenated in enrichIdInCell order.
      std::vector<std::vector<size_type>> d_cellEnrichIdToExtendedAtomId;

    }; // end of class EnrichmentIdsPartition
  }    // end of namespace basis
} // end of namespace dftefe
#include "EnrichmentIdsPartition.t.cpp"
#endif // dftefeEnrichement_h
