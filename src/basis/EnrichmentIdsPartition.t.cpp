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
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <climits>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
#include <utils/Exceptions.h>
#include <algorithm>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <map>
#include <iterator>

namespace dftefe
{
  namespace basis
  {
    namespace EnrichmentIdsPartitionInternal
    {
      /*Function to populate the vector of offset . It considers the newAtomIds.
       * For example getNewAtomIdToEnrichmentIdOffset(0) = the no of enrichment
       * fns in new atom id 0... getNewAtomIdToEnrichmentIdOffset(1) = the no of
       * enrichment fns in new atom id 0 + enrichment fns in new atom id 1...
       * and so on.*/
      template <size_type dim>
      void
      getNewAtomIdToEnrichmentIdOffset(
        std::vector<global_size_type> &newAtomIdToEnrichmentIdOffset,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                                     atomSphericalDataContainer,
        std::shared_ptr<const AtomIdsPartition<dim>> atomIdsPartition,
        const std::vector<std::string> &             atomSymbol,
        const std::string                            fieldName,
        const utils::mpi::MPIComm &                  comm)
      {
        // find newAtomIdToEnrichmentIdOffset vector
        std::vector<global_size_type> newAtomIdToEnrichmentIdOffsetTmp;
        size_type                     nAtomIds = atomSymbol.size();
        newAtomIdToEnrichmentIdOffsetTmp.resize(
          nAtomIds, basis::MaxSizeDefaults::GLOBAL_SIZE_TYPE_MAX);
        newAtomIdToEnrichmentIdOffset.resize(
          nAtomIds, basis::MaxSizeDefaults::GLOBAL_SIZE_TYPE_MAX);

        std::vector<size_type> localAtomIds =
          atomIdsPartition->locallyOwnedAtomIds();
        std::vector<size_type> newAtomIds = atomIdsPartition->newAtomIds();
        std::vector<size_type> oldAtomIds = atomIdsPartition->oldAtomIds();
        for (auto i : localAtomIds)
          {
            size_type        newId  = newAtomIds[i];
            global_size_type offset = 0;
            for (size_type j = 0; j <= newId; j++)
              {
                size_type oldId = oldAtomIds[j];
                offset =
                  offset +
                  atomSphericalDataContainer->nSphericalData(atomSymbol[oldId],
                                                             fieldName);
              }
            newAtomIdToEnrichmentIdOffsetTmp[newId] = offset;
          }

        int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          newAtomIdToEnrichmentIdOffsetTmp.data(),
          newAtomIdToEnrichmentIdOffset.data(),
          newAtomIdToEnrichmentIdOffsetTmp.size(),
          utils::mpi::Types<global_size_type>::getMPIDatatype(),
          utils::mpi::MPIMin,
          comm);
        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(err);
        utils::throwException(mpiIsSuccessAndMsg.first,
                              "MPI Error:" + mpiIsSuccessAndMsg.second);
        newAtomIdToEnrichmentIdOffsetTmp.clear();
      }

      /**
       * Function to populate the pair local enrichment ids in the processor. It
       * returns the pair [a,b) where all the enrichment ids in a to b-1 are
       * there in that processor.
       */

      template <size_type dim>
      void
      getLocallyOwnedEnrichmentIds(
        std::pair<global_size_type, global_size_type>
          &                                  locallyOwnedEnrichmentIds,
        const std::vector<global_size_type> &newAtomIdToEnrichmentIdOffset,
        std::shared_ptr<const AtomIdsPartition<dim>> atomIdsPartition)
      {
        std::vector<size_type> localAtomIds =
          atomIdsPartition->locallyOwnedAtomIds();
        std::vector<size_type> newAtomIds = atomIdsPartition->newAtomIds();
        if (localAtomIds.size() != 0)
          {
            size_type front = newAtomIds[localAtomIds.front()];
            size_type back  = newAtomIds[localAtomIds.back()];
            if (front == 0)
              locallyOwnedEnrichmentIds.first = 0;
            else
              locallyOwnedEnrichmentIds.first =
                newAtomIdToEnrichmentIdOffset[front - 1];
            locallyOwnedEnrichmentIds.second =
              newAtomIdToEnrichmentIdOffset[back];
          }
      }

      /**
       * Function to populate the vector of overlapping atom ids based on the
       * maximum cutoff of each atoms enrichment id in a field.*/

      template <size_type dim>
      void
      getOverlappingAtomIdsInBox(
        std::vector<size_type> &         atomIds,
        const std::vector<double> &      rCutoffMax,
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      minbound,
        const std::vector<double> &      maxbound,
        std::shared_ptr<const PeriodicImageAtomGenerator> imageAtomGenerator)
      {
        atomIds.resize(atomCoordinates.size(), 0);
        size_type numAtomIds = 0;
        size_type Id         = 0;
        bool      flag;

        for (auto it : atomCoordinates)
          {
            flag = false;
            for (size_type i = 0; i < dim; i++)
              {
                double a = minbound[i];
                double b = maxbound[i];
                double c = it[i] - rCutoffMax[Id];
                double d = it[i] + rCutoffMax[Id];

                DFTEFE_Assert(b >= a);
                DFTEFE_Assert(d >= c);
                if (!((c < a && d < a) || (c > b && d > b)))
                  flag = true;
                else
                  {
                    flag = false;
                    break;
                  }
              }
            // An atom whose own ball misses the box may still reach it
            // through one of its periodic images. The master id is what gets
            // recorded either way, since an image creates no new enrichment
            // id of its own.
            if (!flag && imageAtomGenerator != nullptr)
              {
                for (auto iImage :
                     imageAtomGenerator->getImageIdsForMasterTrunc(Id))
                  {
                    const utils::Point &imagePosition =
                      imageAtomGenerator->getImagePositionsTrunc()[iImage];
                    bool imageFlag = false;
                    for (size_type i = 0; i < dim; i++)
                      {
                        double a = minbound[i];
                        double b = maxbound[i];
                        double c = imagePosition[i] - rCutoffMax[Id];
                        double d = imagePosition[i] + rCutoffMax[Id];

                        if (!((c < a && d < a) || (c > b && d > b)))
                          imageFlag = true;
                        else
                          {
                            imageFlag = false;
                            break;
                          }
                      }
                    if (imageFlag)
                      {
                        flag = true;
                        break;
                      }
                  }
              }

            if (flag)
              {
                atomIds[numAtomIds] = Id;
                numAtomIds++;
              }
            Id++;
          }
        atomIds.resize(numAtomIds);
      }

      // get the vector of enrichment ids overlapping with a cell given the cell
      // vertices, these vectors are theselves stored as a vector eg.
      // {{10,19,100},{50,150},...} where each integer is an enrichment id.
      /**
       * Function to populate the vector of overlapping enrichment ids in cells.
       */

      template <size_type dim>
      void
      getOverlappingEnrichmentIdsInCells(
        std::vector<std::vector<global_size_type>>
          &                                  overlappingEnrichmentIdsInCells,
        const std::vector<size_type> &       atomIds,
        const std::vector<global_size_type> &newAtomIdToEnrichmentIdOffset,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                                     atomSphericalDataContainer,
        std::shared_ptr<const AtomIdsPartition<dim>> atomIdsPartition,
        const std::vector<std::string> &             atomSymbol,
        const std::vector<utils::Point> &            atomCoordinates,
        const std::string                            fieldName,
        const std::vector<double> &                  minbound,
        const std::vector<double> &                  maxbound,
        double                                       additionalCutoff,
        const std::vector<bool> &                    isPeriodicFlags,
        const std::vector<std::vector<utils::Point>> &cellVerticesVector,
        const utils::mpi::MPIComm &                   comm,
        std::vector<std::vector<size_type>> &cellEnrichIdToExtendedAtomIdOffset,
        std::vector<std::vector<size_type>> &cellEnrichIdToExtendedAtomId,
        std::shared_ptr<const PeriodicImageAtomGenerator> imageAtomGenerator,
        const size_type                                   nMasterAtoms)
      {
        const size_type numCells = cellVerticesVector.size();
        overlappingEnrichmentIdsInCells.resize(numCells);
        cellEnrichIdToExtendedAtomIdOffset.resize(numCells);
        cellEnrichIdToExtendedAtomId.resize(numCells);

        // Upper bounds on what one cell can hold: every orbital of every
        // candidate atom, each with all of that atom's images. They do not
        // depend on the cell, so the scratch below is allocated once and
        // refilled per cell, and each cell's exact-sized vectors are copied
        // out of its used prefix.
        size_type maxEnrichInCell = 0, maxExtendedAtomIdsInCell = 0;
        for (auto i : atomIds)
          {
            const size_type numOrbitals =
              atomSphericalDataContainer->nSphericalData(atomSymbol[i],
                                                         fieldName);
            const size_type numImages =
              (imageAtomGenerator != nullptr) ?
                imageAtomGenerator->getImageIdsForMasterTrunc(i).size() :
                0;
            maxEnrichInCell += numOrbitals;
            maxExtendedAtomIdsInCell += numOrbitals * (1 + numImages);
          }

        std::vector<global_size_type> enrichmentIdScratch(maxEnrichInCell, 0);
        std::vector<size_type>        extendedAtomIdScratch(maxExtendedAtomIdsInCell, 0);
        std::vector<size_type>        extendedAtomIdOffsetScratch(maxEnrichInCell + 1, 0);

        std::vector<double> minboundGlobalDomain(dim, 0.),
          maxboundGlobalDomain(dim, 0.);
        if (!(std::all_of(isPeriodicFlags.begin(),
                          isPeriodicFlags.end(),
                          [](bool v) { return v; })))
          {
            int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              minbound.data(),
              minboundGlobalDomain.data(),
              minbound.size(),
              utils::mpi::MPIDouble,
              utils::mpi::MPIMin,
              comm);

            err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              maxbound.data(),
              maxboundGlobalDomain.data(),
              maxbound.size(),
              utils::mpi::MPIDouble,
              utils::mpi::MPIMax,
              comm);
          }

        std::vector<size_type> newAtomIds = atomIdsPartition->newAtomIds();
        std::vector<double>    minCellBound;
        std::vector<double>    maxCellBound;
        global_size_type              enrichmentId;

        size_type iCell    = 0;
        auto      cellIter = cellVerticesVector.begin();
        for (; cellIter != cellVerticesVector.end(); ++cellIter)
          {
            maxCellBound.resize(dim, 0);
            minCellBound.resize(dim, 0);
            for (size_type k = 0; k < dim; k++)
              {
                auto   cellVertices = cellIter->begin();
                double maxtmp       = *(cellVertices->begin() + k),
                       mintmp       = *(cellVertices->begin() + k);
                for (; cellVertices != cellIter->end(); ++cellVertices)
                  {
                    if (maxtmp < *(cellVertices->begin() + k))
                      maxtmp = *(cellVertices->begin() + k);
                    if (mintmp > *(cellVertices->begin() + k))
                      mintmp = *(cellVertices->begin() + k);
                  }
                maxCellBound[k] = maxtmp;
                minCellBound[k] = mintmp;
              }

            size_type numEnrichInCell = 0, numExtendedAtomIdsInCell = 0;
            extendedAtomIdOffsetScratch[0] = 0;

            for (auto i : atomIds)
              {
                auto it   = atomCoordinates.begin();
                auto iter = atomSymbol.begin();
                it        = it + i;
                iter      = iter + i;
                std::vector<std::vector<int>> qNumberVector =
                  atomSphericalDataContainer->getQNumbers(*(iter), fieldName);
                auto      qNumberIter = qNumberVector.begin();
                size_type count       = 0;
                for (; qNumberIter != qNumberVector.end(); qNumberIter++)
                  {
                    bool flag = false;
                    // get the sphericaldata struct for the given atom_symbol,
                    // field and qnumbers
                    auto sphericalData =
                      atomSphericalDataContainer->getSphericalData(
                        *(iter), fieldName, *(qNumberIter));
                    double cutoff = sphericalData->getCutoff() +
                                    sphericalData->getCutoff() /
                                      sphericalData->getSmoothness() +
                                    additionalCutoff;
                    for (size_type k = 0; k < dim; k++)
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
                        double c = (*it)[k] - cutoff;
                        double d = (*it)[k] + cutoff;

                        DFTEFE_Assert(b >= a);
                        DFTEFE_Assert(d >= c);

                        // // check that enrichment functions do not spill
                        // bounding
                        // // box for non-periodic cases.
                        // if (!isPeriodicFlags[k] &&
                        //     !(c - additionalCutoff > minboundGlobalDomain[k]
                        //     &&
                        //       d + additionalCutoff <
                        //       maxboundGlobalDomain[k]))
                        //   {
                        //     std::stringstream ss;
                        //     std::copy(qNumberIter->begin(),
                        //               qNumberIter->end(),
                        //               std::ostream_iterator<int>(ss, " "));
                        //     std::string s = ss.str();
                        //     std::string msg =
                        //       "The enrichment functions for " + fieldName +
                        //       " " + s + " for the atom " + *(iter) +
                        //       " at the coordinates [" +
                        //       std::to_string((*it)[0]) + " , " +
                        //       std::to_string((*it)[1]) + " , " +
                        //       std::to_string((*it)[2]) +
                        //       "] may spill to a"
                        //       " non-periodic face of the triangulation domain
                        //       which is not allowed." " Increase the " "
                        //       domain boundary or reduce to the ball radius of
                        //       the enrichment " " function cutoff. ";
                        //     if (additionalCutoff != 0)
                        //       msg +=
                        //         "Recommended domain boundary increase, if
                        //         wanted, can be by " +
                        //         std::to_string(additionalCutoff) +
                        //         " bohr on each non-periodic side of origin.";
                        //     utils::throwException<utils::InvalidArgument>(false,
                        //                                                   msg);
                        //   }

                        if (!((c < a && d < a) || (c > b && d > b)))
                          flag = true;
                        else
                          {
                            flag = false;
                            break;
                          }
                      }
                    bool pushed = false;
                    if (flag)
                      {
                        if (newAtomIds[i] != 0)
                          enrichmentId =
                            newAtomIdToEnrichmentIdOffset[newAtomIds[i] - 1] +
                            count;
                        else
                          enrichmentId = count;
                        enrichmentIdScratch[numEnrichInCell] = enrichmentId;
                        // Master here, images below; relies on the generator
                        // excluding (0,0,0) or it would be counted twice.
                        extendedAtomIdScratch[numExtendedAtomIdsInCell]      = i;
                        numExtendedAtomIdsInCell++;
                        pushed = true;
                      }

                    // The same orbital may additionally be reached by some of
                    // this atom's periodic images. They add no enrichment id
                    // of their own, only extra origins to sum over at
                    // evaluation time, and they are admitted by this orbital's
                    // own cutoff: two orbitals of one atom can disagree on
                    // which images reach this cell.
                    if (imageAtomGenerator != nullptr)
                      {
                        for (auto iImage :
                             imageAtomGenerator->getImageIdsForMasterTrunc(i))
                          {
                            const utils::Point &imagePosition =
                              imageAtomGenerator
                                ->getImagePositionsTrunc()[iImage];
                            bool imageFlag = false;
                            for (size_type k = 0; k < dim; k++)
                              {
                                double a = minCellBound[k];
                                double b = maxCellBound[k];
                                double c = imagePosition[k] - cutoff;
                                double d = imagePosition[k] + cutoff;

                                if (!((c < a && d < a) || (c > b && d > b)))
                                  imageFlag = true;
                                else
                                  {
                                    imageFlag = false;
                                    break;
                                  }
                              }

                            if (imageFlag)
                              {
                                if (!pushed)
                                  {
                                    if (newAtomIds[i] != 0)
                                      enrichmentId =
                                        newAtomIdToEnrichmentIdOffset
                                          [newAtomIds[i] - 1] +
                                        count;
                                    else
                                      enrichmentId = count;
                                    enrichmentIdScratch[numEnrichInCell] =
                                      enrichmentId;
                                    pushed = true;
                                  }
                                extendedAtomIdScratch[numExtendedAtomIdsInCell] =
                                  nMasterAtoms + iImage;
                                numExtendedAtomIdsInCell++;
                              }
                          }
                      }

                    if (pushed)
                      {
                        numEnrichInCell++;
                        extendedAtomIdOffsetScratch[numEnrichInCell] = numExtendedAtomIdsInCell;
                      }

                    count = count + 1;
                  }
              }
            overlappingEnrichmentIdsInCells[iCell].resize(numEnrichInCell, 0);
            std::copy(enrichmentIdScratch.begin(),
                      enrichmentIdScratch.begin() + numEnrichInCell,
                      overlappingEnrichmentIdsInCells[iCell].begin());

            cellEnrichIdToExtendedAtomIdOffset[iCell].resize(numEnrichInCell + 1, 0);
            std::copy(extendedAtomIdOffsetScratch.begin(),
                      extendedAtomIdOffsetScratch.begin() + numEnrichInCell + 1,
                      cellEnrichIdToExtendedAtomIdOffset[iCell].begin());

            cellEnrichIdToExtendedAtomId[iCell].resize(numExtendedAtomIdsInCell, 0);
            std::copy(extendedAtomIdScratch.begin(),
                      extendedAtomIdScratch.begin() + numExtendedAtomIdsInCell,
                      cellEnrichIdToExtendedAtomId[iCell].begin());
            iCell++;
          }
      }

      // if an Enrichmenet id  overlapping in the processor is outside the
      // locallyowned range of enrichmentids then it is ghost enrichment id
      /**
       * Function to return the ghost enrichment ids in the processor.
       */

      template <size_type dim>
      void
      getGhostEnrichmentIds(
        std::vector<global_size_type> &localToGlobalEnrichmentIds,
        std::unordered_map<global_size_type, size_type>
          &enrichmentIdToOldAtomIdMap,
        std::unordered_map<global_size_type, size_type>
          &                            enrichmentIdToQuantumIdMap,
        std::vector<global_size_type> &ghostEnrichmentIds,
        std::vector<size_type> &       atomIdsForLocalEnrichments,
        std::vector<size_type> &       overlappingCellsWithLocalEnrichmentIds,
        std::vector<size_type> &       localToCellLocalEIdsVec,
        std::vector<size_type> &       cellsInLocalEIdVec,
        std::shared_ptr<const AtomIdsPartition<dim>> atomIdsPartition,
        const std::pair<global_size_type, global_size_type>
          &locallyOwnedEnrichmentIds,
        const std::vector<std::vector<global_size_type>>
          &                                  overlappingEnrichmentIdsInCells,
        const std::vector<global_size_type> &newAtomIdToEnrichmentIdOffset)
      {
        ghostEnrichmentIds.clear();
        enrichmentIdToOldAtomIdMap.clear();
        enrichmentIdToQuantumIdMap.clear();
        localToGlobalEnrichmentIds.clear();
        std::vector<global_size_type> enrichmentIdsInProcessor(0);
        atomIdsForLocalEnrichments.clear();
        overlappingCellsWithLocalEnrichmentIds.clear();
        localToCellLocalEIdsVec.clear();
        cellsInLocalEIdVec.clear();

        std::vector<size_type>     oldAtomIds = atomIdsPartition->oldAtomIds();
        std::set<size_type>        atomIdsForLocalEnrichmentsSet;
        std::set<global_size_type> enrichmentIdsInProcessorTmp;
        size_type                  newAtomId, qIdPosition;
        auto iter = overlappingEnrichmentIdsInCells.begin();
        for (; iter != overlappingEnrichmentIdsInCells.end(); iter++)
          {
            auto it = iter->begin();
            for (; it != iter->end(); it++)
              {
                enrichmentIdsInProcessorTmp.insert(*(it));
              }
          }
        for (auto i : enrichmentIdsInProcessorTmp)
          enrichmentIdsInProcessor.push_back(i);
        enrichmentIdsInProcessorTmp.clear();

        // define the map from enrichment id to newatomid and quantum number id.
        // the map is local to a processor bust stores info of all ghost and
        // local eids of the processor.
        for (auto i : enrichmentIdsInProcessor)
          {
            bool foundInVec = false;
            auto j          = newAtomIdToEnrichmentIdOffset.begin();
            for (; j != newAtomIdToEnrichmentIdOffset.end(); j++)
              {
                if (*(j) > i)
                  {
                    newAtomId = j - newAtomIdToEnrichmentIdOffset.begin();
                    if (newAtomId != 0)
                      qIdPosition =
                        i - newAtomIdToEnrichmentIdOffset[newAtomId - 1];
                    else
                      qIdPosition = i;
                    foundInVec = true;
                    break;
                  }
              }
            if (!foundInVec)
              {
                utils::throwException(
                  false,
                  "Enrichment Id " + std::to_string(i) +
                    " in Processor not there in  "
                    "newAtomIdToEnrichmentIdOffset vector."
                    "This is an enrichment degrees of freedom partitioning bug in DFTEFE.");
              }

            enrichmentIdToOldAtomIdMap.insert({i, oldAtomIds[newAtomId]});
            enrichmentIdToQuantumIdMap.insert({i, qIdPosition});
            if (i < locallyOwnedEnrichmentIds.first ||
                i >= locallyOwnedEnrichmentIds.second)
              {
                ghostEnrichmentIds.push_back(i);
              }
            atomIdsForLocalEnrichmentsSet.insert(oldAtomIds[newAtomId]);
          }
        std::copy(atomIdsForLocalEnrichmentsSet.begin(),
                  atomIdsForLocalEnrichmentsSet.end(),
                  std::back_inserter(atomIdsForLocalEnrichments));

        for (size_type i = locallyOwnedEnrichmentIds.first;
             i < locallyOwnedEnrichmentIds.second;
             i++)
          localToGlobalEnrichmentIds.push_back(i);
        for (auto i : ghostEnrichmentIds)
          localToGlobalEnrichmentIds.push_back(i);

        for (size_type iLocalEnrich = 0;
             iLocalEnrich < localToGlobalEnrichmentIds.size();
             iLocalEnrich++)
          {
            size_type numCellForLocalEnrich = 0;
            for (size_type iCell = 0;
                 iCell < overlappingEnrichmentIdsInCells.size();
                 iCell++)
              {
                auto enrichInCellVec = overlappingEnrichmentIdsInCells[iCell];
                for (size_type iEnrichInCell = 0;
                     iEnrichInCell < enrichInCellVec.size();
                     iEnrichInCell++)
                  {
                    if (localToGlobalEnrichmentIds[iLocalEnrich] ==
                        enrichInCellVec[iEnrichInCell])
                      {
                        numCellForLocalEnrich += 1;
                        overlappingCellsWithLocalEnrichmentIds.push_back(iCell);
                        localToCellLocalEIdsVec.push_back(iEnrichInCell);
                      }
                  }
              }
            cellsInLocalEIdVec.push_back(numCellForLocalEnrich);
          }
      }
    } // end of namespace EnrichmentIdsPartitionInternal

    template <size_type dim>
    EnrichmentIdsPartition<dim>::EnrichmentIdsPartition(
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                                    atomSphericalDataContainer,
      std::shared_ptr<const AtomIdsPartition<dim>>  atomIdsPartition,
      const std::vector<std::string> &              atomSymbol,
      const std::vector<utils::Point> &             atomCoordinates,
      const std::string                             fieldName,
      const std::vector<double> &                   minbound,
      const std::vector<double> &                   maxbound,
      double                                        additionalCutoff,
      const std::vector<utils::Point> &             globalDomainBoundVec,
      const std::vector<bool> &                     isPeriodicFlags,
      const std::vector<std::vector<utils::Point>> &cellVerticesVector,
      const utils::mpi::MPIComm &                   comm,
      std::shared_ptr<const PeriodicImageAtomGenerator> imageAtomGenerator)
      : d_atomIdsPartition(atomIdsPartition)
      , d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_fieldName(fieldName)
      , d_atomSymbol(atomSymbol)
      , d_imageAtomGenerator(imageAtomGenerator)
      , d_nMasterAtoms(atomCoordinates.size())
      , d_masterAtomCoordinates(atomCoordinates)
    {
      // Without a generator every enrichment is evaluated about its master
      // alone -- right when nothing is periodic, silently wrong otherwise.
      utils::throwException<utils::InvalidArgument>(
        !std::any_of(isPeriodicFlags.begin(),
                     isPeriodicFlags.end(),
                     [](bool v) { return v; }) ||
          imageAtomGenerator != nullptr,
        "EnrichmentIdsPartition: a periodic direction is flagged but no "
        "PeriodicImageAtomGenerator was given, so every enrichment would be "
        "evaluated about its master atom alone and the periodic image "
        "contributions silently dropped.");

      // The image list is a geometric envelope that each consumer then
      // filters by its own cutoff, so check it covers the widest orbital. The
      // formula must match getOverlappingEnrichmentIdsInCells exactly.
      if (imageAtomGenerator != nullptr)
        {
          // Images are looked up by master atom id, so the generator must
          // have been built from this same list in this same order.
          utils::throwException<utils::InvalidArgument>(
            imageAtomGenerator->nMasterAtoms() == atomCoordinates.size(),
            "EnrichmentIdsPartition: the PeriodicImageAtomGenerator was built "
            "from " +
              std::to_string(imageAtomGenerator->nMasterAtoms()) +
              " atoms but this partition has " +
              std::to_string(atomCoordinates.size()) + ".");

          const std::vector<utils::Point> &generatorAtomCoordinates =
            imageAtomGenerator->getAtomCoordinates();
          for (size_type iAtom = 0; iAtom < atomCoordinates.size(); iAtom++)
            for (size_type j = 0; j < dim; j++)
              utils::throwException<utils::InvalidArgument>(
                std::abs(generatorAtomCoordinates[iAtom][j] -
                         atomCoordinates[iAtom][j]) < 1e-12,
                "EnrichmentIdsPartition: the atom ordering differs from the "
                "one the PeriodicImageAtomGenerator was built with, so image "
                "lookups by atom id would return another atom's images.");

          double maxEnrichmentReach = 0.0;
          for (auto it : atomSymbol)
            for (auto i :
                 atomSphericalDataContainer->getSphericalData(it, fieldName))
              maxEnrichmentReach =
                std::max(maxEnrichmentReach,
                         i->getCutoff() + i->getCutoff() / i->getSmoothness() +
                           additionalCutoff);

          utils::throwException<utils::InvalidArgument>(
            maxEnrichmentReach <= imageAtomGenerator->getCutOffTrunc(),
            "EnrichmentIdsPartition: the enrichment reach (" +
              std::to_string(maxEnrichmentReach) +
              " bohr) exceeds the truncated envelope given to "
              "PeriodicImageAtomGenerator (" +
              std::to_string(imageAtomGenerator->getCutOffTrunc()) +
              " bohr), so images that the enrichments need would be missing. "
              "The caller should size that envelope as the max of the default "
              "and the largest enrichment reach.");
        }

      double sum = 0.0;
      for (size_type i = 0; i < globalDomainBoundVec.size(); i++)
        {
          for (size_type j = 0; j < dim; j++)
            {
              if (i != j)
                sum += globalDomainBoundVec[i][j];
            }
        }
      utils::throwException<utils::InvalidArgument>(
        sum < 1e-12,
        "EnrichmentIdsPartition can only handle orthogonal domains with cartesian"
        "coordinate domain vectors {(1,0,0), (0,1,0, (0,0,1)}. Contact Developers"
        " to get it extended to non orthogonal systems.");
      // Note this class cannot handle rotated orthogonal domain also...

      std::vector<double>    rCutoffMax;
      std::vector<size_type> atomIds;
      rCutoffMax.resize(atomSymbol.size(), 0.);
      size_type           count = 0;
      std::vector<double> cutoff;
      for (auto it : atomSymbol)
        {
          cutoff.resize(0, 0.);
          for (auto i :
               atomSphericalDataContainer->getSphericalData(it, fieldName))
            {
              cutoff.push_back(i->getCutoff() +
                               i->getCutoff() / i->getSmoothness() +
                               additionalCutoff);
            }
          double maxcutoff  = *(std::max_element(cutoff.begin(), cutoff.end()));
          rCutoffMax[count] = maxcutoff;
          count             = count + 1;
        }

      EnrichmentIdsPartitionInternal::getNewAtomIdToEnrichmentIdOffset<dim>(
        d_newAtomIdToEnrichmentIdOffset,
        atomSphericalDataContainer,
        atomIdsPartition,
        atomSymbol,
        fieldName,
        comm);

      EnrichmentIdsPartitionInternal::getLocallyOwnedEnrichmentIds<dim>(
        d_locallyOwnedEnrichmentIds,
        d_newAtomIdToEnrichmentIdOffset,
        atomIdsPartition);

      EnrichmentIdsPartitionInternal::getOverlappingAtomIdsInBox<dim>(
        atomIds,
        rCutoffMax,
        atomCoordinates,
        minbound,
        maxbound,
        imageAtomGenerator);

      EnrichmentIdsPartitionInternal::getOverlappingEnrichmentIdsInCells<dim>(
        d_overlappingEnrichmentIdsInCells,
        atomIds,
        d_newAtomIdToEnrichmentIdOffset,
        atomSphericalDataContainer,
        atomIdsPartition,
        atomSymbol,
        atomCoordinates,
        fieldName,
        minbound,
        maxbound,
        additionalCutoff,
        isPeriodicFlags,
        cellVerticesVector,
        comm,
        d_cellEnrichIdToExtendedAtomIdOffset,
        d_cellEnrichIdToExtendedAtomId,
        imageAtomGenerator,
        d_nMasterAtoms);

      EnrichmentIdsPartitionInternal::getGhostEnrichmentIds<dim>(
        d_localToGlobalEnrichmentIds,
        d_enrichmentIdToOldAtomIdMap,
        d_enrichmentIdToQuantumIdMap,
        d_ghostEnrichmentIds,
        d_atomIdsForLocalEnrichments,
        d_overlappingCellsWithLocalEnrichmentIds,
        d_localToCellLocalEIdsVec,
        d_cellsInLocalEIdVec,
        atomIdsPartition,
        d_locallyOwnedEnrichmentIds,
        d_overlappingEnrichmentIdsInCells,
        d_newAtomIdToEnrichmentIdOffset);

      d_oldAtomIdsVec = atomIdsPartition->oldAtomIds();

      for (auto i : d_atomIdsForLocalEnrichments)
        {
          if (std::find(d_atomSymbolsForLocalEnrichments.begin(),
                        d_atomSymbolsForLocalEnrichments.end(),
                        d_atomSymbol[i]) ==
              d_atomSymbolsForLocalEnrichments.end())
            {
              d_atomSymbolsForLocalEnrichments.push_back(d_atomSymbol[i]);
            }
        }
    }

    // Only change overlap of enrich ids in cells,
    // Note: locallyOwnedEnrichedIds depend on atom partitioning
    // hence do not change
    template <size_type dim>
    void
    EnrichmentIdsPartition<dim>::modifyNumCellsOverlapWithEnrichments(
      const std::vector<std::vector<global_size_type>>
        &overlappingEnrichmentIdsInCells)
    {
      // The incoming map keeps a subset of each cell's enrichments, in a
      // different order, so the origins are projected onto the new positions
      // before the old map is overwritten. This is a pure index remap: the
      // geometry has not changed, so a surviving enrichment in a surviving
      // cell has exactly the origins it already had.
      const size_type numCells = overlappingEnrichmentIdsInCells.size();
      std::vector<std::vector<size_type>> cellEnrichIdToExtendedAtomIdOffset(numCells);
      std::vector<std::vector<size_type>> cellEnrichIdToExtendedAtomId(numCells);

      for (size_type iCell = 0; iCell < numCells; iCell++)
        {
          const std::vector<global_size_type> &newEnrichmentIds =
            overlappingEnrichmentIdsInCells[iCell];
          const std::vector<global_size_type> &oldEnrichmentIds =
            d_overlappingEnrichmentIdsInCells[iCell];
          const std::vector<size_type> &oldExtendedAtomIdOffset =
            d_cellEnrichIdToExtendedAtomIdOffset[iCell];
          const std::vector<size_type> &oldExtendedAtomId =
            d_cellEnrichIdToExtendedAtomId[iCell];

          // newToOldEnrichIdInCell[newPosition] = oldPosition, so one entry
          // per surviving enrichment.
          std::vector<size_type> newToOldEnrichIdInCell(
            newEnrichmentIds.size(), 0);
          size_type              numExtendedAtomIdsInCell = 0;
          for (size_type i = 0; i < newEnrichmentIds.size(); i++)
            {
              bool found = false;
              for (size_type j = 0; j < oldEnrichmentIds.size(); j++)
                if (oldEnrichmentIds[j] == newEnrichmentIds[i])
                  {
                    newToOldEnrichIdInCell[i] = j;
                    found                = true;
                    break;
                  }
              // The caller only asserts this, which is compiled out in a
              // release build, so enforce it here where the origins would
              // otherwise be silently wrong.
              // utils::throwException<utils::InvalidArgument>(
              //   found,
              //   "modifyNumCellsOverlapWithEnrichments was given an enrichment "
              //   "id that the cell did not previously overlap, so its origins "
              //   "are unknown. The overlap this partition was built with should "
              //   "cover the orthogonalized enrichment support, which is what "
              //   "additionalCutoff is for.");
              numExtendedAtomIdsInCell += oldExtendedAtomIdOffset[newToOldEnrichIdInCell[i] + 1] -
                                  oldExtendedAtomIdOffset[newToOldEnrichIdInCell[i]];
            }

          cellEnrichIdToExtendedAtomIdOffset[iCell].resize(newEnrichmentIds.size() + 1,
                                                   0);
          cellEnrichIdToExtendedAtomId[iCell].resize(numExtendedAtomIdsInCell, 0);

          size_type numFilled = 0;
          for (size_type i = 0; i < newEnrichmentIds.size(); i++)
            {
              const size_type begin = oldExtendedAtomIdOffset[newToOldEnrichIdInCell[i]];
              const size_type end   = oldExtendedAtomIdOffset[newToOldEnrichIdInCell[i] + 1];
              std::copy(oldExtendedAtomId.begin() + begin,
                        oldExtendedAtomId.begin() + end,
                        cellEnrichIdToExtendedAtomId[iCell].begin() + numFilled);
              numFilled += end - begin;
              cellEnrichIdToExtendedAtomIdOffset[iCell][i + 1] = numFilled;
            }
        }

      d_cellEnrichIdToExtendedAtomIdOffset = cellEnrichIdToExtendedAtomIdOffset;
      d_cellEnrichIdToExtendedAtomId       = cellEnrichIdToExtendedAtomId;

      d_overlappingEnrichmentIdsInCells.resize(0);
      d_overlappingEnrichmentIdsInCells = overlappingEnrichmentIdsInCells;

      EnrichmentIdsPartitionInternal::getGhostEnrichmentIds<dim>(
        d_localToGlobalEnrichmentIds,
        d_enrichmentIdToOldAtomIdMap,
        d_enrichmentIdToQuantumIdMap,
        d_ghostEnrichmentIds,
        d_atomIdsForLocalEnrichments,
        d_overlappingCellsWithLocalEnrichmentIds,
        d_localToCellLocalEIdsVec,
        d_cellsInLocalEIdVec,
        d_atomIdsPartition,
        d_locallyOwnedEnrichmentIds,
        d_overlappingEnrichmentIdsInCells,
        d_newAtomIdToEnrichmentIdOffset);

      for (auto i : d_atomIdsForLocalEnrichments)
        {
          if (std::find(d_atomSymbolsForLocalEnrichments.begin(),
                        d_atomSymbolsForLocalEnrichments.end(),
                        d_atomSymbol[i]) ==
              d_atomSymbolsForLocalEnrichments.end())
            {
              d_atomSymbolsForLocalEnrichments.push_back(d_atomSymbol[i]);
            }
        }
    }

    template <size_type dim>
    std::vector<global_size_type>
    EnrichmentIdsPartition<dim>::newAtomIdToEnrichmentIdOffset() const
    {
      return d_newAtomIdToEnrichmentIdOffset;
    }

    template <size_type dim>
    std::vector<std::vector<global_size_type>>
    EnrichmentIdsPartition<dim>::overlappingEnrichmentIdsInCells() const
    {
      return d_overlappingEnrichmentIdsInCells;
    }

    template <size_type dim>
    std::pair<global_size_type, global_size_type>
    EnrichmentIdsPartition<dim>::locallyOwnedEnrichmentIds() const
    {
      return d_locallyOwnedEnrichmentIds;
    }

    template <size_type dim>
    std::vector<global_size_type>
    EnrichmentIdsPartition<dim>::ghostEnrichmentIds() const
    {
      return d_ghostEnrichmentIds;
    }

    template <size_type dim>
    size_type
    EnrichmentIdsPartition<dim>::getAtomId(
      const global_size_type enrichmentId) const
    {
      auto it = d_enrichmentIdToOldAtomIdMap.find(enrichmentId);
      utils::throwException(
        it != d_enrichmentIdToOldAtomIdMap.end(),
        "Cannot find the enrichmentId " + std::to_string(enrichmentId) +
          " in locally Owned or Ghost Enrichment Ids of the processor");
      return it->second;

      // auto it = std::find(d_enrichmentIdsVec.begin(),
      // d_enrichmentIdsVec.end(), enrichmentId); DFTEFE_AssertWithMsg(it !=
      // d_enrichmentIdsVec.end(),
      //   "Cannot find the enrichmentId in locally Owned or Ghost Enrichment
      //   Ids of the processor");
      // int index = std::distance(d_enrichmentIdsVec.begin(), it);

      // return d_oldAtomIdsFromEnrichIdsVec[index];
    }

    template <size_type dim>
    EnrichmentIdAttribute
    EnrichmentIdsPartition<dim>::getEnrichmentIdAttribute(
      const global_size_type enrichmentId) const
    {
      auto it = d_enrichmentIdToQuantumIdMap.find(enrichmentId);
      DFTEFE_AssertWithMsg(
        it != d_enrichmentIdToQuantumIdMap.end(),
        "Cannot find the enrichmentId in locally Owned or Ghost Enrichment Ids of the processor");
      EnrichmentIdAttribute retStruct;
      retStruct.atomId =
        (d_enrichmentIdToOldAtomIdMap.find(enrichmentId))->second;
      retStruct.localIdInAtom = it->second;

      // auto it = std::find(d_enrichmentIdsVec.begin(),
      // d_enrichmentIdsVec.end(), enrichmentId); DFTEFE_AssertWithMsg(it !=
      // d_enrichmentIdsVec.end(),
      //   "Cannot find the enrichmentId in locally Owned or Ghost Enrichment
      //   Ids of the processor");
      // int index = std::distance(d_enrichmentIdsVec.begin(), it);
      // EnrichmentIdAttribute retStruct;
      // retStruct.atomId = d_oldAtomIdsFromEnrichIdsVec[index];
      // retStruct.localIdInAtom = d_quantumIdsFromEnrichIdsVec[index];

      // EnrichmentIdAttribute retStruct;
      // auto it = std::upper_bound(d_newAtomIdToEnrichmentIdOffset.begin(),
      //   d_newAtomIdToEnrichmentIdOffset.end(), enrichmentId);
      // DFTEFE_AssertWithMsg(
      //   it != d_newAtomIdToEnrichmentIdOffset.end(),
      //   "Cannot find the enrichmentId in locally Owned or Ghost Enrichment
      //   Ids of the processor");
      // size_type newAtomId = it - d_newAtomIdToEnrichmentIdOffset.begin();
      // retStruct.atomId = d_oldAtomIdsVec[newAtomId];
      // retStruct.localIdInAtom =  (newAtomId != 0) ?
      //     (enrichmentId - d_newAtomIdToEnrichmentIdOffset[newAtomId - 1]) :
      //     enrichmentId;

      return retStruct;
    }

    template <size_type dim>
    size_type
    EnrichmentIdsPartition<dim>::nLocallyOwnedEnrichmentIds() const
    {
      return (d_locallyOwnedEnrichmentIds.second -
              d_locallyOwnedEnrichmentIds.first);
    }

    template <size_type dim>
    global_size_type
    EnrichmentIdsPartition<dim>::nTotalEnrichmentIds() const
    {
      return d_newAtomIdToEnrichmentIdOffset.back();
    }

    template <size_type dim>
    std::shared_ptr<const AtomIdsPartition<dim>>
    EnrichmentIdsPartition<dim>::getAtomIdsPartition() const
    {
      return d_atomIdsPartition;
    }

    template <size_type dim>
    size_type
    EnrichmentIdsPartition<dim>::nEnrichmentIds(const size_type atomId) const
    {
      return d_atomSphericalDataContainer->nSphericalData(d_atomSymbol[atomId],
                                                          d_fieldName);
    }

    template <size_type dim>
    std::vector<size_type>
    EnrichmentIdsPartition<dim>::getAtomIdsForLocalEnrichments() const
    {
      return d_atomIdsForLocalEnrichments;
    }

    template <size_type dim>
    std::vector<std::string>
    EnrichmentIdsPartition<dim>::getAtomSymbolsForLocalEnrichments() const
    {
      return d_atomSymbolsForLocalEnrichments;
    }

    template <size_type dim>
    size_type
    EnrichmentIdsPartition<dim>::nLocalEnrichmentIds() const
    {
      return (d_locallyOwnedEnrichmentIds.second -
              d_locallyOwnedEnrichmentIds.first) +
             d_ghostEnrichmentIds.size();
    }

    template <size_type dim>
    std::vector<size_type>
    EnrichmentIdsPartition<dim>::overlappingCellsWithLocalEnrichmentIds() const
    {
      return d_overlappingCellsWithLocalEnrichmentIds;
    }

    template <size_type dim>
    std::vector<size_type>
    EnrichmentIdsPartition<dim>::localToCellLocalEIdsVec() const
    {
      return d_localToCellLocalEIdsVec;
    }

    template <size_type dim>
    std::vector<global_size_type>
    EnrichmentIdsPartition<dim>::localToGlobalEnrichmentIds() const
    {
      return d_localToGlobalEnrichmentIds;
    }

    template <size_type dim>
    std::vector<size_type>
    EnrichmentIdsPartition<dim>::getExtendedAtomIdsForCellEnrich(
      const size_type cellIdx,
      const size_type enrichIdInCell) const
    {
      const std::vector<size_type> &ids = getExtendedAtomIdsForAllEnrichInCell(cellIdx);
      const std::vector<size_type> &offset =
        getExtendedAtomIdOffsetsForAllEnrichInCell(cellIdx);
      return std::vector<size_type>(ids.begin() + offset[enrichIdInCell],
                                    ids.begin() + offset[enrichIdInCell + 1]);
    }

    template <size_type dim>
    const std::vector<size_type> &
    EnrichmentIdsPartition<dim>::getExtendedAtomIdsForAllEnrichInCell(
      const size_type cellIdx) const
    {
      return d_cellEnrichIdToExtendedAtomId[cellIdx];
    }

    template <size_type dim>
    const std::vector<size_type> &
    EnrichmentIdsPartition<dim>::getExtendedAtomIdOffsetsForAllEnrichInCell(
      const size_type cellIdx) const
    {
      return d_cellEnrichIdToExtendedAtomIdOffset[cellIdx];
    }

    template <size_type dim>
    utils::Point
    EnrichmentIdsPartition<dim>::getPositionOfExtendedAtomId(
      const size_type extendedAtomId) const
    {
      if (extendedAtomId < d_nMasterAtoms)
        return d_masterAtomCoordinates[extendedAtomId];

      DFTEFE_AssertWithMsg(
        d_imageAtomGenerator != nullptr,
        "An extended atom id refers to a periodic image, but no "
        "PeriodicImageAtomGenerator was given to EnrichmentIdsPartition.");
      return d_imageAtomGenerator
        ->getImagePositionsTrunc()[extendedAtomId - d_nMasterAtoms];
    }

    template <size_type dim>
    std::vector<size_type>
    EnrichmentIdsPartition<dim>::cellsInLocalEIdVec() const
    {
      return d_cellsInLocalEIdVec;
    }

  } // end of namespace basis
} // end of namespace dftefe
