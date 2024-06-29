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
#include <climits>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
#include <utils/Exceptions.h>
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
      template <unsigned int dim>
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
        newAtomIdToEnrichmentIdOffsetTmp.resize(nAtomIds, UINT_MAX);
        newAtomIdToEnrichmentIdOffset.resize(nAtomIds, UINT_MAX);

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
          utils::mpi::MPIUnsignedLong,
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

      template <unsigned int dim>
      void
      getLocalEnrichmentIds(
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

      template <unsigned int dim>
      void
      getOverlappingAtomIdsInBox(
        std::vector<size_type> &         atomIds,
        const std::vector<double> &      rCutoffMax,
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      minbound,
        const std::vector<double> &      maxbound)
      {
        atomIds.resize(0);
        size_type Id = 0;
        bool      flag;

        for (auto it : atomCoordinates)
          {
            flag = false;
            for (unsigned int i = 0; i < dim; i++)
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
            if (flag)
              atomIds.push_back(Id);
            Id++;
          }
      }

      // get the vector of enrichment ids overlapping with a cell given the cell
      // vertices, these vectors are theselves stored as a vector eg.
      // {{10,19,100},{50,150},...} where each integer is an enrichment id.
      /**
       * Function to populate the vector of overlapping enrichment ids in cells.
       */

      template <unsigned int dim>
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
        const utils::mpi::MPIComm &                   comm)
      {
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
        std::vector<global_size_type> enrichmentIdVector;
        global_size_type              enrichmentId;

        auto cellIter = cellVerticesVector.begin();
        for (; cellIter != cellVerticesVector.end(); ++cellIter)
          {
            maxCellBound.resize(dim, 0);
            minCellBound.resize(dim, 0);
            for (unsigned int k = 0; k < dim; k++)
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

            enrichmentIdVector.resize(0);

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
                        double c = (*it)[k] - cutoff;
                        double d = (*it)[k] + cutoff;

                        DFTEFE_Assert(b >= a);
                        DFTEFE_Assert(d >= c);

                        // check that enrichment functions do not spill bounding
                        // box for non-periodic cases.
                        if (!isPeriodicFlags[k] &&
                            !(c - additionalCutoff > minboundGlobalDomain[k] &&
                              d + additionalCutoff < maxboundGlobalDomain[k]))
                          {
                            std::stringstream ss;
                            std::copy(qNumberIter->begin(),
                                      qNumberIter->end(),
                                      std::ostream_iterator<int>(ss, " "));
                            std::string s = ss.str();
                            std::string msg =
                              "The enrichment functions for " + fieldName +
                              " " + s +
                              " may spill to a"
                              " non-periodic face of the triangulation domain which is not allowed."
                              " Increase the "
                              " domain boundary or reduce to the ball radius of the enrichment "
                              " function cutoff. ";
                            if (additionalCutoff != 0)
                              msg +=
                                "Recommended domain boundary increase, if wanted, can be by " +
                                std::to_string(additionalCutoff) +
                                " bohr on each non-periodic side of origin.";
                            utils::throwException<utils::InvalidArgument>(false,
                                                                          msg);
                          }

                        if (!((c < a && d < a) || (c > b && d > b)))
                          flag = true;
                        else
                          {
                            flag = false;
                            break;
                          }
                      }
                    if (flag)
                      {
                        if (newAtomIds[i] != 0)
                          enrichmentId =
                            newAtomIdToEnrichmentIdOffset[newAtomIds[i] - 1] +
                            count;
                        else
                          enrichmentId = count;
                        enrichmentIdVector.push_back(enrichmentId);
                      }
                    count = count + 1;
                  }
              }
            overlappingEnrichmentIdsInCells.push_back(enrichmentIdVector);
          }
      }

      // if an Enrichmenet id  overlapping in the processor is outside the
      // locallyowned range of enrichmentids then it is ghost enrichment id
      /**
       * Function to return the ghost enrichment ids in the processor.
       */

      template <unsigned int dim>
      void
      getGhostEnrichmentIds(
        std::shared_ptr<const AtomIdsPartition<dim>> atomIdsPartition,
        std::vector<global_size_type> &              enrichmentIdsInProcessor,
        std::map<global_size_type, size_type> &      enrichmentIdToOldAtomIdMap,
        std::map<global_size_type, size_type> &      enrichmentIdToQuantumIdMap,
        std::vector<global_size_type> &              ghostEnrichmentIds,
        const std::pair<global_size_type, global_size_type>
          &locallyOwnedEnrichmentIds,
        const std::vector<std::vector<global_size_type>>
          &                                  overlappingEnrichmentIdsInCells,
        const std::vector<global_size_type> &newAtomIdToEnrichmentIdOffset)
      {
        std::vector<size_type>     oldAtomIds = atomIdsPartition->oldAtomIds();
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
            auto j = newAtomIdToEnrichmentIdOffset.begin();
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
                    break;
                  }
              }
            enrichmentIdToOldAtomIdMap.insert({i, oldAtomIds[newAtomId]});
            enrichmentIdToQuantumIdMap.insert({i, qIdPosition});
            if (i < locallyOwnedEnrichmentIds.first ||
                i >= locallyOwnedEnrichmentIds.second)
              {
                ghostEnrichmentIds.push_back(i);
              }
          }
      }
    } // end of namespace EnrichmentIdsPartitionInternal

    template <unsigned int dim>
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
      const utils::mpi::MPIComm &                   comm)
    {
      utils::throwException<utils::InvalidArgument>(
        !(std::any_of(isPeriodicFlags.begin(),
                      isPeriodicFlags.end(),
                      [](bool v) { return v; })),
        "EnrichmentIdsPartition can only handle non-periodic boundary conditions."
        " Contact Developers to get it extended to periodic systems.");

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

      EnrichmentIdsPartitionInternal::getLocalEnrichmentIds<dim>(
        d_locallyOwnedEnrichmentIds,
        d_newAtomIdToEnrichmentIdOffset,
        atomIdsPartition);

      EnrichmentIdsPartitionInternal::getOverlappingAtomIdsInBox<dim>(
        atomIds, rCutoffMax, atomCoordinates, minbound, maxbound);

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
        comm);

      EnrichmentIdsPartitionInternal::getGhostEnrichmentIds<dim>(
        atomIdsPartition,
        d_enrichmentIdsInProcessor,
        d_enrichmentIdToOldAtomIdMap,
        d_enrichmentIdToQuantumIdMap,
        d_ghostEnrichmentIds,
        d_locallyOwnedEnrichmentIds,
        d_overlappingEnrichmentIdsInCells,
        d_newAtomIdToEnrichmentIdOffset);
    }

    template <unsigned int dim>
    std::vector<global_size_type>
    EnrichmentIdsPartition<dim>::newAtomIdToEnrichmentIdOffset() const
    {
      return d_newAtomIdToEnrichmentIdOffset;
    }

    template <unsigned int dim>
    std::vector<std::vector<global_size_type>>
    EnrichmentIdsPartition<dim>::overlappingEnrichmentIdsInCells() const
    {
      return d_overlappingEnrichmentIdsInCells;
    }

    template <unsigned int dim>
    std::pair<global_size_type, global_size_type>
    EnrichmentIdsPartition<dim>::locallyOwnedEnrichmentIds() const
    {
      return d_locallyOwnedEnrichmentIds;
    }

    template <unsigned int dim>
    std::vector<global_size_type>
    EnrichmentIdsPartition<dim>::ghostEnrichmentIds() const
    {
      return d_ghostEnrichmentIds;
    }

    template <unsigned int dim>
    size_type
    EnrichmentIdsPartition<dim>::getAtomId(
      const global_size_type enrichmentId) const
    {
      auto it = d_enrichmentIdToOldAtomIdMap.find(enrichmentId);
      utils::throwException<utils::InvalidArgument>(
        it != d_enrichmentIdToOldAtomIdMap.end(),
        "Cannot find the enrichmentId in locally Owned or Ghost Enrichment Ids of the processor");
      return it->second;
    }

    template <unsigned int dim>
    EnrichmentIdAttribute
    EnrichmentIdsPartition<dim>::getEnrichmentIdAttribute(
      const global_size_type enrichmentId) const
    {
      auto it = d_enrichmentIdToQuantumIdMap.find(enrichmentId);
      utils::throwException<utils::InvalidArgument>(
        it != d_enrichmentIdToQuantumIdMap.end(),
        "Cannot find the enrichmentId in locally Owned or Ghost Enrichment Ids of the processor");
      EnrichmentIdAttribute retStruct;
      retStruct.atomId =
        (d_enrichmentIdToOldAtomIdMap.find(enrichmentId))->second;
      retStruct.localIdInAtom = it->second;
      return retStruct;
    }

    template <unsigned int dim>
    size_type
    EnrichmentIdsPartition<dim>::nLocallyOwnedEnrichmentIds() const
    {
      return (d_locallyOwnedEnrichmentIds.second -
              d_locallyOwnedEnrichmentIds.first);
    }

    template <unsigned int dim>
    size_type
    EnrichmentIdsPartition<dim>::nLocalEnrichmentIds() const
    {
      return (d_locallyOwnedEnrichmentIds.second -
              d_locallyOwnedEnrichmentIds.first) +
             d_ghostEnrichmentIds.size();
    }

    template <unsigned int dim>
    global_size_type
    EnrichmentIdsPartition<dim>::nTotalEnrichmentIds() const
    {
      return d_newAtomIdToEnrichmentIdOffset.back();
    }

    // template <unsigned int dim>
    // std::map<global_size_type, size_type>
    // EnrichmentIdsPartition<dim>::enrichmentIdToNewAtomIdMap() const
    // {
    //   return d_enrichmentIdToNewAtomIdMap;
    // }

    // template <unsigned int dim>
    // std::map<global_size_type, size_type>
    // EnrichmentIdsPartition<dim>::enrichmentIdToQuantumIdMap() const
    // {
    //   return d_enrichmentIdToQuantumIdMap;
    // }

  } // end of namespace basis
} // end of namespace dftefe
