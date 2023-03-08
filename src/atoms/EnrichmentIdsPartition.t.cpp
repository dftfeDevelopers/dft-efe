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
#include <set>
#include <string>
#include <vector>
#include <atoms/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <atoms/EnrichmentIdsPartition.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <map>

namespace dftefe
{
  namespace atoms
  {
    template <unsigned int dim>
    EnrichmentIdsPartition::EnrichmentIdsPartition( const AtomIdsPartition &                         atomIdsPartition,
                                                    const AtomSphericalDataContainer &               atomSphericalDataContainer,
                                                    const std::vector<std::string> &                 atomSymbol,
                                                    const std::vector<utils::Point> &                atomCoordinates,
                                                    const std::string                                fieldName,                   
                                                    const std::vector<double> &                      minbound,  
                                                    const std::vector<double> &                      maxbound,
                                                    const std::vector<std::vector<utils::Point>> &   cellVerticesVector,
                                                    const MPIComm &                                  comm)
    : d_atomSymbol(atomSymbol)
    , d_atomCoordinates(atomCoordinates)
    , d_fieldName(fieldName)
    , d_minbound(minbound)
    , d_maxbound(maxbound)
    , d_cellVerticesVector(cellVerticesVector)
    {
      d_rCutoffMax.resize(d_atomSymbol.size(), 0.);
      size_type count = 0;
      for (auto it : d_atomSymbol)
      {
        cutoff.resize(0, 0.);
        for (auto i : atomSphericalDataContainer.getSphericalData(it,d_fieldName))  
        {
          atomEnrichmentId.resize(0, 0);
          cutoff.push_back(i.cutoff + i.cutoff / i.smoothness);
        }
        double maxcutoff = std::max_element(cutoff.begin(), cutoff.end());
        d_rCutoffMax[count] = maxcutoff;
        count=count+1;
      }
    }

    template <unsigned int dim>
    void
    EnrichmentIdsPartition::getNewAtomIdToEnrichedIdOffset() const
    {
      // find newAtomIdToEnrichedIdOffset vector
      std::vector<int> newAtomIdToEnrichedIdOffsetTmp;
      size_type nAtomIds = d_atomSymbol.size();
      newAtomIdToEnrichedIdOffsetTmp.resize(nAtomIds,-1);

      std::vector<size_type> localAtomIds = atomIdsPartition.locallyOwnedAtomIds();
      std::vector<size_type> newAtomIds = atomIdsPartition.newAtomIds();
      std::vector<size_type> oldAtomIds = atomIdsPartition.oldAtomIds();
      for( auto i:localAtomIds )
      {
        size_type newId = newAtomIds[i];
        size_type offset = 0;
        for( size_type j=0; j<=newId; j++)
        {
          size_type oldId = oldAtomIds[j];
          offset = offset + atomSphericalDataContainer.nSphericalData(d_atomSymbol[oldId],d_fieldName);
        }
        newAtomIdToEnrichedIdOffsetTmp[newId] = offset;
      }

      int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(&newAtomIdToEnrichedIdOffsetTmp[0], 
        &d_newAtomIdToEnrichedIdOffset[0], newAtomIdToEnrichedIdOffsetTmp.size(), utils::mpi::MPIUnsigned, utils::mpi::MPIMax, comm);
      mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first, "MPI Error:" + mpiIsSuccessAndMsg.second);
      newAtomIdToEnrichedIdOffsetTmp.clear(); 
    }

    template <unsigned int dim>
    void
    EnrichmentIdsPartition::getLocalEnrichedIds() const
    {
      std::vector<size_type> localAtomIds = atomIdsPartition.locallyOwnedAtomIds();
      std::vector<size_type> newAtomIds = atomIdsPartition.newAtomIds();
      size_type front = newAtomIds[localAtomIds.front()];
      size_type back = newAtomIds[localAtomIds.back()];
      if(newAtomIds[front] == 0)
        d_locallyOwnedEnrichedIds.first = 0;
      else
        d_locallyOwnedEnrichedIds.first = d_newAtomIdToEnrichedIdOffset[front-1];
      d_locallyOwnedEnrichedIds.second = d_newAtomIdToEnrichedIdOffset[back];
    }

    // get atom ids with their maximum enrichment ids overlapping with a box , eg:- the processor

    template <unsigned int dim>
    void
    EnrichmentIdsPartition::getOverlappingAtomIdsInBox(std::vector<size_type> & atomIds) const
    {
      atomIds.resize(0);
      size_type           Id = 0;
      bool                flag;

      for (auto it : d_atomCoordinates)
        {
          flag = false;
          for (unsigned int i = 0; i < dim; i++)
          {
            double a = d_minbound[i];
            double b = d_maxbound[i];
            double c = it[i] - d_rCutoffMax[Id];
            double d = it[i] + d_rCutoffMax[Id];

            DFTEFE_Assert(b>=a);
            DFTEFE_Assert(d>=c);
            if (!((c<=a && d<=a) || (c>=b && d>=b)))
              flag = true;
          }
          if (flag)
            atomIds.push_back(Id);
          Id++;
        }
    }

    // get the vector of enriched ids overlapping with a cell given the cell vertices, 
    // these vectors are theselves stored as a vector eg. {{10,19,100},{50,150},...}
    // where each integer is an enrichment id.
   
    template<unsigned int dim>
    void
    EnrichmentIdsPartition::getOverlappingEnrichedIdsInCells() const
    {
      std::vector<size_type> newAtomIds = atomIdsPartition.newAtomIds();
      std::vector<double> minCellBound;
      std::vector<double> maxCellBound;
      std::vector<size_type> enrichedIdVector;
      std::vector<size_type> atomIds;

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

        enrichedIdVector.resize(0);
        atomIds.resize(0);
        getOverlappingAtomIdsInBox(atomIds);


        for (auto i : atomIds)
        {
          auto it = d_atomCoordinates.begin();
          auto iter = d_atomSymbol.begin();
          it = it+i;
          iter = iter+i;
          auto qNumberIter = atomSphericalDataContainer.getQNumbers(*(iter)).begin();
          size_type count = 0;
          for ( ; qNumberIter != atomSphericalDataContainer.getQNumbers(*(iter)).end() ; qNumberIter++)
          {
            bool flag = false;
            // get the sphericaldata struct for the given atom_symbol, field and qnumbers
            auto sphericalData = atomSphericalDataContainer.getSphericalData(*(iter), d_fieldName, *(qNumberIter));
            double cutoff = sphericalData.cutoff + sphericalData.cutoff / sphericalData.smoothness;
            for (unsigned int k = 0; k < dim; i++)
            {
              // assert for the cell and processor bounds
              DFTEFE_AssertWithMsg(minCellBound[k]>=d_minbound[k] && maxCellBound[k]>=d_minbound[k]
                && minCellBound[k]<=d_maxbound[k] && maxCellBound[k]<=d_maxbound[k]
                ,"Cell Vertices are outside the processor maximum and minimum bounds");
              double a = minCellBound[k];
              double b = maxCellBound[k];
              double c = (*it)[k] - cutoff;
              double d = (*it)[k] + cutoff;

              DFTEFE_Assert(b>=a);
              DFTEFE_Assert(d>=c);
              if (!((c<=a && d<=a) || (c>=b && d>=b)))
                flag = true;
            }
            if (flag)
            {
              if(newAtomIds[i] != 0)
                size_type enrichedId = d_newAtomIdToEnrichedIdOffset[newAtomIds[i]-1]+count;
              else
                size_type enrichedId = count;
              enrichedIdVector.push_back(enrichedId);
            }
            count = count+1;
          }
        }
      d_overlappingEnrichedIdsInCells.push_back(enrichedIdVector);
      }
    }

    // if an enrichemnet id  overlapping in the processor is outside the locallyowned range of enrichedids
    // then it is ghost enriched id

    template <unsigned int dim>
    void
    EnrichmentIdsPartition::getGhostEnrichedIds() const
    {
      std::set<size_type> enrichedIdsInProcessorTmp;
      auto iter = d_overlappingEnrichedIdsInCells.begin();
      for( ; iter != d_overlappingEnrichedIdsInCells.end() ; iter++)
      {
        auto it = iter->begin();
        for ( ; it != iter->end() ; it++)
        {
          enrichedIdsInProcessorTmp.insert(*(it));
        }
      }
      for(auto i:enrichedIdsInProcessorTmp)
        d_enrichedIdsInProcessor.push_back(i);
      enrichedIdsInProcessorTmp.clear();

      // define the map from enriched id to newatomid and quantum number id.
      // the map is local to a processor bust stores info of all ghost and local eids of the processor.
      for(auto i:d_enrichedIdsInProcessor )
      {
        auto j = d_newAtomIdToEnrichedIdOffset.begin();        
        for ( ; j!=d_newAtomIdToEnrichedIdOffset.end() ; j++)
        {
          if(*(j) > i)
          {
            newAtomId = j-d_newAtomIdToEnrichedIdOffset.begin();
            if(newAtomId ! = 0)
              qIdPosition = i-d_newAtomIdToEnrichedIdOffset[newAtomId];
            else
              qIdPosition = i;
            break;
          }
        }
        d_enrichedIdToNewAtomIdMap.insert(i , newAtomId);
        d_enrichedIdToQuantumIdMap.insert(i , qIdPosition);
        if( i<d_locallyOwnedEnrichedIds.first || i>=d_locallyOwnedEnrichedIds.second)
        {
          d_ghostEnrichedIds.push_back(i);
        }
      }
    }

    template <unsigned int dim>
    std::vector<size_type>
    EnrichmentIdsPartition::newAtomIdToEnrichedIdOffset() const
    {
      return d_newAtomIdToEnrichedIdOffset;
    }

    template <unsigned int dim>
    std::vector<std::vector<size_type>>
    EnrichmentIdsPartition::overlappingEnrichedIdsInCells() const
    {
      return d_overlappingEnrichedIdsInCells;
    }

    template <unsigned int dim>
    std::pair<size_type,size_type> 
    EnrichmentIdsPartition::locallyOwnedEnrichedIds() const
    {
      return d_locallyOwnedEnrichedIds;
    }

    template <unsigned int dim>
    std::vector<size_type> 
    EnrichmentIdsPartition::ghostEnrichedIds() const
    {
      return d_ghostEnrichedIds;
    }

    template <unsigned int dim>
    std::map<size_type,size_type>
    EnrichmentIdsPartition::enrichedIdToNewAtomIdMap() const
    {
      return d_enrichedIdToNewAtomIdMap;
    }

    template <unsigned int dim>
    std::map<size_type,size_type>
    EnrichmentIdsPartition::enrichedIdToQuantumIdMap() const
    {
      return d_enrichedIdToQuantumIdMap;
    }

  } // end of namespace atoms
} // end of namespace dftefe