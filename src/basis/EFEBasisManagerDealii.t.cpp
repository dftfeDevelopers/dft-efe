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
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_tools.h>
#include "TriangulationDealiiParallel.h"
#include "TriangulationDealiiSerial.h"
#include "FECellDealii.h"
#include <utils/Point.h>
#include <utils/PointImpl.h>


namespace dftefe
{
  namespace basis
  {

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>::EFEBasisManagerDealii(
      std::shared_ptr<const TriangulationBase> triangulation,
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const size_type                  feOrder,
      const double                     atomPartitionTolerance,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      const utils::mpi::MPIComm &      comm)
      : d_isVariableDofsPerCell(true)
      , d_totalRanges(2) // Classical and Enriched
      , d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(0, utils::Point(dim, 0.0))
      , d_fieldName(fieldName)
      , d_comm(comm)
      , d_atomPartitionTolerance(atomPartitionTolerance)
      , d_overlappingEnrichmentIdsInCells(0)
      , d_locallyOwnedRanges(0)
      , d_globalRanges(0)
      , d_ghostEnrichmentGlobalIds(0)
      , d_enrichmentIdsPartition(nullptr)
      , d_atomIdsPartition(nullptr)
    {
      d_isOrthogonalized = false;
      d_dofHandler         = std::make_shared<dealii::DoFHandler<dim>>();
      // making the classical and enriched dofs in the dealii mesh here
      reinit(triangulation, feOrder, atomSymbolVec, atomCoordinatesVec);
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>::EFEBasisManagerDealii(
      std::shared_ptr<const TriangulationBase> triangulation,
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const size_type                  feOrder,
      const double                     atomPartitionTolerance,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      const utils::mpi::MPIComm &      comm,
      std::shared_ptr<const quadrature::QuadratureRuleAttributes> l2ProjQuadAttr,
      std::shared_ptr<const linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContext)
      : d_isVariableDofsPerCell(true)
      , d_totalRanges(2) // Classical and Enriched
      , d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(0, utils::Point(dim, 0.0))
      , d_fieldName(fieldName)
      , d_comm(comm)
      , d_atomPartitionTolerance(atomPartitionTolerance)
      , d_overlappingEnrichmentIdsInCells(0)
      , d_locallyOwnedRanges(0)
      , d_globalRanges(0)
      , d_ghostEnrichmentGlobalIds(0)
      , d_enrichmentIdsPartition(nullptr)
      , d_atomIdsPartition(nullptr)
      , d_linAlgOpContext(linAlgOpContext)
    {
      d_isOrthogonalized = true;
      d_dofHandler         = std::make_shared<dealii::DoFHandler<dim>>();
      // making the classical and enriched dofs in the dealii mesh here
      reinit(triangulation, feOrder, atomSymbolVec, atomCoordinatesVec);
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    void
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>::reinit(
      std::shared_ptr<const TriangulationBase> triangulation,
      const size_type                          feOrder,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec)
    {
      d_atomCoordinatesVec = atomCoordinatesVec;
      d_atomSymbolVec = atomSymbolVec;
      // Create Classical FE dof_handler
      dealii::FE_Q<dim>                       feElem(feOrder);
      const TriangulationDealiiParallel<dim> *dealiiParallelTria =
        dynamic_cast<const TriangulationDealiiParallel<dim> *>(
          triangulation.get());

      if (!(dealiiParallelTria == nullptr))
        {
          //          d_dofHandler->initialize(dealiiParallelTria->returnDealiiTria(),
          //          feElem);
          d_dofHandler->reinit(dealiiParallelTria->returnDealiiTria());
          d_dofHandler->distribute_dofs(feElem);
        }
      else
        {
          const TriangulationDealiiSerial<dim> *dealiiSerialTria =
            dynamic_cast<const TriangulationDealiiSerial<dim> *>(
              triangulation.get());

          if (!(dealiiSerialTria == nullptr))
            {
              //              d_dofHandler->initialize(dealiiSerialTria->returnDealiiTria(),
              //              feElem);
              d_dofHandler->reinit(dealiiSerialTria->returnDealiiTria());
              d_dofHandler->distribute_dofs(feElem);
            }
          else
            {
              utils::throwException(
                false,
                "reinit() in EFEBasisManagerDealii is not able to re cast the Triangulation.");
            }
        }

      // TODO check how to pass the triangulation to dofHandler
      d_triangulation = triangulation;

      typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        d_dofHandler->begin_active();
      typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        d_dofHandler->end();

      cell = d_dofHandler->begin_active();
      endc = d_dofHandler->end();

      for (; cell != endc; cell++)
        if (cell->is_locally_owned())
          {
            std::shared_ptr<FECellDealii<dim>> cellDealii =
              std::make_shared<FECellDealii<dim>>(cell);

            d_localCells.push_back(cellDealii);
            d_locallyOwnedCells.push_back(cellDealii);
          }


      cell = d_dofHandler->begin_active();
      for (; cell != endc; cell++)
        if (cell->is_ghost())
          {
            std::shared_ptr<FECellDealii<dim>> cellDealii =
              std::make_shared<FECellDealii<dim>>(cell);
            d_localCells.push_back(cellDealii);
          }

      //----------------ENRICHEMNT---------------------//

      // Add enriched FE dofs with on top of the classical dofs.
      // Note that the enriched FE dofs are already partitioned elswhere
      // This implies we pass mpi communicator here also.
      // So we only work with local enrichement ids which are created.

      if(d_isOrthogonalized)
      {
        // Set the CFE basis data
        std::shared_ptr<FEBasisManager> basisManager =   std::make_shared<FEBasisManagerDealii<dim>>(triangulationBase, feDegree);
        if (std::any_of(triangulation->getPeriodicFlags().begin(), triangulation->getPeriodicFlags().end(), [](bool i) { return i == true }))
          utils::throwException(
          false,
          "The domain has to be non-periodic only.");
        std::string constraintName = "HangingWithHomogeneous";
        std::vector<FEConstraintsDealii<ValueTypeBasisData, memorySpace>>> constraintsVec;
        constraintsVec.resize(1);
        for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
          constraintsVec[i]();
        
        constraintsVec[0].clear();
        constraintsVec[0].makeHangingNodeConstraint(basisManager);
        constraintsVec[0].setHomogeneousDirichletBC();
        constraintsVec[0].close();

        std::map<std::string,
                Constraints<ValueTypeBasisData, memorySpace>> constraintsMap;

        constraintsMap[constraintName] = constraintsVec[0];

        BasisStorageAttributesBoolMap basisAttrMap;
        basisAttrMap[BasisStorageAttributes::StoreValues] = true;
        basisAttrMap[BasisStorageAttributes::StoreGradient] = false;
        basisAttrMap[BasisStorageAttributes::StoreHessian] = false;
        basisAttrMap[BasisStorageAttributes::StoreOverlap] = false;
        basisAttrMap[BasisStorageAttributes::StoreGradNiGradNj] = false;
        basisAttrMap[BasisStorageAttributes::StoreJxW] = false;
        basisAttrMap[BasisStorageAttributes::StoreQuadRealPoints] = false;

        // Set up the FE Basis Data Storage
        std::shared_ptr<BasisDataStorage<ValueTypeBasisData, memorySpace>> cfeBasisDataStorage =
          std::make_shared<FEBasisDataStorageDealii<ValueTypeBasisData, memorySpace, dim>>
          (basisManager, l2ProjQuadAttr, basisAttrMap);

        if (quadFamily == quadrature::QuadratureFamily::GAUSS || quadrature::QuadratureFamily::GLL)
        {
          feBasisData->evaluateBasisData(l2ProjQuadAttr, basisAttrMap);
        }
        else
          utils::throwException<utils::InvalidArgument>(
            false, "QuadratureFamily can be GAUSS or GLL only");

        // // Set up BasisHandler
        std::shared_ptr<BasisHandler<ValueTypeBasisData, memorySpace>> cfeBasisHandler =
          std::make_shared<FEBasisHandlerDealii<ValueTypeBasisData, memorySpace, dim>>
          (basisManager, constraintsMap, d_comm);

        d_enrichClassIntfce(cfeBasisDataStorage,
                            cfeBasisHandler,
                            l2ProjQuadAttr,
                            d_atomSphericalDataContainer,
                            d_atomPartitionTolerance,
                            d_atomSymbolVec,
                            d_atomCoordinatesVec,
                            d_fieldName,
                            constraintName,
                            d_linAlgOpContext,
                            d_comm);
      }
      else
      {
        d_enrichClassIntfce(d_triangulation,
                            d_atomSphericalDataContainer,
                            d_atomPartitionTolerance,
                            d_atomSymbolVec,
                            d_atomCoordinatesVec,
                            d_fieldName,
                            d_comm);
      }

      std::shared_ptr<const EnrichmentIdsPartition<dim>> d_enrichmentIdsPartition = d_enrichClassIntfce.getEnrichmentidsPartition()
      std::shared_ptr<const AtomIdsPartition<dim>> d_atomIdsPartition = d_enrichClassIntfce.getAtomIdsPartition()

      d_overlappingEnrichmentIdsInCells =
        d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();

      // populate the global ranges range.The range would be as follows,
      // The order is chosen as : Classical Ranges, Enrichment Range1,
      // Enrichment Range2 ,....
      d_globalRanges.resize(d_totalRanges);
      d_globalRanges[0].first  = 0;
      d_globalRanges[0].second = d_dofHandler->n_dofs();
      for (unsigned int rangeId = 1; rangeId < d_totalRanges; rangeId++)
        {
          d_globalRanges[rangeId].first = d_globalRanges[rangeId - 1].second;
          d_globalRanges[rangeId].second =
            d_globalRanges[rangeId].first +
            d_enrichmentIdsPartition->nTotalEnrichmentIds();
        }

      // populate the locally owned ranges range.The range would be as follows,
      // The order is chosen as : Classical Ranges, Enrichment Range1,
      // Enrichment Range2 ,....
      d_locallyOwnedRanges.resize(d_totalRanges);
      auto             dealiiIndexSet = d_dofHandler->locally_owned_dofs();
      global_size_type startId        = *(dealiiIndexSet.begin());
      global_size_type endId  = startId + d_dofHandler->n_locally_owned_dofs();
      d_locallyOwnedRanges[0] = std::make_pair(startId, endId);
      for (unsigned int rangeId = 1; rangeId < d_totalRanges; rangeId++)
        {
          d_locallyOwnedRanges[rangeId].first =
            d_globalRanges[0].second +
            d_enrichmentIdsPartition->locallyOwnedEnrichmentIds().first;
          d_locallyOwnedRanges[rangeId].second =
            d_globalRanges[0].second +
            d_enrichmentIdsPartition->locallyOwnedEnrichmentIds().second;
        }

      // shift the ghost enriched ids by the total classical dofs
      for (auto i : d_enrichmentIdsPartition->ghostEnrichmentIds())
        d_ghostEnrichmentGlobalIds.push_back(i + d_dofHandler->n_dofs());

      d_numCumulativeLocallyOwnedCellDofs = 0;
      d_numCumulativeLocalCellDofs        = 0;
      for (size_type iCell = 0; iCell < d_locallyOwnedCells.size(); ++iCell)
        d_numCumulativeLocallyOwnedCellDofs += nCellDofs(iCell);

      for (size_type iCell = 0; iCell < d_localCells.size(); ++iCell)
        d_numCumulativeLocalCellDofs += nCellDofs(iCell);
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    double
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getBasisFunctionValue(
      const size_type     basisId,
      const utils::Point &point) const
    {
      utils::throwException(
        false,
        "getBasisFunctionValue() in EFEBasisManagerDealii not yet implemented.");
      return 0;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<double>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getBasisFunctionDerivative(
      const size_type     basisId,
      const utils::Point &point,
      const size_type     derivativeOrder) const
    {
      utils::throwException(
        false,
        "getBasisFunctionDerivative() in EFEBasisManagerDealii not yet implemented.");

      std::vector<double> vecReturn;
      return vecReturn;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const TriangulationBase>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getTriangulation() const
    {
      return d_triangulation;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nLocalCells() const
    {
      return d_localCells.size();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.size();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nGlobalCells() const
    {
      return d_triangulation->nGlobalCells();
    }

    // TODO put an assert condition to check if p refined is false
    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getFEOrder(size_type cellId) const
    {
      return (d_dofHandler->get_fe().degree);
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nCellDofs(size_type cellId) const
    {
      size_type classicalDofs = d_dofHandler->get_fe().n_dofs_per_cell();
      size_type enrichedDofs = d_overlappingEnrichmentIdsInCells[cellId].size();
      return (classicalDofs + enrichedDofs);
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    bool
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::isVariableDofsPerCell() const
    {
      return d_isVariableDofsPerCell;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nLocalNodes() const
    {
      global_size_type retValue = 0;
      for (unsigned int rangeId = 0; rangeId < d_totalRanges; rangeId++)
        {
          retValue = retValue + d_locallyOwnedRanges[rangeId].second -
                     d_locallyOwnedRanges[rangeId].first;
        }
      return retValue;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    global_size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nGlobalNodes() const
    {
      global_size_type retValue = 0;
      for (unsigned int rangeId = 0; rangeId < d_totalRanges; rangeId++)
        {
          retValue = retValue + d_globalRanges[rangeId].second -
                     d_globalRanges[rangeId].first;
        }
      return retValue;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getLocallyOwnedRanges() const
    {
      return d_locallyOwnedRanges;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getGlobalRanges() const
    {
      return d_globalRanges;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::map<BasisIdAttribute, size_type>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getBasisAttributeToRangeIdMap() const
    {
      std::map<BasisIdAttribute, size_type>         returnValue;
      std::pair<global_size_type, global_size_type> classicalRange =
        d_locallyOwnedRanges[0];
      std::pair<global_size_type, global_size_type> enrichedRange =
        d_locallyOwnedRanges[1];
      std::vector<std::pair<global_size_type, global_size_type>>
        locallyOwnedRangeVec(0);
      locallyOwnedRangeVec = getLocallyOwnedRanges();

      if (classicalRange.first == locallyOwnedRangeVec[0].first)
        {
          returnValue[BasisIdAttribute::CLASSICAL] = 0;
          returnValue[BasisIdAttribute::ENRICHED]  = 1;
        }
      else // given for completion
        {
          utils::throwException(
            false,
            "The Ranges are not stored as Classical, Enriched1, Enriched2 ... in EFEBasisManager ");
        }
      return returnValue;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<size_type>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getLocalNodeIds(size_type cellId) const
    {
      utils::throwException(
        false,
        "getLocalNodeIds() in EFEBasisManagerDealii is not yet implemented.");
      std::vector<size_type> vec;
      return vec;
      /// implement this now
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<size_type>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getGlobalNodeIds() const
    {
      utils::throwException(
        false,
        "getGlobalNodeIds() in EFEBasisManagerDealii is not yet implemented.");
      std::vector<size_type> vec;
      return vec;

      /// implement this now
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    void
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getCellDofsGlobalIds(
      size_type                      cellId,
      std::vector<global_size_type> &vecGlobalNodeId) const
    {
      std::vector<global_size_type> vecGlobalClassicalNodeId(0);
      vecGlobalNodeId.resize(nCellDofs(cellId), 0);
      vecGlobalClassicalNodeId.resize(d_dofHandler->get_fe().n_dofs_per_cell(),
                                      0);

      d_locallyOwnedCells[cellId]->cellNodeIdtoGlobalNodeId(
        vecGlobalClassicalNodeId);

      size_type                     classicalcount = 0, enrichedcount = 0;
      std::vector<global_size_type> vecEnrichedNodeId(0);
      vecEnrichedNodeId = d_overlappingEnrichmentIdsInCells[cellId];

      for (size_type count = 0; count < vecGlobalNodeId.size(); count++)
        {
          if (count < d_dofHandler->get_fe().n_dofs_per_cell())
            {
              vecGlobalNodeId[count] = vecGlobalClassicalNodeId[classicalcount];
              classicalcount += 1;
            }
          else
            {
              vecGlobalNodeId[count] =
                vecEnrichedNodeId[enrichedcount] + d_globalRanges[0].second;
              enrichedcount += 1;
            }
        }
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<size_type>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getBoundaryIds() const
    {
      utils::throwException(
        false,
        "getBoundaryIds() in EFEBasisManagerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::beginLocallyOwnedCells()
    {
      return d_locallyOwnedCells.begin();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::endLocallyOwnedCells()
    {
      return d_locallyOwnedCells.end();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::beginLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.begin();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::endLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.end();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::beginLocalCells()
    {
      return d_localCells.begin();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::endLocalCells()
    {
      return d_localCells.end();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::beginLocalCells() const
    {
      return d_localCells.begin();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::endLocalCells() const
    {
      return d_localCells.end();
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    unsigned int
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getDim() const
    {
      return dim;
    }

    //
    // dealii specific functions
    //
    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const dealii::DoFHandler<dim>>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getDoFHandler() const
    {
      return d_dofHandler;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    const dealii::FiniteElement<dim> &
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getReferenceFE(const size_type cellId) const
    {
      //
      // NOTE: The implementation is only restricted to
      // h-refinement (uniform p) and hence the reference FE
      // is same for all cellId. As a result, we pass index
      // 0 to dealii's dofHandler
      //
      return d_dofHandler->get_fe(0);
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    void
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getBasisCenters(
      std::map<global_size_type, utils::Point> &dofCoords) const
    {
      // TODO if the creation of linear mapping is inefficient, then this has to
      // be improved
      std::map<global_size_type, dealii::Point<dim, double>> dealiiDofCoords;
      dealii::MappingQ1<dim, dim>                            mappingQ1;
      dealii::DoFTools::map_dofs_to_support_points<dim, dim>(
        mappingQ1, *(d_dofHandler.get()), dealiiDofCoords);

      convertToDftefePoint<dim>(dealiiDofCoords, dofCoords);

      // add for the enrichment case

      for (unsigned int rangeId = 1; rangeId < d_totalRanges; rangeId++)
        {
          for (global_size_type i = d_locallyOwnedRanges[rangeId].first;
               i < d_locallyOwnedRanges[rangeId].second;
               i++)
            dofCoords.insert(
              {i,
               d_atomCoordinatesVec[d_enrichmentIdsPartition->getAtomId(
                 i - d_globalRanges[0].second)]});
          for (auto i : d_ghostEnrichmentGlobalIds)
            dofCoords.insert(
              {i,
               d_atomCoordinatesVec[d_enrichmentIdsPartition->getAtomId(
                 i - d_globalRanges[0].second)]});
        }
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nCumulativeLocallyOwnedCellDofs() const
    {
      return d_numCumulativeLocallyOwnedCellDofs;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nCumulativeLocalCellDofs() const
    {
      return d_numCumulativeLocalCellDofs;
    }

    // Enrichment functions with dealii mesh. The enrichedid is the cell local
    // id.
    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    double
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getEnrichmentValue(
      const size_type             cellId,
      const size_type             cellLocalEnrichmentId,
      const dftefe::utils::Point &point) const
    {
      double retValue = 0;
      if (!d_overlappingEnrichmentIdsInCells[cellId].empty())
        {
          if (d_overlappingEnrichmentIdsInCells[cellId].size() >
              cellLocalEnrichmentId)
            {
              size_type enrichmentId =
                d_overlappingEnrichmentIdsInCells[cellId]
                                                 [cellLocalEnrichmentId];
              size_type atomId =
                d_enrichmentIdsPartition->getAtomId(enrichmentId);
              size_type qNumberId =
                (d_enrichmentIdsPartition->getEnrichmentIdAttribute(
                   enrichmentId))
                  .localIdInAtom;
              std::string  atomSymbol = d_atomSymbolVec[atomId];
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              std::vector<std::vector<int>> qNumbers(0);
              qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                   d_fieldName);
              auto sphericalData =
                d_atomSphericalDataContainer->getSphericalData(
                  atomSymbol, d_fieldName, qNumbers[qNumberId]);
              retValue = sphericalData->getValue(point, origin);
            }
          else
            {
              utils::throwException(
                false,
                "The requested cell local enrichment id does not exist.");
            }
        }
      else
        {
          utils::throwException(
            false,
            "The requested cell does not have any enrichment ids overlapping with it.");
        }
      return retValue;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<double>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getEnrichmentDerivative(
      const size_type             cellId,
      const size_type             cellLocalEnrichmentId,
      const dftefe::utils::Point &point) const
    {
      std::vector<double> retValue(0);
      if (!d_overlappingEnrichmentIdsInCells[cellId].empty())
        {
          if (d_overlappingEnrichmentIdsInCells[cellId].size() >
              cellLocalEnrichmentId)
            {
              size_type enrichmentId =
                d_overlappingEnrichmentIdsInCells[cellId]
                                                 [cellLocalEnrichmentId];
              size_type atomId =
                d_enrichmentIdsPartition->getAtomId(enrichmentId);
              size_type qNumberId =
                (d_enrichmentIdsPartition->getEnrichmentIdAttribute(
                   enrichmentId))
                  .localIdInAtom;
              std::string  atomSymbol = d_atomSymbolVec[atomId];
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              std::vector<std::vector<int>> qNumbers(0);
              qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                   d_fieldName);
              auto sphericalData =
                d_atomSphericalDataContainer->getSphericalData(
                  atomSymbol, d_fieldName, qNumbers[qNumberId]);
              retValue = sphericalData->getGradientValue(point, origin);
            }
          else
            {
              utils::throwException(
                false,
                "The requested cell local enrichment id does not exist.");
            }
        }
      else
        {
          utils::throwException(
            false,
            "The requested cellid does not have any enrichment ids overlapping with it.");
        }
      return retValue;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<double>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getEnrichmentHessian(
      const size_type             cellId,
      const size_type             cellLocalEnrichmentId,
      const dftefe::utils::Point &point) const
    {
      std::vector<double> retValue(0);
      if (!d_overlappingEnrichmentIdsInCells[cellId].empty())
        {
          if (d_overlappingEnrichmentIdsInCells[cellId].size() >
              cellLocalEnrichmentId)
            {
              size_type enrichmentId =
                d_overlappingEnrichmentIdsInCells[cellId]
                                                 [cellLocalEnrichmentId];
              size_type atomId =
                d_enrichmentIdsPartition->getAtomId(enrichmentId);
              size_type qNumberId =
                (d_enrichmentIdsPartition->getEnrichmentIdAttribute(
                   enrichmentId))
                  .localIdInAtom;
              std::string  atomSymbol = d_atomSymbolVec[atomId];
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              std::vector<std::vector<int>> qNumbers(0);
              qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                   d_fieldName);
              auto sphericalData =
                d_atomSphericalDataContainer->getSphericalData(
                  atomSymbol, d_fieldName, qNumbers[qNumberId]);
              retValue = sphericalData->getHessianValue(point, origin);
            }
          else
            {
              utils::throwException(
                false,
                "The requested cell local enrichment id does not exist.");
            }
        }
      else
        {
          utils::throwException(
            false,
            "The requested cell does not have any enrichment ids overlapping with it.");
        }
      return retValue;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::vector<global_size_type>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getGhostEnrichmentGlobalIds() const
    {
      return d_ghostEnrichmentGlobalIds;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    global_size_type
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::nGlobalEnrichmentNodes() const
    {
      return (d_enrichmentIdsPartition->nTotalEnrichmentIds());
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const EnrichmentIdsPartition<dim>>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getEnrichmentIdsPartition() const
    {
      return d_enrichmentIdsPartition;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace, dim>>
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::getEnrichmentClassicalInterface() const
    {
      return d_enrichClassIntfce;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    bool
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::isOrthgonalized() const
    {
      return d_isOrthgonalized;
    }

    template <typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace, size_type dim>
    bool
    EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>::totalRanges() const
    {
      return d_totalRanges;
    }

  } // namespace basis
} // namespace dftefe
