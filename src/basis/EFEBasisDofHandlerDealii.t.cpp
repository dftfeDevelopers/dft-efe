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
#include <quadrature/QuadratureValuesContainer.h>
#include <quadrature/QuadratureRuleContainer.h>
#include <quadrature/QuadratureAttributes.h>
#include "CFEBasisDofHandlerDealii.h"
#include "EFEConstraintsLocalDealii.h"
#include "CFEBasisDataStorageDealii.h"
#include "FEBasisManager.h"


#include <utils/Exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>


namespace dftefe
{
  namespace basis
  {
    namespace EFEBasisDofHandlerInternal
    {
      template <typename ValueTypeBasisCoeff,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      setDealiiMatrixFreeLight(
        dealii::DoFHandler<dim> &dealiiDofHandler,
        dealii::AffineConstraints<ValueTypeBasisCoeff>
          &dealiiAffineConstraintMatrix,
        dealii::MatrixFree<dim, ValueTypeBasisCoeff> &dealiiMatrixFree)
      {
        typename dealii::MatrixFree<dim>::AdditionalData dealiiAdditionalData;
        dealiiAdditionalData.tasks_parallel_scheme =
          dealii::MatrixFree<dim>::AdditionalData::partition_partition;
        dealii::UpdateFlags dealiiUpdateFlags     = dealii::update_default;
        dealiiAdditionalData.mapping_update_flags = dealiiUpdateFlags;
        dealii::Quadrature<dim> dealiiQuadratureType(dealii::QGauss<dim>(1));
        dealiiMatrixFree.clear();
        dealii::MappingQ1<dim> mappingDealii;
        dealiiMatrixFree.reinit(mappingDealii,
                                dealiiDofHandler,
                                dealiiAffineConstraintMatrix,
                                dealiiQuadratureType,
                                dealiiAdditionalData);
      }

      template <typename ValueTypeBasisCoeff,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getGhostIndices(
        std::vector<global_size_type> &ghostEnrichmentGlobalIds,
        const dealii::MatrixFree<dim, ValueTypeBasisCoeff> &dealiiMatrixFree,
        std::vector<global_size_type> &                     ghostIndices)
      {
        const dealii::Utilities::MPI::Partitioner &dealiiPartitioner =
          *(dealiiMatrixFree.get_vector_partitioner());
        const dealii::IndexSet &ghostIndexSet =
          dealiiPartitioner.ghost_indices();
        const size_type numGhostIndicesClassical = ghostIndexSet.n_elements();
        std::vector<global_size_type> ghostIndicesClassical(0);
        ghostIndicesClassical.resize(numGhostIndicesClassical, 0);
        ghostIndexSet.fill_index_vector(ghostIndicesClassical);

        // get the enriched ghost ids
        ghostIndices.clear();
        ghostIndices.insert(ghostIndices.begin(),
                            ghostIndicesClassical.begin(),
                            ghostIndicesClassical.end());
        ghostIndices.insert(ghostIndices.end(),
                            ghostEnrichmentGlobalIds.begin(),
                            ghostEnrichmentGlobalIds.end());
      }
    } // namespace EFEBasisDofHandlerInternal

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::
      EFEBasisDofHandlerDealii(
        std::shared_ptr<const EnrichmentClassicalInterfaceSpherical<
          ValueTypeBasisData,
          memorySpace,
          dim>>                    EnrichmentClassicalInterface,
        const size_type            feOrder,
        const utils::mpi::MPIComm &mpiComm)
      : d_isVariableDofsPerCell(true)
      , d_totalRanges(2) // Classical and Enriched
      , d_overlappingEnrichmentIdsInCells(0)
      , d_locallyOwnedRanges(0)
      , d_globalRanges(0)
      , d_ghostEnrichmentGlobalIds(0)
      , d_enrichmentIdsPartition(nullptr)
      , d_boundaryIds(0)
    {
      d_dofHandler = std::make_shared<dealii::DoFHandler<dim>>();
      // making the classical and enriched dofs in the dealii mesh here
      reinit(EnrichmentClassicalInterface, feOrder, mpiComm);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::
      EFEBasisDofHandlerDealii(
        std::shared_ptr<const EnrichmentClassicalInterfaceSpherical<
          ValueTypeBasisData,
          memorySpace,
          dim>>         EnrichmentClassicalInterface,
        const size_type feOrder)
      : d_isVariableDofsPerCell(true)
      , d_totalRanges(2) // Classical and Enriched
      , d_overlappingEnrichmentIdsInCells(0)
      , d_locallyOwnedRanges(0)
      , d_globalRanges(0)
      , d_ghostEnrichmentGlobalIds(0)
      , d_enrichmentIdsPartition(nullptr)
      , d_boundaryIds(0)
    {
      d_dofHandler = std::make_shared<dealii::DoFHandler<dim>>();
      // making the classical and enriched dofs in the dealii mesh here
      reinit(EnrichmentClassicalInterface, feOrder);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::
      reinit(std::shared_ptr<const EnrichmentClassicalInterfaceSpherical<
               ValueTypeBasisData,
               memorySpace,
               dim>>                    enrichmentClassicalInterface,
             const size_type            feOrder,
             const utils::mpi::MPIComm &mpiComm)
    {
      d_isDistributed = true;
      // Create Classical FE dof_handler
      d_enrichClassIntfce = enrichmentClassicalInterface;
      d_isOrthogonalized  = enrichmentClassicalInterface->isOrthgonalized();
      d_atomSphericalDataContainer =
        enrichmentClassicalInterface->getAtomSphericalDataContainer();
      d_atomSymbolVec = enrichmentClassicalInterface->getAtomSymbolVec();
      d_atomCoordinatesVec =
        enrichmentClassicalInterface->getAtomCoordinatesVec();
      d_fieldName = enrichmentClassicalInterface->getFieldName();
      dealii::FE_Q<dim>                       feElem(feOrder);
      const TriangulationDealiiParallel<dim> *dealiiParallelTria =
        dynamic_cast<const TriangulationDealiiParallel<dim> *>(
          enrichmentClassicalInterface->getTriangulation().get());

      // TODO - If remeshing then update triangulation

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
              enrichmentClassicalInterface->getTriangulation().get());

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
                "reinit() in EFEBasisDofHandlerDealii is not able to re cast the Triangulation.");
            }
        }

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

      d_enrichmentIdsPartition =
        d_enrichClassIntfce->getEnrichmentIdsPartition();

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

      // Make dealii affine constraint matrix for base constraints only.
      // This will only contain the constraints related to the geometric
      // nature of the finite elemnt mesh. One needs dof_handler for this
      // part as the constriant matrix needs to trim out the locally

      dealii::AffineConstraints<ValueTypeBasisCoeff>
        dealiiAffineConstraintMatrix;

      dealiiAffineConstraintMatrix.clear();
      dealii::IndexSet locally_relevant_dofs;
      locally_relevant_dofs.clear();
      dealii::DoFTools::extract_locally_relevant_dofs(*(this->getDoFHandler()),
                                                      locally_relevant_dofs);
      dealiiAffineConstraintMatrix.reinit(locally_relevant_dofs);
      dealii::DoFTools::make_hanging_node_constraints(
        *(this->getDoFHandler()), dealiiAffineConstraintMatrix);

      // dealiiAffineConstraintMatrix->makePeriodicConstriants();
      dealiiAffineConstraintMatrix.close();

      // Next one can further trim out the ghost set of the locally relevant set
      // for the constriants that are needed only for the geometric constriants
      // defined above. So we obtain an optimized ghost set.

      // One will add the Non-geometric constrints like Inhomogeneous
      // constriants. This will be added to the affine constraint matrix
      // in the feBasisHandler as they do not require the information
      // of the geometry or dof_handler.
      // Note: For more efficiency,  this step of getting optimized ghost ids
      // could be technically done after making the constriant matrix in its
      // full glory i.e. after the inhomogeneous constraints , but that would
      // cause the feBasisManager to be convoluted. (this ghost trimming using
      // dealii::matrixfree) Since the trimming in this process does not lead to
      // much benefit (as only extra nodes on or near the boundary may be
      // trimmed) hence this is avoided for the ease of implementation.

      //
      // NOTE: Since our purpose is to create the dealii MatrixFree object
      // only to access the partitioning of DoFs for each constraint
      // (i.e., the ghost indices for each constraint), we need not create
      // the MatrixFree object with its full glory (i.e., with all the relevant
      // quadrature rule and with all the relevant update flags). Instead for
      // cheap construction of the MatrixFree object, we can just create it
      // the MatrixFree object for a dummy quadrature rule
      // and with default update flags
      //

      // Hence this step is to repartition the dofs so that the ghost set is
      // reduced a nd trimmed with only those remain which are required for
      // satisfying hanging and periodic for the current processor.

      dealii::MatrixFree<dim, ValueTypeBasisCoeff> dealiiMatrixFree;

      EFEBasisDofHandlerInternal::
        setDealiiMatrixFreeLight<ValueTypeBasisCoeff, memorySpace, dim>(
          *d_dofHandler, dealiiAffineConstraintMatrix, dealiiMatrixFree);

      std::vector<global_size_type> ghostIndicesSTLVec;
      EFEBasisDofHandlerInternal::
        getGhostIndices<ValueTypeBasisCoeff, memorySpace, dim>(
          d_ghostEnrichmentGlobalIds, dealiiMatrixFree, ghostIndicesSTLVec);

      //
      // populate d_mpiPatternP2P - nbx consensus map for P2P communication
      //
      d_mpiPatternP2P =
        std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
          getLocallyOwnedRanges(), ghostIndicesSTLVec, mpiComm);

      // Get the required parameters for creating ConstraintsLocal Object

      std::vector<std::pair<global_size_type, global_size_type>>
        locallyOwnedRanges = d_mpiPatternP2P->getLocallyOwnedRanges();
      std::vector<global_size_type> ghostIndices =
        d_mpiPatternP2P->getGhostIndices();
      std::unordered_map<global_size_type, size_type> globalToLocalMapLocalDofs;
      globalToLocalMapLocalDofs.clear();

      for (auto j : locallyOwnedRanges)
        {
          for (global_size_type i = j.first; i < j.second; i++)
            {
              globalToLocalMapLocalDofs.insert(
                {i, d_mpiPatternP2P->globalToLocal(i)});
            }
        }
      for (auto j : ghostIndices)
        {
          globalToLocalMapLocalDofs.insert(
            {j, d_mpiPatternP2P->globalToLocal(j)});
        }

      // Creation of geometric / intrinsic constraint
      // matrix having only trimmed constraint ids.

      d_constraintsLocal = std::make_shared<
        const EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
        dealiiAffineConstraintMatrix,
        locallyOwnedRanges,
        ghostIndices,
        globalToLocalMapLocalDofs);

      // get boundary node ids of all locally owned and ghost cells. This is
      // because the dofs of locally owned and ghost cells
      // (locally_relevant_cells in dealii terminology) is the one that is used
      // to set_inhomogeneity as constraint equations are solved in
      // locally_relevant dofs domain which is superset of locally_owned dofs
      // set.

      const unsigned int vertices_per_cell =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      const unsigned int dofs_per_cell =
        this->getDoFHandler()->get_fe().dofs_per_cell;
      const unsigned int faces_per_cell =
        dealii::GeometryInfo<dim>::faces_per_cell;
      const unsigned int dofs_per_face =
        this->getDoFHandler()->get_fe().dofs_per_face;

      std::vector<global_size_type> cellGlobalDofIndices(dofs_per_cell);
      std::vector<global_size_type> iFaceGlobalDofIndices(dofs_per_face);

      std::vector<bool> dofs_touched(this->nGlobalNodes(), false);
      auto cellIter = this->beginLocalCells(), endIter = this->endLocalCells();
      for (; cellIter != endIter; ++cellIter)
        {
          (*cellIter)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              (*cellIter)->getFaceDoFGlobalIndices(iFace,
                                                   iFaceGlobalDofIndices);
              const size_type boundaryId =
                (*cellIter)->getFaceBoundaryId(iFace);
              if (boundaryId == 0)
                {
                  for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      const dealii::types::global_dof_index nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      if (dofs_touched[nodeId])
                        continue;
                      dofs_touched[nodeId] = true;
                      // check if a node is not hanging and periodic
                      if (!dealiiAffineConstraintMatrix.is_constrained(nodeId))
                        {
                          d_boundaryIds.push_back(nodeId);
                        }
                    }
                }
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::
      reinit(std::shared_ptr<const EnrichmentClassicalInterfaceSpherical<
               ValueTypeBasisData,
               memorySpace,
               dim>>         enrichmentClassicalInterface,
             const size_type feOrder)
    {
      d_isDistributed = false;
      // Create Classical FE dof_handler
      d_enrichClassIntfce = enrichmentClassicalInterface;
      d_isOrthogonalized  = enrichmentClassicalInterface->isOrthgonalized();
      d_atomSphericalDataContainer =
        enrichmentClassicalInterface->getAtomSphericalDataContainer();
      d_atomSymbolVec = enrichmentClassicalInterface->getAtomSymbolVec();
      d_atomCoordinatesVec =
        enrichmentClassicalInterface->getAtomCoordinatesVec();
      d_fieldName = enrichmentClassicalInterface->getFieldName();
      dealii::FE_Q<dim>                       feElem(feOrder);
      const TriangulationDealiiParallel<dim> *dealiiParallelTria =
        dynamic_cast<const TriangulationDealiiParallel<dim> *>(
          enrichmentClassicalInterface->getTriangulation().get());

      // TODO - If remeshing then update triangulation

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
              enrichmentClassicalInterface->getTriangulation().get());

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
                "reinit() in EFEBasisDofHandlerDealii is not able to re cast the Triangulation.");
            }
        }

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

      d_enrichmentIdsPartition =
        d_enrichClassIntfce->getEnrichmentIdsPartition();

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

      // Make dealii affine constraint matrix for base constraints only.
      // This will only contain the constraints related to the geometric
      // nature of the finite elemnt mesh. One needs dof_handler for this
      // part as the constriant matrix needs to trim out the locally

      dealii::AffineConstraints<ValueTypeBasisCoeff>
        dealiiAffineConstraintMatrix;

      dealiiAffineConstraintMatrix.clear();
      dealii::IndexSet locally_relevant_dofs;
      locally_relevant_dofs.clear();
      dealii::DoFTools::extract_locally_relevant_dofs(*(this->getDoFHandler()),
                                                      locally_relevant_dofs);
      dealiiAffineConstraintMatrix.reinit(locally_relevant_dofs);
      dealii::DoFTools::make_hanging_node_constraints(
        *(this->getDoFHandler()), dealiiAffineConstraintMatrix);

      // dealiiAffineConstraintMatrix->makePeriodicConstriants();
      dealiiAffineConstraintMatrix.close();

      // Next one can further trim out the ghost set of the locally relevant set
      // for the constriants that are needed only for the geometric constriants
      // defined above. So we obtain an optimized ghost set.

      // One will add the Non-geometric constrints like Inhomogeneous
      // constriants. This will be added to the affine constraint matrix
      // in the feBasisHandler as they do not require the information
      // of the geometry or dof_handler.
      // Note: For more efficiency,  this step of getting optimized ghost ids
      // could be technically done after making the constriant matrix in its
      // full glory i.e. after the inhomogeneous constraints , but that would
      // cause the feBasisManager to be convoluted. (this ghost trimming using
      // dealii::matrixfree) Since the trimming in this process does not lead to
      // much benefit (as only extra nodes on or near the boundary may be
      // trimmed) hence this is avoided for the ease of implementation.

      //
      // NOTE: Since our purpose is to create the dealii MatrixFree object
      // only to access the partitioning of DoFs for each constraint
      // (i.e., the ghost indices for each constraint), we need not create
      // the MatrixFree object with its full glory (i.e., with all the relevant
      // quadrature rule and with all the relevant update flags). Instead for
      // cheap construction of the MatrixFree object, we can just create it
      // the MatrixFree object for a dummy quadrature rule
      // and with default update flags
      //

      // Hence this step is to repartition the dofs so that the ghost set is
      // reduced a nd trimmed with only those remain which are required for
      // satisfying hanging and periodic for the current processor.

      dealii::MatrixFree<dim, ValueTypeBasisCoeff> dealiiMatrixFree;

      EFEBasisDofHandlerInternal::
        setDealiiMatrixFreeLight<ValueTypeBasisCoeff, memorySpace, dim>(
          *d_dofHandler, dealiiAffineConstraintMatrix, dealiiMatrixFree);

      std::vector<global_size_type> ghostIndicesSTLVec;
      EFEBasisDofHandlerInternal::
        getGhostIndices<ValueTypeBasisCoeff, memorySpace, dim>(
          d_ghostEnrichmentGlobalIds, dealiiMatrixFree, ghostIndicesSTLVec);

      //
      // populate d_mpiPatternP2P - nbx consensus map for P2P communication
      //
      std::vector<size_type> locallyOwnedRangesSizeVec(0);
      for (auto i : getLocallyOwnedRanges())
        {
          locallyOwnedRangesSizeVec.push_back(i.second - i.first);
        }
      d_mpiPatternP2P =
        std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRangesSizeVec);

      // Get the required parameters for creating ConstraintsLocal Object

      std::vector<std::pair<global_size_type, global_size_type>>
        locallyOwnedRanges = d_mpiPatternP2P->getLocallyOwnedRanges();
      std::vector<global_size_type> ghostIndices =
        d_mpiPatternP2P->getGhostIndices();
      std::unordered_map<global_size_type, size_type> globalToLocalMapLocalDofs;
      globalToLocalMapLocalDofs.clear();

      for (auto j : locallyOwnedRanges)
        {
          for (global_size_type i = j.first; i < j.second; i++)
            {
              globalToLocalMapLocalDofs.insert(
                {i, d_mpiPatternP2P->globalToLocal(i)});
            }
        }
      for (auto j : ghostIndices)
        {
          globalToLocalMapLocalDofs.insert(
            {j, d_mpiPatternP2P->globalToLocal(j)});
        }

      // Creation of geometric / intrinsic constraint
      // matrix having only trimmed constraint ids.

      d_constraintsLocal = std::make_shared<
        const EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
        dealiiAffineConstraintMatrix,
        locallyOwnedRanges,
        ghostIndices,
        globalToLocalMapLocalDofs);

      // get boundary node ids of all locally owned and ghost cells. This is
      // because the dofs of locally owned and ghost cells
      // (locally_relevant_cells in dealii terminology) is the one that is used
      // to set_inhomogeneity as constraint equations are solved in
      // locally_relevant dofs domain which is superset of locally_owned dofs
      // set.

      const unsigned int vertices_per_cell =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      const unsigned int dofs_per_cell =
        this->getDoFHandler()->get_fe().dofs_per_cell;
      const unsigned int faces_per_cell =
        dealii::GeometryInfo<dim>::faces_per_cell;
      const unsigned int dofs_per_face =
        this->getDoFHandler()->get_fe().dofs_per_face;

      std::vector<global_size_type> cellGlobalDofIndices(dofs_per_cell);
      std::vector<global_size_type> iFaceGlobalDofIndices(dofs_per_face);

      std::vector<bool> dofs_touched(this->nGlobalNodes(), false);
      auto cellIter = this->beginLocalCells(), endIter = this->endLocalCells();
      for (; cellIter != endIter; ++cellIter)
        {
          (*cellIter)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              (*cellIter)->getFaceDoFGlobalIndices(iFace,
                                                   iFaceGlobalDofIndices);
              const size_type boundaryId =
                (*cellIter)->getFaceBoundaryId(iFace);
              if (boundaryId == 0)
                {
                  for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                       ++iFaceDof)
                    {
                      const dealii::types::global_dof_index nodeId =
                        iFaceGlobalDofIndices[iFaceDof];
                      if (dofs_touched[nodeId])
                        continue;
                      dofs_touched[nodeId] = true;
                      // check if a node is not hanging and periodic
                      if (!dealiiAffineConstraintMatrix.is_constrained(nodeId))
                        {
                          d_boundaryIds.push_back(nodeId);
                        }
                    }
                }
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    double
    EFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisFunctionValue(const size_type     basisId,
                                  const utils::Point &point) const
    {
      utils::throwException(
        false,
        "getBasisFunctionValue() in EFEBasisDofHandlerDealii not yet implemented.");
      return 0;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<double>
    EFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getBasisFunctionDerivative(const size_type     basisId,
                                       const utils::Point &point,
                                       const size_type derivativeOrder) const
    {
      utils::throwException(
        false,
        "getBasisFunctionDerivative() in EFEBasisDofHandlerDealii not yet implemented.");

      std::vector<double> vecReturn;
      return vecReturn;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const TriangulationBase>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getTriangulation() const
    {
      return d_enrichClassIntfce->getTriangulation();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nLocalCells() const
    {
      return d_localCells.size();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.size();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nGlobalCells() const
    {
      return d_enrichClassIntfce->getTriangulation()->nGlobalCells();
    }

    // TODO put an assert condition to check if p refined is false
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getFEOrder(size_type cellId) const
    {
      return (d_dofHandler->get_fe().degree);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nCellDofs(size_type cellId) const
    {
      size_type classicalDofs = d_dofHandler->get_fe().n_dofs_per_cell();
      size_type enrichedDofs = d_overlappingEnrichmentIdsInCells[cellId].size();
      return (classicalDofs + enrichedDofs);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::isVariableDofsPerCell() const
    {
      return d_isVariableDofsPerCell;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nLocalNodes() const
    {
      global_size_type retValue = 0;
      for (unsigned int rangeId = 0; rangeId < d_totalRanges; rangeId++)
        {
          retValue = retValue + d_locallyOwnedRanges[rangeId].second -
                     d_locallyOwnedRanges[rangeId].first;
        }
      return retValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    global_size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nGlobalNodes() const
    {
      global_size_type retValue = 0;
      for (unsigned int rangeId = 0; rangeId < d_totalRanges; rangeId++)
        {
          retValue = retValue + d_globalRanges[rangeId].second -
                     d_globalRanges[rangeId].first;
        }
      return retValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getLocallyOwnedRanges() const
    {
      return d_locallyOwnedRanges;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getGlobalRanges() const
    {
      return d_globalRanges;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::map<BasisIdAttribute, size_type>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getBasisAttributeToRangeIdMap() const
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
            "The Ranges are not stored as Classical, Enriched1, Enriched2 ... in EFEBasisDofHandler ");
        }
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<size_type>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getLocalNodeIds(size_type cellId) const
    {
      utils::throwException(
        false,
        "getLocalNodeIds() in EFEBasisDofHandlerDealii is not yet implemented.");
      std::vector<size_type> vec;
      return vec;
      /// implement this now
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<size_type>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getGlobalNodeIds() const
    {
      utils::throwException(
        false,
        "getGlobalNodeIds() in EFEBasisDofHandlerDealii is not yet implemented.");
      std::vector<size_type> vec;
      return vec;

      /// implement this now
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::
      getCellDofsGlobalIds(size_type                      cellId,
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

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const std::vector<global_size_type> &
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getBoundaryIds() const
    {
      return d_boundaryIds;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::
      FECellIterator
      EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim>::beginLocallyOwnedCells()
    {
      return d_locallyOwnedCells.begin();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::
      FECellIterator
      EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim>::endLocallyOwnedCells()
    {
      return d_locallyOwnedCells.end();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::
      const_FECellIterator
      EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim>::beginLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.begin();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::
      const_FECellIterator
      EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim>::endLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.end();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::
      FECellIterator
      EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim>::beginLocalCells()
    {
      return d_localCells.begin();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::
      FECellIterator
      EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim>::endLocalCells()
    {
      return d_localCells.end();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::
      const_FECellIterator
      EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim>::beginLocalCells() const
    {
      return d_localCells.begin();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::
      const_FECellIterator
      EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                               ValueTypeBasisData,
                               memorySpace,
                               dim>::endLocalCells() const
    {
      return d_localCells.end();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    unsigned int
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getDim() const
    {
      return dim;
    }

    //
    // dealii specific functions
    //
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const dealii::DoFHandler<dim>>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getDoFHandler() const
    {
      return d_dofHandler;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const dealii::FiniteElement<dim> &
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getReferenceFE(const size_type cellId) const
    {
      //
      // NOTE: The implementation is only restricted to
      // h-refinement (uniform p) and hence the reference FE
      // is same for all cellId. As a result, we pass index
      // 0 to dealii's dofHandler
      //
      return d_dofHandler->get_fe(0);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::
      getBasisCenters(std::map<global_size_type, utils::Point> &dofCoords) const
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

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nCumulativeLocallyOwnedCellDofs() const
    {
      return d_numCumulativeLocallyOwnedCellDofs;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nCumulativeLocalCellDofs() const
    {
      return d_numCumulativeLocalCellDofs;
    }

    // Enrichment functions with dealii mesh. The enrichedid is the cell local
    // id.
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    double
    EFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getEnrichmentValue(const size_type cellId,
                               const size_type cellLocalEnrichmentId,
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

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<double>
    EFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getEnrichmentDerivative(const size_type cellId,
                                    const size_type cellLocalEnrichmentId,
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

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<double>
    EFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace,
      dim>::getEnrichmentHessian(const size_type cellId,
                                 const size_type cellLocalEnrichmentId,
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

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<global_size_type>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getGhostEnrichmentGlobalIds() const
    {
      return d_ghostEnrichmentGlobalIds;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    global_size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::nGlobalEnrichmentNodes() const
    {
      return (d_enrichmentIdsPartition->nTotalEnrichmentIds());
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const EnrichmentIdsPartition<dim>>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getEnrichmentIdsPartition() const
    {
      return d_enrichmentIdsPartition;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<
      const EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getEnrichmentClassicalInterface() const
    {
      return d_enrichClassIntfce;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::isOrthogonalized() const
    {
      return d_isOrthogonalized;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::totalRanges() const
    {
      return d_totalRanges;
    }

    // Some additional functions for getting geometric constriants matrix
    // and MPIPatternP2P

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getIntrinsicConstraints() const
    {
      return d_constraintsLocal;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::createConstraintsStart() const
    {
      std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
        constraintsLocal = std::make_shared<
          EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>>();

      constraintsLocal->copyFrom(*d_constraintsLocal);

      return (constraintsLocal);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::
      createConstraintsEnd(
        std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
          constraintsLocal) const
    {
      constraintsLocal->close();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::getMPIPatternP2P() const
    {
      return d_mpiPatternP2P;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    EFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             ValueTypeBasisData,
                             memorySpace,
                             dim>::isDistributed() const
    {
      return d_isDistributed;
    }

  } // namespace basis
} // namespace dftefe
