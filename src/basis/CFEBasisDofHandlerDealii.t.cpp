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
 * @author Bikash Kanungo, Vishal Subramanian
 */
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_tools.h>
#include "TriangulationDealiiParallel.h"
#include "TriangulationDealiiSerial.h"
#include "FECellDealii.h"
#include <deal.II/fe/mapping_q1.h>

#include <utils/Exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/dofs/dof_tools.h>


namespace dftefe
{
  namespace basis
  {
    namespace CFEBasisDofHandlerInternal
    {
      template <typename ValueTypeBasisCoeff,
                utils::MemorySpace memorySpace,
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
                utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getGhostIndices(
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
      }
    } // namespace CFEBasisDofHandlerInternal

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    CFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      memorySpace,
      dim>::CFEBasisDofHandlerDealii(std::shared_ptr<const TriangulationBase>
                                                                triangulation,
                                     const size_type            feOrder,
                                     const utils::mpi::MPIComm &mpiComm)
      : d_isVariableDofsPerCell(false)
      , d_totalRanges(1)
      , d_boundaryIds(0)
    {
      d_dofHandler = std::make_shared<dealii::DoFHandler<dim>>();
      reinit(triangulation, feOrder, mpiComm);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    CFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      memorySpace,
      dim>::CFEBasisDofHandlerDealii(std::shared_ptr<const TriangulationBase>
                                                     triangulation,
                                     const size_type feOrder)
      : d_isVariableDofsPerCell(false)
      , d_totalRanges(1)
      , d_boundaryIds(0)
    {
      d_dofHandler = std::make_shared<dealii::DoFHandler<dim>>();
      reinit(triangulation, feOrder);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      memorySpace,
      dim>::reinit(std::shared_ptr<const TriangulationBase> triangulation,
                   const size_type                          feOrder,
                   const utils::mpi::MPIComm &              mpiComm)
    {
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
                "reinit() in CFEBasisDofHandlerDealii is not able to re cast the Triangulation.");
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

      CFEBasisDofHandlerInternal::
        setDealiiMatrixFreeLight<ValueTypeBasisCoeff, memorySpace, dim>(
          *d_dofHandler, dealiiAffineConstraintMatrix, dealiiMatrixFree);

      std::vector<global_size_type> ghostIndicesSTLVec;
      CFEBasisDofHandlerInternal::getGhostIndices<ValueTypeBasisCoeff, memorySpace, dim>(
        dealiiMatrixFree, ghostIndicesSTLVec);

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

      for(auto j : locallyOwnedRanges)
      {
        for (global_size_type i = j.first;
            i < j.second;
            i++)
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
          const CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
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
              (*cellIter)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
              const size_type boundaryId = (*cellIter)->getFaceBoundaryId(iFace);
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
              utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      memorySpace,
      dim>::reinit(std::shared_ptr<const TriangulationBase> triangulation,
                   const size_type                          feOrder)
    {
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
                "reinit() in CFEBasisDofHandlerDealii is not able to re cast the Triangulation.");
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

      CFEBasisDofHandlerInternal::
        setDealiiMatrixFreeLight<ValueTypeBasisCoeff, memorySpace, dim>(
          *d_dofHandler, dealiiAffineConstraintMatrix, dealiiMatrixFree);

      std::vector<global_size_type> ghostIndicesSTLVec;
      CFEBasisDofHandlerInternal::getGhostIndices<ValueTypeBasisCoeff, memorySpace, dim>(
        dealiiMatrixFree, ghostIndicesSTLVec);

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

      for(auto j : locallyOwnedRanges)
      {
        for (global_size_type i = j.first;
            i < j.second;
            i++)
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
          const CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
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
              (*cellIter)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
              const size_type boundaryId = (*cellIter)->getFaceBoundaryId(iFace);
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
              utils::MemorySpace memorySpace,
              size_type                  dim>
    double
    CFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      memorySpace,
      dim>::getBasisFunctionValue(const size_type     basisId,
                                  const utils::Point &point) const
    {
      utils::throwException(
        false,
        "getBasisFunctionValue() in CFEBasisDofHandlerDealii not yet implemented.");
      return 0;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<double>
    CFEBasisDofHandlerDealii<
      ValueTypeBasisCoeff,
      memorySpace,
      dim>::getBasisFunctionDerivative(const size_type     basisId,
                                       const utils::Point &point,
                                       const size_type derivativeOrder) const
    {
      utils::throwException(
        false,
        "getBasisFunctionDerivative() in CFEBasisDofHandlerDealii not yet implemented.");

      std::vector<double> vecReturn;
      return vecReturn;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const TriangulationBase>
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getTriangulation() const
    {
      return d_triangulation;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::nLocalCells() const
    {
      return d_localCells.size();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::nLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.size();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::nGlobalCells() const
    {
      return d_triangulation->nGlobalCells();
    }

    // TODO put an assert condition to check if p refined is false
    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getFEOrder(size_type cellId) const
    {
      return (d_dofHandler->get_fe().degree);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::nCellDofs(size_type cellId) const
    {
      return d_dofHandler->get_fe().n_dofs_per_cell();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::isVariableDofsPerCell() const
    {
      return d_isVariableDofsPerCell;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::nLocalNodes() const
    {
      return d_dofHandler->n_locally_owned_dofs();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    global_size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::nGlobalNodes() const
    {
      return d_dofHandler->n_dofs();
    }

        template <typename ValueTypeBasisCoeff,
                  utils::MemorySpace memorySpace,
                  size_type                  dim>
        std::vector<std::pair<global_size_type, global_size_type>>
        CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                 memorySpace,
                                 dim>::getLocallyOwnedRanges() const
    {
      std::vector<std::pair<global_size_type, global_size_type>> returnValue(0);
      auto             dealiiIndexSet = d_dofHandler->locally_owned_dofs();
      global_size_type startId        = *(dealiiIndexSet.begin());
      global_size_type endId = startId + d_dofHandler->n_locally_owned_dofs();
      std::pair<global_size_type, global_size_type> classicalRange =
        std::make_pair(startId, endId);

      returnValue.push_back(classicalRange);

      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getGlobalRanges() const
    {
      std::vector<std::pair<global_size_type, global_size_type>> retValue(0);
      retValue.resize(1);
      retValue[0].first  = 0;
      retValue[0].second = d_dofHandler->n_dofs();
      return retValue;
    }


    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::map<BasisIdAttribute, size_type>
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getBasisAttributeToRangeIdMap() const
    {
      std::map<BasisIdAttribute, size_type> returnValue;
      returnValue[BasisIdAttribute::CLASSICAL] = 0;
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<size_type>
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getLocalNodeIds(size_type cellId) const
    {
      utils::throwException(
        false,
        "getLocalNodeIds() in CFEBasisDofHandlerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;
      /// implement this now
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<size_type>
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getGlobalNodeIds() const
    {
      utils::throwException(
        false,
        "getGlobalNodeIds() in CFEBasisDofHandlerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;

      /// implement this now
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::
      getCellDofsGlobalIds(size_type                      cellId,
                           std::vector<global_size_type> &vecGlobalNodeId) const
    {
      vecGlobalNodeId.resize(nCellDofs(cellId), 0);

      d_locallyOwnedCells[cellId]->cellNodeIdtoGlobalNodeId(vecGlobalNodeId);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    const std::vector<global_size_type> &
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getBoundaryIds() const
    {
      return d_boundaryIds;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::FECellIterator
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::beginLocallyOwnedCells()
    {
      return d_locallyOwnedCells.begin();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::FECellIterator
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::endLocallyOwnedCells()
    {
      return d_locallyOwnedCells.end();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::const_FECellIterator
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::beginLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.begin();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::const_FECellIterator
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::endLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.end();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::FECellIterator
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::beginLocalCells()
    {
      return d_localCells.begin();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::FECellIterator
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::endLocalCells()
    {
      return d_localCells.end();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::const_FECellIterator
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::beginLocalCells() const
    {
      return d_localCells.begin();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::const_FECellIterator
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::endLocalCells() const
    {
      return d_localCells.end();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    unsigned int
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getDim() const
    {
      return dim;
    }

    //
    // dealii specific functions
    //
    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const dealii::DoFHandler<dim>>
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getDoFHandler() const
    {
      return d_dofHandler;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    const dealii::FiniteElement<dim> &
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getReferenceFE(const size_type cellId) const
    {
      //
      // NOTE: The implementation is only restricted to
      // h-refinement (uniform p) and hence the reference FE
      // is same for all cellId. As a result, we pass index
      // 0 to dealii's dofHandler
      //
      // if (d_isHPRefined)
      //   {
      //     utils::throwException(
      //       false,
      //       "Support for hp-refined finite element mesh is not supported
      //       yet.");
      //   }

      return d_dofHandler->get_fe(0);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
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
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::nCumulativeLocallyOwnedCellDofs() const
    {
      return d_numCumulativeLocallyOwnedCellDofs;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::nCumulativeLocalCellDofs() const
    {
      return d_numCumulativeLocalCellDofs;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::totalRanges() const
    {
      return d_totalRanges;
    }


    // Some additional functions for getting geometric constriants matrix
    // and MPIPatternP2P

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                memorySpace,
                                dim>::getIntrinsicConstraints() const
    {
      return d_constraintsLocal;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
      CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                                memorySpace,
                                dim>::createConstraintsStart() const
    {
      std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
          constraintsLocal =
        std::make_shared<CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>>();

      constraintsLocal->copyFrom(*d_constraintsLocal);

      return(constraintsLocal);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::
      createConstraintsEnd(std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, 
                            memorySpace>> constraintsLocal) const
    {
      constraintsLocal->close();
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getMPIPatternP2P() const
    {
      return d_mpiPatternP2P;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    CFEBasisDofHandlerDealii<ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::isDistributed() const
    {
      return d_isDistributed;
    }

  } // namespace basis
} // namespace dftefe
