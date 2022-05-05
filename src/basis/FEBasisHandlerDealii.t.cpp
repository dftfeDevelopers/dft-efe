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
#include <utils/Exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/partitioner.h>
namespace dftefe
{
  namespace basis
  {
    namespace FEBasisHandlerDealiiInternal
    {
      template <size_type dim>
      size_type
      getLocallyOwnedCellsCumulativeDofs(
        const FEBasisManagerDealii<dim> *feBMDealii)
      {
        size_type returnValue          = 0;
        size_type numLocallyOwnedCells = feBMDealii->nLocallyOwnedCells();
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          returnValue += feBMDealii->nCellDofs(iCell);

        return returnValue;
      }


      template <size_type dim>
      void
      getLocallyOwnedCellStartIds(
        const FEBasisManagerDealii<dim> *feBMDealii,
        std::vector<size_type> &         locallyOwnedCellStartIds)
      {
        size_type numLocallyOwnedCells = feBMDealii->nLocallyOwnedCells();
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            if (iCell == 0)
              {
                locallyOwnedCellStartIds[iCell] = 0;
              }
            else
              {
                locallyOwnedCellStartIds[iCell] =
                  locallyOwnedCellStartIds[iCell - 1] +
                  feBMDealii->nCellDofs(iCell - 1);
              }
          }
      }

      template <size_type dim>
      void
      getLocallyOwnedCellGlobalIndices<dim>(
        const FEBasisManagerDealii<dim> *feBMDealii,
        std::vector<global_size_type> &  locallyOwnedCellGlobalIndices)
      {
        size_type numLocallyOwnedCells = feBMDealii->nLocallyOwnedCells();
        size_type cumulativeCellDofs   = 0;
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            const size_type numCellDofs = d_feBMDealii->nCellDofs(iCell);
            std::vector<global_size_type> cellGlobalIds(numCellDofs);
            feBMDealii->getCellDofsGlobalIds(iCell, cellGlobalIds);
            std::copy(cellGlobalIds.begin(),
                      cellGlobalIds.end(),
                      locallyOwnedCellGlobalIndices.begin() +
                        cumulativeCellDofs);
            cumulativeCellDofs += numCellDofs;
          }
      }

      template <typename ValueType, size_type dim>
      void
      setDealiiMatrixFreeLight(
        const FEBasisManagerDealii<dim> *feBMDealii,
        const std::map<
          std::string,
          std::shared_ptr<const FEBasisConstraintsDealii<dim, ValueType>>>
          &feConstraintsDealiiMap,
        dealii::MatrixFree<dim, ValueType> > &dealiiMatrixFree)
      {
        typename dealii::MatrixFree<dim>::AdditionalData dealiiAdditionalData;
        dealiiAdditionalData.tasks_parallel_scheme =
          dealii::MatrixFree<dim>::AdditionalData::partition_partition;
        dealii::UpdateFlags dealiiUpdateFlags     = dealii::update_default;
        dealiiAdditionalData.mapping_update_flags = dealiiUpdateFlags;

        std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
          feBMDealii->getDoFHandler();
        std::vector<const dealii::DoFHandler<dim> *> dofHandlerVec(
          numConstraints, dofHandler.get());
        std::vector<const dealii::AffineConstraints<ValueType> *>
                  dealiiAffineConstraintsVec(numConstraints, nullptr);
        size_type iConstraint = 0;
        for (auto it = feConstraintsDealiiMap.begin();
             it != d_feConstraintsDealiiMap.end();
             ++it)
          {
            dealiiAffineConstraintsVec[iConstraint] =
              &((it->second)->getAffineConstraints());
            iConstraint++;
          }

        std::vector<dealii::Quadrature<dim>> dealiiQuadratureTypeVec(
          1, dealii::QGauss<dim>(1));
        dealiiMatrixFree.clear();
        dealiiMatrixFree.reinit(dofHandlerVec,
                                dealiiAffineConstraintsVec,
                                dealiiQuadratureTypeVec,
                                dealiiAdditionalData);
      }

      template <typename ValueType, size_type dim>
      void
      getGhostIndices(const dealii::MatrixFree<dim, ValueType> >
                        &dealiiMatrixFree,
                      const size_type                constraintId,
                      std::vector<global_size_type> &ghostIndices)
      {
        const dealii::Utilities::MPI::Partitioner &dealiiPartitioner =
          dealiiMatrixFree.get_vector_partitioner(constraintId);
        dealii::IndexSet &ghostIndexSet   = dealiiPartitioner.ghost_indices();
        const size_type   numGhostIndices = ghostIndexSet.n_elements();
        ghostIndices.resize(numGhostIndices, 0);
        ghostIndexSet.fill_index_vector(ghostIndices);
      }

      template <typename ValueType,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      getLocallyOwnedCellLocalIndices(
        const FEBasisManagerDealii<dim> *        feBMDealii,
        const utils::MPIPatternP2P<memorySpace> *mpiPatternP2P,
        const std::vector<global_size_type> &    locallyOwnedCellGlobalIndices,
        std::vector<size_type> &                 locallyOwnedCellLocalIndices)
      {
        size_type numLocallyOwnedCells = feBMDealii->nLocallyOwnedCells();
        size_type cumulativeDofs       = 0;
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            const size_type numCellDofs = d_feBMDealii->nCellDofs(iCell);
            for (size_type iDof = 0; iDof < numCellDofs; ++iDof)
              {
                const global_size_type globalId =
                  locallyOwnedCellGlobalIndices[cumulativeDofs + iDof];
                locallyOwnedCellLocalIndices[cumulativeDofs + iDof] =
                  mpiPatternP2P->globalToLocal(globalId);
              }
            cumulativeDofs += numCellDofs;
          }
      }
    } // end of namespace FEBasisHandlerDealiiInternal


#ifdef DFTEFE_WITH_MPI
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::FEBasisHandlerDealii(
      std::shared_ptr<const BasisManager>                       basisManager,
      std::map<std::string, std::shared_ptr<const Constraints>> constraintsMap,
      const MPI_Comm &                                          mpiComm)
    {
      reinit(basisManager, constraintsMap, mpiComm);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::reinit(
      std::shared_ptr<const BasisManager>                       basisManager,
      std::map<std::string, std::shared_ptr<const Constraints>> constraintsMap,
      const MPI_Comm &                                          mpiComm)
    {
      d_mpiComm    = mpiComm;
      d_feBMDealii = std::dynamic_pointer_cast<const FEBasisManagerDealii<dim>>(
        basisManager);
      utils::throwException(
        d_feBMDealii != nullptr,
        "Error in casting the input basis manager in FEBasisHandlerDealii to FEBasisParitionerDealii");
      const size_type numConstraints = constraintsMap.size();
      for (auto it = constraintsMap.begin(), it != constraintsMap.end(); ++it)
        {
          std::string constraintsName = it->first;
          std::shared_ptr<const FEBasisConstraintsDealii<dim, ValueType>>
            feBasisConstraintsDealii = std::dynamic_pointer_cast<
              const FEBasisConstraintsDealii<dim, ValueType>>(it->second);
          utils::throwException(
            feBasisConstraintsDealii != nullptr,
            "Error in casting the input constraints to FEBasisConstraintsDealii in FEBasisHandlerDealii");
          d_feConstraintsDealiiMap[constraintsName] = feBasisConstraintsDealii;
        }

      d_locallyOwnedRange = d_feBMDealii->getLocallyOwnedRange();

      size_type       numLocallyOwnedCells = d_feBMDealii->nLocallyOwnedCells();
      const size_type cumulativeCellDofs =
        FEBasisHandlerDealiiInternal::getLocallyOwnedCellsCumulativeDofs<dim>(
          d_feBMDealii.get());

      //
      // populate d_locallyOwnedCellStartIds
      //
      std::vector<size_type> locallyOwnedCellStartIdsTmp(numLocallyOwnedCells,
                                                         0);
      FEBasisHandlerDealiiInternal::getLocallyOwnedCellStartIds<dim>(
        d_feBMDealii.get(), locallyOwnedCellStartIdsTmp);
      d_locallyOwnedCellStartIds.resize(numLocallyOwnedCells);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        numLocallyOwnedCells,
        d_locallyOwnedCellStartIds.data(),
        locallyOwnedCellStartIdsTmp.begin());

      //
      // populate d_locallyOwnedCellGlobalIndices
      //
      std::vector<global_size_type> locallyOwnedCellGlobalIndicesTmp(
        cumulativeCellDofs, 0);
      FEBasisHandlerDealiiInternal::getLocallyOwnedCellGlobalIndices<dim>(
        d_feBMDealii.get(), locallyOwnedCellGlobalIndicesTmp);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        cumulativeCellDofs,
        d_locallyOwnedCellGlobalIndices.data(),
        locallyOwnedCellGlobalIndicesTmp.begin());

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
      dealii::MatrixFree<dim, ValueType> > dealiiMatrixFree;
      FEBasisHandlerDealiiInternal::setDealiiMatrixFreeLight<ValueType, dim>(
        d_feBMDealii.get(), d_feConstraintsDealiiMap, dealiiMatrixFree);

      iConstraint = 0;
      for (auto it = d_feConstraintsDealiiMap.begin();
           it != d_feConstraintsDealiiMap.end();
           ++it)
        {
          const std::string constraintName = it->first;

          //
          // populate d_ghostIndicesMap
          //
          std::vector<global_size_type> ghostIndicesTmp(0);
          FEBasisHandlerDealiiInternal::getGhostIndices<ValueType, dim>(
            dealiiMatrixFree, iConstraint, ghostIndicesTmp);
          auto globalSizeVector = std::make_shared<
            typename BasisHandler<memorySpace>::GlobalSizeTypeVector>(
            numGhostIndices);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numGhostIndices, globalSizeVector->data(), ghostIndicesTmp.begin());
          d_ghostIndicesMap[constraintName] = globalSizeVector;

          //
          // populate d_mpiPatternP2PMap
          //
          auto mpiPatternP2P =
            std::make_shared<utils::MPIPatternP2P<memorySpace>>(
              d_locallyOwnedRange, ghostIndicesTmp, d_mpiComm);
          d_mpiPatternP2PMap[constraintName] = mpiPatternP2P;

          std::vector<size_type> locallyOwnedCellLocalIndicesTmp(
            cumulativeCellDofs, 0);
          FEBasisHandlerDealiiInternal::getLocallyOwnedCellLocalIndices(
            d_feBMDealii.get(),
            mpiPatternP2P.get(),
            locallyOwnedCellGlobalIndicesTmp,
            locallyOwnedCellLocalIndicesTmp);
          auto globalSizeVector = std::make_shared<
            typename BasisHandler<memorySpace>::GlobalSizeTypeVector>(
            cumulativeDofs, 0);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            cumulativeDofs,
            globalSizeVector->data(),
            locallyOwnedCellLocalIndicesTmp.begin());

          d_locallyOwnedCellLocalIndicesMap[constraintName] = globalSizeVector;

          iConstraint++;
        }
    }
#else
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::FEBasisHandlerDealii(
      std::shared_ptr<const BasisManager>                       basisManager,
      std::map<std::string, std::shared_ptr<const Constraints>> constraintsMap)
    {
      reinit(basisManager, constraintsMap);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::reinit(
      std::shared_ptr<const BasisManager>                       basisManager,
      std::map<std::string, std::shared_ptr<const Constraints>> constraintsMap)
    {}
#endif // DFTEFE_WITH_MPI

  } // end of namespace basis
} // end of namespace dftefe
