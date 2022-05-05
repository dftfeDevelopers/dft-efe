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
#include <deal.II/matrix_free/matrix_free.h>
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
      getNumLocallyOwnedCellDofs(const FEBasisManagerDealii<dim> *feBMDealii,
                                 std::vector<size_type> &locallyOwnedCellDofs)
      {
        size_type numLocallyOwnedCells = feBMDealii->nLocallyOwnedCells();
        locallyOwnedCellDofs.resize(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          locallyOwnedCellDofs[iCell] = feBMDealii->nCellDofs(iCell);
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
      getLocallyOwnedCellGlobalIndices(
        const FEBasisManagerDealii<dim> *feBMDealii,
        std::vector<global_size_type> &  locallyOwnedCellGlobalIndices)
      {
        size_type numLocallyOwnedCells = feBMDealii->nLocallyOwnedCells();
        size_type cumulativeCellDofs   = 0;
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            const size_type numCellDofs = feBMDealii->nCellDofs(iCell);
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
          std::shared_ptr<const FEConstraintsDealii<dim, ValueType>>>
          &feConstraintsDealiiMap,
        dealii::MatrixFree<dim, ValueType> &dealiiMatrixFree)
      {
        typename dealii::MatrixFree<dim>::AdditionalData dealiiAdditionalData;
        dealiiAdditionalData.tasks_parallel_scheme =
          dealii::MatrixFree<dim>::AdditionalData::partition_partition;
        dealii::UpdateFlags dealiiUpdateFlags     = dealii::update_default;
        dealiiAdditionalData.mapping_update_flags = dealiiUpdateFlags;

        std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
          feBMDealii->getDoFHandler();
  
        size_type numConstraints = feConstraintsDealiiMap.size();
        std::vector<const dealii::DoFHandler<dim> *> dofHandlerVec(
          numConstraints, dofHandler.get());
        std::vector<const dealii::AffineConstraints<ValueType> *>
                  dealiiAffineConstraintsVec(numConstraints, nullptr);
        size_type iConstraint = 0;
        for (auto it = feConstraintsDealiiMap.begin();
             it != feConstraintsDealiiMap.end();
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
      getGhostIndices(const dealii::MatrixFree<dim, ValueType>
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
      void getLocallyOwnedCellLocalIndices(
        const FEBasisManagerDealii<dim> *        feBMDealii,
        const utils::MPIPatternP2P<memorySpace> *mpiPatternP2P,
        const std::vector<global_size_type> &    locallyOwnedCellGlobalIndices,
        std::vector<size_type> &                 locallyOwnedCellLocalIndices)
      {
        size_type numLocallyOwnedCells = feBMDealii->nLocallyOwnedCells();
        size_type cumulativeDofs       = 0;
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            const size_type numCellDofs = feBMDealii->nCellDofs(iCell);
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
      std::map<std::string, std::shared_ptr<const Constraints<ValueType>>> constraintsMap,
      const MPI_Comm &                                          mpiComm)
    {
      reinit(basisManager, constraintsMap, mpiComm);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void FEBasisHandlerDealii<ValueType, memorySpace, dim>::reinit(
      std::shared_ptr<const BasisManager>                       basisManager,
      std::map<std::string, std::shared_ptr<const Constraints<ValueType>>> constraintsMap,
      const MPI_Comm &                                          mpiComm)
    {
      d_mpiComm    = mpiComm;
      d_feBMDealii = std::dynamic_pointer_cast<const FEBasisManagerDealii<dim>>(
        basisManager);
      utils::throwException(
        d_feBMDealii != nullptr,
        "Error in casting the input basis manager in FEBasisHandlerDealii to FEBasisParitionerDealii");
      const size_type numConstraints = constraintsMap.size();
      for (auto it = constraintsMap.begin(); it != constraintsMap.end(); ++it)
        {
          std::string constraintsName = it->first;
          std::shared_ptr<const FEConstraintsDealii<dim, ValueType>>
            feBasisConstraintsDealii = std::dynamic_pointer_cast<
              const FEConstraintsDealii<dim, ValueType>>(it->second);
          utils::throwException(
            feBasisConstraintsDealii != nullptr,
            "Error in casting the input constraints to FEConstraintsDealii in FEBasisHandlerDealii");
          d_feConstraintsDealiiMap[constraintsName] = feBasisConstraintsDealii;
        }

      d_locallyOwnedRange = d_feBMDealii->getLocallyOwnedRange();

      size_type numLocallyOwnedCells = d_feBMDealii->nLocallyOwnedCells();
      FEBasisHandlerDealiiInternal::getNumLocallyOwnedCellDofs<dim>(
        d_feBMDealii.get(), d_numLocallyOwnedCellDofs);
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
      dealii::MatrixFree<dim, ValueType>  dealiiMatrixFree;
      FEBasisHandlerDealiiInternal::setDealiiMatrixFreeLight<ValueType, dim>(
        d_feBMDealii.get(), d_feConstraintsDealiiMap, dealiiMatrixFree);

      size_type iConstraint = 0;
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
          const size_type numGhostIndices  = ghostIndicesTmp.size();
          auto            globalSizeVectorGhostMap = std::make_shared<
            typename BasisHandler<ValueType,memorySpace>::GlobalSizeTypeVector>(
            numGhostIndices);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numGhostIndices, globalSizeVectorGhostMap->data(), ghostIndicesTmp.begin());
          d_ghostIndicesMap[constraintName] = globalSizeVectorGhostMap;

          //
          // populate d_mpiPatternP2PMap
          //
          auto mpiPatternP2P =
            std::make_shared<utils::MPIPatternP2P<memorySpace>>(
              d_locallyOwnedRange, ghostIndicesTmp, d_mpiComm);
          d_mpiPatternP2PMap[constraintName] = mpiPatternP2P;

          //
          // populate d_locallyOwnedCellLocalIndices
          //
          std::vector<size_type> locallyOwnedCellLocalIndicesTmp(
            cumulativeCellDofs, 0);
          FEBasisHandlerDealiiInternal::
            getLocallyOwnedCellLocalIndices<ValueType, memorySpace, dim>(
              d_feBMDealii.get(),
              mpiPatternP2P.get(),
              locallyOwnedCellGlobalIndicesTmp,
              locallyOwnedCellLocalIndicesTmp);
          auto globalSizeVectorLocalIndicies = std::make_shared<
            typename BasisHandler<ValueType,memorySpace>::GlobalSizeTypeVector>(
            cumulativeCellDofs, 0);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            cumulativeCellDofs,
            globalSizeVectorLocalIndicies->data(),
            locallyOwnedCellLocalIndicesTmp.begin());

          d_locallyOwnedCellLocalIndicesMap[constraintName] = globalSizeVectorLocalIndicies;

          iConstraint++;
        }
    }
#else
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::FEBasisHandlerDealii(
      std::shared_ptr<const BasisManager>                       basisManager,
      std::map<std::string, std::shared_ptr<const Constraints<ValueType>>> constraintsMap)
    {
      reinit(basisManager, constraintsMap);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void FEBasisHandlerDealii<ValueType, memorySpace, dim>::reinit(
      std::shared_ptr<const BasisManager>                       basisManager,
      std::map<std::string, std::shared_ptr<const Constraints<ValueType>>> constraintsMap)
    {
      d_feBMDealii = std::dynamic_pointer_cast<const FEBasisManagerDealii<dim>>(
        basisManager);
      utils::throwException(
        d_feBMDealii != nullptr,
        "Error in casting the input basis manager in FEBasisHandlerDealii to FEBasisParitionerDealii");
      const size_type numConstraints = constraintsMap.size();
      for (auto it = constraintsMap.begin(); it != constraintsMap.end(); ++it)
        {
          std::string constraintsName = it->first;
          std::shared_ptr<const FEConstraintsDealii<dim, ValueType>>
            feBasisConstraintsDealii = std::dynamic_pointer_cast<
              const FEConstraintsDealii<dim, ValueType>>(it->second);
          utils::throwException(
            feBasisConstraintsDealii != nullptr,
            "Error in casting the input constraints to FEConstraintsDealii in FEBasisHandlerDealii");
          d_feConstraintsDealiiMap[constraintsName] = feBasisConstraintsDealii;
        }

      d_locallyOwnedRange = d_feBMDealii->getLocallyOwnedRange();

      size_type numLocallyOwnedCells = d_feBMDealii->nLocallyOwnedCells();
      FEBasisHandlerDealiiInternal::getNumLocallyOwnedCellDofs<dim>(
        d_feBMDealii.get(), d_numLocallyOwnedCellDofs);
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
      dealii::MatrixFree<dim, ValueType> dealiiMatrixFree;
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
          const size_type numGhostIndices  = ghostIndicesTmp.size();
          auto            globalSizeVector = std::make_shared<
            typename BasisHandler<ValueType,memorySpace>::GlobalSizeTypeVector>(
            numGhostIndices);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numGhostIndices, globalSizeVector->data(), ghostIndicesTmp.begin());
          d_ghostIndicesMap[constraintName] = globalSizeVector;

          //
          // populate d_mpiPatternP2PMap
          //
          auto mpiPatternP2P =
            std::make_shared<utils::MPIPatternP2P<memorySpace>>(
              d_locallyOwnedRange);
          d_mpiPatternP2PMap[constraintName] = mpiPatternP2P;

          //
          // populate d_locallyOwnedCellLocalIndices
          //
          std::vector<size_type> locallyOwnedCellLocalIndicesTmp(
            cumulativeCellDofs, 0);
          FEBasisHandlerDealiiInternal::
            getLocallyOwnedCellLocalIndices<ValueType, memorySpace, dim>(
              d_feBMDealii.get(),
              mpiPatternP2P.get(),
              locallyOwnedCellGlobalIndicesTmp,
              locallyOwnedCellLocalIndicesTmp);
          auto globalSizeVector = std::make_shared<
            typename BasisHandler<ValueType,memorySpace>::GlobalSizeTypeVector>(
            cumulativeDofs, 0);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            cumulativeDofs,
            globalSizeVector->data(),
            locallyOwnedCellLocalIndicesTmp.begin());

          d_locallyOwnedCellLocalIndicesMap[constraintName] = globalSizeVector;

          iConstraint++;
        }
    }
#endif // DFTEFE_WITH_MPI

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::pair<global_size_type, global_size_type>
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::getLocallyOwnedRange(
      const std::string constraintsName) const
    {
      return d_locallyOwnedRange;
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::nLocal(
      const std::string constraintsName) const
    {
      auto it = d_ghostIndicesMap.find(constraintsName);
      if (it == d_ghostIndicesMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The ghost indices in FEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      const size_type numLocallyOwned =
        d_locallyOwnedRange.second - d_locallyOwnedRange.first;
      const size_type numGhost = (*(it->second)).size();
      return (numLocallyOwned + numGhost);
    }
    
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::nLocallyOwned(
      const std::string constraintsName) const
    {
      return (d_locallyOwnedRange.second - d_locallyOwnedRange.first);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::nGhost(
      const std::string constraintsName) const
    {
      auto it = d_ghostIndicesMap.find(constraintsName);
      if (it == d_ghostIndicesMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The ghost indices in FEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      const size_type numGhost = (*(it->second)).size();
      return numGhost;
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::numLocallyOwnedCellDofs(
      const size_type cellId) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to numLocallyOwnedCellDofs is greater than or "
        " equal to the number of locally owned cells.");
      return d_numLocallyOwnedCellDofs[cellId];
    }


    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename FEBasisHandlerDealii<ValueType, memorySpace, dim>::GlobalSizeTypeVector &
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::getGhostIndices(
      const std::string constraintsName) const
    {
      auto it = d_ghostIndicesMap.find(constraintsName);
      if (it == d_ghostIndicesMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The ghost indices in FEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return *(it->second());
    }


    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::inLocallyOwnedRange(
      const global_size_type globalId,
      const std::string      constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in FEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return (it->second)->inLocallyOwnedRange(globalId);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::isGhostEntry(
      const global_size_type ghostId,
      const std::string      constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in FEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return (it->second)->isGhostEntry(ghostId);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::globalToLocalIndex(
      const global_size_type globalId,
      const std::string      constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in FEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return (it->second)->globalToLocal(globalId);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    global_size_type
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::localToGlobalIndex(
      const size_type   localId,
      const std::string constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in FEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return (it->second)->localToGlobal(localId);
    }

    //
    // FE specific functions
    //
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisHandlerDealii<ValueType, memorySpace, dim>::const_GlobalIndexIter
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsBegin(const std::string constraintsName) const
    {
      return d_locallyOwnedCellGlobalIndices.begin();
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisHandlerDealii<ValueType, memorySpace, dim>::const_GlobalIndexIter
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsBegin(const size_type   cellId,
                                        const std::string constraintsName) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellGlobalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");
      return d_locallyOwnedCellGlobalIndices.begin() +
             d_locallyOwnedCellStartIds[cellId];
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisHandlerDealii<ValueType, memorySpace, dim>::const_GlobalIndexIter
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsEnd(const size_type   cellId,
                                      const std::string constraintsName) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellGlobalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");
      return d_locallyOwnedCellGlobalIndices.begin() +
             d_locallyOwnedCellStartIds[cellId] +
             d_numLocallyOwnedCellDofs[cellId];
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisHandlerDealii<ValueType, memorySpace, dim>::const_LocalIndexIter
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsBegin(const std::string constraintsName) const
    {
      auto it = d_locallyOwnedCellLocalIndicesMap.find(constraintsName);
      utils::throwException<utils::InvalidArgument>(
        it != d_locallyOwnedCellLocalIndicesMap.end(),
        "The cell local indices is not created for the constraints " +
          constraintsName);
      return (*(it->second)).begin();
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisHandlerDealii<ValueType, memorySpace, dim>::const_LocalIndexIter
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsBegin(const size_type   cellId,
                                       const std::string constraintsName) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellLocalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");

      auto it = d_locallyOwnedCellLocalIndicesMap.find(constraintsName);
      DFTEFE_AssertWithMsg(
        it != d_locallyOwnedCellLocalIndicesMap.end(),
        "The cell local indices for the given constraintsName is not created");
      return (*(it->second)).begin() + d_locallyOwnedCellStartIds[cellId];
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisHandlerDealii<ValueType, memorySpace, dim>::const_LocalIndexIter
    FEBasisHandlerDealii<ValueType, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsEnd(const size_type   cellId,
                                     const std::string constraintsName) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellLocalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");

      auto it = d_locallyOwnedCellLocalIndicesMap.find(constraintsName);
      DFTEFE_AssertWithMsg(
        it != d_locallyOwnedCellLocalIndicesMap.end(),
        "The cell local indices for the given constraintsName is not created");
      return (*(it->second)).begin() + d_locallyOwnedCellStartIds[cellId] +
             d_numLocallyOwnedCellDofs[cellId];
    }
  } // end of namespace basis
} // end of namespace dftefe
