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
#include <utils/Exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>
#include <string>
namespace dftefe
{
  namespace basis
  {
    namespace EFEBasisHandlerDealiiInternal
    {
      template <typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      size_type
      getLocallyOwnedCellsCumulativeDofs(
        const basis::EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>
          *efeBMDealii)
      {
        size_type returnValue          = 0;
        size_type numLocallyOwnedCells = efeBMDealii->nLocallyOwnedCells();
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          returnValue += efeBMDealii->nCellDofs(iCell);

        return returnValue;
      }

      template <typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getNumLocallyOwnedCellDofs(
        const basis::EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>
          *                     efeBMDealii,
        std::vector<size_type> &locallyOwnedCellDofs)
      {
        size_type numLocallyOwnedCells = efeBMDealii->nLocallyOwnedCells();
        locallyOwnedCellDofs.resize(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          locallyOwnedCellDofs[iCell] = efeBMDealii->nCellDofs(iCell);
      }

      template <typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getLocallyOwnedCellStartIds(
        const basis::EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>
          *                     efeBMDealii,
        std::vector<size_type> &locallyOwnedCellStartIds)
      {
        size_type numLocallyOwnedCells = efeBMDealii->nLocallyOwnedCells();
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
                  efeBMDealii->nCellDofs(iCell - 1);
              }
          }
      }

      template <typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getLocallyOwnedCellGlobalIndices(
        const basis::EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>
          *                            efeBMDealii,
        std::vector<global_size_type> &locallyOwnedCellGlobalIndices)
      {
        size_type numLocallyOwnedCells = efeBMDealii->nLocallyOwnedCells();
        size_type cumulativeCellDofs   = 0;
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            const size_type numCellDofs = efeBMDealii->nCellDofs(iCell);
            std::vector<global_size_type> cellGlobalIds(numCellDofs);
            efeBMDealii->getCellDofsGlobalIds(iCell, cellGlobalIds);
            std::copy(cellGlobalIds.begin(),
                      cellGlobalIds.end(),
                      locallyOwnedCellGlobalIndices.begin() +
                        cumulativeCellDofs);
            cumulativeCellDofs += numCellDofs;
          }
      }

      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      setDealiiMatrixFreeLight(
        const EFEBasisManagerDealii<dim> *efeBMDealii,
        const std::shared_ptr<
          const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>> 
            efeConstraintsDealii,
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
        dealiiMatrixFree.reinit(mappingDealii, *(efeBMDealii->getDoFHandler()),
                                (efeConstraintsDealii)->getAffineConstraints(),
                                dealiiQuadratureType,
                                dealiiAdditionalData);
      }

      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getGhostIndices(
        const dealii::MatrixFree<dim, ValueTypeBasisCoeff> &dealiiMatrixFree,
        const EFEBasisManagerDealii<dim> *                  efeBMDealii,
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
        std::vector<global_size_type> ghostIndicesEnriched(0);
        ghostIndicesEnriched = efeBMDealii->getGhostEnrichmentGlobalIds();

        ghostIndices.clear();
        ghostIndices.insert(ghostIndices.begin(),
                            ghostIndicesClassical.begin(),
                            ghostIndicesClassical.end());
        ghostIndices.insert(ghostIndices.end(),
                            ghostIndicesEnriched.begin(),
                            ghostIndicesEnriched.end());
      }

      template <typename ValueTypeBasisCoeff,
                typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getLocallyOwnedCellLocalIndices(
        const basis::EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>
          *                                           efeBMDealii,
        const utils::mpi::MPIPatternP2P<memorySpace> *mpiPatternP2P,
        const std::vector<global_size_type> &locallyOwnedCellGlobalIndices,
        std::vector<size_type> &             locallyOwnedCellLocalIndices)
      {
        size_type numLocallyOwnedCells = efeBMDealii->nLocallyOwnedCells();
        size_type cumulativeDofs       = 0;
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            const size_type numCellDofs = efeBMDealii->nCellDofs(iCell);
            for (size_type iDof = 0; iDof < numCellDofs; ++iDof)
              {
                const global_size_type globalId =
                  locallyOwnedCellGlobalIndices[cumulativeDofs + iDof];
                locallyOwnedCellLocalIndices[cumulativeDofs + iDof] =
                  mpiPatternP2P->globalToLocal(globalId);
                /*
                if (!locallyOwnedCellLocalIndices[cumulativeDofs + iDof] ==
                    globalId)
                  std::cout
                    << " Error in mpiP2P global id to local not correct for id =
                "
                    << globalId << std::endl;
                */
              }
            cumulativeDofs += numCellDofs;
          }
      }
    } // end of namespace EFEBasisHandlerDealiiInternal


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>::
      EFEBasisHandlerDealii(
        std::shared_ptr<const BasisManager> basisManager,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>
                                   constraints,
        const utils::mpi::MPIComm &mpiComm)
    {
      reinit(basisManager, constraints, mpiComm);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::reinit(
      std::shared_ptr<const BasisManager> basisManager,
      std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>
                                   constraints,
      const utils::mpi::MPIComm &mpiComm)
    {
      // initialize the private data members.
      d_isDistributed = true;
      d_mpiComm       = mpiComm;
      d_efeBMDealii   = std::dynamic_pointer_cast<
        const EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>(
        basisManager);
      utils::throwException(
        d_efeBMDealii != nullptr,
        "Error in casting the input basis manager in EFEBasisHandlerDealii to EFEBasisManagerDealii");

      std::shared_ptr<
        const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
          efeConstraintsDealii;

      std::shared_ptr<
        const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
          efeBasisConstraintsDealii = std::dynamic_pointer_cast<
            const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
              constraints);
      utils::throwException(
        efeBasisConstraintsDealii != nullptr,
        "Error in casting the input constraints to EFEConstraintsDealii in EFEBasisHandlerDealii");

      d_locallyOwnedRanges = d_efeBMDealii->getLocallyOwnedRanges();

      size_type numLocallyOwnedCells = d_efeBMDealii->nLocallyOwnedCells();
      EFEBasisHandlerDealiiInternal::
        getNumLocallyOwnedCellDofs<ValueTypeBasisData, memorySpace, dim>(
          d_efeBMDealii.get(), d_numLocallyOwnedCellDofs);
      const size_type cumulativeCellDofs =
        EFEBasisHandlerDealiiInternal::getLocallyOwnedCellsCumulativeDofs<
          ValueTypeBasisData,
          memorySpace,
          dim>(d_efeBMDealii.get());

      //
      // populate d_locallyOwnedCellStartIds
      //
      d_locallyOwnedCellStartIds.resize(numLocallyOwnedCells, 0);
      EFEBasisHandlerDealiiInternal::
        getLocallyOwnedCellStartIds<ValueTypeBasisData, memorySpace, dim>(
          d_efeBMDealii.get(), d_locallyOwnedCellStartIds);

      //
      // populate d_locallyOwnedCellGlobalIndices
      //
      std::vector<global_size_type> locallyOwnedCellGlobalIndicesTmp(
        cumulativeCellDofs, 0);
      d_locallyOwnedCellGlobalIndices.resize(cumulativeCellDofs);
      EFEBasisHandlerDealiiInternal::
        getLocallyOwnedCellGlobalIndices<ValueTypeBasisData, memorySpace, dim>(
          d_efeBMDealii.get(), locallyOwnedCellGlobalIndicesTmp);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        cumulativeCellDofs,
        d_locallyOwnedCellGlobalIndices.data(),
        locallyOwnedCellGlobalIndicesTmp.data());

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

      dealii::MatrixFree<dim, ValueTypeBasisCoeff> dealiiMatrixFree;
      EFEBasisHandlerDealiiInternal::
        setDealiiMatrixFreeLight<ValueTypeBasisCoeff, memorySpace, dim>(
          d_efeBMDealii.get(), efeConstraintsDealii, dealiiMatrixFree);

      size_type classicalAttributeId =
        d_efeBMDealii
          ->getBasisAttributeToRangeIdMap()[BasisIdAttribute::CLASSICAL];

          //
          // populate d_ghostIndices
          //
          std::vector<global_size_type> ghostIndicesTmp(0);
          EFEBasisHandlerDealiiInternal::getGhostIndices<ValueTypeBasisCoeff,
                                                         dim>(dealiiMatrixFree,
                                                              d_efeBMDealii.get(),
                                                              ghostIndicesTmp);
          const size_type numGhostIndices          = ghostIndicesTmp.size();
          auto  d_ghostIndices = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::GlobalSizeTypeVector>(
            numGhostIndices);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numGhostIndices,
            d_ghostIndices->data(),
            ghostIndicesTmp.data());

          //
          // populate d_mpiPatternP2P
          //
          auto d_mpiPatternP2P =
            std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
              d_locallyOwnedRanges, ghostIndicesTmp, d_mpiComm);

          // Creation of optimized constraint matrix having only
          // dealiimatrixfree trimmed constraint ids.
          // d_efeConstraintsDealiiOpt
          std::shared_ptr<
            EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
            d_efeConstraintsDealiiOpt = std::make_shared<
              EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>();
          d_efeConstraintsDealiiOpt->copyConstraintsData(
            *(efeConstraintsDealii), *d_mpiPatternP2P, classicalAttributeId);
          d_efeConstraintsDealiiOpt->populateConstraintsData(
            *d_mpiPatternP2P, classicalAttributeId);

          //
          // populate d_locallyOwnedCellLocalIndices
          //
          std::vector<size_type> locallyOwnedCellLocalIndicesTmp(
            cumulativeCellDofs, 0);
          EFEBasisHandlerDealiiInternal::getLocallyOwnedCellLocalIndices<
            ValueTypeBasisCoeff,
            ValueTypeBasisData,
            memorySpace,
            dim>(d_efeBMDealii.get(),
                 d_mpiPatternP2P.get(),
                 locallyOwnedCellGlobalIndicesTmp,
                 locallyOwnedCellLocalIndicesTmp);
          auto d_locallyOwnedCellLocalIndices = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::SizeTypeVector>(
            cumulativeCellDofs, 0);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            cumulativeCellDofs,
            d_locallyOwnedCellLocalIndices->data(),
            locallyOwnedCellLocalIndicesTmp.data());

      d_efeBMDealii->getBasisCenters(d_supportPoints);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>::
      EFEBasisHandlerDealii(
        std::shared_ptr<const BasisManager> basisManager,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>
                                   constraints)
    {
      reinit(basisManager, constraints);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::reinit(
      std::shared_ptr<const BasisManager> basisManager,
      std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>
                                   constraints)
    {
      d_isDistributed = false;
      d_efeBMDealii   = std::dynamic_pointer_cast<
        const EFEBasisManagerDealii<ValueTypeBasisData, memorySpace, dim>>(
        basisManager);
      utils::throwException(
        d_efeBMDealii != nullptr,
        "Error in casting the input basis manager in EFEBasisHandlerDealii to EFEBasisManagerDealii");

      std::shared_ptr<
        const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
          efeConstraintsDealii;

      std::shared_ptr<
        const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
          efeBasisConstraintsDealii = std::dynamic_pointer_cast<
            const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
              constraints);
      utils::throwException(
        efeBasisConstraintsDealii != nullptr,
        "Error in casting the input constraints to EFEConstraintsDealii in EFEBasisHandlerDealii");

      d_locallyOwnedRanges = d_efeBMDealii->getLocallyOwnedRanges();

      size_type numLocallyOwnedCells = d_efeBMDealii->nLocallyOwnedCells();
      EFEBasisHandlerDealiiInternal::
        getNumLocallyOwnedCellDofs<ValueTypeBasisData, memorySpace, dim>(
          d_efeBMDealii.get(), d_numLocallyOwnedCellDofs);
      const size_type cumulativeCellDofs =
        EFEBasisHandlerDealiiInternal::getLocallyOwnedCellsCumulativeDofs<
          ValueTypeBasisData,
          memorySpace,
          dim>(d_efeBMDealii.get());

      //
      // populate d_locallyOwnedCellStartIds
      //
      d_locallyOwnedCellStartIds.resize(numLocallyOwnedCells, 0);
      EFEBasisHandlerDealiiInternal::
        getLocallyOwnedCellStartIds<ValueTypeBasisData, memorySpace, dim>(
          d_efeBMDealii.get(), d_locallyOwnedCellStartIds);

      //
      // populate d_locallyOwnedCellGlobalIndices
      //
      std::vector<global_size_type> locallyOwnedCellGlobalIndicesTmp(
        cumulativeCellDofs, 0);
      d_locallyOwnedCellGlobalIndices.resize(cumulativeCellDofs);
      EFEBasisHandlerDealiiInternal::
        getLocallyOwnedCellGlobalIndices<ValueTypeBasisData, memorySpace, dim>(
          d_efeBMDealii.get(), locallyOwnedCellGlobalIndicesTmp);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        cumulativeCellDofs,
        d_locallyOwnedCellGlobalIndices.data(),
        locallyOwnedCellGlobalIndicesTmp.data());

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

      dealii::MatrixFree<dim, ValueTypeBasisCoeff> dealiiMatrixFree;
      EFEBasisHandlerDealiiInternal::
        setDealiiMatrixFreeLight<ValueTypeBasisCoeff, memorySpace, dim>(
          d_efeBMDealii.get(), efeConstraintsDealii, dealiiMatrixFree);

      size_type classicalAttributeId =
        d_efeBMDealii
          ->getBasisAttributeToRangeIdMap()[BasisIdAttribute::CLASSICAL];

          //
          // populate d_ghostIndices
          //
          std::vector<global_size_type> ghostIndicesTmp(0);
          EFEBasisHandlerDealiiInternal::getGhostIndices<ValueTypeBasisCoeff,
                                                         dim>(dealiiMatrixFree,
                                                              d_efeBMDealii.get(),
                                                              ghostIndicesTmp);
          const size_type numGhostIndices          = ghostIndicesTmp.size();
          auto  d_ghostIndices = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::GlobalSizeTypeVector>(
            numGhostIndices);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numGhostIndices,
            d_ghostIndices->data(),
            ghostIndicesTmp.data());

          //
          // populate d_mpiPatternP2P
          //
          //
          // create std vector of sizes
          std::vector<size_type> locallyOwnedRangesSizeVec(0);
          for (auto i : d_locallyOwnedRanges)
            {
              locallyOwnedRangesSizeVec.push_back(i.second - i.first);
            }
          auto d_mpiPatternP2P =
            std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
              locallyOwnedRangesSizeVec);

          // Creation of optimized constraint matrix having only
          // dealiimatrixfree trimmed constraint ids.
          // d_efeConstraintsDealiiOpt
          std::shared_ptr<
            EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
            d_efeConstraintsDealiiOpt = std::make_shared<
              EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>();
          d_efeConstraintsDealiiOpt->copyConstraintsData(
            *(efeConstraintsDealii), *d_mpiPatternP2P, classicalAttributeId);
          d_efeConstraintsDealiiOpt->populateConstraintsData(
            *d_mpiPatternP2P, classicalAttributeId);

          //
          // populate d_locallyOwnedCellLocalIndices
          //
          std::vector<size_type> locallyOwnedCellLocalIndicesTmp(
            cumulativeCellDofs, 0);
          EFEBasisHandlerDealiiInternal::getLocallyOwnedCellLocalIndices<
            ValueTypeBasisCoeff,
            ValueTypeBasisData,
            memorySpace,
            dim>(d_efeBMDealii.get(),
                 d_mpiPatternP2P.get(),
                 locallyOwnedCellGlobalIndicesTmp,
                 locallyOwnedCellLocalIndicesTmp);
          auto d_locallyOwnedCellLocalIndices = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::SizeTypeVector>(
            cumulativeCellDofs, 0);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            cumulativeCellDofs,
            d_locallyOwnedCellLocalIndices->data(),
            locallyOwnedCellLocalIndicesTmp.data());

      //
      // FIXME: Assumes linear mapping from reference cell to real cell
      //
      d_efeBMDealii->getBasisCenters(d_supportPoints);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>::isDistributed() const
    {
      return d_isDistributed;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const Constraints<ValueTypeBasisCoeff, memorySpace> &
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getConstraints() const
    {
      return *(d_efeConstraintsDealiiOpt);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getMPIPatternP2P() const
    {
      return d_mpiPatternP2P;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getLocallyOwnedRanges() const
    {
      return d_locallyOwnedRanges;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::nLocal() const
    {
      // add locallyownedranges here
      size_type numLocallyOwned = 0;
      for (auto i : d_locallyOwnedRanges)
        {
          numLocallyOwned = numLocallyOwned + i.second - i.first;
        }
      const size_type numGhost = (*(d_ghostIndices)).size();
      return (numLocallyOwned + numGhost);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::nLocallyOwned() const
    {
      size_type numLocallyOwned = 0;
      for (auto i : d_locallyOwnedRanges)
        {
          numLocallyOwned = numLocallyOwned + i.second - i.first;
        }
      return numLocallyOwned;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::nGhost() const
    {
      const size_type numGhost = (*(d_ghostIndices)).size();
      return numGhost;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getBasisCenters(const size_type       localId,
                      dftefe::utils::Point &basisCenter) const
    {
      global_size_type globalId = localToGlobalIndex(localId);

      if (d_supportPoints.find(globalId) != d_supportPoints.end())
        basisCenter = d_supportPoints.find(globalId)->second;
      else
        {
          std::string msg =
            "The localId does not have any point in the EFE mesh for id no ";
          msg = msg + std::to_string(globalId);
          utils::throwException(false, msg);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>::nLocallyOwnedCells() const
    {
      return d_efeBMDealii->nLocallyOwnedCells();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>::nCumulativeLocallyOwnedCellDofs() const
    {
      return d_efeBMDealii->nCumulativeLocallyOwnedCellDofs();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>::nLocallyOwnedCellDofs(const size_type cellId)
      const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to nLocallyOwnedCellDofs is greater than or "
        " equal to the number of locally owned cells.");
      return d_numLocallyOwnedCellDofs[cellId];
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                         ValueTypeBasisData,
                                         memorySpace,
                                         dim>::GlobalSizeTypeVector &
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getGhostIndices() const
    {
      return *(d_ghostIndices);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::pair<bool, size_type>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      inLocallyOwnedRanges(const global_size_type globalId) const
    {
      return (d_mpiPatternP2P)->inLocallyOwnedRanges(globalId);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::pair<bool, size_type>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::isGhostEntry(
      const global_size_type ghostId) const
    {
      return (d_mpiPatternP2P)->isGhostEntry(ghostId);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      globalToLocalIndex(const global_size_type globalId) const
    {
      return (d_mpiPatternP2P)->globalToLocal(globalId);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    global_size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      localToGlobalIndex(const size_type   localId) const
    {
      return (d_mpiPatternP2P)->localToGlobal(localId);
    }

    //
    // FE specific functions
    //
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                   ValueTypeBasisData,
                                   memorySpace,
                                   dim>::const_GlobalIndexIter
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsBegin() const
    {
      return d_locallyOwnedCellGlobalIndices.begin();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                   ValueTypeBasisData,
                                   memorySpace,
                                   dim>::const_GlobalIndexIter
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsBegin(const size_type   cellId) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellGlobalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");
      return d_locallyOwnedCellGlobalIndices.begin() +
             d_locallyOwnedCellStartIds[cellId];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      const_GlobalIndexIter
      EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
        locallyOwnedCellGlobalDofIdsEnd(const size_type   cellId) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellGlobalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");
      return d_locallyOwnedCellGlobalIndices.begin() +
             d_locallyOwnedCellStartIds[cellId] +
             d_numLocallyOwnedCellDofs[cellId];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                   ValueTypeBasisData,
                                   memorySpace,
                                   dim>::const_LocalIndexIter
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsBegin() const
    {
      return (*(d_locallyOwnedCellLocalIndices)).begin();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                   ValueTypeBasisData,
                                   memorySpace,
                                   dim>::const_LocalIndexIter
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsBegin(const size_type   cellId) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellLocalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");

      return (*(d_locallyOwnedCellLocalIndices)).begin() + d_locallyOwnedCellStartIds[cellId];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      const_LocalIndexIter
      EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
        locallyOwnedCellLocalDofIdsEnd(const size_type   cellId) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellLocalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");

      return (*(d_locallyOwnedCellLocalIndices)).begin() + d_locallyOwnedCellStartIds[cellId] +
             d_numLocallyOwnedCellDofs[cellId];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const BasisManager &
    EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace,
                          dim>::getBasisManager() const
    {
      return *d_efeBMDealii;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getCellDofsLocalIds(const size_type         cellId,
                          std::vector<size_type> &vecLocalNodeId) const
    {
      std::vector<global_size_type> vecGlobalNodeId(0);
      d_efeBMDealii->getCellDofsGlobalIds(cellId, vecGlobalNodeId);
      for (auto i : vecGlobalNodeId)
        {
          vecLocalNodeId.push_back(globalToLocalIndex(i));
        }
    }

  } // end of namespace basis
} // end of namespace dftefe
