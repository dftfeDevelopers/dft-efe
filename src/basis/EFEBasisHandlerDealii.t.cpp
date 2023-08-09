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
      template <size_type dim>
      size_type
      getLocallyOwnedCellsCumulativeDofs(
        const EFEBasisManagerDealii<dim> *efeBMDealii)
      {
        size_type returnValue          = 0;
        size_type numLocallyOwnedCells = efeBMDealii->nLocallyOwnedCells();
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          returnValue += efeBMDealii->nCellDofs(iCell);

        return returnValue;
      }

      template <size_type dim>
      void
      getNumLocallyOwnedCellDofs(const EFEBasisManagerDealii<dim> *efeBMDealii,
                                 std::vector<size_type> &locallyOwnedCellDofs)
      {
        size_type numLocallyOwnedCells = efeBMDealii->nLocallyOwnedCells();
        locallyOwnedCellDofs.resize(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          locallyOwnedCellDofs[iCell] = efeBMDealii->nCellDofs(iCell);
      }

      template <size_type dim>
      void
      getLocallyOwnedCellStartIds(
        const EFEBasisManagerDealii<dim> *efeBMDealii,
        std::vector<size_type> &          locallyOwnedCellStartIds)
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

      template <size_type dim>
      void
      getLocallyOwnedCellGlobalIndices(
        const EFEBasisManagerDealii<dim> *efeBMDealii,
        std::vector<global_size_type> &   locallyOwnedCellGlobalIndices)
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
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      setDealiiMatrixFreeLight(
        const EFEBasisManagerDealii<dim> *efeBMDealii,
        const std::map<
          std::string,
          std::shared_ptr<
            const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>>
          &                                           efeConstraintsDealiiMap,
        dealii::MatrixFree<dim, ValueTypeBasisCoeff> &dealiiMatrixFree)
      {
        typename dealii::MatrixFree<dim>::AdditionalData dealiiAdditionalData;
        dealiiAdditionalData.tasks_parallel_scheme =
          dealii::MatrixFree<dim>::AdditionalData::partition_partition;
        dealii::UpdateFlags dealiiUpdateFlags     = dealii::update_default;
        dealiiAdditionalData.mapping_update_flags = dealiiUpdateFlags;

        std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
          efeBMDealii->getDoFHandler();

        size_type numConstraints = efeConstraintsDealiiMap.size();
        std::vector<const dealii::DoFHandler<dim> *> dofHandlerVec(
          numConstraints, dofHandler.get());
        std::vector<const dealii::AffineConstraints<ValueTypeBasisCoeff> *>
                  dealiiAffineConstraintsVec(numConstraints, nullptr);
        size_type iConstraint = 0;
        for (auto it = efeConstraintsDealiiMap.begin();
             it != efeConstraintsDealiiMap.end();
             ++it)
          {
            dealiiAffineConstraintsVec[iConstraint] =
              &((it->second)->getAffineConstraints());
            iConstraint++;
          }

        std::vector<dealii::Quadrature<dim>> dealiiQuadratureTypeVec(
          1, dealii::QGauss<dim>(1));
        dealiiMatrixFree.clear();
        dealii::MappingQ1<dim> mappingDealii;
        dealiiMatrixFree.reinit(mappingDealii, dofHandlerVec,
                                dealiiAffineConstraintsVec,
                                dealiiQuadratureTypeVec,
                                dealiiAdditionalData);
      }

      template <typename ValueTypeBasisCoeff, size_type dim>
      void
      getGhostIndices(
        const dealii::MatrixFree<dim, ValueTypeBasisCoeff> &dealiiMatrixFree,
        const size_type                                     constraintId,
        const EFEBasisManagerDealii<dim> *                  efeBMDealii,
        std::vector<global_size_type> &                     ghostIndices)
      {
        const dealii::Utilities::MPI::Partitioner &dealiiPartitioner =
          *(dealiiMatrixFree.get_vector_partitioner(constraintId));
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
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getLocallyOwnedCellLocalIndices(
        const EFEBasisManagerDealii<dim> *            efeBMDealii,
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      EFEBasisHandlerDealii(
        std::shared_ptr<const BasisManager> basisManager,
        std::map<
          std::string,
          std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>>
                                   constraintsMap,
        const utils::mpi::MPIComm &mpiComm)
    {
      reinit(basisManager, constraintsMap, mpiComm);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::reinit(
      std::shared_ptr<const BasisManager> basisManager,
      std::map<
        std::string,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>>
                                 constraintsMap,
      const utils::mpi::MPIComm &mpiComm)
    {
      // initialize the private data members.
      d_isDistributed = true;
      d_mpiComm       = mpiComm;
      d_efeBMDealii =
        std::dynamic_pointer_cast<const EFEBasisManagerDealii<dim>>(
          basisManager);
      utils::throwException(
        d_efeBMDealii != nullptr,
        "Error in casting the input basis manager in EFEBasisHandlerDealii to EFEBasisManagerDealii");
      const size_type numConstraints = constraintsMap.size();

      std::map<
        std::string,
        std::shared_ptr<
          const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>>
        efeConstraintsDealiiMap;

      // Create the efeConstraintsDealiiMap. It will map from constraintsName to
      // efeConstraintsDealii object.
      for (auto it = constraintsMap.begin(); it != constraintsMap.end(); ++it)
        {
          std::string constraintsName = it->first;
          std::shared_ptr<
            const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
            efeBasisConstraintsDealii = std::dynamic_pointer_cast<
              const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
              it->second);
          utils::throwException(
            efeBasisConstraintsDealii != nullptr,
            "Error in casting the input constraints to EFEConstraintsDealii in EFEBasisHandlerDealii");
          efeConstraintsDealiiMap[constraintsName] = efeBasisConstraintsDealii;
        }

      d_locallyOwnedRanges = d_efeBMDealii->getLocallyOwnedRanges();

      size_type numLocallyOwnedCells = d_efeBMDealii->nLocallyOwnedCells();
      EFEBasisHandlerDealiiInternal::getNumLocallyOwnedCellDofs<dim>(
        d_efeBMDealii.get(), d_numLocallyOwnedCellDofs);
      const size_type cumulativeCellDofs =
        EFEBasisHandlerDealiiInternal::getLocallyOwnedCellsCumulativeDofs<dim>(
          d_efeBMDealii.get());

      //
      // populate d_locallyOwnedCellStartIds
      //
      d_locallyOwnedCellStartIds.resize(numLocallyOwnedCells, 0);
      EFEBasisHandlerDealiiInternal::getLocallyOwnedCellStartIds<dim>(
        d_efeBMDealii.get(), d_locallyOwnedCellStartIds);

      //
      // populate d_locallyOwnedCellGlobalIndices
      //
      std::vector<global_size_type> locallyOwnedCellGlobalIndicesTmp(
        cumulativeCellDofs, 0);
      d_locallyOwnedCellGlobalIndices.resize(cumulativeCellDofs);
      EFEBasisHandlerDealiiInternal::getLocallyOwnedCellGlobalIndices<dim>(
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
          d_efeBMDealii.get(), efeConstraintsDealiiMap, dealiiMatrixFree);

      size_type iConstraint = 0;
      size_type classicalAttributeId =
        d_efeBMDealii
          ->getBasisAttributeToRangeIdMap()[BasisIdAttribute::CLASSICAL];
      for (auto it = efeConstraintsDealiiMap.begin();
           it != efeConstraintsDealiiMap.end();
           ++it)
        {
          const std::string constraintName = it->first;

          //
          // populate d_ghostIndicesMap
          //
          std::vector<global_size_type> ghostIndicesTmp(0);
          EFEBasisHandlerDealiiInternal::getGhostIndices<ValueTypeBasisCoeff,
                                                         dim>(
            dealiiMatrixFree,
            iConstraint,
            d_efeBMDealii.get(),
            ghostIndicesTmp);
          const size_type numGhostIndices          = ghostIndicesTmp.size();
          auto            globalSizeVectorGhostMap = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::GlobalSizeTypeVector>(
            numGhostIndices);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numGhostIndices,
            globalSizeVectorGhostMap->data(),
            ghostIndicesTmp.data());
          d_ghostIndicesMap[constraintName] = globalSizeVectorGhostMap;

          //
          // populate d_mpiPatternP2PMap
          //
          auto mpiPatternP2P =
            std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
              d_locallyOwnedRanges, ghostIndicesTmp, d_mpiComm);

            int rank;
  dftefe::utils::mpi::MPICommRank(d_mpiComm, &rank);

      // Get the number of processes
    int numProcs;
    dftefe::utils::mpi::MPICommSize(d_mpiComm, &numProcs);

// for(unsigned int iProc = 0 ; iProc < numProcs; iProc++)
// {
//   if(iProc == rank)
//   {
//     std::cout<<" Proc id = "<<rank<<"\n";
//     std::cout<<" printing local range \n";
//     for(unsigned int iRange = 0; iRange <  d_locallyOwnedRanges.size();iRange++)
//     {
//       std::cout<<"iRange = "<<iRange<<" start = "<<d_locallyOwnedRanges[iRange].first<<" end = "<<d_locallyOwnedRanges[iRange].second<< " proc local id start = " << mpiPatternP2P->globalToLocal(d_locallyOwnedRanges[iRange].first) << "\n"; //<< " proc local id end = " << mpiPatternP2P->globalToLocal(d_locallyOwnedRanges[iRange].second - 1)  << "\n";
//     }
//     std::cout<<" printing ghost\n";
//     for(unsigned int iRange = 0; iRange <  ghostIndicesTmp.size();iRange++)
//     {
//        std::cout<<"iRange = "<<iRange<<" global id = "<<ghostIndicesTmp[iRange]<< " proc local id = " << mpiPatternP2P->globalToLocal(ghostIndicesTmp[iRange]) << "\n";
//     }
//   }
//   std::cout << std::flush ;
//   dftefe::utils::mpi::MPIBarrier(d_mpiComm);
// }
          d_mpiPatternP2PMap[constraintName] = mpiPatternP2P;

          // Creation of optimized constraint matrix having only
          // dealiimatrixfree trimmed constraint ids.
          std::shared_ptr<
            EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
            efeBasisConstraintsDealiiOpt = std::make_shared<
              EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>();
          efeBasisConstraintsDealiiOpt->copyConstraintsData(
            *(it->second), *mpiPatternP2P, classicalAttributeId);
          efeBasisConstraintsDealiiOpt->populateConstraintsData(
            *mpiPatternP2P, classicalAttributeId);
          d_efeConstraintsDealiiOptMap[constraintName] =
            efeBasisConstraintsDealiiOpt;

          //
          // populate d_locallyOwnedCellLocalIndices
          //
          std::vector<size_type> locallyOwnedCellLocalIndicesTmp(
            cumulativeCellDofs, 0);
          EFEBasisHandlerDealiiInternal::getLocallyOwnedCellLocalIndices<
            ValueTypeBasisCoeff,
            memorySpace,
            dim>(d_efeBMDealii.get(),
                 mpiPatternP2P.get(),
                 locallyOwnedCellGlobalIndicesTmp,
                 locallyOwnedCellLocalIndicesTmp);
          auto sizeVectorLocalIndicies = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::SizeTypeVector>(
            cumulativeCellDofs, 0);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            cumulativeCellDofs,
            sizeVectorLocalIndicies->data(),
            locallyOwnedCellLocalIndicesTmp.data());

          d_locallyOwnedCellLocalIndicesMap[constraintName] =
            sizeVectorLocalIndicies;

          iConstraint++;
        }

      d_efeBMDealii->getBasisCenters(d_supportPoints);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      EFEBasisHandlerDealii(
        std::shared_ptr<const BasisManager> basisManager,
        std::map<
          std::string,
          std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>>
          constraintsMap)
    {
      reinit(basisManager, constraintsMap);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::reinit(
      std::shared_ptr<const BasisManager> basisManager,
      std::map<
        std::string,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>>
        constraintsMap)
    {
      d_isDistributed = false;
      d_efeBMDealii =
        std::dynamic_pointer_cast<const EFEBasisManagerDealii<dim>>(
          basisManager);
      utils::throwException(
        d_efeBMDealii != nullptr,
        "Error in casting the input basis manager in EFEBasisHandlerDealii to EFEBasisParitionerDealii");
      const size_type numConstraints = constraintsMap.size();
      std::map<
        std::string,
        std::shared_ptr<
          const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>>
        efeConstraintsDealiiMap;
      for (auto it = constraintsMap.begin(); it != constraintsMap.end(); ++it)
        {
          std::string constraintsName = it->first;
          std::shared_ptr<
            const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
            efeBasisConstraintsDealii = std::dynamic_pointer_cast<
              const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
              it->second);
          utils::throwException(
            efeBasisConstraintsDealii != nullptr,
            "Error in casting the input constraints to EFEConstraintsDealii in EFEBasisHandlerDealii");
          efeConstraintsDealiiMap[constraintsName] = efeBasisConstraintsDealii;
        }

      d_locallyOwnedRanges = d_efeBMDealii->getLocallyOwnedRanges();

      size_type numLocallyOwnedCells = d_efeBMDealii->nLocallyOwnedCells();
      EFEBasisHandlerDealiiInternal::getNumLocallyOwnedCellDofs<dim>(
        d_efeBMDealii.get(), d_numLocallyOwnedCellDofs);
      const size_type cumulativeCellDofs =
        EFEBasisHandlerDealiiInternal::getLocallyOwnedCellsCumulativeDofs<dim>(
          d_efeBMDealii.get());

      //
      // populate d_locallyOwnedCellStartIds
      //
      d_locallyOwnedCellStartIds.resize(numLocallyOwnedCells, 0);
      EFEBasisHandlerDealiiInternal::getLocallyOwnedCellStartIds<dim>(
        d_efeBMDealii.get(), d_locallyOwnedCellStartIds);

      //
      // populate d_locallyOwnedCellGlobalIndices
      //
      std::vector<global_size_type> locallyOwnedCellGlobalIndicesTmp(
        cumulativeCellDofs, 0);
      d_locallyOwnedCellGlobalIndices.resize(cumulativeCellDofs);
      EFEBasisHandlerDealiiInternal::getLocallyOwnedCellGlobalIndices<dim>(
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
          d_efeBMDealii.get(), efeConstraintsDealiiMap, dealiiMatrixFree);

      size_type classicalAttributeId =
        d_efeBMDealii
          ->getBasisAttributeToRangeIdMap()[BasisIdAttribute::CLASSICAL];
      size_type iConstraint = 0;
      for (auto it = efeConstraintsDealiiMap.begin();
           it != efeConstraintsDealiiMap.end();
           ++it)
        {
          const std::string constraintName = it->first;

          //
          // populate d_ghostIndicesMap
          //
          std::vector<global_size_type> ghostIndicesTmp(0);
          EFEBasisHandlerDealiiInternal::getGhostIndices<ValueTypeBasisCoeff,
                                                         dim>(
            dealiiMatrixFree,
            iConstraint,
            d_efeBMDealii.get(),
            ghostIndicesTmp);
          const size_type numGhostIndices  = ghostIndicesTmp.size();
          auto            globalSizeVector = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::GlobalSizeTypeVector>(
            numGhostIndices);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numGhostIndices, globalSizeVector->data(), ghostIndicesTmp.data());
          d_ghostIndicesMap[constraintName] = globalSizeVector;

          //
          // populate d_mpiPatternP2PMap
          //
          //
          // create std vector of sizes
          std::vector<size_type> locallyOwnedRangesSizeVec(0);
          for (auto i : d_locallyOwnedRanges)
            {
              locallyOwnedRangesSizeVec.push_back(i.second - i.first);
            }
          auto mpiPatternP2P =
            std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
              locallyOwnedRangesSizeVec);
          d_mpiPatternP2PMap[constraintName] = mpiPatternP2P;

          //
          // populate d_locallyOwnedCellLocalIndices
          //
          std::vector<size_type> locallyOwnedCellLocalIndicesTmp(
            cumulativeCellDofs, 0);
          EFEBasisHandlerDealiiInternal::getLocallyOwnedCellLocalIndices<
            ValueTypeBasisCoeff,
            memorySpace,
            dim>(d_efeBMDealii.get(),
                 mpiPatternP2P.get(),
                 locallyOwnedCellGlobalIndicesTmp,
                 locallyOwnedCellLocalIndicesTmp);

          size_type cumulativeDofs    = nCumulativeLocallyOwnedCellDofs();
          auto      sizeTypeVectorPtr = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::SizeTypeVector>(cumulativeDofs,
                                                                0);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            cumulativeDofs,
            sizeTypeVectorPtr->data(),
            locallyOwnedCellLocalIndicesTmp.data());

          d_locallyOwnedCellLocalIndicesMap[constraintName] = sizeTypeVectorPtr;

          std::shared_ptr<
            EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
            efeBasisConstraintsDealiiOpt = std::make_shared<
              EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>();
          efeBasisConstraintsDealiiOpt->copyConstraintsData(
            *(it->second), *mpiPatternP2P, classicalAttributeId);
          efeBasisConstraintsDealiiOpt->populateConstraintsData(
            *mpiPatternP2P, classicalAttributeId);
          d_efeConstraintsDealiiOptMap[constraintName] =
            efeBasisConstraintsDealiiOpt;

          iConstraint++;
        }

      //
      // FIXME: Assumes linear mapping from reference cell to real cell
      //
      d_efeBMDealii->getBasisCenters(d_supportPoints);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::setConstraints(
      std::map<
        std::string,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>>
        constraintsMap)
    {
      const size_type numConstraints = constraintsMap.size();
      std::map<
        std::string,
        std::shared_ptr<
          const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>>
        efeConstraintsDealiiMap;
      for (auto it = constraintsMap.begin(); it != constraintsMap.end(); ++it)
        {
          std::string constraintsName = it->first;
          std::shared_ptr<
            const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
            efeBasisConstraintsDealii = std::dynamic_pointer_cast<
              const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>(
              it->second);
          utils::throwException(
            efeBasisConstraintsDealii != nullptr,
            "Error in casting the input constraints to EFEConstraintsDealii in EFEBasisHandlerDealii");
          efeConstraintsDealiiMap[constraintsName] = efeBasisConstraintsDealii;
        }

      //
      // get cumulativeCellDofs
      //
      const size_type cumulativeCellDofs =
        EFEBasisHandlerDealiiInternal::getLocallyOwnedCellsCumulativeDofs<dim>(
          d_efeBMDealii.get());

      //
      // get locallyOwnedCellGlobalIndicesTmp
      //
      std::vector<global_size_type> locallyOwnedCellGlobalIndicesTmp(
        cumulativeCellDofs, 0);
      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        cumulativeCellDofs,
        locallyOwnedCellGlobalIndicesTmp.data(),
        d_locallyOwnedCellGlobalIndices.data());


      dealii::MatrixFree<dim, ValueTypeBasisCoeff> dealiiMatrixFree;
      EFEBasisHandlerDealiiInternal::
        setDealiiMatrixFreeLight<ValueTypeBasisCoeff, memorySpace, dim>(
          d_efeBMDealii.get(), efeConstraintsDealiiMap, dealiiMatrixFree);

      size_type classicalAttributeId =
        d_efeBMDealii
          ->getBasisAttributeToRangeIdMap()[BasisIdAttribute::CLASSICAL];

      size_type iConstraint = 0;
      for (auto it = efeConstraintsDealiiMap.begin();
           it != efeConstraintsDealiiMap.end();
           ++it)
        {
          const std::string constraintName = it->first;

          //
          // push into d_ghostIndicesMap
          //
          std::vector<global_size_type> ghostIndicesTmp(0);
          EFEBasisHandlerDealiiInternal::getGhostIndices<ValueTypeBasisCoeff,
                                                        dim>(dealiiMatrixFree,
                                                             iConstraint,
                                                             d_efeBMDealii.get(),
                                                             ghostIndicesTmp);
          const size_type numGhostIndices  = ghostIndicesTmp.size();
          auto            globalSizeVector = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::GlobalSizeTypeVector>(
            numGhostIndices);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            numGhostIndices, globalSizeVector->data(), ghostIndicesTmp.data());

          this->d_ghostIndicesMap.insert({ constraintName, globalSizeVector });

          //
          // push into d_mpiPatternP2PMap
          //
          if( d_isDistributed == false)
          {
            std::vector<size_type> locallyOwnedRangesSizeVec(0);
            for (auto i : d_locallyOwnedRanges)
              {
                locallyOwnedRangesSizeVec.push_back(i.second - i.first);
              }
            auto mpiPatternP2P =
              std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
                locallyOwnedRangesSizeVec);

            this->d_mpiPatternP2PMap.insert({ constraintName, mpiPatternP2P });
          }
          else
          {
            auto mpiPatternP2P =
              std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
                d_locallyOwnedRanges, ghostIndicesTmp, d_mpiComm);

            this->d_mpiPatternP2PMap.insert({ constraintName, mpiPatternP2P });
          }

          auto mpiPatternP2P = d_mpiPatternP2PMap[constraintName];

          //
          // push into d_locallyOwnedCellLocalIndices
          //
          std::vector<size_type> locallyOwnedCellLocalIndicesTmp(
            cumulativeCellDofs, 0);
          EFEBasisHandlerDealiiInternal::getLocallyOwnedCellLocalIndices<
            ValueTypeBasisCoeff,
            memorySpace,
            dim>(d_efeBMDealii.get(),
                 mpiPatternP2P.get(),
                 locallyOwnedCellGlobalIndicesTmp,
                 locallyOwnedCellLocalIndicesTmp);

          size_type cumulativeDofs    = nCumulativeLocallyOwnedCellDofs();
          auto      sizeTypeVectorPtr = std::make_shared<
            typename BasisHandler<ValueTypeBasisCoeff,
                                  memorySpace>::SizeTypeVector>(cumulativeDofs,
                                                                0);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            cumulativeDofs,
            sizeTypeVectorPtr->data(),
            locallyOwnedCellLocalIndicesTmp.data());

          this->d_locallyOwnedCellLocalIndicesMap.insert({ constraintName, sizeTypeVectorPtr });

          std::shared_ptr<
            EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
            efeBasisConstraintsDealiiOpt = std::make_shared<
              EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>();
          efeBasisConstraintsDealiiOpt->copyConstraintsData(
            *(it->second), *mpiPatternP2P, classicalAttributeId);
          efeBasisConstraintsDealiiOpt->populateConstraintsData(
            *mpiPatternP2P, classicalAttributeId);

          this->d_efeConstraintsDealiiOptMap.insert({ constraintName, efeBasisConstraintsDealiiOpt });

          iConstraint++;
        }
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isDistributed() const
    {
      return d_isDistributed;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const Constraints<ValueTypeBasisCoeff, memorySpace> &
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getConstraints(const std::string constraintsName) const
    {
      auto it = d_efeConstraintsDealiiOptMap.find(constraintsName);
      if (it == d_efeConstraintsDealiiOptMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "FEBasisHandlerDealii does not contain the constraints "
            "corresponding to " +
              constraintsName);
        }

      return *(it->second);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getMPIPatternP2P(const std::string constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in EFEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return it->second;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getLocallyOwnedRanges(const std::string constraintsName) const
    {
      return d_locallyOwnedRanges;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::nLocal(
      const std::string constraintsName) const
    {
      auto it = d_ghostIndicesMap.find(constraintsName);
      if (it == d_ghostIndicesMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The ghost indices in EFEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      // add locallyownedranges here
      size_type numLocallyOwned = 0;
      for (auto i : d_locallyOwnedRanges)
        {
          numLocallyOwned = numLocallyOwned + i.second - i.first;
        }
      const size_type numGhost = (*(it->second)).size();
      return (numLocallyOwned + numGhost);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::nLocallyOwned(
      const std::string constraintsName) const
    {
      size_type numLocallyOwned = 0;
      for (auto i : d_locallyOwnedRanges)
        {
          numLocallyOwned = numLocallyOwned + i.second - i.first;
        }
      return numLocallyOwned;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::nGhost(
      const std::string constraintsName) const
    {
      auto it = d_ghostIndicesMap.find(constraintsName);
      if (it == d_ghostIndicesMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The ghost indices in EFEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      const size_type numGhost = (*(it->second)).size();
      return numGhost;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getBasisCenters(const size_type       localId,
                      const std::string     constraintsName,
                      dftefe::utils::Point &basisCenter) const
    {
      global_size_type globalId = localToGlobalIndex(localId, constraintsName);

      if (d_supportPoints.find(globalId) != d_supportPoints.end())
        basisCenter = d_supportPoints.find(globalId)->second;
      else
      {
        std::string msg = "The localId does not have any point in the EFE mesh for id no ";
        msg = msg + std::to_string(globalId);
        utils::throwException(false, msg);
      }
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      nLocallyOwnedCells() const
    {
      return d_efeBMDealii->nLocallyOwnedCells();
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      nCumulativeLocallyOwnedCellDofs() const
    {
      return d_efeBMDealii->nCumulativeLocallyOwnedCellDofs();
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      nLocallyOwnedCellDofs(const size_type cellId) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to nLocallyOwnedCellDofs is greater than or "
        " equal to the number of locally owned cells.");
      return d_numLocallyOwnedCellDofs[cellId];
    }


    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::GlobalSizeTypeVector &
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getGhostIndices(const std::string constraintsName) const
    {
      auto it = d_ghostIndicesMap.find(constraintsName);
      if (it == d_ghostIndicesMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The ghost indices in EFEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return *(it->second);
    }


    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::pair<bool, size_type>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      inLocallyOwnedRanges(const global_size_type globalId,
                           const std::string      constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in EFEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return (it->second)->inLocallyOwnedRanges(globalId);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::pair<bool, size_type>
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::isGhostEntry(
      const global_size_type ghostId,
      const std::string      constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in EFEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return (it->second)->isGhostEntry(ghostId);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      globalToLocalIndex(const global_size_type globalId,
                         const std::string      constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in EFEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return (it->second)->globalToLocal(globalId);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    global_size_type
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      localToGlobalIndex(const size_type   localId,
                         const std::string constraintsName) const
    {
      auto it = d_mpiPatternP2PMap.find(constraintsName);
      if (it == d_mpiPatternP2PMap.end())
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "The MPIPatternP2P in EFEBasisHandlerDealii is not created for "
            "the constraint " +
              constraintsName);
        }
      return (it->second)->localToGlobal(localId);
    }

    //
    // FE specific functions
    //
    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                   memorySpace,
                                   dim>::const_GlobalIndexIter
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsBegin(const std::string constraintsName) const
    {
      return d_locallyOwnedCellGlobalIndices.begin();
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                   memorySpace,
                                   dim>::const_GlobalIndexIter
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      const_GlobalIndexIter
      EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                   memorySpace,
                                   dim>::const_LocalIndexIter
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsBegin(const std::string constraintsName) const
    {
      auto it = d_locallyOwnedCellLocalIndicesMap.find(constraintsName);
      utils::throwException<utils::InvalidArgument>(
        it != d_locallyOwnedCellLocalIndicesMap.end(),
        "The cell local indices is not created for the constraints " +
          constraintsName);
      return (*(it->second)).begin();
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff,
                                   memorySpace,
                                   dim>::const_LocalIndexIter
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      const_LocalIndexIter
      EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const BasisManager &
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getBasisManager() const
    {
      return *d_efeBMDealii;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    EFEBasisHandlerDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getCellDofsLocalIds(const size_type         cellId,
                          const std::string       constraintsName,
                          std::vector<size_type> &vecLocalNodeId) const
    {
      std::vector<global_size_type> vecGlobalNodeId(0);
      d_efeBMDealii->getCellDofsGlobalIds(cellId, vecGlobalNodeId);
      for (auto i : vecGlobalNodeId)
        {
          vecLocalNodeId.push_back(globalToLocalIndex(i, constraintsName));
        }
    }

  } // end of namespace basis
} // end of namespace dftefe
