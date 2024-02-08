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
#include <string>
#include <utils/Exceptions.h>
namespace dftefe
{
  namespace basis
  {
    namespace FEBasisManagerInternal
    {
      template <typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      size_type
      getLocallyOwnedCellsCumulativeDofs(
        const basis::FEBasisDofHandler<ValueTypeBasisData, memorySpace, dim>
          *feBDH)
      {
        size_type returnValue          = 0;
        size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          returnValue += feBDH->nCellDofs(iCell);

        return returnValue;
      }

      template <typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getNumLocallyOwnedCellDofs(
        const basis::FEBasisDofHandler<ValueTypeBasisData, memorySpace, dim>
          *                     feBDH,
        std::vector<size_type> &locallyOwnedCellDofs)
      {
        size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
        locallyOwnedCellDofs.resize(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          locallyOwnedCellDofs[iCell] = feBDH->nCellDofs(iCell);
      }

      template <typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getLocallyOwnedCellStartIds(
        const basis::FEBasisDofHandler<ValueTypeBasisData, memorySpace, dim>
          *                     feBDH,
        std::vector<size_type> &locallyOwnedCellStartIds)
      {
        size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
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
                  feBDH->nCellDofs(iCell - 1);
              }
          }
      }

      template <typename ValueTypeBasisData,
                dftefe::utils::MemorySpace memorySpace,
                size_type                  dim>
      void
      getLocallyOwnedCellGlobalIndices(
        const basis::FEBasisDofHandler<ValueTypeBasisData, memorySpace, dim>
          *                            feBDH,
        std::vector<global_size_type> &locallyOwnedCellGlobalIndices)
      {
        size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
        size_type cumulativeCellDofs   = 0;
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            const size_type               numCellDofs = feBDH->nCellDofs(iCell);
            std::vector<global_size_type> cellGlobalIds(numCellDofs);
            feBDH->getCellDofsGlobalIds(iCell, cellGlobalIds);
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
      getLocallyOwnedCellLocalIndices(
        const basis::FEBasisDofHandler<ValueTypeBasisData, memorySpace, dim>
          *                                           feBDH,
        const utils::mpi::MPIPatternP2P<memorySpace> *mpiPatternP2P,
        const std::vector<global_size_type> &locallyOwnedCellGlobalIndices,
        std::vector<size_type> &             locallyOwnedCellLocalIndices)
      {
        size_type numLocallyOwnedCells = feBDH->nLocallyOwnedCells();
        size_type cumulativeDofs       = 0;
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          {
            const size_type numCellDofs = feBDH->nCellDofs(iCell);
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
    } // end of namespace FEBasisManagerInternal


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      FEBasisManager(std::shared_ptr<const BasisDofHandler> basisDofHandler,
                     std::shared_ptr<const utils::ScalarSpatialFunctionReal>
                       dirichletBoundaryCondition)
    {
      reinit(basisDofHandler, dirichletBoundaryCondition);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      reinit(std::shared_ptr<const BasisDofHandler> basisDofHandler,
             std::shared_ptr<const utils::ScalarSpatialFunctionReal>
               dirichletBoundaryCondition)
    {
      // initialize the private data members.
      d_feBDH = std::dynamic_pointer_cast<
        const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
        basisDofHandler);
      utils::throwException(
        d_feBDH != nullptr,
        "Error in casting the input basisDofHandler in FEBasisManager to FEBasisDofHandler");

      if (dirichletBoundaryCondition != nullptr)
        {
          std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
            constraintsLocalIntrinsic = d_feBDH->createConstraintsStart();

          std::vector<global_size_type> boundaryIds = d_feBDH->getBoundaryIds();

          std::map<global_size_type, utils::Point> boundaryCoord;
          boundaryCoord.clear();
          d_feBDH->getBasisCenters(boundaryCoord);
          for (auto nodeId : boundaryIds)
            {
              if (boundaryCoord.find(nodeId) != boundaryCoord.end())
                {
                  auto inhomoValue = (*dirichletBoundaryCondition)(
                    boundaryCoord.find(nodeId)->second);
                  constraintsLocalIntrinsic->setInhomogeneity(nodeId,
                                                              inhomoValue);
                }
              else
                utils::throwException(
                  false, "Could not find boundary nodeId in FEBasisManager.");
            }
          d_feBDH->createConstraintsEnd(constraintsLocalIntrinsic);

          d_constraintsLocal = constraintsLocalIntrinsic;
        }
      else
        {
          d_constraintsLocal = d_feBDH->getIntrinsicConstraints();
        }

      d_locallyOwnedRanges = d_feBDH->getLocallyOwnedRanges();

      size_type numLocallyOwnedCells = d_feBDH->nLocallyOwnedCells();
      FEBasisManagerInternal::
        getNumLocallyOwnedCellDofs<ValueTypeBasisData, memorySpace, dim>(
          d_feBDH.get(), d_numLocallyOwnedCellDofs);
      const size_type cumulativeCellDofs =
        FEBasisManagerInternal::getLocallyOwnedCellsCumulativeDofs<
          ValueTypeBasisData,
          memorySpace,
          dim>(d_feBDH.get());

      //
      // populate d_locallyOwnedCellStartIds
      //
      d_locallyOwnedCellStartIds.resize(numLocallyOwnedCells, 0);
      FEBasisManagerInternal::
        getLocallyOwnedCellStartIds<ValueTypeBasisData, memorySpace, dim>(
          d_feBDH.get(), d_locallyOwnedCellStartIds);

      //
      // populate d_locallyOwnedCellGlobalIndices
      //
      std::vector<global_size_type> locallyOwnedCellGlobalIndicesTmp(
        cumulativeCellDofs, 0);
      d_locallyOwnedCellGlobalIndices.resize(cumulativeCellDofs);
      FEBasisManagerInternal::
        getLocallyOwnedCellGlobalIndices<ValueTypeBasisData, memorySpace, dim>(
          d_feBDH.get(), locallyOwnedCellGlobalIndicesTmp);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        cumulativeCellDofs,
        d_locallyOwnedCellGlobalIndices.data(),
        locallyOwnedCellGlobalIndicesTmp.data());

      //
      // populate d_mpiPatternP2P
      //
      d_mpiPatternP2P = d_feBDH->getMPIPatternP2P();

      //
      // populate d_ghostIndices
      //
      std::vector<global_size_type> ghostIndicesTmp(0);

      ghostIndicesTmp = d_mpiPatternP2P->getGhostIndices();

      const size_type numGhostIndices = ghostIndicesTmp.size();
      d_ghostIndices                  = std::make_shared<
        typename BasisManager<ValueTypeBasisCoeff,
                              memorySpace>::GlobalSizeTypeVector>(
        numGhostIndices);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        numGhostIndices, d_ghostIndices->data(), ghostIndicesTmp.data());

      //
      // populate d_locallyOwnedCellLocalIndices
      //
      std::vector<size_type> locallyOwnedCellLocalIndicesTmp(cumulativeCellDofs,
                                                             0);
      FEBasisManagerInternal::getLocallyOwnedCellLocalIndices<
        ValueTypeBasisCoeff,
        ValueTypeBasisData,
        memorySpace,
        dim>(d_feBDH.get(),
             d_mpiPatternP2P.get(),
             locallyOwnedCellGlobalIndicesTmp,
             locallyOwnedCellLocalIndicesTmp);
      d_locallyOwnedCellLocalIndices =
        std::make_shared<typename BasisManager<ValueTypeBasisCoeff,
                                               memorySpace>::SizeTypeVector>(
          cumulativeCellDofs, 0);
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        cumulativeCellDofs,
        d_locallyOwnedCellLocalIndices->data(),
        locallyOwnedCellLocalIndicesTmp.data());

      d_feBDH->getBasisCenters(d_supportPoints);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace> &
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      getConstraints() const
    {
      return *(d_constraintsLocal);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      getMPIPatternP2P() const
    {
      return d_mpiPatternP2P;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      getLocallyOwnedRanges() const
    {
      return d_locallyOwnedRanges;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      nLocal() const
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
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      nLocallyOwned() const
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
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      nGhost() const
    {
      const size_type numGhost = (*(d_ghostIndices)).size();
      return numGhost;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
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
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      nLocallyOwnedCells() const
    {
      return d_feBDH->nLocallyOwnedCells();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      nCumulativeLocallyOwnedCellDofs() const
    {
      return d_feBDH->nCumulativeLocallyOwnedCellDofs();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      nLocallyOwnedCellDofs(const size_type cellId) const
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
    const typename FEBasisManager<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  memorySpace,
                                  dim>::GlobalSizeTypeVector &
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      getGhostIndices() const
    {
      return *(d_ghostIndices);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::pair<bool, size_type>
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      inLocallyOwnedRanges(const global_size_type globalId) const
    {
      return (d_mpiPatternP2P)->inLocallyOwnedRanges(globalId);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::pair<bool, size_type>
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      isGhostEntry(const global_size_type ghostId) const
    {
      return (d_mpiPatternP2P)->isGhostEntry(ghostId);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      globalToLocalIndex(const global_size_type globalId) const
    {
      return (d_mpiPatternP2P)->globalToLocal(globalId);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    global_size_type
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      localToGlobalIndex(const size_type localId) const
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
    typename FEBasisManager<ValueTypeBasisCoeff,
                            ValueTypeBasisData,
                            memorySpace,
                            dim>::const_GlobalIndexIter
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsBegin() const
    {
      return d_locallyOwnedCellGlobalIndices.begin();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisManager<ValueTypeBasisCoeff,
                            ValueTypeBasisData,
                            memorySpace,
                            dim>::const_GlobalIndexIter
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsBegin(const size_type cellId) const
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
    typename FEBasisManager<ValueTypeBasisCoeff,
                            ValueTypeBasisData,
                            memorySpace,
                            dim>::const_GlobalIndexIter
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      locallyOwnedCellGlobalDofIdsEnd(const size_type cellId) const
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
    typename FEBasisManager<ValueTypeBasisCoeff,
                            ValueTypeBasisData,
                            memorySpace,
                            dim>::const_LocalIndexIter
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsBegin() const
    {
      return (*(d_locallyOwnedCellLocalIndices)).begin();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisManager<ValueTypeBasisCoeff,
                            ValueTypeBasisData,
                            memorySpace,
                            dim>::const_LocalIndexIter
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsBegin(const size_type cellId) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellLocalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");

      return (*(d_locallyOwnedCellLocalIndices)).begin() +
             d_locallyOwnedCellStartIds[cellId];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    typename FEBasisManager<ValueTypeBasisCoeff,
                            ValueTypeBasisData,
                            memorySpace,
                            dim>::const_LocalIndexIter
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      locallyOwnedCellLocalDofIdsEnd(const size_type cellId) const
    {
      DFTEFE_AssertWithMsg(
        cellId < d_numLocallyOwnedCellDofs.size(),
        "Cell Id provided to locallyOwnedCellLocalDofIdsBegin() must be "
        " smaller than the number of locally owned cells.");

      return (*(d_locallyOwnedCellLocalIndices)).begin() +
             d_locallyOwnedCellStartIds[cellId] +
             d_numLocallyOwnedCellDofs[cellId];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const BasisDofHandler &
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      getBasisDofHandler() const
    {
      return *d_feBDH;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>::
      getCellDofsLocalIds(const size_type         cellId,
                          std::vector<size_type> &vecLocalNodeId) const
    {
      std::vector<global_size_type> vecGlobalNodeId(0);
      d_feBDH->getCellDofsGlobalIds(cellId, vecGlobalNodeId);
      for (auto i : vecGlobalNodeId)
        {
          vecLocalNodeId.push_back(globalToLocalIndex(i));
        }
    }

  } // end of namespace basis
} // end of namespace dftefe
