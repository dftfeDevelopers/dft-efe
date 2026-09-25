
#include <deal.II/dofs/dof_tools.h>
#include <basis/FECellBase.h>
#include <memory>
#include <basis/ConstraintsInternal.h>
#include <utils/Defaults.h>
#include <utils/DataTypeOverloads.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/MemoryTransfer.h>
#include <utils/MPIWrapper.h>
#include <utils/MPITypes.h>
#include <set>
#include <vector>


namespace dftefe
{
  namespace basis
  {
    // default constructor
    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      EFEConstraintsLocalDealii(const dealii::IndexSet &locally_owned_dofs,
                                const dealii::IndexSet &locally_relevant_dofs)
      : d_isCleared(false)
      , d_isClosed(false)
      , d_isMeanValueConstraintActive(false)
      , d_meanValueConstraintNodeIdLocal(0)
      , d_meanValueConstraintProcId(0)
      , d_meanValueConstraintNodeIdGlobal(0)
      , d_meanValueMpiComm(utils::mpi::MPICommSelf)
      , d_meanValueMyRank(0)
    {
      d_locallyOwnedRanges.resize(0);
      d_ghostIndices.resize(0);
      d_ghostIndicesSet.clear();
      d_globalToLocalMap.clear();
      d_dealiiAffineConstraintMatrix.clear();
      d_dealiiAffineConstraintMatrix.reinit(locally_owned_dofs,
                                            locally_relevant_dofs);
    }

    // constructor taking the closed dealiiAffineConstraintMatrix and
    // partitioning information to pass to dftefe objects
    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      EFEConstraintsLocalDealii(
        dealii::AffineConstraints<RealTypeBasisCoeff>
          &dealiiAffineConstraintMatrix,
        std::vector<std::pair<global_size_type, global_size_type>>
          &                            locallyOwnedRanges,
        std::vector<global_size_type> &ghostIndices,
        std::unordered_map<global_size_type, size_type>
          &globalToLocalMapLocalDofs)
      : d_dealiiAffineConstraintMatrix(dealiiAffineConstraintMatrix)
      , d_locallyOwnedRanges(locallyOwnedRanges)
      , d_ghostIndices(ghostIndices)
      , d_ghostIndicesSet(ghostIndices.begin(), ghostIndices.end())
      , d_globalToLocalMap(globalToLocalMapLocalDofs)
      , d_isCleared(false)
      , d_isClosed(true)
      , d_isMeanValueConstraintActive(false)
      , d_meanValueConstraintNodeIdLocal(0)
      , d_meanValueConstraintProcId(0)
      , d_meanValueConstraintNodeIdGlobal(0)
      , d_meanValueMpiComm(utils::mpi::MPICommSelf)
      , d_meanValueMyRank(0)
    {
      copyConstraintsDataFromDealiiToDftefe();
    }

    //
    // Copy function - note one has to call close after calling copyFrom
    //
    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::copyFrom(
      const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>
        &constraintsLocalIn)
    {
      const EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>
        &EFEConstraintsLocalDealiiIn = dynamic_cast<
          const EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>
            &>(constraintsLocalIn);

      utils::throwException(
        &EFEConstraintsLocalDealiiIn != nullptr,
        " Could not typecast ConstraintsLocal to EFEConstraintsLocalDealii in EFEConstraintsLocalDealii.h");

      d_isClosed           = false;
      d_isCleared          = false;
      d_locallyOwnedRanges = EFEConstraintsLocalDealiiIn.d_locallyOwnedRanges;
      d_ghostIndices       = EFEConstraintsLocalDealiiIn.d_ghostIndices;
      d_ghostIndicesSet    = EFEConstraintsLocalDealiiIn.d_ghostIndicesSet;
      d_globalToLocalMap   = EFEConstraintsLocalDealiiIn.d_globalToLocalMap;
      copyConstraintsDataFromDealiiToDealii(EFEConstraintsLocalDealiiIn);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::clear()
    {
      d_dealiiAffineConstraintMatrix.clear();
      d_isCleared = true;
      d_isClosed  = false;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::close()
    {
      d_dealiiAffineConstraintMatrix.close();
      copyConstraintsDataFromDealiiToDftefe();
      d_isCleared = false;
      d_isClosed  = true;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      setInhomogeneity(global_size_type    basisId,
                       ValueTypeBasisCoeff constraintValue)
    {
      utils::throwException(
        !d_isClosed,
        " Clear the constraint matrix before setting inhomogeneities. Cannot setInhomogeneity after close().");

      // If condition is removed
      // add_line does not do anything if the basisId already exists.
      addLine(basisId);
      d_dealiiAffineConstraintMatrix.set_inhomogeneity(
        basisId, utils::realPart(constraintValue));
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::isClosed()
      const
    {
      return d_isClosed;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isConstrained(global_size_type basisId) const
    {
      return d_dealiiAffineConstraintMatrix.is_constrained(basisId);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const dealii::AffineConstraints<
      typename EFEConstraintsLocalDealii<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::RealTypeBasisCoeff> &
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getAffineConstraints() const
    {
      return d_dealiiAffineConstraintMatrix;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const std::vector<std::pair<
      global_size_type,
      typename EFEConstraintsLocalDealii<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::RealTypeBasisCoeff>> *
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getConstraintEntries(const global_size_type lineDof) const
    {
      return d_dealiiAffineConstraintMatrix.get_constraint_entries(lineDof);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isInhomogeneouslyConstrained(const global_size_type lineDof) const
    {
      return (
        d_dealiiAffineConstraintMatrix.is_inhomogeneously_constrained(lineDof));
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ValueTypeBasisCoeff
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getInhomogeneity(const global_size_type lineDof) const
    {
      return (d_dealiiAffineConstraintMatrix.get_inhomogeneity(lineDof));
    }


    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      copyConstraintsDataFromDealiiToDealii(
        const EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>
          &constraintsDataIn)
    {
      this->clear();
      std::vector<std::pair<global_size_type, global_size_type>>
        locallyOwnedRanges = d_locallyOwnedRanges;

      auto locallyOwnedRange = locallyOwnedRanges[0];

      bool printWarning = false;
      for (auto locallyOwnedId = locallyOwnedRange.first;
           locallyOwnedId < locallyOwnedRange.second;
           locallyOwnedId++)
        {
          if (constraintsDataIn.isConstrained(locallyOwnedId))
            {
              const global_size_type lineDof = locallyOwnedId;
              this->addLine(lineDof);
              if (constraintsDataIn.isInhomogeneouslyConstrained(lineDof))
                {
                  this->setInhomogeneity(
                    lineDof, constraintsDataIn.getInhomogeneity(lineDof));
                }
              const std::vector<
                std::pair<global_size_type, RealTypeBasisCoeff>> *rowData =
                constraintsDataIn.getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (size_type j = 0; j < rowData->size(); ++j)
                {
                  if (!(isGhostEntry((*rowData)[j].first) ||
                        inLocallyOwnedRanges((*rowData)[j].first)))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      printWarning                          = true;
                      break;
                    }
                }

              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;

              this->addEntries(lineDof, *rowData);
            }
        }

      auto ghostIndices = d_ghostIndices; // can be optimized .. checking
                                          // enriched ghosts also

      for (auto ghostIter = ghostIndices.begin();
           ghostIter != ghostIndices.end();
           ghostIter++)
        {
          if (constraintsDataIn.isConstrained(*ghostIter))
            {
              const global_size_type lineDof = *ghostIter;
              this->addLine(lineDof);
              if (constraintsDataIn.isInhomogeneouslyConstrained(lineDof))
                {
                  this->setInhomogeneity(
                    lineDof, constraintsDataIn.getInhomogeneity(lineDof));
                }
              const std::vector<
                std::pair<global_size_type, RealTypeBasisCoeff>> *rowData =
                constraintsDataIn.getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (size_type j = 0; j < rowData->size(); ++j)
                {
                  if (!(isGhostEntry((*rowData)[j].first) ||
                        inLocallyOwnedRanges((*rowData)[j].first)))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      printWarning                          = true;
                      break;
                    }
                }

              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;
              this->addEntries(lineDof, *rowData);
            }
        }

      if (printWarning)
        {
          std::cout
            << "DFT-EFE Warning : the ghost indices provided is not complete....Check if the ghost indices are sufficient\n";
        }
    }


    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      copyConstraintsDataFromDealiiToDftefe()
    {
      bool printWarning = false;

      std::vector<global_size_type> rowConstraintsIdsGlobalTmp(0);
      std::vector<size_type>        rowConstraintsIdsLocalTmp(0);
      std::vector<size_type>        columnConstraintsIdsLocalTmp(0);
      std::vector<size_type>        constraintRowSizesAccumulatedTmp(0);
      std::vector<global_size_type> columnConstraintsIdsGlobalTmp(0);

      std::vector<RealTypeBasisCoeff>              columnConstraintsValuesTmp(0);
      std::vector<RealTypeBasisCoeff> constraintsInhomogenitiesTmp(0);

      std::vector<size_type> rowConstraintsSizesTmp(0);

      std::vector<std::pair<global_size_type, global_size_type>>
        locallyOwnedRanges = d_locallyOwnedRanges;

      auto locallyOwnedRange = locallyOwnedRanges[0];

      size_type columnIdStart = 0;

      for (auto locallyOwnedId = locallyOwnedRange.first;
           locallyOwnedId < locallyOwnedRange.second;
           locallyOwnedId++)
        {
          if (this->isConstrained(locallyOwnedId))
            {
              const global_size_type lineDof = locallyOwnedId;
              const std::vector<
                std::pair<global_size_type, RealTypeBasisCoeff>> *rowData =
                this->getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (size_type j = 0; j < rowData->size(); ++j)
                {
                  if (!(isGhostEntry((*rowData)[j].first) ||
                        inLocallyOwnedRanges((*rowData)[j].first)))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      printWarning                          = true;
                      break;
                    }
                }



              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;

              rowConstraintsIdsLocalTmp.push_back(globalToLocal(lineDof));
              rowConstraintsIdsGlobalTmp.push_back(lineDof);
              constraintsInhomogenitiesTmp.push_back(
                utils::realPart(getInhomogeneity(lineDof)));
              rowConstraintsSizesTmp.push_back(rowData->size());
              for (size_type j = 0; j < rowData->size(); ++j)
                {
                  columnConstraintsIdsGlobalTmp.push_back((*rowData)[j].first);
                  columnConstraintsIdsLocalTmp.push_back(
                    globalToLocal((*rowData)[j].first));
                  RealTypeBasisCoeff realPart = utils::realPart((*rowData)[j].second);
                  columnConstraintsValuesTmp.push_back(realPart);
                }

              constraintRowSizesAccumulatedTmp.push_back(columnIdStart);
              columnIdStart += rowData->size();
            }
        }

      auto ghostIndices = d_ghostIndices;

      for (auto ghostIter = ghostIndices.begin();
           ghostIter != ghostIndices.end();
           ghostIter++)
        {
          if (this->isConstrained(*ghostIter))
            {
              const global_size_type lineDof = *ghostIter;

              const std::vector<
                std::pair<global_size_type, RealTypeBasisCoeff>> *rowData =
                this->getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (size_type j = 0; j < rowData->size(); ++j)
                {
                  if (!(isGhostEntry((*rowData)[j].first) ||
                        inLocallyOwnedRanges((*rowData)[j].first)))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      printWarning                          = true;
                      break;
                    }
                }

              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;

              rowConstraintsIdsLocalTmp.push_back(globalToLocal(lineDof));
              rowConstraintsIdsGlobalTmp.push_back(lineDof);
              constraintsInhomogenitiesTmp.push_back(
                utils::realPart(getInhomogeneity(lineDof)));
              rowConstraintsSizesTmp.push_back(rowData->size());
              for (size_type j = 0; j < rowData->size(); ++j)
                {
                  columnConstraintsIdsGlobalTmp.push_back((*rowData)[j].first);
                  columnConstraintsIdsLocalTmp.push_back(
                    globalToLocal((*rowData)[j].first));
                  RealTypeBasisCoeff realPart = utils::realPart((*rowData)[j].second);
                  columnConstraintsValuesTmp.push_back(realPart);
                }
              constraintRowSizesAccumulatedTmp.push_back(columnIdStart);
              columnIdStart += rowData->size();
            }
        }

      if (printWarning)
        {
          std::cout
            << "DFT-EFE Warning : the ghost indices provided is not complete....Check if the ghost indices\n";
        }


      d_rowConstraintsIdsGlobal.resize(rowConstraintsIdsGlobalTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        rowConstraintsIdsGlobalTmp.size(),
        d_rowConstraintsIdsGlobal.data(),
        rowConstraintsIdsGlobalTmp.data());

      d_rowConstraintsIdsLocal.resize(rowConstraintsIdsLocalTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        rowConstraintsIdsLocalTmp.size(),
        d_rowConstraintsIdsLocal.data(),
        rowConstraintsIdsLocalTmp.data());

      d_columnConstraintsIdsLocal.resize(columnConstraintsIdsLocalTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        columnConstraintsIdsLocalTmp.size(),
        d_columnConstraintsIdsLocal.data(),
        columnConstraintsIdsLocalTmp.data());

      d_columnConstraintsIdsGlobal.resize(columnConstraintsIdsGlobalTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        columnConstraintsIdsGlobalTmp.size(),
        d_columnConstraintsIdsGlobal.data(),
        columnConstraintsIdsGlobalTmp.data());

      d_columnConstraintsValues.resize(columnConstraintsValuesTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        columnConstraintsValuesTmp.size(),
        d_columnConstraintsValues.data(),
        columnConstraintsValuesTmp.data());

      d_constraintRowSizesAccumulated.resize(
        constraintRowSizesAccumulatedTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        constraintRowSizesAccumulatedTmp.size(),
        d_constraintRowSizesAccumulated.data(),
        constraintRowSizesAccumulatedTmp.data());


      d_constraintsInhomogenities.resize(constraintsInhomogenitiesTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        constraintsInhomogenitiesTmp.size(),
        d_constraintsInhomogenities.data(),
        constraintsInhomogenitiesTmp.data());

      d_rowConstraintsSizes.resize(rowConstraintsSizesTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        rowConstraintsSizesTmp.size(),
        d_rowConstraintsSizes.data(),
        rowConstraintsSizesTmp.data());
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      addEntries(
        const global_size_type constrainedDofIndex,
        const std::vector<std::pair<global_size_type, RealTypeBasisCoeff>>
          &colWeightPairs)
    {
      d_dealiiAffineConstraintMatrix.add_entries(constrainedDofIndex,
                                                 colWeightPairs);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::addLine(
      const global_size_type lineDof)
    {
      d_dealiiAffineConstraintMatrix.add_line(lineDof);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      distributeParentToChild(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const
    {
      ConstraintsInternal<ValueTypeBasisCoeff, memorySpace>::
        constraintsDistributeParentToChild(vectorData,
                                           blockSize,
                                           d_rowConstraintsIdsLocal,
                                           d_rowConstraintsSizes,
                                           d_columnConstraintsIdsLocal,
                                           d_constraintRowSizesAccumulated,
                                           d_columnConstraintsValues,
                                           d_constraintsInhomogenities,
                                           *vectorData.getLinAlgOpContext());

    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      applyMeanValueConstraintDistributeP2C(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const
    {
      // Set the pinned dof from its masters, vec[o] = dot(a, vec), which is
      // the slave-from-masters half of the mean value constraint. The dot runs
      // over locally owned entries only and is then summed across processors,
      // and only the processor owning o writes the result.
      // dftfe analog: meanValueConstraintDistribute
      // (poissonSolverProblem.cc:445-457)
      if (!d_isMeanValueConstraintActive)
        return;

      {
          std::vector<ValueTypeBasisCoeff> dotProd(blockSize);
          for (size_type iVec = 0; iVec < blockSize; ++iVec)
            dotProd[iVec] =
              linearAlgebra::blasLapack::dot<ValueTypeBasisCoeff,
                                             ValueTypeBasisCoeff,
                                             memorySpace>(
                d_meanValueConstraintVec.locallyOwnedSize(),
                d_meanValueConstraintVec.data(),
                1,
                vectorData.data() + iVec,
                blockSize,
                *vectorData.getLinAlgOpContext());

          utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            dotProd.data(),
            (int)blockSize,
            utils::mpi::Types<ValueTypeBasisCoeff>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_meanValueMpiComm);

          if (d_meanValueMyRank == (int)d_meanValueConstraintProcId)
            utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
              blockSize,
              vectorData.data() + d_meanValueConstraintNodeIdLocal * blockSize,
              dotProd.data());
          vectorData.updateGhostValues();
        }
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      applyMeanValueConstraintDistributeC2P(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const
    {
      // Transpose of the above: whatever has accumulated on the pinned dof is
      // redistributed onto its masters as vec += vec[o] * a, after which the
      // pinned entry is zeroed so it contributes nothing further. vec[o] is
      // broadcast first, since only the processor owning o holds it.
      // dftfe analog: meanValueConstraintDistributeSlaveToMaster
      // (poissonSolverProblem.cc:461-482), with the zeroing step folded in as
      // in the device variant at poissonSolverProblemDevice.cc:548
      if (!d_isMeanValueConstraintActive)
        return;

      {
          std::vector<ValueTypeBasisCoeff> valueAtNode(
            blockSize, utils::Types<ValueTypeBasisCoeff>::zero);
          if (d_meanValueMyRank == (int)d_meanValueConstraintProcId)
            utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
              blockSize,
              valueAtNode.data(),
              vectorData.data() +
                d_meanValueConstraintNodeIdLocal * blockSize);

          utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
            valueAtNode.data(),
            (int)blockSize,
            utils::mpi::Types<ValueTypeBasisCoeff>::getMPIDatatype(),
            (int)d_meanValueConstraintProcId,
            d_meanValueMpiComm);

          for (size_type iVec = 0; iVec < blockSize; ++iVec)
            linearAlgebra::blasLapack::axpby<ValueTypeBasisCoeff,
                                             ValueTypeBasisCoeff,
                                             memorySpace>(
              d_meanValueConstraintVec.locallyOwnedSize(),
              valueAtNode[iVec],
              d_meanValueConstraintVec.data(),
              (ValueTypeBasisCoeff)1.0,
              vectorData.data() + iVec,
              vectorData.data() + iVec,
              *vectorData.getLinAlgOpContext());

          if (d_meanValueMyRank == (int)d_meanValueConstraintProcId)
            {
              std::vector<ValueTypeBasisCoeff> zeros(
                blockSize, utils::Types<ValueTypeBasisCoeff>::zero);
              utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::
                copy(blockSize,
                     vectorData.data() +
                       d_meanValueConstraintNodeIdLocal * blockSize,
                     zeros.data());
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      setMeanValueConstraint(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                        basisIntegrals,
        const utils::mpi::MPIComm &mpiComm)
    {
      d_meanValueMpiComm = mpiComm;
      int nProcs         = 1;
      utils::mpi::MPICommRank(mpiComm, &d_meanValueMyRank);
      utils::mpi::MPICommSize(mpiComm, &nProcs);

      d_meanValueConstraintVec = basisIntegrals;
      const size_type localSize = d_meanValueConstraintVec.localSize();

      // Fold the ghost-side contributions into the locally owned entries and
      // apply the dealii constraints already held here (hanging + periodic +
      // Dirichlet). The mean-value branch inside distributeChildToParent is
      // still dead at this point, since d_isMeanValueConstraintActive is only
      // set at the end of this function.
      this->distributeChildToParent(d_meanValueConstraintVec, 1);
      d_meanValueConstraintVec.accumulateAddLocallyOwned();
      d_meanValueConstraintVec.updateGhostValues();

      std::vector<ValueTypeBasisCoeff> wHost(localSize);
      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        localSize, wHost.data(), d_meanValueConstraintVec.data());

      //
      // Pick the pinned dof. Selection mirrors dftfe
      // (poissonSolverProblem.cc:551-619): a candidate is a locally owned dof
      // that appears in no constraint equation, neither as a slave nor as a
      // master; the first processor holding any candidate owns the pinned dof
      // and takes its first candidate.
      //
      std::set<global_size_type> indicesTouchedByConstraints;
      const dealii::IndexSet     locallyRelevantElements =
        d_dealiiAffineConstraintMatrix.get_local_lines();
      for (auto it = locallyRelevantElements.begin();
           it != locallyRelevantElements.end();
           ++it)
        {
          if (d_dealiiAffineConstraintMatrix.is_constrained(*it))
            {
              indicesTouchedByConstraints.insert(*it);
              const std::vector<
                std::pair<global_size_type, RealTypeBasisCoeff>> *rowData =
                d_dealiiAffineConstraintMatrix.get_constraint_entries(*it);
              for (size_type j = 0; j < rowData->size(); ++j)
                indicesTouchedByConstraints.insert((*rowData)[j].first);
            }
        }

      size_type numCandidates = 0;
      for (size_type iRange = 0; iRange < d_locallyOwnedRanges.size(); ++iRange)
        for (global_size_type g = d_locallyOwnedRanges[iRange].first;
             g < d_locallyOwnedRanges[iRange].second;
             ++g)
          if (indicesTouchedByConstraints.count(g) == 0)
            numCandidates++;

      std::vector<global_size_type> candidates(numCandidates, 0);
      size_type                     iCandidate = 0;
      for (size_type iRange = 0; iRange < d_locallyOwnedRanges.size(); ++iRange)
        for (global_size_type g = d_locallyOwnedRanges[iRange].first;
             g < d_locallyOwnedRanges[iRange].second;
             ++g)
          if (indicesTouchedByConstraints.count(g) == 0)
            candidates[iCandidate++] = g;

      size_type              localNumCandidates = candidates.size();
      std::vector<size_type> allNumCandidates(nProcs, 0);
      utils::mpi::MPIAllgather<utils::MemorySpace::HOST>(
        &localNumCandidates,
        1,
        utils::mpi::Types<size_type>::getMPIDatatype(),
        allNumCandidates.data(),
        1,
        utils::mpi::Types<size_type>::getMPIDatatype(),
        mpiComm);

      bool foundCandidate         = false;
      d_meanValueConstraintProcId = 0;
      for (int iProc = 0; iProc < nProcs; ++iProc)
        if (allNumCandidates[iProc] > 0)
          {
            d_meanValueConstraintProcId = (size_type)iProc;
            foundCandidate              = true;
            break;
          }
      utils::throwException(
        foundCandidate,
        "MeanValueConstraint: no unconstrained dof is available for pinning "
        "the null space of the Poisson operator.");

      ValueTypeBasisCoeff valueAtConstraintNode =
        utils::Types<ValueTypeBasisCoeff>::zero;
      if (d_meanValueMyRank == (int)d_meanValueConstraintProcId)
        {
          d_meanValueConstraintNodeIdGlobal = candidates[0];
          d_meanValueConstraintNodeIdLocal =
            globalToLocal(d_meanValueConstraintNodeIdGlobal);
          valueAtConstraintNode = wHost[d_meanValueConstraintNodeIdLocal];
        }

      utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
        &d_meanValueConstraintNodeIdGlobal,
        1,
        utils::mpi::Types<global_size_type>::getMPIDatatype(),
        (int)d_meanValueConstraintProcId,
        mpiComm);
      utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
        &valueAtConstraintNode,
        1,
        utils::mpi::Types<ValueTypeBasisCoeff>::getMPIDatatype(),
        (int)d_meanValueConstraintProcId,
        mpiComm);

      utils::throwException(
        utils::abs_(valueAtConstraintNode) > 1e-14,
        "MeanValueConstraint: the pinned dof has a vanishing mass integral.");

      //
      // Rescale w -> a = -w / w_o and zero the pinned entry.
      //
      linearAlgebra::blasLapack::ascale<ValueTypeBasisCoeff,
                                        ValueTypeBasisCoeff,
                                        memorySpace>(
        localSize,
        (ValueTypeBasisCoeff)(-1.0) / valueAtConstraintNode,
        d_meanValueConstraintVec.data(),
        d_meanValueConstraintVec.data(),
        *d_meanValueConstraintVec.getLinAlgOpContext());

      if (d_meanValueMyRank == (int)d_meanValueConstraintProcId)
        {
          const ValueTypeBasisCoeff zero =
            utils::Types<ValueTypeBasisCoeff>::zero;
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            1,
            d_meanValueConstraintVec.data() + d_meanValueConstraintNodeIdLocal,
            &zero);
        }
      d_meanValueConstraintVec.updateGhostValues();

      d_isMeanValueConstraintActive = true;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      hasMeanValueConstraint() const
    {
      return d_isMeanValueConstraintActive;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getMeanValueConstraintVec() const
    {
      return d_meanValueConstraintVec;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    global_size_type
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getMeanValueConstraintNodeIdGlobal() const
    {
      return d_meanValueConstraintNodeIdGlobal;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    size_type
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getMeanValueConstraintProcId() const
    {
      return d_meanValueConstraintProcId;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      distributeChildToParent(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const
    {
      ConstraintsInternal<ValueTypeBasisCoeff, memorySpace>::
        constraintsDistributeChildToParent(vectorData,
                                           blockSize,
                                           d_rowConstraintsIdsLocal,
                                           d_rowConstraintsSizes,
                                           d_columnConstraintsIdsLocal,
                                           d_constraintRowSizesAccumulated,
                                           d_columnConstraintsValues,
                                           *vectorData.getLinAlgOpContext());
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      setConstrainedNodesToZero(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &       vectorData,
        size_type blockSize) const
    {
      ConstraintsInternal<ValueTypeBasisCoeff, memorySpace>::
        constraintsSetConstrainedNodesToZero(vectorData,
                                             blockSize,
                                             d_rowConstraintsIdsLocal,
                                             *vectorData.getLinAlgOpContext());


      if (d_isMeanValueConstraintActive &&
          d_meanValueMyRank == (int)d_meanValueConstraintProcId)
        {
          std::vector<ValueTypeBasisCoeff> zeros(
            blockSize, utils::Types<ValueTypeBasisCoeff>::zero);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            blockSize,
            vectorData.data() + d_meanValueConstraintNodeIdLocal * blockSize,
            zeros.data());
        }
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      setConstrainedNodes(linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                     memorySpace> &vectorData,
                          size_type                                blockSize,
                          ValueTypeBasisCoeff                      alpha) const
    {
      ConstraintsInternal<ValueTypeBasisCoeff, memorySpace>::
        constraintsSetConstrainedNodes(vectorData,
                                       blockSize,
                                       d_rowConstraintsIdsLocal,
                                       alpha,
                                       *vectorData.getLinAlgOpContext());


      // Keeps the Jacobi preconditioner invertible at the pinned dof, which
      // dftfe gets implicitly through distribute_local_to_global on d_diagonalA
      if (d_isMeanValueConstraintActive &&
          d_meanValueMyRank == (int)d_meanValueConstraintProcId)
        {
          std::vector<ValueTypeBasisCoeff> alphas(blockSize, alpha);
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            blockSize,
            vectorData.data() + d_meanValueConstraintNodeIdLocal * blockSize,
            alphas.data());
        }
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    inline bool
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isGhostEntry(const global_size_type globalId) const
    {
      return d_ghostIndicesSet.count(globalId) > 0;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    inline bool
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      inLocallyOwnedRanges(const global_size_type globalId) const
    {
      for (const auto &i : d_locallyOwnedRanges)
        {
          if (globalId >= i.first && globalId < i.second)
            return true;
        }
      return false;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    size_type
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      globalToLocal(const global_size_type globalId) const
    {
      utils::throwException(
        (d_globalToLocalMap.find(globalId) != d_globalToLocalMap.end()),
        " Could not find the globalId in locally owned or ghost ids in EFEConstraintsDealii.h");
      return d_globalToLocalMap.find(globalId)->second;
    }

  } // namespace basis
} // namespace dftefe
