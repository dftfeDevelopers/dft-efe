
#include <deal.II/dofs/dof_tools.h>
#include <utils/NumberUtils.h>
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      CFEConstraintsLocalDealii(const dealii::IndexSet &locally_owned_dofs,
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      CFEConstraintsLocalDealii(
        dealii::AffineConstraints<ValueTypeBasisCoeff>
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::copyFrom(
      const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>
        &constraintsLocalIn)
    {
      const CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>
        &cfeConstraintsLocalDealiiIn = dynamic_cast<
          const CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>
            &>(constraintsLocalIn);

      utils::throwException(
        &cfeConstraintsLocalDealiiIn != nullptr,
        " Could not typecast ConstraintsLocal to CFEConstraintsLocalDealii in CFEConstraintsLocalDealii.h");

      d_isClosed           = false;
      d_isCleared          = false;
      d_locallyOwnedRanges = cfeConstraintsLocalDealiiIn.d_locallyOwnedRanges;
      d_ghostIndices       = cfeConstraintsLocalDealiiIn.d_ghostIndices;
      d_ghostIndicesSet    = cfeConstraintsLocalDealiiIn.d_ghostIndicesSet;
      d_globalToLocalMap   = cfeConstraintsLocalDealiiIn.d_globalToLocalMap;
      copyConstraintsDataFromDealiiToDealii(cfeConstraintsLocalDealiiIn);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::clear()
    {
      d_dealiiAffineConstraintMatrix.clear();
      d_isCleared = true;
      d_isClosed  = false;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::close()
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      setInhomogeneity(global_size_type    basisId,
                       ValueTypeBasisCoeff constraintValue)
    {
      utils::throwException(
        !d_isClosed,
        " Clear the constraint matrix before setting inhomogeneities. Cannot setInhomogeneity after close().");

      // If condition is removed
      // add_line does not do anything if the basisId already exists.
      addLine(basisId);
      d_dealiiAffineConstraintMatrix.set_inhomogeneity(basisId,
                                                       constraintValue);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::isClosed()
      const
    {
      return d_isClosed;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isConstrained(global_size_type basisId) const
    {
      return d_dealiiAffineConstraintMatrix.is_constrained(basisId);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const dealii::AffineConstraints<ValueTypeBasisCoeff> &
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getAffineConstraints() const
    {
      return d_dealiiAffineConstraintMatrix;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const std::vector<std::pair<global_size_type, ValueTypeBasisCoeff>> *
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getConstraintEntries(const global_size_type lineDof) const
    {
      return d_dealiiAffineConstraintMatrix.get_constraint_entries(lineDof);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isInhomogeneouslyConstrained(const global_size_type lineDof) const
    {
      return (
        d_dealiiAffineConstraintMatrix.is_inhomogeneously_constrained(lineDof));
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ValueTypeBasisCoeff
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getInhomogeneity(const global_size_type lineDof) const
    {
      return (d_dealiiAffineConstraintMatrix.get_inhomogeneity(lineDof));
    }


    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      copyConstraintsDataFromDealiiToDealii(
        const CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>
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
                std::pair<global_size_type, ValueTypeBasisCoeff>> *rowData =
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
                std::pair<global_size_type, ValueTypeBasisCoeff>> *rowData =
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      copyConstraintsDataFromDealiiToDftefe()
    {
      bool printWarning = false;

      std::vector<global_size_type> rowConstraintsIdsGlobalTmp(0);
      std::vector<size_type>        rowConstraintsIdsLocalTmp(0);
      std::vector<size_type>        columnConstraintsIdsLocalTmp(0);
      std::vector<size_type>        constraintRowSizesAccumulatedTmp(0);
      std::vector<global_size_type> columnConstraintsIdsGlobalTmp(0);

      std::vector<double>              columnConstraintsValuesTmp(0);
      std::vector<ValueTypeBasisCoeff> constraintsInhomogenitiesTmp(0);

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
                std::pair<global_size_type, ValueTypeBasisCoeff>> *rowData =
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
              constraintsInhomogenitiesTmp.push_back(getInhomogeneity(lineDof));
              rowConstraintsSizesTmp.push_back(rowData->size());
              for (size_type j = 0; j < rowData->size(); ++j)
                {
                  columnConstraintsIdsGlobalTmp.push_back((*rowData)[j].first);
                  columnConstraintsIdsLocalTmp.push_back(
                    globalToLocal((*rowData)[j].first));
                  double realPart = utils::getRealPart((*rowData)[j].second);
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
                std::pair<global_size_type, ValueTypeBasisCoeff>> *rowData =
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
              constraintsInhomogenitiesTmp.push_back(getInhomogeneity(lineDof));
              rowConstraintsSizesTmp.push_back(rowData->size());
              for (size_type j = 0; j < rowData->size(); ++j)
                {
                  columnConstraintsIdsGlobalTmp.push_back((*rowData)[j].first);
                  columnConstraintsIdsLocalTmp.push_back(
                    globalToLocal((*rowData)[j].first));
                  double realPart = utils::getRealPart((*rowData)[j].second);
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      addEntries(
        const global_size_type constrainedDofIndex,
        const std::vector<std::pair<global_size_type, ValueTypeBasisCoeff>>
          &colWeightPairs)
    {
      d_dealiiAffineConstraintMatrix.add_entries(constrainedDofIndex,
                                                 colWeightPairs);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::addLine(
      const global_size_type lineDof)
    {
      d_dealiiAffineConstraintMatrix.add_line(lineDof);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
      // Pick the pinned dof: a locally owned dof in no constraint equation,
      // as in dftfe (poissonSolverProblem.cc:551-568). The relevant indices
      // come from this class's own owned ranges plus ghosts, not from
      // get_local_lines(), which clear() leaves empty here.
      size_type numLocallyRelevant = d_ghostIndices.size();
      for (size_type iRange = 0; iRange < d_locallyOwnedRanges.size(); ++iRange)
        numLocallyRelevant += d_locallyOwnedRanges[iRange].second -
                              d_locallyOwnedRanges[iRange].first;

      std::vector<global_size_type> locallyRelevantElements(numLocallyRelevant,
                                                            0);
      size_type iRelevant = 0;
      for (size_type iRange = 0; iRange < d_locallyOwnedRanges.size(); ++iRange)
        for (global_size_type g = d_locallyOwnedRanges[iRange].first;
             g < d_locallyOwnedRanges[iRange].second;
             ++g)
          {
            locallyRelevantElements[iRelevant] = g;
            iRelevant++;
          }
      for (size_type i = 0; i < d_ghostIndices.size(); ++i)
        {
          locallyRelevantElements[iRelevant] = d_ghostIndices[i];
          iRelevant++;
        }

      // A dof is excluded if it appears in any constraint equation, as the
      // slave or as one of its masters, exactly as dftfe does
      // (poissonSolverProblem.cc:551-568).
      std::set<global_size_type> indicesTouchedByConstraints;
      for (size_type i = 0; i < numLocallyRelevant; ++i)
        {
          const global_size_type lineDof = locallyRelevantElements[i];
          if (d_dealiiAffineConstraintMatrix.is_constrained(lineDof))
            {
              indicesTouchedByConstraints.insert(lineDof);
              const std::vector<
                std::pair<global_size_type, ValueTypeBasisCoeff>> *rowData =
                d_dealiiAffineConstraintMatrix.get_constraint_entries(lineDof);
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

      size_type localNumCandidates = candidates.size();

      //
      // Among the candidates, pin the one with the largest mass integral
      // |w_i|: a = -w/w_o divides by it, and it puts the pinned dof in the
      // interior where dealii would not make it a periodic master. 
      //
      double           localBestAbsW = -1.0;
      global_size_type localBestId   = 0;
      for (size_type i = 0; i < localNumCandidates; ++i)
        {
          const double absW =
            (double)utils::abs_(wHost[globalToLocal(candidates[i])]);
          if (absW > localBestAbsW)
            {
              localBestAbsW = absW;
              localBestId   = candidates[i];
            }
        }

      double globalBestAbsW = -1.0;
      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        &localBestAbsW,
        &globalBestAbsW,
        1,
        utils::mpi::Types<double>::getMPIDatatype(),
        utils::mpi::MPIMax,
        mpiComm);

      utils::throwException(
        globalBestAbsW >= 0.0,
        "MeanValueConstraint: no unconstrained dof is available for pinning "
        "the null space of the Poisson operator.");

      // nProcs stands for "no claim", so the minimum is the lowest ranked
      // processor that holds the winning value and ties break deterministically.
      int localClaim = (localBestAbsW == globalBestAbsW) ? d_meanValueMyRank :
                                                           nProcs;
      int winningProc = nProcs;
      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        &localClaim,
        &winningProc,
        1,
        utils::mpi::Types<int>::getMPIDatatype(),
        utils::mpi::MPIMin,
        mpiComm);
      d_meanValueConstraintProcId = (size_type)winningProc;

      ValueTypeBasisCoeff valueAtConstraintNode =
        utils::Types<ValueTypeBasisCoeff>::zero;
      if (d_meanValueMyRank == (int)d_meanValueConstraintProcId)
        {
          d_meanValueConstraintNodeIdGlobal = localBestId;
          // dftfe asserts the same thing right after its election
          // (poissonSolverProblem.cc:613-616). Kept as a throw: the one cheap
          // check that the exclusion above actually excluded something.
          utils::throwException<utils::InvalidArgument>(
            !d_dealiiAffineConstraintMatrix.is_constrained(
              d_meanValueConstraintNodeIdGlobal),
            "MeanValueConstraint: the elected dof is itself constrained, so "
            "the dealii distribute would overwrite the value the mean value "
            "constraint puts there. The candidate exclusion did not see this "
            "dof's constraint row.");
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      hasMeanValueConstraint() const
    {
      return d_isMeanValueConstraintActive;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getMeanValueConstraintVec() const
    {
      return d_meanValueConstraintVec;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    global_size_type
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getMeanValueConstraintNodeIdGlobal() const
    {
      return d_meanValueConstraintNodeIdGlobal;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    size_type
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getMeanValueConstraintProcId() const
    {
      return d_meanValueConstraintProcId;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isGhostEntry(const global_size_type globalId) const
    {
      return d_ghostIndicesSet.count(globalId) > 0;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    inline bool
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
    CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      globalToLocal(const global_size_type globalId) const
    {
      utils::throwException(
        (d_globalToLocalMap.find(globalId) != d_globalToLocalMap.end()),
        " Could not find the globalId in locally owned or ghost ids in CFEConstraintsDealii.h");
      return d_globalToLocalMap.find(globalId)->second;
    }

    // template <typename ValueTypeBasisCoeff,
    //           utils::MemorySpace memorySpace,
    //           size_type          dim>
    // void
    // CFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
    //   getHomogeneousDirichletBCMatrixFree(const dealii::DoFHandler<dim, dim>
    //   &dealiiDofHandler,
    //                                       dealii::MatrixFree<dim,
    //                                       ValueTypeBasisCoeff>
    //                                       &dealiiMatrixFree,
    //                                        dealii::AffineConstraints<ValueTypeBasisCoeff>
    //                                        &constraintMatrix) const
    // {
    //   dealii::AffineConstraints<ValueTypeBasisCoeff>
    //     onlyHangingNodeConstraints;
    //   onlyHangingNodeConstraints.clear();
    //   constraintMatrix.clear();
    //   dealii::IndexSet locally_relevant_dofs;
    //   locally_relevant_dofs.clear();
    //   dealii::DoFTools::extract_locally_relevant_dofs(dealiiDofHandler,
    //                                                   locally_relevant_dofs);
    //   onlyHangingNodeConstraints.reinit(
    //     dealiiDofHandler.locally_owned_dofs(), locally_relevant_dofs);
    //   dealii::DoFTools::make_hanging_node_constraints(
    //     dealiiDofHandler, onlyHangingNodeConstraints);
    //   onlyHangingNodeConstraints.close();

    //   constraintMatrix.reinit(
    //     dealiiDofHandler.locally_owned_dofs(), locally_relevant_dofs);
    //   dealii::DoFTools::make_hanging_node_constraints(
    //     dealiiDofHandler, constraintMatrix);

    //   const uInt vertices_per_cell =
    //       dealii::GeometryInfo<dim>::vertices_per_cell;
    //   const uInt dofs_per_cell  = dealiiDofHandler.get_fe().dofs_per_cell;
    //   const uInt faces_per_cell = dealii::GeometryInfo<dim>::faces_per_cell;
    //   const uInt dofs_per_face  = dealiiDofHandler.get_fe().dofs_per_face;

    //   std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
    //     dofs_per_cell);
    //   std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
    //     dofs_per_face);

    //   std::vector<bool> dofs_touched(dealiiDofHandler.n_dofs(), false);
    // dealii::DoFHandler<3>::active_cell_iterator cell =
    //                                               dealiiDofHandler.begin_active(),
    //                                             endc =
    //                                             dealiiDofHandler.end();
    //   for (; cell != endc; ++cell)
    //   if (cell->is_locally_owned() || cell->is_ghost())
    //     {
    //       cell->get_dof_indices(cellGlobalDofIndices);
    //       for (uInt iFace = 0; iFace < faces_per_cell; ++iFace)
    //         {
    //           const uInt boundaryId = cell->face(iFace)->boundary_id();
    //           if (boundaryId == 0)
    //             {
    //               cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
    //               for (uInt iFaceDof = 0; iFaceDof < dofs_per_face;
    //                   ++iFaceDof)
    //                 {
    //                   const dealii::types::global_dof_index nodeId =
    //                     iFaceGlobalDofIndices[iFaceDof];
    //                   if (dofs_touched[nodeId])
    //                     continue;
    //                   dofs_touched[nodeId] = true;
    //                   if (!onlyHangingNodeConstraints.is_constrained(nodeId))
    //                     {
    //                       constraintMatrix.add_line(nodeId);
    //                       constraintMatrix.set_inhomogeneity(nodeId, 0);
    //                     } // non-hanging node check
    //                 }     // Face dof loop
    //             }         // non-periodic boundary id
    //         }             // Face loop
    //     }                 // cell locally owned
    //     constraintMatrix.close();

    //   typename dealii::MatrixFree<dim>::AdditionalData dealiiAdditionalData;
    //   dealiiAdditionalData.tasks_parallel_scheme =
    //     dealii::MatrixFree<dim>::AdditionalData::partition_partition;
    //   dealii::UpdateFlags dealiiUpdateFlags  = dealii::update_values | dealii::update_gradients |
    //     dealii::update_JxW_values | dealii::update_quadrature_points;
    //   dealiiAdditionalData.mapping_update_flags = dealiiUpdateFlags;
    //   dealii::Quadrature<dim> dealiiQuadratureType(dealii::QGauss<dim>(1));
    //   dealiiMatrixFree.clear();
    //   dealii::MappingQ1<dim> mappingDealii;
    //   dealiiMatrixFree.reinit(dealii::MappingQ1<dim, dim>(),
    //                           dealiiDofHandler,
    //                           constraintMatrix,
    //                           dealiiQuadratureType,
    //                           dealiiAdditionalData);
    // }
  } // namespace basis
} // namespace dftefe
