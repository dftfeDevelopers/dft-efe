
#include <deal.II/dofs/dof_tools.h>
#include <utils/NumberUtils.h>
#include <basis/FECellBase.h>
#include <memory>
#include <basis/ConstraintsInternal.h>


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
    {
      d_locallyOwnedRanges.resize(0);
      d_ghostIndices.resize(0);
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
      , d_globalToLocalMap(globalToLocalMapLocalDofs)
      , d_isCleared(false)
      , d_isClosed(true)
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
      d_dealiiAffineConstraintMatrix.set_inhomogeneity(basisId,
                                                       constraintValue);
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
    const dealii::AffineConstraints<ValueTypeBasisCoeff> &
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getAffineConstraints() const
    {
      return d_dealiiAffineConstraintMatrix;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const std::vector<std::pair<global_size_type, ValueTypeBasisCoeff>> *
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
                std::pair<global_size_type, ValueTypeBasisCoeff>> *rowData =
                constraintsDataIn.getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (unsigned int j = 0; j < rowData->size(); ++j)
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
              for (unsigned int j = 0; j < rowData->size(); ++j)
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
              for (unsigned int j = 0; j < rowData->size(); ++j)
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
              for (unsigned int j = 0; j < rowData->size(); ++j)
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
              for (unsigned int j = 0; j < rowData->size(); ++j)
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
              for (unsigned int j = 0; j < rowData->size(); ++j)
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
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
                                             d_rowConstraintsIdsLocal);
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
                                       alpha);
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isGhostEntry(const global_size_type globalId) const
    {
      bool returnValue = false;
      auto it =
        std::find(d_ghostIndices.begin(), d_ghostIndices.end(), globalId);

      if (it != d_ghostIndices.end())
        returnValue = true;
      return returnValue;
    }

    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    EFEConstraintsLocalDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      inLocallyOwnedRanges(const global_size_type globalId) const
    {
      bool returnValue = false;
      for (auto i : d_locallyOwnedRanges)
        {
          if (globalId >= i.first && globalId < i.second)
            {
              returnValue = true;
              break;
            }
        }
      return returnValue;
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
