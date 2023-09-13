
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      FEConstraintsDealii()
      : d_isCleared(false)
      , d_isClosed(false)
    {
      //      d_constraintMatrix =
      //      dealii::AffineConstraints<ValueTypeBasisCoeff>();
    }


    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::clear()
    {
      d_constraintMatrix.clear();
      d_isCleared = true;
      d_isClosed  = false;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::close()
    {
      d_constraintMatrix.close();
      d_isCleared = false;
      d_isClosed  = true;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      makeHangingNodeConstraint(std::shared_ptr<FEBasisManager> feBasis)
    {
      utils::throwException(
        d_isCleared && !d_isClosed,
        " Clear the constraint matrix before making hanging node constraints");

      d_feBasisManager =
        std::dynamic_pointer_cast<const FEBasisManagerDealii<dim>>(feBasis);

      utils::throwException(
        d_feBasisManager != nullptr,
        " Could not cast the FEBasisManager to FEBasisManagerDealii in make hanging node constraints");


      dealii::IndexSet locally_relevant_dofs;
      locally_relevant_dofs.clear();
      dealii::DoFTools::extract_locally_relevant_dofs(
        *(d_feBasisManager->getDoFHandler()), locally_relevant_dofs);
      d_constraintMatrix.reinit(locally_relevant_dofs);
      dealii::DoFTools::make_hanging_node_constraints(
        *(d_feBasisManager->getDoFHandler()), d_constraintMatrix);
      d_isCleared = false;
      d_isClosed  = false;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      setInhomogeneity(global_size_type    basisId,
                       ValueTypeBasisCoeff constraintValue)
    {
      utils::throwException(
        !d_isClosed,
        " Clear the constraint matrix before setting inhomogeneities");

      // If condition is removed
      // add_line does not do anything if the basisId already exists.
      addLine(basisId);
      d_constraintMatrix.set_inhomogeneity(basisId, constraintValue);
      d_isCleared = false;
      d_isClosed  = false;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::isClosed() const
    {
      return d_isClosed;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::isConstrained(
      global_size_type basisId) const
    {
      return d_constraintMatrix.is_constrained(basisId);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      setHomogeneousDirichletBC()
    {
      dealii::IndexSet locallyRelevantDofs;
      dealii::DoFTools::extract_locally_relevant_dofs(
        *(d_feBasisManager->getDoFHandler()), locallyRelevantDofs);

      const unsigned int vertices_per_cell =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      const unsigned int dofs_per_cell =
        d_feBasisManager->getDoFHandler()->get_fe().dofs_per_cell;
      const unsigned int faces_per_cell =
        dealii::GeometryInfo<dim>::faces_per_cell;
      const unsigned int dofs_per_face =
        d_feBasisManager->getDoFHandler()->get_fe().dofs_per_face;

      std::vector<global_size_type> cellGlobalDofIndices(dofs_per_cell);
      std::vector<global_size_type> iFaceGlobalDofIndices(dofs_per_face);

      std::vector<bool> dofs_touched(d_feBasisManager->nGlobalNodes(), false);
      auto              cell = d_feBasisManager->beginLocalCells(),
           endc              = d_feBasisManager->endLocalCells();
      for (; cell != endc; ++cell)
        {
          (*cell)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              (*cell)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
              const size_type boundaryId = (*cell)->getFaceBoundaryId(iFace);
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
                      if (!isConstrained(nodeId))
                        {
                          setInhomogeneity(nodeId, 0);
                        } // non-hanging node check
                    }     // Face dof loop
                }
            } // Face loop
        }     // cell locally owned
    }
    /*
        template <typename ValueTypeBasisCoeff,
                  dftefe::utils::MemorySpace memorySpace,
                  size_type                  dim>
        void
        FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
          setInhomogeneousDirichletBC(utils::ScalarSpatialFunctionReal
       &boundaryValues)
        {
          dealii::IndexSet locallyRelevantDofs;
          dealii::DoFTools::extract_locally_relevant_dofs(
            *(d_feBasisManager->getDoFHandler()), locallyRelevantDofs);

          const unsigned int vertices_per_cell =
            dealii::GeometryInfo<dim>::vertices_per_cell;
          const unsigned int dofs_per_cell =
            d_feBasisManager->getDoFHandler()->get_fe().dofs_per_cell;
          const unsigned int faces_per_cell =
            dealii::GeometryInfo<dim>::faces_per_cell;
          const unsigned int dofs_per_face =
            d_feBasisManager->getDoFHandler()->get_fe().dofs_per_face;

          std::vector<global_size_type> cellGlobalDofIndices(dofs_per_cell);
          std::vector<global_size_type> iFaceGlobalDofIndices(dofs_per_face);
          std::map<global_size_type, utils::Point> boundaryCoord;
          d_feBasisManager->getBasisCenters(boundaryCoord);

          std::vector<bool> dofs_touched(d_feBasisManager->nGlobalNodes(),
       false); auto              cell =
       d_feBasisManager->beginLocallyOwnedCells(), endc              =
       d_feBasisManager->endLocallyOwnedCells(); for (; cell != endc; ++cell)
            {
              (*cell)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);

              for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
                {
                  (*cell)->getFaceDoFGlobalIndices(iFace,
       iFaceGlobalDofIndices); const size_type boundaryId =
       (*cell)->getFaceBoundaryId(iFace); if (boundaryId == 0)
                    {
                      for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                           ++iFaceDof)
                        {
                          const dealii::types::global_dof_index nodeId =
                            iFaceGlobalDofIndices[iFaceDof];
                          if (dofs_touched[nodeId])
                            continue;
                          auto inhomoValue =
                            boundaryValues(boundaryCoord[nodeId]);
                          dofs_touched[nodeId] = true;
                          if (!isConstrained(nodeId))
                            {
                              setInhomogeneity(nodeId, inhomoValue);
                            } // non-hanging node check
                        }     // Face dof loop
                    }
                } // Face loop
            }     // cell locally owned
        }
    */
    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const dealii::AffineConstraints<ValueTypeBasisCoeff> &
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getAffineConstraints() const
    {
      return d_constraintMatrix;
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const std::vector<std::pair<global_size_type, ValueTypeBasisCoeff>> *
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getConstraintEntries(const global_size_type lineDof) const
    {
      return d_constraintMatrix.get_constraint_entries(lineDof);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      isInhomogeneouslyConstrained(const global_size_type lineDof) const
    {
      return (d_constraintMatrix.is_inhomogeneously_constrained(lineDof));
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    ValueTypeBasisCoeff
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      getInhomogeneity(const global_size_type lineDof) const
    {
      return (d_constraintMatrix.get_inhomogeneity(lineDof));
    }


    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      copyConstraintsData(
        const Constraints<ValueTypeBasisCoeff, memorySpace> &constraintsDataIn,
        const utils::mpi::MPIPatternP2P<memorySpace> &       mpiPattern,
        const size_type                                      classicalId)
    {
      this->clear();
      std::vector<std::pair<global_size_type, global_size_type>>
        locallyOwnedRanges = mpiPattern.getLocallyOwnedRanges();

      auto locallyOwnedRange = locallyOwnedRanges[classicalId];
      // auto locallyOwnedRange = mpiPattern.getLocallyOwnedRange();

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
                  if (!(mpiPattern.isGhostEntry((*rowData)[j].first).first ||
                        mpiPattern.inLocallyOwnedRanges((*rowData)[j].first)
                          .first))
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

      auto ghostIndices =
        mpiPattern.getGhostIndices(); // can be optimized .. checking enriched
                                      // ghosts also

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
                  if (!(mpiPattern.isGhostEntry((*rowData)[j].first).first ||
                        mpiPattern.inLocallyOwnedRanges((*rowData)[j].first)
                          .first))
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
      populateConstraintsData(
        const utils::mpi::MPIPatternP2P<memorySpace> &mpiPattern,
        const size_type                               classicalId)
    {
      bool printWarning = false;

      std::vector<global_size_type> rowConstraintsIdsGlobalTmp;
      std::vector<size_type>        rowConstraintsIdsLocalTmp;
      std::vector<size_type>        columnConstraintsIdsLocalTmp;
      std::vector<size_type>        constraintRowSizesAccumulatedTmp;
      std::vector<global_size_type> columnConstraintsIdsGlobalTmp;

      std::vector<double>              columnConstraintsValuesTmp;
      std::vector<ValueTypeBasisCoeff> constraintsInhomogenitiesTmp;

      std::vector<size_type> rowConstraintsSizesTmp;

      std::vector<std::pair<global_size_type, global_size_type>>
        locallyOwnedRanges = mpiPattern.getLocallyOwnedRanges();

      auto locallyOwnedRange = locallyOwnedRanges[classicalId];
      // auto locallyOwnedRange = mpiPattern.getLocallyOwnedRange();

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
                  if (!(mpiPattern.isGhostEntry((*rowData)[j].first).first ||
                        mpiPattern.inLocallyOwnedRanges((*rowData)[j].first)
                          .first))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      printWarning                          = true;
                      break;
                    }
                }



              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;

              rowConstraintsIdsLocalTmp.push_back(
                mpiPattern.globalToLocal(lineDof));
              rowConstraintsIdsGlobalTmp.push_back(lineDof);
              constraintsInhomogenitiesTmp.push_back(getInhomogeneity(lineDof));
              rowConstraintsSizesTmp.push_back(rowData->size());
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  columnConstraintsIdsGlobalTmp.push_back((*rowData)[j].first);
                  columnConstraintsIdsLocalTmp.push_back(
                    mpiPattern.globalToLocal((*rowData)[j].first));
                  double realPart = utils::getRealPart((*rowData)[j].second);
                  columnConstraintsValuesTmp.push_back(realPart);
                }

              constraintRowSizesAccumulatedTmp.push_back(columnIdStart);
              columnIdStart += rowData->size();
            }
        }

      auto ghostIndices = mpiPattern.getGhostIndices();

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
                  if (!(mpiPattern.isGhostEntry((*rowData)[j].first).first ||
                        mpiPattern.inLocallyOwnedRanges((*rowData)[j].first)
                          .first))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      printWarning                          = true;
                      break;
                    }
                }

              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;

              rowConstraintsIdsLocalTmp.push_back(
                mpiPattern.globalToLocal(lineDof));
              rowConstraintsIdsGlobalTmp.push_back(lineDof);
              constraintsInhomogenitiesTmp.push_back(getInhomogeneity(lineDof));
              rowConstraintsSizesTmp.push_back(rowData->size());
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  columnConstraintsIdsGlobalTmp.push_back((*rowData)[j].first);
                  columnConstraintsIdsLocalTmp.push_back(
                    mpiPattern.globalToLocal((*rowData)[j].first));
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::addEntries(
      const global_size_type constrainedDofIndex,
      const std::vector<std::pair<global_size_type, ValueTypeBasisCoeff>>
        &colWeightPairs)
    {
      d_constraintMatrix.add_entries(constrainedDofIndex, colWeightPairs);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::addLine(
      const global_size_type lineDof)
    {
      d_constraintMatrix.add_line(lineDof);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
                                           d_constraintsInhomogenities);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
                                           d_columnConstraintsValues);
    }

    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>::
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


  } // namespace basis
} // namespace dftefe
