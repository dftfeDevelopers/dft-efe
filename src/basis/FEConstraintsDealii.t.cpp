
#include <deal.II/dofs/dof_tools.h>
#include "FECellBase.h"
#include <memory>
#include "ConstraintsInternal.h"


namespace dftefe
{
  namespace basis
  {
    // default constructor
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    FEConstraintsDealii<ValueType, memorySpace, dim>::FEConstraintsDealii()
      : d_isCleared(false)
      , d_isClosed(false)
    {
      //      d_constraintMatrix = dealii::AffineConstraints<ValueType>();
    }


    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::clear()
    {
      d_constraintMatrix.clear();
      d_isCleared = true;
      d_isClosed  = false;
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::close()
    {
      d_constraintMatrix.close();
      d_isCleared = false;
      d_isClosed  = true;
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::makeHangingNodeConstraint(
      std::shared_ptr<FEBasisManager> feBasis)
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

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::setInhomogeneity(
      global_size_type basisId,
      ValueType        constraintValue)
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

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    FEConstraintsDealii<ValueType, memorySpace, dim>::isClosed() const
    {
      return d_isClosed;
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    FEConstraintsDealii<ValueType, memorySpace, dim>::isConstrained(
      global_size_type basisId) const
    {
      return d_constraintMatrix.is_constrained(basisId);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::
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
      auto              cell = d_feBasisManager->beginLocallyOwnedCells(),
           endc              = d_feBasisManager->endLocallyOwnedCells();
      for (; cell != endc; ++cell)
        {
          (*cell)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
          for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
            {
              (*cell)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
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
            }         // Face loop
        }             // cell locally owned
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const dealii::AffineConstraints<ValueType> &
    FEConstraintsDealii<ValueType, memorySpace, dim>::getAffineConstraints()
      const
    {
      return d_constraintMatrix;
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::pair<global_size_type, ValueType> *
    FEConstraintsDealii<ValueType, memorySpace, dim>::getConstraintEntries(
      const global_size_type lineDof) const
    {
      return d_constraintMatrix.get_constraint_entries(lineDof);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    FEConstraintsDealii<ValueType, memorySpace, dim>::
      isInhomogeneouslyConstrained(const global_size_type lineDof) const
    {
      return (d_constraintMatrix.is_inhomogeneously_constrained(lineDof));
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    ValueType
    FEConstraintsDealii<ValueType, memorySpace, dim>::getInhomogeneity(
      const global_size_type lineDof) const
    {
      return (d_constraintMatrix.get_inhomogeneity(lineDof));
    }


    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::copyConstraintsData(
      const Constraints<ValueType, memorySpace> &constraintsDataIn,
      const utils::MPIPatternP2P<memorySpace> &  mpiPattern)
    {
      this->clear();
      auto locallyOwnedIndices = mpiPattern.getLocallyOwnedIndices();

      bool printWarning = false;
      for (auto locallyOwnedIter = locallyOwnedIndices.begin();
           locallyOwnedIter != locallyOwnedIndices.end();
           locallyOwnedIter++)
        {
          if (constraintsDataIn.isConstrained(*locallyOwnedIter))
            {
              const global_size_type lineDof = *locallyOwnedIter;
              this->addLine(lineDof);
              if (constraintsDataIn.isInhomogeneouslyConstrained(lineDof))
                {
                  this->setInhomogeneity(
                    lineDof, constraintsDataIn.getInhomogeneity(lineDof));
                }
              const std::vector<std::pair<global_size_type, ValueType>>
                *rowData = constraintsDataIn.getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  if (!(mpiPattern.isGhostEntry((*rowData)[j].first) ||
                        mpiPattern.inLocallyOwnedRange((*rowData)[j].first)))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      printWarning                          = true;
                      break;
                    }
                }

              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;

              this->addEntries(lineDof, rowData);
            }
        }

      auto ghostIndices = mpiPattern.getGhostIndices();

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
              const std::vector<std::pair<global_size_type, ValueType>>
                *rowData = constraintsDataIn.getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  if (!(mpiPattern.isGhostEntry((*rowData)[j].first) ||
                        mpiPattern.inLocallyOwnedRange((*rowData)[j].first)))
                    {
                      isConstraintRhsExpandingOutOfIndexSet = true;
                      printWarning                          = true;
                      break;
                    }
                }

              if (isConstraintRhsExpandingOutOfIndexSet)
                continue;
              this->addEntries(lineDof, rowData);
            }
        }

      if (printWarning)
        {
          std::cout
            << "DFT-EFE Warning : the ghost indices provided is not complete....Check if the ghost indices are sufficient\n";
        }
    }


    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::populateConstraintsData(
      const utils::MPIPatternP2P<memorySpace> &mpiPattern)
    {
      bool printWarning = false;
      bool isComplex    = std::is_same<std::complex<float>, ValueType>::value ||
                       std::is_same<std::complex<double>, ValueType>::value;


      std::vector<global_size_type> rowConstraintsIdsGlobalTmp;
      std::vector<global_size_type> rowConstraintsIdsLocalTmp;
      std::vector<global_size_type> columnConstraintsIdsLocalTmp;
      std::vector<global_size_type> columnConstraintsIdsGlobalTmp;

      std::vector<double>    columnConstraintsValuesTmp;
      std::vector<ValueType> constraintsInhomogenitiesTmp;

      std::vector<global_size_type> rowConstraintsSizesTmp;
      std::vector<global_size_type> localIndexMapUnflattenedToFlattenedTmp;

      auto locallyOwnedIndices = mpiPattern.getLocallyOwnedIndices();

      for (auto locallyOwnedIter = locallyOwnedIndices.begin();
           locallyOwnedIter != locallyOwnedIndices.end();
           locallyOwnedIter++)
        {
          if (this->isConstrained(*locallyOwnedIter))
            {
              const dealii::types::global_dof_index lineDof = *locallyOwnedIter;
              const std::vector<std::pair<dealii::types::global_dof_index,
                                          ValueType>> *rowData =
                this->getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  if (!(mpiPattern.isGhostEntry((*rowData)[j].first) ||
                        mpiPattern.inLocallyOwnedRange((*rowData)[j].first)))
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
                  if (isComplex)
                    {
                      columnConstraintsValuesTmp.push_back(
                        (*rowData)[j].second.real);
                    }
                  else
                    {
                      columnConstraintsValuesTmp.push_back(
                        (*rowData)[j].second);
                    }
                }
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

              const std::vector<std::pair<dealii::types::global_dof_index,
                                          ValueType>> *rowData =
                this->getConstraintEntries(lineDof);

              bool isConstraintRhsExpandingOutOfIndexSet = false;
              for (unsigned int j = 0; j < rowData->size(); ++j)
                {
                  if (!(mpiPattern.isGhostEntry((*rowData)[j].first) ||
                        mpiPattern.inLocallyOwnedRange((*rowData)[j].first)))
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
                  if (isComplex)
                    {
                      columnConstraintsValuesTmp.push_back(
                        (*rowData)[j].second.real);
                    }
                  else
                    {
                      columnConstraintsValuesTmp.push_back(
                        (*rowData)[j].second);
                    }
                }
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
        rowConstraintsIdsGlobalTmp.begin());

      d_rowConstraintsIdsLocal.resize(rowConstraintsIdsLocalTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        rowConstraintsIdsLocalTmp.size(),
        d_rowConstraintsIdsLocal.data(),
        rowConstraintsIdsLocalTmp.begin());

      d_columnConstraintsIdsLocal.resize(columnConstraintsIdsLocalTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        columnConstraintsIdsLocalTmp.size(),
        d_columnConstraintsIdsLocal.data(),
        columnConstraintsIdsLocalTmp.begin());

      d_columnConstraintsIdsGlobal.resize(columnConstraintsIdsGlobalTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        columnConstraintsIdsGlobalTmp.size(),
        d_columnConstraintsIdsGlobal.data(),
        columnConstraintsIdsGlobalTmp.begin());

      d_columnConstraintsValues.resize(columnConstraintsValuesTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        columnConstraintsValuesTmp.size(),
        d_columnConstraintsValues.data(),
        columnConstraintsValuesTmp.begin());

      d_constraintsInhomogenities.resize(constraintsInhomogenitiesTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        constraintsInhomogenitiesTmp.size(),
        d_constraintsInhomogenities.data(),
        constraintsInhomogenitiesTmp.begin());

      d_rowConstraintsSizes.resize(rowConstraintsSizesTmp.size());
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        rowConstraintsSizesTmp.size(),
        d_rowConstraintsSizes.data(),
        rowConstraintsSizesTmp.begin());
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::addEntries(
      const global_size_type constrainedDofIndex,
      const std::vector<std::pair<global_size_type, ValueType>> &colWeightPairs)
    {
      d_constraintMatrix.add_entries(constrainedDofIndex, colWeightPairs);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::addLine(
      const global_size_type lineDof)
    {
      d_constraintMatrix.add_line(lineDof);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::distributeParentToChild(
      linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
      size_type blockSize ) const
    {
      ConstraintsInternal<ValueType, memorySpace>::constraintsDistributeParentToChild(
        vectorData,
        blockSize,
        d_rowConstraintsIdsLocal,
        d_rowConstraintsSizes,
        d_columnConstraintsIdsLocal,
        d_columnConstraintsValues,
        d_constraintsInhomogenities);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::distributeChildToParent(
      linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
      size_type blockSize ) const
    {
      ConstraintsInternal<ValueType, memorySpace>::constraintsDistributeChildToParent(
        vectorData,
        blockSize,
        d_rowConstraintsIdsLocal,
        d_rowConstraintsSizes,
        d_columnConstraintsIdsLocal,
        d_columnConstraintsValues);
    }

    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    FEConstraintsDealii<ValueType, memorySpace, dim>::setConstrainedNodesToZero(
      linearAlgebra::Vector<ValueType, memorySpace> &vectorData,
      size_type blockSize) const
    {
      ConstraintsInternal<ValueType, memorySpace>::constraintsSetConstrainedNodesToZero(
        vectorData,
        blockSize,
        d_rowConstraintsIdsLocal);
    }


  } // namespace basis
} // namespace dftefe
