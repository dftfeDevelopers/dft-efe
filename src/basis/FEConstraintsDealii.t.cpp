
#include <deal.II/dofs/dof_tools.h>
#include "FECellBase.h"
#include <memory>


namespace dftefe
{
  namespace basis
  {
    // default constructor
    template <size_type dim, typename ValueType>
    FEConstraintsDealii<dim, ValueType>::FEConstraintsDealii()
      : d_isCleared(false)
      , d_isClosed(false)
    {
      d_constraintMatrix =
        std::make_shared<dealii::AffineConstraints<ValueType>>();
    }


    template <size_type dim, typename ValueType>
    void
    FEConstraintsDealii<dim, ValueType>::clear()
    {
      d_constraintMatrix->clear();
      d_isCleared = true;
      d_isClosed  = false;
    }

    template <size_type dim, typename ValueType>
    void
    FEConstraintsDealii<dim, ValueType>::close()
    {
      d_constraintMatrix->close();
      d_isCleared = false;
      d_isClosed  = true;
    }

    template <size_type dim, typename ValueType>
    void
    FEConstraintsDealii<dim, ValueType>::makeHangingNodeConstraint(
      std::shared_ptr<FEBasisManager> feBasis)
    {
      utils::throwException(
        d_isCleared && !d_isClosed,
        " Clear the constraint matrix before making hanging node constraints");

      std::shared_ptr<const FEBasisManagerDealii<dim>> dealiiDoFHandler =
        std::dynamic_pointer_cast<const FEBasisManagerDealii<dim>>(feBasis);

      utils::throwException(
        dealiiDoFHandler != nullptr,
        " Could not cast the FEBasisManager to FEBasisManagerDealii in make hanging node constraints");

      // check if this will work
      d_dofHandler =
        std::dynamic_pointer_cast<const FEBasisManagerDealii<dim>>(feBasis);
      dealii::IndexSet locally_relevant_dofs;
      locally_relevant_dofs.clear();
      dealii::DoFTools::extract_locally_relevant_dofs(
        *(dealiiDoFHandler->getDoFHandler()), locally_relevant_dofs);
      d_constraintMatrix->reinit(locally_relevant_dofs);
      dealii::DoFTools::make_hanging_node_constraints(
        *(dealiiDoFHandler->getDoFHandler()), *d_constraintMatrix);
      d_isCleared = false;
      d_isClosed  = false;
    }

    template <size_type dim, typename ValueType>
    void
    FEConstraintsDealii<dim, ValueType>::setInhomogeneity(
      size_type basisId,
      ValueType constraintValue)
    {
      utils::throwException(
        !d_isClosed,
        " Clear the constraint matrix before setting inhomogeneities");
      if (this->isConstrained(basisId) == false)
        {
          d_constraintMatrix->add_line(basisId);
        }
      d_constraintMatrix->set_inhomogeneity(basisId, constraintValue);
      d_isCleared = false;
      d_isClosed  = false;
    }

    template <size_type dim, typename ValueType>
    bool
    FEConstraintsDealii<dim, ValueType>::isClosed()
    {
      return d_isClosed;
    }

    template <size_type dim, typename ValueType>
    bool
    FEConstraintsDealii<dim, ValueType>::isConstrained(size_type basisId)
    {
      return d_constraintMatrix->is_constrained(basisId);
    }

    template <size_type dim, typename ValueType>
    void
    FEConstraintsDealii<dim, ValueType>::setHomogeneousDirichletBC()
    {
      dealii::IndexSet locallyRelevantDofs;
      dealii::DoFTools::extract_locally_relevant_dofs(
        *(d_dofHandler->getDoFHandler()), locallyRelevantDofs);

      const unsigned int vertices_per_cell =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      const unsigned int dofs_per_cell =
        d_dofHandler->getDoFHandler()->get_fe().dofs_per_cell;
      const unsigned int faces_per_cell =
        dealii::GeometryInfo<dim>::faces_per_cell;
      const unsigned int dofs_per_face =
        d_dofHandler->getDoFHandler()->get_fe().dofs_per_face;

      std::vector<global_size_type> cellGlobalDofIndices(dofs_per_cell);
      std::vector<global_size_type> iFaceGlobalDofIndices(dofs_per_face);

      std::vector<bool> dofs_touched(d_dofHandler->nGlobalNodes(), false);
      auto              cell = d_dofHandler->beginLocallyOwnedCells(),
           endc              = d_dofHandler->endLocallyOwnedCells();
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

    template <size_type dim, typename ValueType>
    const dealii::AffineConstraints<ValueType> &
    FEConstraintsDealii<dim, ValueType>::getAffineConstraints() const
    {
      return *d_constraintMatrix;
    }
  } // namespace basis
} // namespace dftefe
