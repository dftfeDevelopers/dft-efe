#include "FEBasisManagerDealii.h"
#include <deal.II/dofs/dof_tools.h>


namespace dftefe
{
  namespace basis
  {
    // default constructor
    template <size_type dim, typename ValueType>
    FEConstraintsDealii<dim,ValueType>::FEConstraintsDealii()
    :d_isCleared(false),
      d_isClosed(false)
    {
      d_constraintMatrix = std::make_shared<dealii::AffineConstraints<ValueType>>();

    }

    // default destructor
    template <size_type dim, typename ValueType>
    FEConstraintsDealii<dim,ValueType>::~FEConstraintsDealii()
    {

    }


    template <size_type dim,typename ValueType>
    void FEConstraintsDealii<dim,ValueType>::clear()
    {
      d_constraintMatrix->clear();
      d_isCleared = true;
      d_isClosed = false;
    }

    template <size_type dim,typename ValueType>
    void FEConstraintsDealii<dim,ValueType>::close()
    {
      d_constraintMatrix->close();
      d_isCleared = false;
      d_isClosed = true;
    }

    template <size_type dim,typename ValueType>
    void FEConstraintsDealii<dim,ValueType>::makeHangingNodeConstraint(
      FEBasisManager &feBasis)
    {
      utils::throwException(
        d_isCleared && !d_isClosed,
        " Clear the constraint matrix before making hanging node constraints");

      const FEBasisManagerDealii<dim>  *dealiiDoFHandler =
        dynamic_cast<const FEBasisManagerDealii<dim> *>(&feBasis);

      utils::throwException(
        dealiiDoFHandler != nullptr,
        " Could not cast the FEBasisManager to FEBasisManagerDealii in make hanging node constraints");
      dealii::IndexSet locally_relevant_dofs;
      locally_relevant_dofs.clear();
      dealii::DoFTools::extract_locally_relevant_dofs(dealiiDoFHandler->getDoFHandler(), locally_relevant_dofs);
      d_constraintMatrix->reinit(locally_relevant_dofs);
      dealii::DoFTools::make_hanging_node_constraints(dealiiDoFHandler->getDoFHandler(), d_constraintMatrix);
      d_isCleared = false;
      d_isClosed = false;
    }

    template <size_type dim,typename ValueType>
    void FEConstraintsDealii<dim,ValueType>::addLine( size_type lineId)
    {
      utils::throwException(
        !d_isClosed,
        " Clear the constraint matrix before adding constraints");
      d_constraintMatrix->add_line(lineId);
      d_isCleared = false;
      d_isClosed = false;
    }
    template <size_type dim,typename ValueType>
    void FEConstraintsDealii<dim,ValueType>::setInhomogeneity(size_type lineId,
                                                      ValueType constraintValue)
    {
      utils::throwException(
        !d_isClosed,
        " Clear the constraint matrix before setting inhomogeneities");
      d_constraintMatrix->set_inhomogeneity(lineId,constraintValue );
      d_isCleared = false;
      d_isClosed = false;

    }

    template <size_type dim,typename ValueType>
    bool FEConstraintsDealii<dim,ValueType>::isClosed()
    {
      return d_isClosed;
    }
  }
}
