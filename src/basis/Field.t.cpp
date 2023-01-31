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
 * @author Bikash Kanungo, Vishal Subramanian
 */

#include <linearAlgebra/Vector.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    Field<ValueTypeBasisCoeff, memorySpace>::Field(
      std::shared_ptr<const BasisHandler<ValueTypeBasisCoeff, memorySpace>>
                                                   basisHandler,
      const std::string                            constraintsName,
      linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext)
    {
      reinit(basisHandler, constraintsName, linAlgOpContext);
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::reinit(
      std::shared_ptr<const BasisHandler<ValueTypeBasisCoeff, memorySpace>>
                                                   basisHandler,
      const std::string                            constraintsName,
      linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext)
    {
      d_basisHandler     = basisHandler;
      d_constraintsName  = constraintsName;
      d_linAlgOpContext  = linAlgOpContext;
      auto mpiPatternP2P = basisHandler->getMPIPatternP2P(constraintsName);

      //
      // create the vector
      //

      d_vector = std::make_shared<
        linearAlgebra::Vector<ValueTypeBasisCoeff, memorySpace>>(
        mpiPatternP2P, d_linAlgOpContext, ValueTypeBasisCoeff());
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::applyConstraintsParentToChild()
    {
      const Constraints<ValueTypeBasisCoeff, memorySpace> &constraints =
        d_basisHandler->getConstraints(d_constraintsName);
      constraints.distributeParentToChild(*d_vector);
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::applyConstraintsChildToParent()
    {
      const Constraints<ValueTypeBasisCoeff, memorySpace> &constraints =
        d_basisHandler->getConstraints(d_constraintsName);
      constraints.distributeChildToParent(*d_vector);
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    const linearAlgebra::Vector<ValueTypeBasisCoeff, memorySpace> &
    Field<ValueTypeBasisCoeff, memorySpace>::getVector()
    {
      return *d_vector;
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    const BasisHandler<ValueTypeBasisCoeff, memorySpace> &
    Field<ValueTypeBasisCoeff, memorySpace>::getBasisHandler() const
    {
      return *d_basisHandler;
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    typename Field<ValueTypeBasisCoeff, memorySpace>::iterator
    Field<ValueTypeBasisCoeff, memorySpace>::begin()
    {
      return d_vector->begin();
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    typename Field<ValueTypeBasisCoeff, memorySpace>::const_iterator
    Field<ValueTypeBasisCoeff, memorySpace>::begin() const
    {
      return d_vector->begin();
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    typename Field<ValueTypeBasisCoeff, memorySpace>::iterator
    Field<ValueTypeBasisCoeff, memorySpace>::end()
    {
      return d_vector->end();
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    typename Field<ValueTypeBasisCoeff, memorySpace>::const_iterator
    Field<ValueTypeBasisCoeff, memorySpace>::end() const
    {
      return d_vector->end();
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::updateGhostValues(
      const size_type communicationChannel /*= 0*/)
    {
      d_vector->updateGhostValues(communicationChannel);
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::accumulateAddLocallyOwned(
      const size_type communicationChannel /*= 0*/)
    {
      d_vector->accumulateAddLocallyOwned(communicationChannel);
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::updateGhostValuesBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_vector->updateGhostValuesBegin(communicationChannel);
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::updateGhostValuesEnd()
    {
      d_vector->updateGhostValuesEnd();
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::accumulateAddLocallyOwnedBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_vector->accumulateAddLocallyOwnedBegin();
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    void
    Field<ValueTypeBasisCoeff, memorySpace>::accumulateAddLocallyOwnedEnd()
    {
      d_vector->accumulateAddLocallyOwnedEnd();
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    std::string
    Field<ValueTypeBasisCoeff, memorySpace>::getConstraintsName() const
    {
      return d_constraintsName;
    }

    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    linearAlgebra::LinAlgOpContext<memorySpace> &
    Field<ValueTypeBasisCoeff, memorySpace>::getLinAlgOpContext() const
    {
      return *d_linAlgOpContext;
    }
  } // end of namespace basis
} // end of namespace dftefe
