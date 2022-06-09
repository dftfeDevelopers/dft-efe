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

#include <linearAlgebra/DistributedVector.h>
#include <linearAlgebra/SerialVector.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueType, utils::MemorySpace memorySpace>
    Field<ValueType, memorySpace>::Field(
      std::shared_ptr<const BasisHandler<ValueType, memorySpace>>   basisHandler,
      const std::string                     constraintsName,
      const linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
      reinit(basisHandler, constraintsName, linAlgOpContext);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void Field<ValueType, memorySpace>::reinit(
      std::shared_ptr<const BasisHandler<ValueType,memorySpace>>   basisHandler,
      const std::string                     constraintsName,
      const linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
      d_basisHandler     = basisHandler;
      d_constraintsName  = constraintsName;
      d_linAlgOpContext  = linAlgOpContext;
      auto mpiPatternP2P = basisHandler->getMPIPatternP2P(constraintsName);

      //
      // create the vector
      //
      if (d_basisHandler->isDistributed())
        {
          d_vector = std::make_shared<
            linearAlgebra::DistributedVector<ValueType, memorySpace>>(
            mpiPatternP2P, d_linAlgOpContext, ValueType());
        }
      else
        {
          auto locallyOwnedRange = d_basisHandler->getLocallyOwnedRange();
          const size_type size =
            locallyOwnedRange.second - locallyOwnedRange.first;
          d_vector = std::make_shared<
            linearAlgebra::SerialVector<ValueType, memorySpace>>(
            size, d_linAlgOpContext, ValueType());
        }
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    Field<ValueType, memorySpace>::applyConstraintsParentToChild()
    {
      const Constraints<ValueType, memorySpace> &constraints =
        d_basisHandler->getConstraints(d_constraintsName);
      constraints.distributeParentToChild(*d_vector);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    Field<ValueType, memorySpace>::applyConstraintsChildToParent()
    {
      const Constraints<ValueType, memorySpace> &constraints =
        d_basisHandler->getConstraints(d_constraintsName);
      constraints.distributeChildToParent(*d_vector);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    const linearAlgebra::Vector<ValueType, memorySpace> &
    Field<ValueType, memorySpace>::getVector()
    {
      return *d_vector;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    const BasisHandler<ValueType,memorySpace> &
    Field<ValueType, memorySpace>::getBasisHandler() const
    {
      return *d_basisHandler;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename Field<ValueType, memorySpace>::iterator
    Field<ValueType, memorySpace>::begin()
    {
      return d_vector->begin();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename Field<ValueType, memorySpace>::const_iterator
    Field<ValueType, memorySpace>::begin() const
    {
      return d_vector->begin();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename Field<ValueType, memorySpace>::iterator
    Field<ValueType, memorySpace>::end()
    {
      return d_vector->end();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename Field<ValueType, memorySpace>::const_iterator
    Field<ValueType, memorySpace>::end() const
    {
      return d_vector->end();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    Field<ValueType, memorySpace>::updateGhostValues(
      const size_type communicationChannel /*= 0*/)
    {
      d_vector->updateGhostValues(communicationChannel);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    Field<ValueType, memorySpace>::accumulateAddLocallyOwned(
      const size_type communicationChannel /*= 0*/)
    {
      d_vector->accumulateAddLocallyOwned(communicationChannel);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    Field<ValueType, memorySpace>::updateGhostValuesBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_vector->updateGhostValuesBegin(communicationChannel);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    Field<ValueType, memorySpace>::updateGhostValuesEnd()
    {
      d_vector->updateGhostValuesEnd();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    Field<ValueType, memorySpace>::accumulateAddLocallyOwnedBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_vector->accumulateAddLocallyOwnedBegin();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    Field<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd()
    {
      d_vector->accumulateAddLocallyOwnedEnd();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    std::string
    Field<ValueType, memorySpace>::getConstraintsName() const
    {
      return d_constraintsName;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    const linearAlgebra::LinAlgOpContext<memorySpace> &
    Field<ValueType, memorySpace>::getLinAlgOpContext() const
    {
      return d_linAlgOpContext;
    }
  } // end of namespace basis
} // end of namespace dftefe
