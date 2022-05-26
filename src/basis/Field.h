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

#ifndef dftefeField_h
#define dftefeField_h

#include <basis/BasisHandler.h>
#include <basis/Constraints.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <utils/MemorySpaceType.h>
#include <utils/MPICommunicatorP2P.h>
#include <string>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * An abstract class to handle a physical field, such as
     * Kohn-Sham orbital, electrostatic potential, etc
     */
    template <typename ValueType, utils::MemorySpace memorySpace>
    class Field
    {
    public:
      //
      // typedefs
      //
      using value_type =
        typename linearAlgebra::Vector<ValueType, memorySpace>::value_type;
      using pointer =
        typename linearAlgebra::Vector<ValueType, memorySpace>::pointer;
      using reference =
        typename linearAlgebra::Vector<ValueType, memorySpace>::reference;
      using const_reference =
        typename linearAlgebra::Vector<ValueType, memorySpace>::const_reference;
      using iterator =
        typename linearAlgebra::Vector<ValueType, memorySpace>::iterator;
      using const_iterator =
        typename linearAlgebra::Vector<ValueType, memorySpace>::const_iterator;

      Field(std::shared_ptr<const BasisHandler<memorySpace>> basisHandler,
            const std::string                                constraintsName,
            const linearAlgebra::LinAlgOpContext &           linAlgOpContext);

      ~Field() = default;

      reinit(std::shared_ptr<const BasisHandler<memorySpace>> basisHandler,
             const std::string                                constraintsName,
             const linearAlgebra::LinAlgOpContext &           linAlgOpContext);

      void
      applyConstraintsParentToChild();

      void
      applyConstraintsChildToParent();

      const Vector<ValueType, memorySpace> &
      getVector();

      const BasisHandler<memorySpace> &
      getBasisHandler() const;

      iterator
      begin();

      const_iterator
      begin() const;

      iterator
      end();

      const_iterator
      end() const;
      
      void
      updateGhostValues(const size_type communicationChannel = 0);

      void
      accumulateAddLocallyOwned(const size_type communicationChannel = 0);
      
      void
      updateGhostValuesBegin(const size_type communicationChannel = 0);

      void
      updateGhostValuesEnd();

      void
      accumulateAddLocallyOwnedBegin(
        const size_type communicationChannel = 0);

      void
      accumulateAddLocallyOwnedEnd();

      const linearAlgebra::LinAlgOpContext &
	getLinAlgContext() const;
    private:
      const std::string                                d_constraintsName;
      linearAlgebra::LinAlgOpContext                   d_linAlgOpContext;
      std::shared_ptr<const BasisHandler<memorySpace>> d_basisHandler;
      std::shared_ptr < linearAlgebra::Vector<ValueType, memorySpace> d_vector;
    }; // end of Field
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/Field.t.cpp>
#endif // dftefeField_h
