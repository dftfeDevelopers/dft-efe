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
    template <typename ValueTypeBasisCoeff, utils::MemorySpace memorySpace>
    class Field
    {
    public:
      //
      // typedefs
      //
      using value_type =
        typename linearAlgebra::Vector<ValueTypeBasisCoeff,
                                       memorySpace>::value_type;
      using pointer   = typename linearAlgebra::Vector<ValueTypeBasisCoeff,
                                                     memorySpace>::pointer;
      using reference = typename linearAlgebra::Vector<ValueTypeBasisCoeff,
                                                       memorySpace>::reference;
      using const_reference =
        typename linearAlgebra::Vector<ValueTypeBasisCoeff,
                                       memorySpace>::const_reference;
      using iterator = typename linearAlgebra::Vector<ValueTypeBasisCoeff,
                                                      memorySpace>::iterator;
      using const_iterator =
        typename linearAlgebra::Vector<ValueTypeBasisCoeff,
                                       memorySpace>::const_iterator;

      Field(
        std::shared_ptr<const BasisHandler<ValueTypeBasisCoeff, memorySpace>>
                                                     basishandler,
        const std::string                            constraintsName,
        linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext);

      ~Field() = default;

      void
      reinit(
        std::shared_ptr<const BasisHandler<ValueTypeBasisCoeff, memorySpace>>
                                                     basisHandler,
        const std::string                            constraintsName,
        linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext);

      void
      applyConstraintsParentToChild();

      void
      applyConstraintsChildToParent();

      const linearAlgebra::Vector<ValueTypeBasisCoeff, memorySpace> &
      getVector();

      const BasisHandler<ValueTypeBasisCoeff, memorySpace> &
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
      accumulateAddLocallyOwnedBegin(const size_type communicationChannel = 0);

      void
      accumulateAddLocallyOwnedEnd();

      linearAlgebra::LinAlgOpContext<memorySpace> &
      getLinAlgOpContext() const;

      std::string
      getConstraintsName() const;

    private:
      std::string                                  d_constraintsName;
      linearAlgebra::LinAlgOpContext<memorySpace> *d_linAlgOpContext;
      std::shared_ptr<const BasisHandler<ValueTypeBasisCoeff, memorySpace>>
        d_basisHandler;
      std::shared_ptr<linearAlgebra::Vector<ValueTypeBasisCoeff, memorySpace>>
        d_vector;
    }; // end of Field
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/Field.t.cpp>
#endif // dftefeField_h
