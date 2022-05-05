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
namespace dftefe
{
  namespace basis
  {
    template <typename ValueType, utils::MemorySpace memorySpace>
    Field<ValueType, memorySpace>::Field(
      std::shared_ptr<const BasisHandler> basisHandler,
      const std::string                   constraintsName)
      : d_basisHandler(basisHandler)
      , d_constraintsName(constraintsName)
      , d_constraints(basisHandler->getConstraints(constraintsName))
    {
      reinit(basisHandler, constraintsName);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    Field<ValueType, memorySpace>::reinit(
      std::shared_ptr<const BasisHandler> basisHandler,
      const std::string                   constraintsName)
    {
      d_basisHandler     = basisHandler;
      d_constraintsName  = constraintsName;
      d_constraints      = &(basisHandler->getConstraints(constraintsName));
      auto mpiPatternP2P = basisHandler->getMPIPatternP2P(constraintsName);
      //
      // Since it is a field with a single component, block size for
      // MPICommunicatorP2P is set to 1
      //
      const size_type blockSize = 1;
      d_mpiCommunicatorP2P =
        std::make_shared<utils::MPICommunicatorP2P<ValueType, memorySpace>>(
          mpiPatternP2P, blockSize);
    }

  } // end of namespace basis
} // end of namespace dftefe
