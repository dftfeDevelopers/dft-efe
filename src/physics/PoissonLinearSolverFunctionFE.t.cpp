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
 * @author Bikash Kanungo
 */

#include <utils/Defaults.h>
namespace dftefe
{
  namespace physics
  {
    //
    // Constructor
    //
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      PoissonLinearSolverFunctionFE(
        const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>
          &feBasisHandler,
        const utils::FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &                                                  feBasisDataStorage,
        const linearAlgebra::Vector<ValueType, memorySpace> &b,
        const std::string                                    constraintsName,
        const linearAlgebra::PreconditionerType              pcType)
      : d_feBasisHandler(&feBasisHandler)
      , d_feBasisDataStorage(&feBasisDataStorage)
      , d_b(b)
      , d_constraintsName(constraintsName)
      , d_pcType(pcType)
      , d_x(b.getMPIPatternP2P(),
            b.getLinAlgOpContext(),
            utils::Types<ValueType>::zero)
    {}

  } // end of namespace physics
} // end of namespace dftefe
