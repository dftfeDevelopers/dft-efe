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

#ifndef dftefePreconditionerFactory_h
#define dftefePreconditionerFactory_h

#include <utils/MemorySpaceType.h>
#include <utils/TypeConfig.h>
#include <linearAlgebra/SolverTypes.h>
#include <linearAlgebra/Preconditioner.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *@brief Factory class create preconditioner objects based on input parameter(s)
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying preconditioner
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     *  on which the preconditioner will act.
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNES, etc.) in which the data of the preconditioner
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    class PreconditionerFactory
    {
    public:
      /**
       *@brief Default Constructor
       *
       */
      PreconditionerFactory() = default;

      /**
       *@brief Default Destructor
       *
       */
      ~PreconditionerFactory() = default;

      static std::shared_ptr<
        Preconditioner<ValueTypeOperator, ValueTypeOperand, memorySpace>>
      create(const SolverTypes::PreconditionerType &pcType) const;
    };

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/PreconditionerFactory.t.cpp>
#endif // dftefePreconditionerFactory_h
