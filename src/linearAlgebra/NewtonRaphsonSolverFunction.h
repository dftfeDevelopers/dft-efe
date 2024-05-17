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
 * @author Avirup Sircar
 */

#ifndef dftefeNewtonRaphsonSolverFunction_h
#define dftefeNewtonRaphsonSolverFunction_h

#include <utils/TypeConfig.h>
#include <linearAlgebra/LinearAlgebraTypes.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType>
    class NewtonRaphsonSolverFunction
    {
    public:
      virtual ~NewtonRaphsonSolverFunction() = default;

      virtual const ValueType
      getValue(ValueType &x) const = 0;

      virtual const ValueType
      getForce(ValueType &x) const = 0;

      virtual void
      setSolution(const ValueType &x) = 0;

      virtual void
      getSolution(ValueType &solution) = 0;

      virtual const ValueType &
      getInitialGuess() const = 0;

    }; // end of class NewtonRaphsonSolverFunction
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeNewtonRaphsonSolverFunction_h
