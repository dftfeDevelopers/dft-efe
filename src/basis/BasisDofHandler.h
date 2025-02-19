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

#ifndef dftefeBasisDofHandler_h
#define dftefeBasisDofHandler_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * An abstract class to handle basis related operations, such as
     * evaluating the value and gradients of any basis function at a
     * point.
     */
    class BasisDofHandler
    {
    public:
      virtual ~BasisDofHandler() = default;
      virtual double
      getBasisFunctionValue(const size_type     basisId,
                            const utils::Point &point) const = 0;
      virtual std::vector<double>
      getBasisFunctionDerivative(const size_type     basisId,
                                 const utils::Point &point,
                                 const size_type derivativeOrder = 1) const = 0;

    }; // end of BasisDofHandler
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeBasisDofHandler_h
