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

#ifndef dftefeQuadratureAttributes_h
#define dftefeQuadratureAttributes_h

#include <utils/TypeConfig.h>
namespace dftefe
{
  namespace quadrature
  {
    /**
     * @brief Class to store the attributes of a quad point, such as
     * the cell Id it belongs, the quadPointId within the cell it belongs to,
     * and the quadrature rule (defined by quadratureRuleId) it is part of.
     */
    class QuadratureAttributes
    {
    public:
      size_type cellId      = 0;
      size_type quadRuleId  = 0;
      size_type quadPointId = 0;
    }; // end of class QuadratureAttributes

  } // end of namespace quadrature
} // end of namespace dftefe
#endif
