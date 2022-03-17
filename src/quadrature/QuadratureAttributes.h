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
    enum class QuadratureFamily
    {
      GAUSS; GLL;
    };

    enum class QuadratureRuleType
    {

      // Gauss quadrature rules
      GAUSS_1,
      GAUSS_2,
      GAUSS_3,
      GAUSS_4,
      GAUSS_5,
      GAUSS_6,
      GAUSS_7,
      GAUSS_8,
      GAUSS_9,
      GAUSS_10,
      GAUSS_11,
      GAUSS_12,
      GAUSS_VARIABLE,

      // Gauss-Legendre-Lobatta quadrature rules
      GLL_1,
      GLL_2,
      GLL_3,
      GLL_4,
      GLL_5,
      GLL_6,
      GLL_7,
      GLL_8,
      GLL_9,
      GLL_10,
      GLL_11,
      GLL_12,
      GLL_VARIABLE,

      // Adaptive quadrature rule
      ADAPTIVE
    };
    /**
     * @brief Class to store the attributes of a quad point, such as
     * the cell Id it belongs, the quadPointId within the cell it belongs to,
     * and the quadrature rule (defined by quadratureRuleId) it is part of.
     */
    class QuadratureAttributes
    {
    public:
      size_type          cellId = 0;
      QuadratureRuleType quadType;
      size_type          quadPointId = 0;
    }; // end of class QuadratureAttributes

    //
    // helper functions
    //
    size_type
    get1DQuadNumPoints(const QuadratureRuleType quadRuleType);
    QuadratureFamily
    getQuadratureFamily(const QuadratureRuleType quadRuleType);
  } // end of namespace quadrature
} // end of namespace dftefe
#endif
