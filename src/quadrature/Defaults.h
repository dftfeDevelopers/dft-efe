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

#ifndef dftefeQuadratureDefaults_h
#define dftefeQuadratureDefaults_h

#include <utils/TypeConfig.h>
#include <string>

namespace dftefe
{
  namespace quadrature
  {
    class QuadratureRuleAdaptiveDefaults
    {
    public:
      //
      // smallest cell volume to recurse to
      //
      static const double SMALLEST_CELL_VOLUME;

      //
      // maximum recursion of divisions for adaptive quad
      //
      static const unsigned int MAX_RECURSION;

      //
      // normalization of Integral threhold criteria to avoid 0/0 form
      //
      static const double INTEGRAL_THRESHOLDS_NORMALIZATION;

    }; // end of class QuadratureRuleAdaptiveDefaults

    class QuadratureRuleAttributesDefaults
    {
    public:
      //
      // default 1d points for checking isCartesianTensorStructured
      //
      static const size_type NUM_1D_POINTS;

      //
      // default string tag
      //
      static const std::string TAG;

    }; // end of class QuadratureRuleAttributesDefaults

    class QuadratureRuleGaussSubdividedDefaults
    {
    public:
      //
      // NUM_CELLS_FOR_ADAPTIVE_REFERENCE
      //
      static const size_type NUM_CELLS_FOR_ADAPTIVE_REFERENCE;

    }; // end of class QuadratureRuleGaussSubdividedDefaults


  } // end of namespace quadrature
} // end of namespace dftefe
#endif // dftefeQuadratureDefaults_h
