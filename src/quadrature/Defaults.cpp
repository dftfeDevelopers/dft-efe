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
#include <quadrature/Defaults.h>
#include <limits.h>

namespace dftefe
{
  namespace quadrature
  {
    /**
     * @brief Setting all the QuadratureRuleAdaptiveDefaults
     */
    const double QuadratureRuleAdaptiveDefaults::SMALLEST_CELL_VOLUME = 1e-12;
    const unsigned int QuadratureRuleAdaptiveDefaults::MAX_RECURSION  = 1000;
    const double
      QuadratureRuleAdaptiveDefaults::INTEGRAL_THRESHOLDS_NORMALIZATION = 1e-16;
    /**
     * @brief Setting all the QuadratureRuleAttributesDefaults
     */
    const size_type   QuadratureRuleAttributesDefaults::NUM_1D_POINTS = 0;
    const std::string QuadratureRuleAttributesDefaults::TAG = std::string();

    /**
     * @brief Setting all the QuadratureRuleGaussSubdividedDefaults
     */
    const size_type
      QuadratureRuleGaussSubdividedDefaults::NUM_CELLS_FOR_ADAPTIVE_REFERENCE =
        5;

  } // end of namespace quadrature
} // end of namespace dftefe
