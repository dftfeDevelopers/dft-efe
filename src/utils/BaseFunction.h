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

#ifndef dftefeBaseFunction_h
#define dftefeBaseFunction_h

#include "Point.h"
#include <complex>

namespace dftefe
{
  namespace utils
  {
    class BaseFunction
    {
    public:
      virtual void
      getValue(const dftefe::utils::Point &realPoint, double outputVal) = 0;

      virtual void
      getValue(const std::vector<dftefe::utils::Point> &realPoint,
               std::vector<double> &                    outputVal) = 0;

      virtual void
      getValue(const dftefe::utils::Point &realPoint,
               std::complex<double>        outputVal) = 0;

      virtual void
      getValue(const std::vector<dftefe::utils::Point> &realPoint,
               std::vector<std::complex<double>> &      outputVal) = 0;
    };

  } // namespace utils
} // namespace dftefe

#endif // dftefeBaseFunction_h
