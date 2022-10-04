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

#include <utils/Exceptions.h>
#include <string>

namespace dftefe
{
  namespace utils
  {
    template <typename T>
    RandNumGen<T>::RandNumGen(T min, T max)
    {
      std::string msg =
        "Invalid template parameter used in "
        "utils::RandNumGen. The allowable values of the template parameters "
        "are: short, int, long, long long, unsigned short, unsigned int, "
        "unsigned long, unsigned long long, float, double, long double, "
        "std:complex<float>, std::complex<double>, std::complex<long double.";
      throwException<InvalidArgument>(false, msg);
    }

    template <typename T>
    T
    RandNumGen<T>::generate()
    {
      std::string msg =
        "Invalid template parameter used in "
        "utils::RandNumGen. The allowable values of the template parameters "
        "are: short, int, long, long long, unsigned short, unsigned int, "
        "unsigned long, unsigned long long, float, double, long double, "
        "std:complex<float>, std::complex<double>, std::complex<long double.";
      throwException<InvalidArgument>(false, msg);
      return T();
    }
  } // end of namespace utils
} // end of namespace dftefe
