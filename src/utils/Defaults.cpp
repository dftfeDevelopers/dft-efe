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
  namespace utils
  {
    const int                  Types<int>::zero                = 0;
    const unsigned int         Types<unsigned int>::zero       = 0;
    const short int            Types<short int>::zero          = 0;
    const unsigned short int   Types<unsigned short int>::zero = 0;
    const long int             Types<long int>::zero           = 0;
    const unsigned long int    Types<unsigned long int>::zero  = 0;
    const float                Types<float>::zero              = 0.0;
    const double               Types<double>::zero             = 0.0;
    const std::complex<double> Types<std::complex<double>>::zero =
      std::complex<double>(0.0, 0.0);
    const std::complex<float> Types<std::complex<float>>::zero =
      std::complex<float>(0.0, 0.0);
    const char        Types<char>::zero        = (char)0;
    const std::string Types<std::string>::zero = "";
  } // end of namespace utils
} // end of namespace dftefe
