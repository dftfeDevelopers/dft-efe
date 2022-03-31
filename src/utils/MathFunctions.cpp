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
namespace dftefe
{
  namespace utils
  {
    namespace mathFunctions
    {
      int
      intPow(int base, unsigned int e)
      {
        int result = 1;
        for (;;)
          {
            if (e & 1)
              result *= base;
            e >>= 1;
            if (!e)
              break;
            base *= base;
          }
        return result;
      }

      size_type
      sizeTypePow(size_type base, size_type e)
      {
        size_type result = 1;
        for (;;)
          {
            if (e & 1)
              result *= base;
            e >>= 1;
            if (!e)
              break;
            base *= base;
          }
        return result;
      }
    } // namespace mathFunctions
  }   // namespace utils
} // namespace dftefe
