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
 * @author Ian C. Lin.
 */

#include <iostream>
#include "Vector.h"

int
main()
{
  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::DEVICE> a(
    4, 0);
  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::DEVICE> b(
    4, 0);
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  a[3] = 4;

  b[0] = 5;
  b[1] = 6;
  b[2] = 7;
  b[3] = 8;

  a += b;

  //  for (int i = 0; i < 4; ++i) {
  //      std::cout << a[i] << ", ";
  //    }
  //  std::cout << std::endl;
}