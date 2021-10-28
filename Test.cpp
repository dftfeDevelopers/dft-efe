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
#ifndef DFTEFE_WITH_DEVICE
  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::HOST> a(4,
                                                                            0);
  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::HOST> b(4,
                                                                            0);
#elif DFTEFE_WITH_DEVICE
  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::DEVICE> a(
    4, 0);
  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::DEVICE> b(
    4, 0);
#endif

  std::vector<double> a_h(4, 0), b_h(4, 0), c_h(4, 0);
  a_h[0] = 1;
  a_h[1] = 2;
  a_h[2] = 3;
  a_h[3] = 4;

  b_h[0] = 5;
  b_h[1] = 6;
  b_h[2] = 7;
  b_h[3] = 8;


#ifndef DFTEFE_WITH_DEVICE
  dftefe::utils::MemoryTransfer<
    double,
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::HOST>::copy(4, a.data(), a_h.data());
  dftefe::utils::MemoryTransfer<
    double,
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::HOST>::copy(4, b.data(), b_h.data());
#elif DFTEFE_WITH_DEVICE
  dftefe::utils::MemoryTransfer<
    double,
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(4, a.data(), a_h.data());
  dftefe::utils::MemoryTransfer<
    double,
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(4, b.data(), b_h.data());
#endif

  a += b;

  for (int i = 0; i < 4; ++i)
    {
      std::cout << c_h[i] << ", ";
    }
  std::cout << std::endl;

#ifndef DFTEFE_WITH_DEVICE
  dftefe::utils::MemoryTransfer<
    double,
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::HOST>::copy(4, c_h.data(), a.data());
#elif DFTEFE_WITH_DEVICE
  dftefe::utils::MemoryTransfer<
    double,
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::DEVICE>::copy(4, c_h.data(), a.data());
#endif
  for (int i = 0; i < 4; ++i)
    {
      std::cout << c_h[i] << ", ";
    }
  std::cout << std::endl;
}