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

#include <complex>
#include <fstream>
#include "linearAlgebra/Vector.h"

int
main()
{
  std::string filename = "TestVectorAggregate2.out";
  std::ofstream fout(filename);

  unsigned int vSize = 10;
  // test double
  // Refs
  std::vector<double> dAddRef = {
    11.78, 36.16, 7.20, 34.73, 20.64, 40.75, 12.83, 24.89, 10.16, 30.96};
  std::vector<double> dAddEqRef = {
    5.4, 14.4, 3.2, 14.3, 8.0, 16.5, 7.7, 11.9, 5.6, 10.4};
  std::vector<double> dMinusEqRef = {
    0.8, -3.8, 0.2, -2.3, -2.8, -3.5, 6.1, 3.1, 3.4, -8.8};


  std::vector<double> dVecStdA = {
    3.1, 5.3, 1.7, 6.0, 2.6, 6.5, 6.9, 7.5, 4.5, 0.8};
  std::vector<double> dVecStdB = {
    2.3, 9.1, 1.5, 8.3, 5.4, 10.0, 0.8, 4.4, 1.1, 9.6};


  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::DEVICE>
    dVecA(vSize, 0), dVecC(vSize, 0), dTemp(vSize, 0);

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            dVecA.data(),
                                            dVecStdA.data());

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            dTemp.data(),
                                            dVecStdB.data());

  // test copy constructor
  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::DEVICE>
    dVecB(dTemp);

  std::vector<double> dPrintCache(vSize, 0.0);
  // test add function
  double dAlpha = 1.5, dBeta = 3.1;
  dftefe::linearAlgebra::add(dAlpha, dVecA, dBeta, dVecB, dVecC);
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::DEVICE>::copy(vSize,
                                              dPrintCache.data(),
                                              dVecC.data());
  fout << "double add ";
  for (auto i : dPrintCache)
    {
      fout << i << ", ";
    }
  fout << std::endl;


  // test +=
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            dVecA.data(),
                                            dVecStdA.data());

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            dVecB.data(),
                                            dVecStdB.data());

  dVecA += dVecB;
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::DEVICE>::copy(vSize,
                                            dPrintCache.data(),
                                            dVecA.data());
  fout << "double += ";
  for (auto i : dPrintCache)
    {
      fout << i << ", ";
    }
  fout << std::endl;
  //  for (auto i : dAddEqRef)
  //    {
  //      fout << i << ", ";
  //    }
  //  fout << std::endl << std::endl;

  // test -=
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            dVecA.data(),
                                            dVecStdA.data());

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            dVecB.data(),
                                            dVecStdB.data());

  dVecA -= dVecB;
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::DEVICE>::copy(vSize,
                                            dPrintCache.data(),
                                            dVecA.data());
  fout << "double -= ";
  for (auto i : dPrintCache)
    {
      fout << i << ", ";
    }
  fout << std::endl;
  //  for (auto i : dMinusEqRef)
  //    {
  //      fout << i << ", ";
  //    }
  //  fout << std::endl << std::endl;


  // test complex
  using namespace std::complex_literals;
  // Refs
  std::vector<std::complex<double>> zAddRef     = {-1.54 + 16.80i,
                                               -1.79 + 20.79i,
                                               20.22 + 15.64i,
                                               20.37 + 20.25i,
                                               -18.68 + 6.62i,
                                               46.55 + 16.70i,
                                               8.30 + 42.53i,
                                               -11.33 + -21.53i,
                                               16.17 + 55.78i,
                                               9.22 + 8.53i};
  std::vector<std::complex<double>> zAddEqRef   = {2.00 + 4.40i,
                                                 -4.60 + -1.10i,
                                                 12.40 + 6.30i,
                                                 14.00 + 7.30i,
                                                 7.20 + -1.70i,
                                                 9.00 + 0.50i,
                                                 8.30 + 11.50i,
                                                 1.70 + 5.30i,
                                                 11.30 + 15.80i,
                                                 16.80 + 1.80i};
  std::vector<std::complex<double>> zMinusEqRef = {2.00 + 0.80i,
                                                   10.80 + -6.30i,
                                                   -4.00 + 3.30i,
                                                   -3.40 + 4.50i,
                                                   5.60 + 15.70i,
                                                   1.00 + -11.10i,
                                                   3.10 + 0.50i,
                                                   -14.30 + 8.10i,
                                                   2.70 + -1.20i,
                                                   -1.40 + 14.20i};


  std::vector<std::complex<double>> zVecStdA = {2.0 + 2.6i,
                                                3.1 - 3.7i,
                                                4.2 + 4.8i,
                                                5.3 + 5.9i,
                                                6.4 + 7.0i,
                                                5.0 - 5.3i,
                                                5.7 + 6.0i,
                                                -6.3 + 6.7i,
                                                7.0 + 7.3i,
                                                7.7 + 8.0i};

  std::vector<std::complex<double>> zVecStdB = {-0.0 + 1.8i,
                                                -7.7 + 2.6i,
                                                8.2 + 1.5i,
                                                8.7 + 1.4i,
                                                0.8 - 8.7i,
                                                4.0 + 5.8i,
                                                2.6 + 5.5i,
                                                8.0 - 1.4i,
                                                4.3 + 8.5i,
                                                9.1 - 6.2i};

  dftefe::linearAlgebra::Vector<std::complex<double>,
                                dftefe::utils::MemorySpace::DEVICE>
    zVecA(vSize, 0), zTemp(vSize, 0), zVecC(vSize, 0);

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            zVecA.data(),
                                            zVecStdA.data());

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            zTemp.data(),
                                            zVecStdB.data());

  // test copy constructor
  dftefe::linearAlgebra::Vector<std::complex<double>,
                                dftefe::utils::MemorySpace::DEVICE>
    zVecB(zTemp);

  // test add function
  std::complex<double> zAlpha = 2.0 + 3.1i, zBeta = 3.0 - 1.4i;
  dftefe::linearAlgebra::add(zAlpha, zVecA, zBeta, zVecB, zVecC);

  std::vector<std::complex<double>> zPrintCache(vSize, 0.0);
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::DEVICE>::copy(vSize,
                                            zPrintCache.data(),
                                            zVecC.data());
  fout << "complex<double> add ";
  for (auto i : zPrintCache)
    {
      fout << i << ", ";
    }
  fout << std::endl;
  //  for (auto i : zAddRef)
  //    {
  //      fout << i << ", ";
  //    }
  //  fout << std::endl << std::endl;


  // test +=
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            zVecA.data(),
                                            zVecStdA.data());

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            zVecB.data(),
                                            zVecStdB.data());

  zVecA += zVecB;
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::DEVICE>::copy(vSize,
                                            zPrintCache.data(),
                                            zVecA.data());
  fout << "complex<double> += ";
  for (auto i : zPrintCache)
    {
      fout << i << ", ";
    }
  fout << std::endl;
  //  for (auto i : zAddEqRef)
  //    {
  //      fout << i << ", ";
  //    }
  //  fout << std::endl << std::endl;

  // test -=
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            zVecA.data(),
                                            zVecStdA.data());

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::DEVICE,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            zVecB.data(),
                                            zVecStdB.data());

  zVecA -= zVecB;
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::DEVICE>::copy(vSize,
                                            zPrintCache.data(),
                                            zVecA.data());
  fout << "complex<double> -= ";
  for (auto i : zPrintCache)
    {
      fout << i << ", ";
    }
  fout << std::endl;
  //  for (auto i : zMinusEqRef)
  //    {
  //      fout << i << ", ";
  //    }
  //  fout << std::endl << std::endl;

  return 0;
}
