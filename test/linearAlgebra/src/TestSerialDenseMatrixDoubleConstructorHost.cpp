/******************************************************************************
 * Copyright (c) 2022.                                                        *
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
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <linearAlgebra/SerialDenseMatrix.h>

int
main()
{
  double tol = 1.0e-12;

  const auto HOST = dftefe::utils::MemorySpace::HOST;
  typedef dftefe::linearAlgebra::blasLapack::blasQueueType<HOST> QUEUE;
  typedef dftefe::linearAlgebra::SerialDenseMatrix<double, HOST> MATRIX;

  int    nRows = 5, nCols = 3;
  QUEUE  queue;
  MATRIX A(nRows, nCols, std::make_shared<QUEUE>(queue), 0);

  double              lo = -10.0, hi = 10.0;
  std::vector<double> AData(nRows * nCols, 0.0);
  for (int i = 0; i < AData.size(); ++i)
    {
      AData[i]        = lo + (hi - lo) * std::rand() / RAND_MAX;
      *(A.data() + i) = AData[i];
    }

  MATRIX B(A);

  for (unsigned int i = 0; i < AData.size(); ++i)
    {
      if (std::fabs(AData[i] - *(B.data() + i)) > tol)
        {
          std::string msg =
            "At index " + std::to_string(i) + ": (" +
            std::to_string(i % nRows) + ", " + std::to_string(i / nCols) + ")" +
            " mismatch of entries after doing dftefe::linearAlgebra::SerialDenseMatrix copy constructor for double at Host. "
            " dftefe::linearAlgebra::SerialDenseMatrix value: (" +
            std::to_string(*(B.data() + i)) + "), reference value: (" +
            std::to_string(AData[i]) + ")";
          throw std::runtime_error(msg);
        }
    }

  MATRIX C;
  C = A;
  for (unsigned int i = 0; i < AData.size(); ++i)
    {
      if (std::fabs(AData[i] - *(C.data() + i)) > tol)
        {
          std::string msg =
            "At index " + std::to_string(i) + ": (" +
            std::to_string(i % nRows) + ", " + std::to_string(i / nCols) + ")" +
            " mismatch of entries after doing dftefe::linearAlgebra::SerialDenseMatrix assignment operator for double at Host. "
            " dftefe::linearAlgebra::SerialDenseMatrix value: (" +
            std::to_string(*(C.data() + i)) + "), reference value: (" +
            std::to_string(AData[i]) + ")";
          throw std::runtime_error(msg);
        }
    }

  MATRIX D;
  D = std::move(A);

  for (unsigned int i = 0; i < AData.size(); ++i)
    {
      if (std::fabs(AData[i] - *(D.data() + i)) > tol)
        {
          std::string msg =
            "At index " + std::to_string(i) + ": (" +
            std::to_string(i % nRows) + ", " + std::to_string(i / nCols) + ")" +
            " mismatch of entries after doing dftefe::linearAlgebra::SerialDenseMatrix move assignment operator for double at Host. "
            " dftefe::linearAlgebra::SerialDenseMatrix value: (" +
            std::to_string(*(D.data() + i)) + "), reference value: (" +
            std::to_string(AData[i]) + ")";
          throw std::runtime_error(msg);
        }
    }


  MATRIX E = std::move(D);

  for (unsigned int i = 0; i < AData.size(); ++i)
    {
      if (std::fabs(AData[i] - *(E.data() + i)) > tol)
        {
          std::string msg =
            "At index " + std::to_string(i) + ": (" +
            std::to_string(i % nRows) + ", " + std::to_string(i / nCols) + ")" +
            " mismatch of entries after doing dftefe::linearAlgebra::SerialDenseMatrix move constructor operator for double at Host. "
            " dftefe::linearAlgebra::SerialDenseMatrix value: (" +
            std::to_string(*(E.data() + i)) + "), reference value: (" +
            std::to_string(AData[i]) + ")";
          throw std::runtime_error(msg);
        }
    }

  return 0;
}