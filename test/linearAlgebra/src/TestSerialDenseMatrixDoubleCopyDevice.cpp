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
  const auto DEVICE = dftefe::utils::MemorySpace::DEVICE;
  typedef dftefe::linearAlgebra::blasLapack::blasQueueType<DEVICE> QUEUE;
  typedef dftefe::linearAlgebra::SerialDenseMatrix<double, DEVICE> MATRIX;

  int    nRows = 5, nCols = 3;
  int device;
  dftefe::utils::deviceGetDevice(&device);
  std::shared_ptr<QUEUE> queue = std::make_shared<QUEUE>(device, 0);
  MATRIX A(nRows, nCols, queue, 0);
  MATRIX B(nRows, nCols, queue, 0);
  MATRIX C(nRows, nCols, queue, 0);

  double              lo = -10.0, hi = 10.0;
  std::vector<double> AData(nRows * nCols, 0.0);
  for (int i = 0; i < AData.size(); ++i)
    {
      AData[i]        = lo + (hi - lo) * std::rand() / RAND_MAX;
    }

  dftefe::utils::MemoryTransfer<DEVICE, HOST>::copy(nRows*nCols, A.data(), AData.data());
  B.copyFrom(A);
  std::vector<double> resutBuffer(nRows*nCols, 0.0);
  dftefe::utils::MemoryTransfer<HOST, DEVICE>::copy(nRows*nCols, resutBuffer.data(), B.data());
  for (dftefe::size_type i = 0; i < AData.size(); ++i)
    {
      if (std::fabs(AData[i] - *(resutBuffer.data() + i)) > tol)
        {
          std::string msg =
            "At index " + std::to_string(i) + ": (" +
            std::to_string(i % nRows) + ", " + std::to_string(i / nCols) + ")" +
            " mismatch of entries after doing dftefe::linearAlgebra::SerialDenseMatrix::copyFrom for double at Device. "
            " dftefe::linearAlgebra::SerialDenseMatrix value: (" +
            std::to_string(*(resutBuffer.data() + i)) + "), reference value: (" +
            std::to_string(AData[i]) + ")";
          throw std::runtime_error(msg);
        }
    }

  B.copyTo(C);
  resutBuffer.assign(nRows*nCols, 0.0);
  dftefe::utils::MemoryTransfer<HOST, DEVICE>::copy(nRows*nCols, resutBuffer.data(), C.data());
  for (dftefe::size_type i = 0; i < AData.size(); ++i)
    {
      if (std::fabs(AData[i] - *(resutBuffer.data() + i)) > tol)
        {
          std::string msg =
            "At index " + std::to_string(i) + ": (" +
            std::to_string(i % nRows) + ", " + std::to_string(i / nCols) + ")" +
            " mismatch of entries after doing dftefe::linearAlgebra::SerialDenseMatrix::copyTo for double at Device. "
            " dftefe::linearAlgebra::SerialDenseMatrix value: (" +
            std::to_string(*(resutBuffer.data() + i)) + "), reference value: (" +
            std::to_string(AData[i]) + ")";
          throw std::runtime_error(msg);
        }
    }

  return 0;
}