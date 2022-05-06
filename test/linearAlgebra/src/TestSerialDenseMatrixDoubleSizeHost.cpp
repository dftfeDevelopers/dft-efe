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

#include <stdexcept>
#include <linearAlgebra/SerialDenseMatrix.h>

int
main()
{
  double tol = 1.0e-12;

  const auto HOST = dftefe::utils::MemorySpace::HOST;
  typedef dftefe::linearAlgebra::blasLapack::BlasQueue<HOST> QUEUE;
  typedef dftefe::linearAlgebra::SerialDenseMatrix<double, HOST> MATRIX;

  dftefe::size_type   nRows = 5, nCols = 3;
  QUEUE  queue;
  MATRIX A(nRows, nCols, std::make_shared<QUEUE>(queue), 0);

  if ((A.getGlobalRows() != nRows) || (A.getGlobalCols() != nCols))
    {
      std::string msg =
        "globalRows and globalCols do not match for dftefe::linearAlgebra::SerialDenseMatrix::copyTo for double on Host. given dimensions: (" + std::to_string(nRows) + ", " + std::to_string(nCols) + "), matrix dimension: (" +
        std::to_string(A.getGlobalRows()) + ", " + std::to_string(A.getGlobalCols()) + ")";
      throw std::runtime_error(msg);
    }

  dftefe::size_type nRow_t = 0, nCol_t = 0;
  A.getGlobalSize(nRow_t, nCol_t);
  if ((nRow_t != nRows) || (nCol_t != nCols))
    {
      std::string msg =
        "getGlobalSize function does not match for dftefe::linearAlgebra::SerialDenseMatrix::copyTo for double on Host. given dimensions: (" + std::to_string(nRows) + ", " + std::to_string(nCols) + "), matrix dimension: (" +
        std::to_string(nRow_t) + ", " + std::to_string(nCol_t) + ")";
      throw std::runtime_error(msg);
    }
}