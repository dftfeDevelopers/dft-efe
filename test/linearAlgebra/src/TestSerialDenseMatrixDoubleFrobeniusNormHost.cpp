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
#include <linearAlgebra/SerialVector.h>

int main () {
  double tol = 1.0e-12;

  const auto HOST = dftefe::utils::MemorySpace::HOST;
  typedef dftefe::linearAlgebra::blasLapack::BlasQueue<HOST> QUEUE;
  dftefe::size_type                                                   nRows = 5, nCols = 3;
  QUEUE                                                  queue;
  dftefe::linearAlgebra::SerialDenseMatrix<double, HOST> A(
    nRows, nCols, std::make_shared<QUEUE>(queue));

  double lo = -10.0, hi = 10.0;
  std::vector<double> AData(nRows*nCols, 0.0);
  for (dftefe::size_type i = 0; i < AData.size(); ++i) {
      AData[i] = lo + (hi - lo)*std::rand()/RAND_MAX;
      *(A.data() + i) = AData[i];
    }

  double vFrob = 0.0;
  for (const double &i : AData) {
      vFrob += i*i;
    }
  vFrob = std::sqrt(vFrob);

  double mFrob = A.frobeniusNorm();

  if (std::fabs(mFrob - vFrob) > tol)
    {
      std::string msg =
        "Mismatch of Frobenius norm of std::vector and dftefe::linearAlgebra::SerialDenseMatrix. "
        "std::vector Frobenius norm: " +
        std::to_string(vFrob) +
        " dftefe::linearAlgebra::SerialDenseMatrix Frobenius norm: " +
        std::to_string(mFrob);
      throw std::runtime_error(msg);
    }
}