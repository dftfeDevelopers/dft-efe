
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
#include <complex>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/SerialVector.h>

bool
absCompare(const std::complex<double> & a,
    const std::complex<double> & b)
{
  return (std::abs(a) < std::abs(b));
}

int
main()
{
  const dftefe::utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
  const double lo = -10.0;
  const double hi = 10.0;
  dftefe::size_type vSize = 3;
  const double tol = 1e-13;
  
  std::vector<std::complex<double>> zVecStd(vSize);
  for(auto & it : zVecStd)
  {
    it.real(lo + (hi-lo)*std::rand()/RAND_MAX);
    it.imag(lo + (hi-lo)*std::rand()/RAND_MAX);
  }
 
  double zVecStdL2Norm = 0.0;
  for(const auto & it : zVecStd)
    zVecStdL2Norm += std::abs(it)*std::abs(it);

  zVecStdL2Norm = std::sqrt(zVecStdL2Norm);
  double zVecStdLInfNorm = std::abs(*std::max_element(zVecStd.begin(), zVecStd.end(), absCompare));

  std::shared_ptr<dftefe::linearAlgebra::blasLapack::blasQueueType<Host>> queue=std::make_shared<dftefe::linearAlgebra::blasLapack::blasQueueType<Host>>();

  std::shared_ptr<dftefe::linearAlgebra::Vector<std::complex<double>, dftefe::utils::MemorySpace::HOST>> zVec
    = std::make_shared<dftefe::linearAlgebra::SerialVector<std::complex<double>, dftefe::utils::MemorySpace::HOST>>(vSize, 0,queue);

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            zVec->data(),
                                            zVecStd.data());

  const double zVecL2Norm = zVec->l2Norm();
  const double zVecLInfNorm = zVec->lInfNorm();
  if(std::fabs(zVecL2Norm-zVecStdL2Norm) > tol)
  { 
    std::string msg = "Mismatch of L2 norm of std::vector and dftefe::linearAlgebra::SerialVector. "
      "std::vector L2 norm: " + std::to_string(zVecStdL2Norm) + 
      " dftefe::linearAlgebra::SerialVector L2 norm: " + std::to_string(zVecL2Norm);
    throw std::runtime_error(msg);
  }
  
  if(std::fabs(zVecLInfNorm-zVecStdLInfNorm) > tol)
  { 
    std::string msg = "Mismatch of LInf norm of std::vector and dftefe::linearAlgebra::SerialVector. "
      "std::vector LInf norm: " + std::to_string(zVecStdLInfNorm) + 
      " dftefe::linearAlgebra::SerialVector LInf norm: " + std::to_string(zVecLInfNorm);
    throw std::runtime_error(msg);
  }
  
  return 0;
}
