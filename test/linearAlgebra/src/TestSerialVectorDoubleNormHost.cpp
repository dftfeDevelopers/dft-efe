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
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/SerialVector.h>

int
main()
{
  const dftefe::utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
  const double lo = -10.0;
  const double hi = 10.0;
  unsigned int vSize = 3;
  const double tol = 1e-13;
  
  // test double
  std::vector<double> dVecStd(vSize);
  for(auto & it : dVecStd)
    it = lo + (hi-lo)*std::rand()/RAND_MAX;
 
  double dVecStdL2Norm = 0.0;
  for(const auto & it : dVecStd)
    dVecStdL2Norm += it*it;

  dVecStdL2Norm = std::sqrt(dVecStdL2Norm);
  double dVecStdLInfNorm = *std::max_element(dVecStd.begin(), dVecStd.end());

  std::shared_ptr<dftefe::linearAlgebra::blasLapack::blasQueueType<Host>> queue=std::make_shared<dftefe::linearAlgebra::blasLapack::blasQueueType<Host>>();

  std::shared_ptr<dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::HOST>> dVec
    = std::make_shared<dftefe::linearAlgebra::SerialVector<double, dftefe::utils::MemorySpace::HOST>>(vSize, 0,queue);

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            dVec->data(),
                                            dVecStd.data());

  const double dVecL2Norm = dVec->l2Norm();
  const double dVecLInfNorm = dVec->lInfNorm();
  if(std::fabs(dVecL2Norm-dVecStdL2Norm) > tol)
  { 
    std::string msg = "Mismatch of L2 norm of std::vector and dftefe::linearAlgebra::SerialVector. "
      "std::vector L2 norm: " + std::to_string(dVecStdL2Norm) + 
      " dftefe::linearAlgebra::SerialVector L2 norm: " + std::to_string(dVecL2Norm);
    throw std::runtime_error(msg);
  }
  
  if(std::fabs(dVecLInfNorm-dVecStdLInfNorm) > tol)
  { 
    std::string msg = "Mismatch of LInf norm of std::vector and dftefe::linearAlgebra::SerialVector. "
      "std::vector LInf norm: " + std::to_string(dVecStdLInfNorm) + 
      " dftefe::linearAlgebra::SerialVector LInf norm: " + std::to_string(dVecLInfNorm);
    throw std::runtime_error(msg);
  }
  
  return 0;
}
