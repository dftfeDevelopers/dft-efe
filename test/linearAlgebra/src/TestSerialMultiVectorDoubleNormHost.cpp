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
#include <utils/DataTypeOverloads.h>
#include <linearAlgebra/SerialMultiVector.h>

int
main()
{
  using namespace dftefe;
  const dftefe::utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
  const double lo = -10.0;
  const double hi = 10.0;
  const size_type vSize = 3;
  const size_type numVec=2;
  const double tol = 1e-13;
  
  // test double
  std::vector<std::vector<double>> dMultiVecStd(numVec,std::vector<double>(vSize));
  std::vector<double> dMultiVecStdFlattened(numVec*vSize);
  for (size_type i=0; i<numVec; i++)  
    for(size_type j=0; j<vSize; j++)
    {
      dMultiVecStd[i][j] = lo + (hi-lo)*std::rand()/RAND_MAX;
      dMultiVecStdFlattened[j*numVec+i]=dMultiVecStd[i][j];
    }
 
  std::vector<double> dMultiVecStdL2Norm(numVec,0.0);
  std::vector<double> dMultiVecStdLInfNorm(numVec,0.0);  
  for (size_type i=0; i<numVec; i++)  
    for(size_type j=0; j<vSize; j++)
      dMultiVecStdL2Norm[i] += std::abs(dMultiVecStd[i][j])*std::abs(dMultiVecStd[i][j]);

  for (size_type i=0; i<numVec; i++) 
  {
    dMultiVecStdL2Norm[i] = std::sqrt(dMultiVecStdL2Norm[i]);
    dMultiVecStdLInfNorm[i] = *std::max_element(dMultiVecStd[i].begin(), dMultiVecStd[i].end(),utils::absCompare<double>);
  }

  std::shared_ptr<dftefe::linearAlgebra::blasLapack::blasQueueType<Host>> queue=std::make_shared<dftefe::linearAlgebra::blasLapack::blasQueueType<Host>>();

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>> dMultiVec
    = std::make_shared<dftefe::linearAlgebra::SerialMultiVector<double, dftefe::utils::MemorySpace::HOST>>(vSize,numVec, 0,queue);

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::HOST>::copy(vSize*numVec,
                                            dMultiVec->data(),
                                            dMultiVecStdFlattened.data());

  const std::vector<double> dMultiVecL2Norm = dMultiVec->l2Norms();
  const std::vector<double> dMultiVecLInfNorm = dMultiVec->lInfNorms();
  for (size_type i=0; i<numVec; i++)
  {
    if(std::fabs(dMultiVecL2Norm[i]-dMultiVecStdL2Norm[i]) > tol)
    { 
      std::string msg = "Mismatch of L2 norms of std::vector and dftefe::linearAlgebra::SerialMultiVector. "
        "std::vector L2 norm: " + std::to_string(dMultiVecStdL2Norm[i]) + 
        " dftefe::linearAlgebra::SerialMultiVector L2 norm: " + std::to_string(dMultiVecL2Norm[i]);
      throw std::runtime_error(msg);
    }
    
    if(std::fabs(dMultiVecLInfNorm[i]-dMultiVecStdLInfNorm[i]) > tol)
    { 
      std::string msg = "Mismatch of LInf norms of std::vector and dftefe::linearAlgebra::SerialMultiVector. "
        "std::vector LInf norm: " + std::to_string(dMultiVecStdLInfNorm[i]) + 
        " dftefe::linearAlgebra::SerialMultiVector LInf norm: " + std::to_string(dMultiVecLInfNorm[i]);
      throw std::runtime_error(msg);
    }
  }
  
  return 0;
}
