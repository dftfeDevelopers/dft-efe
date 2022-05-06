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
#include <memory>
#include <linearAlgebra/SerialMultiVector.h>

using namespace dftefe;
using MemoryStorageDoubleHost    = dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>;
using MultiVectorDoubleHost = dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>;
using SerialMultiVectorDoubleHost = dftefe::linearAlgebra::SerialMultiVector<double, dftefe::utils::MemorySpace::HOST>;

  int
main()
{
  const utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
  std::shared_ptr<dftefe::linearAlgebra::blasLapack::BlasQueue<Host>> queue=std::make_shared<dftefe::linearAlgebra::blasLapack::BlasQueue<Host>>();  
  const double lo = -10.0;
  const double hi = 10.0;
  const dftefe::size_type vSize = 3;
  const dftefe::size_type numVectors=2;
  const double tol = 1e-13;

  // test double
  std::vector<double> dVecStd1(vSize*numVectors);
  for(auto & it : dVecStd1)
    it = lo + (hi-lo)*std::rand()/RAND_MAX;

  std::vector<double> dVecStd2(vSize*numVectors);
  for(auto & it : dVecStd2)
    it = lo + (hi-lo)*std::rand()/RAND_MAX;

  std::vector<double> dVecStd3(vSize*numVectors);
  for(dftefe::size_type i = 0; i < vSize*numVectors; ++i)
    dVecStd3[i] = dVecStd1[i] + dVecStd2[i];

  std::unique_ptr<MemoryStorageDoubleHost> memStorage1 
    = std::make_unique<MemoryStorageDoubleHost>(vSize*numVectors);
  memStorage1->copyFrom<Host>(dVecStd1.data());
  std::shared_ptr<MultiVectorDoubleHost> dVec1
    = std::make_shared<SerialMultiVectorDoubleHost>(vSize,numVectors, 0,queue);
  dVec1->setStorage(memStorage1);

  std::unique_ptr<MemoryStorageDoubleHost> memStorage2 
     = std::make_unique<MemoryStorageDoubleHost>(vSize*numVectors);
  memStorage2->copyFrom<Host>(dVecStd2.data());
  std::shared_ptr<MultiVectorDoubleHost> dVec2
    = std::make_shared<SerialMultiVectorDoubleHost>(vSize,numVectors, 0,queue);
  dVec2->setStorage(memStorage2);

  std::shared_ptr<MultiVectorDoubleHost> dVec3
    = std::make_shared<SerialMultiVectorDoubleHost>(vSize,numVectors, 0,queue);
  linearAlgebra::add<double,Host>(1.0, *dVec1, 1.0, *dVec2, *dVec3);
  const MemoryStorageDoubleHost & dVec3Storage = dVec3->getValues();
  std::vector<double> dVec3HostCopy(vSize*numVectors);
  dVec3Storage.copyTo<Host>(dVec3HostCopy.data()); 

  for(dftefe::size_type i = 0; i < vSize*numVectors; ++i)
  {
    if(std::fabs(dVecStd3[i]-dVec3HostCopy[i]) > tol)
    { 
      std::string msg = "At index " + std::to_string(i) + 
	" mismatch of entries after adding two multi vectors with std::vector and dftefe::linearAlgebra::SerialMultiVector. "
	"std::vector value: " + std::to_string(dVecStd3[i]) + 
	" dftefe::linearAlgebra::SerialMultiVector values: " + std::to_string(dVec3HostCopy[i]);
      throw std::runtime_error(msg);
    }
  }

  return 0;
}
