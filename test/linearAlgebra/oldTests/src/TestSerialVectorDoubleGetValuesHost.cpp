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
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/SerialVector.h>

using namespace dftefe;
using MemoryStorageDoubleHost    = dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>;
using VectorDoubleHost = dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::HOST>;
using SerialVectorDoubleHost = dftefe::linearAlgebra::SerialVector<double, dftefe::utils::MemorySpace::HOST>;

  int
main()
{
  const utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
  dftefe::linearAlgebra::blasLapack::BlasQueue<Host> queue; 

  dftefe::linearAlgebra::LinAlgOpContext<Host> linAlgContext(&queue); 
  const double lo = -10.0;
  const double hi = 10.0;
  dftefe::size_type vSize = 3;
  const double tol = 1e-13;

  // test double
  std::vector<double> dVecStd(vSize);
  for(auto & it : dVecStd)
    it = lo + (hi-lo)*std::rand()/RAND_MAX;

  std::unique_ptr<MemoryStorageDoubleHost> memStorage
    = std::make_unique<MemoryStorageDoubleHost>(vSize);
  memStorage->copyFrom<Host>(dVecStd.data());
  std::shared_ptr<VectorDoubleHost> dVec
    = std::make_shared<SerialVectorDoubleHost>(vSize, &linAlgContext,0);
  dVec->setStorage(memStorage);

  const MemoryStorageDoubleHost & dVecStorage = dVec->getValues();
  std::vector<double> dVecHostCopy(vSize);
  dVecStorage.copyTo<Host>(dVecHostCopy.data()); 

  for(dftefe::size_type i = 0; i < vSize; ++i)
  {
    if(std::fabs(dVecStd[i]-dVecHostCopy[i]) > tol)
    { 
      std::string msg = "At index " + std::to_string(i) + 
	" mismatch of entries using getValues() on dftefe::linearAlgebra::SerialVector. "
	"Expected value: " + std::to_string(dVecStd[i]) + 
	" obtained value: " + std::to_string(dVecHostCopy[i]);
      throw std::runtime_error(msg);
    }
  }

  return 0;
}
