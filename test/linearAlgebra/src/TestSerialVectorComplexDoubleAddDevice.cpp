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
 * @author Sambit Das
 */
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/SerialVector.h>
#include <utils/DeviceAPICalls.h>
#include <iostream>

using namespace dftefe;
using MemoryStorageComplexDoubleDevice    = dftefe::utils::MemoryStorage<std::complex<double>, dftefe::utils::MemorySpace::DEVICE>;
using VectorComplexDoubleDevice = dftefe::linearAlgebra::Vector<std::complex<double>, dftefe::utils::MemorySpace::DEVICE>;
using SerialVectorComplexDoubleDevice = dftefe::linearAlgebra::SerialVector<std::complex<double>, dftefe::utils::MemorySpace::DEVICE>;

  int
main()
{
  const utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;  
  const utils::MemorySpace Device = dftefe::utils::MemorySpace::DEVICE;

  int device;
  utils::deviceGetDevice(&device);
  std::cout<<device<<std::endl;

  std::shared_ptr<dftefe::linearAlgebra::blasLapack::BlasQueue<Device>> queue=std::make_shared<dftefe::linearAlgebra::blasLapack::BlasQueue<Device>>(device,0);

  const double lo = -10.0;
  const double hi = 10.0;
  dftefe::size_type vSize = 3;
  const double tol = 1e-13;

  // test complex double
  std::vector<std::complex<double>> dVecStd1(vSize);
  for(auto & it : dVecStd1)
    it = (lo + (hi-lo)*std::rand()/RAND_MAX,lo + 2.5*(hi-lo)*std::rand()/RAND_MAX);

  std::vector<std::complex<double>> dVecStd2(vSize);
  for(auto & it : dVecStd2)
    it = (lo + (hi-lo)*std::rand()/RAND_MAX,lo -2.5*(hi-lo)*std::rand()/RAND_MAX);

  std::vector<std::complex<double>> dVecStd3(vSize);
  for(dftefe::size_type i = 0; i < vSize; ++i)
    dVecStd3[i] = dVecStd1[i] + dVecStd2[i];

  std::unique_ptr<MemoryStorageComplexDoubleDevice> memStorage1 
    = std::make_unique<MemoryStorageComplexDoubleDevice>(vSize);
  memStorage1->copyFrom<Host>(dVecStd1.data());
  std::shared_ptr<VectorComplexDoubleDevice> dVec1
    = std::make_shared<SerialVectorComplexDoubleDevice>(vSize, 0,queue);
  dVec1->setStorage(memStorage1);

  std::unique_ptr<MemoryStorageComplexDoubleDevice> memStorage2 
     = std::make_unique<MemoryStorageComplexDoubleDevice>(vSize);
  memStorage2->copyFrom<Host>(dVecStd2.data());
  std::shared_ptr<VectorComplexDoubleDevice> dVec2
    = std::make_shared<SerialVectorComplexDoubleDevice>(vSize, 0,queue);
  dVec2->setStorage(memStorage2);

  std::shared_ptr<VectorComplexDoubleDevice> dVec3
    = std::make_shared<SerialVectorComplexDoubleDevice>(vSize, 0,queue);
  linearAlgebra::add<std::complex<double>,Device>(1.0, *dVec1, 1.0, *dVec2, *dVec3);
  const MemoryStorageComplexDoubleDevice & dVec3Storage = dVec3->getValues();
  std::vector<std::complex<double>> dVec3HostCopy(vSize);
  dVec3Storage.copyTo<Host>(dVec3HostCopy.data()); 

  for(dftefe::size_type i = 0; i < vSize; ++i)
  {
    if(std::fabs(dVecStd3[i]-dVec3HostCopy[i]) > tol)
    { 
      std::string msg = "At index " + std::to_string(i) + 
	" mismatch of entries after adding two vectors norm of std::vector and dftefe::linearAlgebra::SerialVector. "
	"std::vector value: " + std::to_string(dVecStd3[i].real()) + " " +  std::to_string(dVecStd3[i].imag())+ 
	" dftefe::linearAlgebra::SerialVector values: " + std::to_string(dVec3HostCopy[i].real()) + " " + std::to_string(dVec3HostCopy[i].imag());
      throw std::runtime_error(msg);
    }
  }
  return 0;
}
