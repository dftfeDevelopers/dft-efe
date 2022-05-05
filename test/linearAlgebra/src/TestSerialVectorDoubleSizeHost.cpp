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
using VectorDoubleHost = dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::HOST>;
using SerialVectorDoubleHost = dftefe::linearAlgebra::SerialVector<double, dftefe::utils::MemorySpace::HOST>;

  int
main()
{
  const utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
  dftefe::size_type vSize = 3;
  std::shared_ptr<dftefe::linearAlgebra::blasLapack::BlasQueue<Host>> queue=std::make_shared<dftefe::linearAlgebra::blasLapack::BlasQueue<Host>>();  
  std::shared_ptr<VectorDoubleHost> dVec
    = std::make_shared<SerialVectorDoubleHost>(vSize, 0,queue);

  const global_size_type size = dVec->size();
  const size_type locallyOwnedSize = dVec->locallyOwnedSize();
  const size_type ghostSize = dVec->ghostSize();
  const size_type localSize = dVec->localSize();

  if(size != vSize)
  { 
    std::string msg = "Mismatch of global size of dftefe::linearAlgebra::SerialVector. "
      "Expected global size: " +  std::to_string(vSize) + " , obtained global size: " +
      std::to_string(size);
    throw std::runtime_error(msg);
  }
  
  if(locallyOwnedSize != vSize)
  { 
    std::string msg = "Mismatch of locallyOwned size of dftefe::linearAlgebra::SerialVector. "
      "Expected locally owned size: " +  std::to_string(vSize) + " , obtained global size: " +
      std::to_string(locallyOwnedSize);
    throw std::runtime_error(msg);
  }
  
  if(ghostSize != 0)
  { 
    std::string msg = "Mismatch of ghost size of dftefe::linearAlgebra::SerialVector. "
      "Expected ghost size: 0"  " , obtained global size: " +
      std::to_string(ghostSize);
    throw std::runtime_error(msg);
  }
  
  if(localSize != vSize)
  { 
    std::string msg = "Mismatch of local size of dftefe::linearAlgebra::SerialVector. "
      "Expected local size: " + std::to_string(vSize) + " , obtained global size: " +
      std::to_string(localSize);
    throw std::runtime_error(msg);
  }

return 0;
}
