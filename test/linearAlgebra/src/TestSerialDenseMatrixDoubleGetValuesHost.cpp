/******************************************************************************
 * Copyright (c) 2022-2022.                                                   *
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
#include <memory>
#include <linearAlgebra/SerialDenseMatrix.h>


using namespace dftefe;
using MemoryStorageDoubleHost =
 dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>;
using MatrixDoubleHost =
 dftefe::linearAlgebra::Matrix<double, dftefe::utils::MemorySpace::HOST>;
using SerialDenseMatrixDoubleHost =
 dftefe::linearAlgebra::SerialDenseMatrix<double,
                                          dftefe::utils::MemorySpace::HOST>;

int
main()
{
 const utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
 std::shared_ptr<dftefe::linearAlgebra::blasLapack::blasQueueType<Host>>
   queue = std::make_shared<
     dftefe::linearAlgebra::blasLapack::blasQueueType<Host>>();
 const double lo    = -10.0;
 const double hi    = 10.0;
 size_type nRows = 5, nCols = 3, vSize = nRows*nCols;
 const double tol   = 1e-13;

 // test double
 std::vector<double> dVecStd(vSize);
 for (auto &it : dVecStd)
   it = lo + (hi - lo) * std::rand() / RAND_MAX;

 std::vector<double> dVecStd2(vSize);
 for(auto & it : dVecStd2)
   it = lo + (hi-lo)*std::rand()/RAND_MAX;

 std::unique_ptr<MemoryStorageDoubleHost> memStorage =
   std::make_unique<MemoryStorageDoubleHost>(vSize);
 memStorage->copyFrom<Host>(dVecStd.data());
 std::shared_ptr<MatrixDoubleHost> dMat =
   std::make_shared<SerialDenseMatrixDoubleHost>(nRows, nCols, queue);
 dMat->setStorage(memStorage);

 const MemoryStorageDoubleHost &dMatrixData = dMat->getValues();
 std::vector<double>            dMatrixHostCopy(vSize);
 dMatrixData.copyTo<Host>(dMatrixHostCopy.data());

 dVecStd[5] = 2.0;

 for (size_type i = 0; i < vSize; ++i)
   {
     if (std::fabs(dVecStd[i] - dMatrixHostCopy[i]) > tol)
       {
         std::string msg =
           "At index (" + std::to_string(i%nRows) + ", " + std::to_string(i/nRows) +
           ") mismatch of entries using setStorage() on dftefe::linearAlgebra::SerialDenseMatrix. "
           "Expected value: " +
           std::to_string(dVecStd[i]) +
           " obtained value: " + std::to_string(dMatrixHostCopy[i]);
         throw std::runtime_error(msg);
       }
   }

 return 0;
}