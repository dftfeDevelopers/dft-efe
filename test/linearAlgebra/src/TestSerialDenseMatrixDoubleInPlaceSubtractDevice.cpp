/******************************************************************************
 * Copyright (c) 2021-2022.                                                   *
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
 * @author Ian C. Lin
 */

#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <linearAlgebra/SerialDenseMatrix.h>

using namespace dftefe;
using MemoryStorageDoubleDevice =
  dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::DEVICE>;
using MatrixDoubleDevice =
  dftefe::linearAlgebra::Matrix<double, dftefe::utils::MemorySpace::DEVICE>;
using SerialDenseMatrixDoubleDevice =
  dftefe::linearAlgebra::SerialDenseMatrix<double,
                                           dftefe::utils::MemorySpace::DEVICE>;

int
main()
{
  const utils::MemorySpace Device = dftefe::utils::MemorySpace::DEVICE;
  const utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
  int device;
  dftefe::utils::deviceGetDevice(&device);
  std::shared_ptr<dftefe::linearAlgebra::blasLapack::blasQueueType<Device>>
    queue = std::make_shared<
      dftefe::linearAlgebra::blasLapack::blasQueueType<Device>>(device, 0);
  const double lo    = -10.0;
  const double hi    = 10.0;
  size_type nRows = 5, nCols = 3;
  size_type vSize = nRows * nCols;
  const double tol   = 1e-13;

  // test double
  std::vector<double> dVecStd1(vSize);
  for (auto &it : dVecStd1)
    it = lo + (hi - lo) * std::rand() / RAND_MAX;

  std::vector<double> dVecStd2(vSize);
  for (auto &it : dVecStd2)
    it = lo + (hi - lo) * std::rand() / RAND_MAX;

  std::vector<double> dVecStd3(vSize);
  for (size_type i = 0; i < vSize; ++i)
    dVecStd3[i] = dVecStd1[i] - dVecStd2[i];

  std::unique_ptr<MemoryStorageDoubleDevice> memStorage1 =
    std::make_unique<MemoryStorageDoubleDevice>(vSize);
  memStorage1->copyFrom<Host>(dVecStd1.data());
  std::shared_ptr<MatrixDoubleDevice> dMat1 =
    std::make_shared<SerialDenseMatrixDoubleDevice>(nRows, nCols, queue);
  dMat1->setStorage(memStorage1);

  std::unique_ptr<MemoryStorageDoubleDevice> memStorage2 =
    std::make_unique<MemoryStorageDoubleDevice>(vSize);
  memStorage2->copyFrom<Host>(dVecStd2.data());
  std::shared_ptr<MatrixDoubleDevice> dMat2 =
    std::make_shared<SerialDenseMatrixDoubleDevice>(nRows, nCols, queue);
  dMat2->setStorage(memStorage2);

  *dMat1 -= *dMat2;
  const MemoryStorageDoubleDevice &dVec3Storage = dMat1->getValues();
  std::vector<double>            dVec3HostCopy(vSize);
  dVec3Storage.copyTo<Host>(dVec3HostCopy.data());

  for (size_type i = 0; i < vSize; ++i)
    {
      if (std::fabs(dVecStd3[i] - dVec3HostCopy[i]) > tol)
        {
          std::string msg =
            "At index (" + std::to_string(i % nRows) + ", " + std::to_string(i / nRows) +
            ") mismatch of entries after adding two matrices norm of std::vector and dftefe::linearAlgebra::SerialDenseMatrix. "
            "std::vector value: " +
            std::to_string(dVecStd3[i]) +
            " dftefe::linearAlgebra::SerialDenseMatrix values: " +
            std::to_string(dVec3HostCopy[i]);
          throw std::runtime_error(msg);
        }
    }

  return 0;
}
