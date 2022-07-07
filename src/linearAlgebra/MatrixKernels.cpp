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

#include <linearAlgebra/MatrixKernels.h>

namespace dftefe
{

  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void MatrixKernels<ValueType, memorySpace>::GeneralMatrixAllocation(slate::Matrix<ValueType> &matrix)
    {
      matrix.insertLocalTiles(slate::Target::Host);
    }

    template class MatrixKernels<double, dftefe::utils::MemorySpace::HOST>;
    template class MatrixKernels<float, dftefe::utils::MemorySpace::HOST>;
    template class MatrixKernels<std::complex<double>, dftefe::utils::MemorySpace::HOST>;
    template class MatrixKernels<std::complex<float>, dftefe::utils::MemorySpace::HOST>;

#ifdef DFTEFE_WITH_DEVICE_CUDA
    template class MatrixKernels<double, dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MatrixKernels<float, dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MatrixKernels<std::complex<double>, dftefe::utils::MemorySpace::HOST_PINNED>;
    template class MatrixKernels<std::complex<float>, dftefe::utils::MemorySpace::HOST_PINNED>;
#endif

  } // namespace linearAlgebra
} // namespace dftefe
