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

#ifndef dftefeDenseMatrixOperations_h
#define dftefeDenseMatrixOperations_h

#include <memory>
#include <blas.hh>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/DenseMatrix.h>
#include <utils/MemorySpaceType.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace DenseMatrixOperations
    {
        template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
        void
        multiply(ValueType alpha,
             DenseMatrix<ValueType,memorySpace> & A,
             DenseMatrix<ValueType,memorySpace> & B,
             ValueType beta,
             DenseMatrix<ValueType,memorySpace> & C);
    }
  } // namespace linearAlgebra

} // namespace dftefe

#include "MatrixOperations.t.cpp"
#endif // dftefeMatrixOperations_h
