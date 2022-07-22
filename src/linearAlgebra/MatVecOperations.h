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

#ifndef dftefeMatVecOperations_h
#define dftefeMatVecOperations_h

#include <memory>
#include <blas.hh>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/Matrix.h>
#include <utils/MemorySpaceType.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace MatVecOperations
    {
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      computeOverlapMatrixAConjugateTransposeA(
        const MultiVector<ValueType, memorySpace> &A,
        HermitianMatrix<ValueType, memorySpace> &  S,
        LinAlgOpContext<memorySpace> &             context,
        const size_type                            vectorsBlockSize = 0);

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      choleskyOrthogonalization(const MultiVector<ValueType, memorySpace> &A,
                                MultiVector<ValueType, memorySpace> &      B,
                                LinAlgOpContext<memorySpace> &context);

    } // namespace MatVecOperations
  }   // namespace linearAlgebra

} // namespace dftefe

#include "MatVecOperations.t.cpp"
#endif // dftefeMatVecOperations_h
