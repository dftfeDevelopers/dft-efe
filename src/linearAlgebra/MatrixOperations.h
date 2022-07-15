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

#ifndef dftefeMatrixOperations_h
#define dftefeMatrixOperations_h

#include <memory>
#include <blas.hh>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/Matrix.h>
#include <utils/MemorySpaceType.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace MatrixOperations
    {
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      multiply(ValueType                              alpha,
               GeneralMatrix<ValueType, memorySpace> &A,
               GeneralMatrix<ValueType, memorySpace> &B,
               ValueType                              beta,
               GeneralMatrix<ValueType, memorySpace> &C);

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      multiply(ValueType                                alpha,
               HermitianMatrix<ValueType, memorySpace> &A,
               GeneralMatrix<ValueType, memorySpace> &  B,
               ValueType                                beta,
               GeneralMatrix<ValueType, memorySpace> &  C);

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      multiply(ValueType                                alpha,
               GeneralMatrix<ValueType, memorySpace> &  A,
               HermitianMatrix<ValueType, memorySpace> &B,
               ValueType                                beta,
               GeneralMatrix<ValueType, memorySpace> &  C);


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      triangular_multiply(TriangularMatrix<ValueType, memorySpace> &A,
                          GeneralMatrix<ValueType, memorySpace> &   B);

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      triangular_multiply(GeneralMatrix<ValueType, memorySpace> &   A,
                          TriangularMatrix<ValueType, memorySpace> &B);


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      triangular_solve(TriangularMatrix<ValueType, memorySpace> &A,
                       GeneralMatrix<ValueType, memorySpace> &   B);

      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      triangular_solve(GeneralMatrix<ValueType, memorySpace> &   A,
                       TriangularMatrix<ValueType, memorySpace> &B);


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      chol_factor(HermitianMatrix<ValueType, memorySpace> &A);


    } // namespace MatrixOperations
  }   // namespace linearAlgebra

} // namespace dftefe

#include "MatrixOperations.t.cpp"
#endif // dftefeMatrixOperations_h
