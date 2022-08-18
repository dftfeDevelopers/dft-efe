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
 * @author Ian C. Lin, Sambit Das
 */

#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::multiply(ValueType                              alpha,
                               GeneralMatrix<ValueType, memorySpace> &A,
                               GeneralMatrix<ValueType, memorySpace> &B,
                               ValueType                              beta,
                               GeneralMatrix<ValueType, memorySpace> &C)
    {
//      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
//        {
//          slate::multiply(alpha,
//                          A.getSlateMatrix(),
//                          B.getSlateMatrix(),
//                          beta,
//                          {slate::Option::Target, slate::Target::Devices});
//        }
//      else
//        {
//          slate::multiply(alpha,
//                          A.getSlateMatrix(),
//                          B.getSlateMatrix(),
//                          beta,
//                          {slate::Option::Target, slate::Target::Host});
          slate::multiply(alpha, A.getSlateMatrix(), B.getSlateMatrix(), beta, C.getSlateMatrix());
//        }
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::multiply(ValueType                                alpha,
                               HermitianMatrix<ValueType, memorySpace> &A,
                               GeneralMatrix<ValueType, memorySpace> &  B,
                               ValueType                                beta,
                               GeneralMatrix<ValueType, memorySpace> &  C)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::Host});
        }
    }



    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::multiply(ValueType                                alpha,
                               GeneralMatrix<ValueType, memorySpace> &  A,
                               HermitianMatrix<ValueType, memorySpace> &B,
                               ValueType                                beta,
                               GeneralMatrix<ValueType, memorySpace> &  C)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::Host});
        }
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::triangular_multiply(
      TriangularMatrix<ValueType, memorySpace> &A,
      GeneralMatrix<ValueType, memorySpace> &   B)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          {slate::Option::Target, slate::Target::Host});
        }
    }



    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::triangular_multiply(
      GeneralMatrix<ValueType, memorySpace> &   A,
      TriangularMatrix<ValueType, memorySpace> &B)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          {slate::Option::Target, slate::Target::Host});
        }
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::triangular_solve(
      TriangularMatrix<ValueType, memorySpace> &A,
      GeneralMatrix<ValueType, memorySpace> &   B)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::triangular_solve(A.getSlateMatrix(),
                                  B.getSlateMatrix(),
                                  {slate::Option::Target,
                                   slate::Target::Devices});
        }
      else
        {
          slate::triangular_solve(A.getSlateMatrix(),
                                  B.getSlateMatrix(),
                                  {slate::Option::Target, slate::Target::Host});
        }
    }



    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::triangular_solve(
      GeneralMatrix<ValueType, memorySpace> &   A,
      TriangularMatrix<ValueType, memorySpace> &B)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::triangular_solve(A.getSlateMatrix(),
                                  B.getSlateMatrix(),
                                  {slate::Option::Target,
                                   slate::Target::Devices});
        }
      else
        {
          slate::triangular_solve(A.getSlateMatrix(),
                                  B.getSlateMatrix(),
                                  {slate::Option::Target, slate::Target::Host});
        }
    }

  } // namespace linearAlgebra
} // namespace dftefe
