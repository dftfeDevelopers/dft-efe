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
 * @author Ian C. Lin, Vishal Subramanian
 */
namespace dftefe
{
  namespace linearAlgebra
  {

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::SerialDenseMatrix(
      const SerialDenseMatrix<ValueType, memorySpace> &u)
    {
      this->d_nGlobalRows = u.d_nGlobalRows;
      this->d_nGlobalCols = u.d_nGlobalCols;
      this->d_nLocalRows  = u.d_nLocalRows;
      this->d_nLocalCols  = u.d_nLocalCols;
      this->d_blasQueue   = u.d_blasQueue;
      this->d_data =
        std::make_shared<typename dftefe::utils::MemoryStorage<ValueType, memorySpace>>(
          this->d_nGlobalRows * this->d_nGlobalCols);
      this->copyFrom<memorySpace>(u);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::SerialDenseMatrix(
      size_type rows,
      size_type cols,
      blasWrapper::blasQueueType<memorySpace> &blasQueueInput,
      ValueType initVal)
    {
      this->d_nGlobalRows = rows;
      this->d_nGlobalCols = cols;
      this->d_nLocalRows  = rows;
      this->d_nLocalCols  = cols;
      this->d_blasQueue = blasQueueInput;
      d_data =
        std::make_shared<typename dftefe::utils::MemoryStorage<ValueType, memorySpace>>(
          this->d_nGlobalRows * this->d_nGlobalCols, initVal);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::~SerialDenseMatrix()
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double SerialDenseMatrix<ValueType, memorySpace>::frobeniusNorm () const
    {
      double value  = 0 ;
      value = nrm2(d_nLocalCols*d_nLocalRows, this->data(), 1 );

      return value;
    }

  } // namespace linearAlgebra
} // namespace dftefe
