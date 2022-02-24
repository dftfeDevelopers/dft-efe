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

#include "SerialDenseMatrix.h"

namespace dftefe
{
  namespace linearAlgebra
  {

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::
      SerialDenseMatrix(const SerialDenseMatrix<ValueType,memorySpace > &u)
    {
      this->d_nGlobalRows = u.d_nGlobalRows;
      this->d_nGlobalCols = u.d_nGlobalCols;
      this->d_nLocalRows = u.d_nLocalRows;
      this->d_nLocalCols = u.d_nLocalCols;
      this->d_blasQueue = u.;

      d_data.allocate(this->d_nGlobalRows * this->d_nGlobalCols);
      this->copyFrom<ValueType,memorySpace> (u);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::
      SerialDenseMatrix(size_type rows, size_type cols,
                      ValueType initVal )
    {

      this->d_nGlobalRows = rows;
      this->d_nGlobalCols = cols;
      this->d_nLocalRows = rows;
      this->d_nLocalCols = cols;
      d_data.allocate(this->d_nGlobalRows * this->d_nGlobalCols, initVal);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::~SerialDenseMatrix()
    {

    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    iterator SerialDenseMatrix<ValueType, memorySpace>::begin()
    {
      return d_data.begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const_iterator SerialDenseMatrix<ValueType, memorySpace>::begin() const
    {
      return d_data.begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    iterator SerialDenseMatrix<ValueType, memorySpace>::end()
    {
      return d_data.end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const_iterator SerialDenseMatrix<ValueType, memorySpace>::end() const
    {
      return d_data.end();
    }

//    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
//    SerialDenseMatrix<ValueType, memorySpace> &
//      SerialDenseMatrix<ValueType, memorySpace>::
//        operator=(const SerialDenseMatrix<ValueType, memorySpace> &rhs)
//    {
//        if ((rhs.getLocalRows() != this->getLocalRows()) || (rhs.getLocalCols() != this->getLocalCols()) )
//        {
//          this->d_nGlobalRows = rhs.d_nGlobalRows;
//          this->d_nGlobalCols = rhs.d_nGlobalCols;
//          this->d_nLocalRows = rhs.d_nLocalRows;
//          this->d_nLocalCols = rhs.d_nLocalCols;
//
//          this->d_data.resize(this->d_nGlobalRows * this->d_nGlobalCols );
//        }
//
//        this->copyFrom<ValueType,memorySpace> (rhs);
//        return (*this);
//    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType * SerialDenseMatrix<ValueType, memorySpace>::data() noexcept
    {
      return (d_data.data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType * SerialDenseMatrix<ValueType, memorySpace>::data() const noexcept
    {
      return (d_data.data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceDst>
    void SerialDenseMatrix<ValueType, memorySpace>::
    copyTo(MatrixBase<ValueType, memorySpaceDst> &dstMatrix) const
    {
      this->d_data.copyTo(dstMatrix.getDataVec());

    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceSrc>
    void SerialDenseMatrix<ValueType, memorySpace>::
    copyFrom( const MatrixBase<ValueType, memorySpaceSrc> &srcMatrix)
    {
      this->d_data.copyFrom(srcMatrix.getDataVec());

    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type SerialDenseMatrix<ValueType, memorySpace>::getLocalRows( ) const
    {
      return (this->d_nLocalRows);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type SerialDenseMatrix<ValueType, memorySpace>::getLocalCols( ) const
    {
      return (this->d_nLocalCols);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type SerialDenseMatrix<ValueType, memorySpace>::getGlobalRows( ) const
    {
      return (this->d_nGlobalRows);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type SerialDenseMatrix<ValueType, memorySpace>::getGlobalCols( ) const
    {
      return (this->d_nGlobalCols);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void SerialDenseMatrix<ValueType, memorySpace>::
      getLocalSize(size_type &rows,size_type &cols ) const
    {
      rows = this->d_nLocalRows;
      cols = this->d_nLocalCols;

    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void SerialDenseMatrix<ValueType, memorySpace>::
    getGlobalSize(size_type &rows,size_type &cols ) const
    {
      rows = this->d_nGlobalRows;
      cols = this->d_nGlobalCols;

    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    dftefe::utils::MemoryManager<ValueType, memorySpace>::MemoryStorage &
    SerialDenseMatrix<ValueType, memorySpace>::getDataVec()
    {
      return (this->d_data);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    blasWrapper::blasQueueType<memorySapce> &
    SerialDenseMatrix<ValueType, memorySpace>::getQueue()
    {
      return d_blasQueue;
    }

  }
}
