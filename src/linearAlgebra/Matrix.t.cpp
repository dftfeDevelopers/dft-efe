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
* @author Vishal subramanian
*/


namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Matrix<ValueType, memorySpace>::iterator
    Matrix<ValueType, memorySpace>::begin()
    {
      return d_data->begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Matrix<ValueType, memorySpace>::const_iterator
    Matrix<ValueType, memorySpace>::begin() const
    {
      return d_data->begin();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Matrix<ValueType, memorySpace>::iterator
    Matrix<ValueType, memorySpace>::end()
    {
      return d_data->end();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    typename Matrix<ValueType, memorySpace>::const_iterator
    Matrix<ValueType, memorySpace>::end() const
    {
      return d_data->end();
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    ValueType *
    Matrix<ValueType, memorySpace>::data()
    {
      return (d_data->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const ValueType *
    Matrix<ValueType, memorySpace>::data() const
    {
      return (d_data->data());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceDst>
    void
    Matrix<ValueType, memorySpace>::copyTo(
      Matrix<ValueType, memorySpaceDst> &dstMatrix) const
    {
      this->d_data->copyTo(dstMatrix.getDataVec());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    template <dftefe::utils::MemorySpace memorySpaceSrc>
    void
    Matrix<ValueType, memorySpace>::copyFrom(
      const Matrix<ValueType, memorySpaceSrc> &srcMatrix)
    {
      this->d_data->copyFrom(srcMatrix.getDataVec());
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Matrix<ValueType, memorySpace>::getLocalRows() const
    {
      return (this->d_nLocalRows);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Matrix<ValueType, memorySpace>::getLocalCols() const
    {
      return (this->d_nLocalCols);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Matrix<ValueType, memorySpace>::getGlobalRows() const
    {
      return (this->d_nGlobalRows);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    Matrix<ValueType, memorySpace>::getGlobalCols() const
    {
      return (this->d_nGlobalCols);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Matrix<ValueType, memorySpace>::getLocalSize(
      size_type &rows,
      size_type &cols) const
    {
      rows = this->d_nLocalRows;
      cols = this->d_nLocalCols;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    Matrix<ValueType, memorySpace>::getGlobalSize(
      size_type &rows,
      size_type &cols) const
    {
      rows = this->d_nGlobalRows;
      cols = this->d_nGlobalCols;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    dftefe::utils::MemoryStorage<ValueType, memorySpace> &
    Matrix<ValueType, memorySpace>::getDataVec()
    {
      return *(this->d_data);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    blasWrapper::blasQueueType<memorySpace> &
    Matrix<ValueType, memorySpace>::getQueue()
    {
      return d_blasQueue;
    }
  }
}
