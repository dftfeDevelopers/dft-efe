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
      d_nGlobalRows = u.d_nGlobalRows;
      d_nGlobalCols = u.d_nGlobalCols;
      d_nLocalRows  = u.d_nLocalRows;
      d_nLocalCols  = u.d_nLocalCols;
      d_blasQueue   = u.d_blasQueue;
      d_property    = u.d_property;
      d_uplo        = u.d_uplo;
      d_layout      = u.d_layout;
      d_data        = std::make_shared<
        typename dftefe::utils::MemoryStorage<ValueType, memorySpace>>(
        d_nGlobalRows * d_nGlobalCols);
      this->copyFrom<memorySpace>(u);
    }

    //
    // Move Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::SerialDenseMatrix(
      SerialDenseMatrix<ValueType, memorySpace> &&u) noexcept
    {
      d_data = std::move(u.d_data);

      d_nGlobalRows = std::move(u.d_nGlobalRows);
      d_nGlobalCols = std::move(u.d_nGlobalCols);
      d_nLocalRows  = std::move(u.d_nLocalRows);
      d_nLocalCols  = std::move(u.d_nLocalCols);
      d_blasQueue   = std::move(u.d_blasQueue);
      d_property    = std::move(u.d_property);
      d_uplo        = std::move(u.d_uplo);
      d_layout      = std::move(u.d_layout);

      // TODO check compatibity
      //      bool areCompatible =
      //        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      //      utils::throwException<utils::LogicError>(
      //        areCompatible,
      //        "Trying to move from an incompatible Vector. One is a
      //        SerialVector and the " " other a DistributedVector.");
    }

    //
    // Copy assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace> &
    SerialDenseMatrix<ValueType, memorySpace>::operator=(
      const SerialDenseMatrix<ValueType, memorySpace> &u)
    {
      d_data =
        std::make_shared<typename Matrix<ValueType, memorySpace>::Storage>(
          (u.d_data)->size());
      *d_data = *(u.d_data);

      d_nGlobalRows = u.d_nGlobalRows;
      d_nGlobalCols = u.d_nGlobalCols;
      d_nLocalRows  = u.d_nLocalRows;
      d_nLocalCols  = u.d_nLocalCols;
      d_blasQueue   = u.d_blasQueue;
      d_property    = u.d_property;
      d_uplo        = u.d_uplo;
      d_layout      = u.d_layout;

      //      VectorAttributes vectorAttributesSerial(
      //        VectorAttributes::Distribution::SERIAL);
      //      bool areCompatible =
      //        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      //      utils::throwException<utils::LogicError>(
      //        areCompatible,
      //        "Trying to copy assign from an incompatible Vector. One is a
      //        SerialVector and the " " other a DistributedVector.");
      return *this;
    }

    //
    // Move assignment
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace> &
    SerialDenseMatrix<ValueType, memorySpace>::operator=(
      SerialDenseMatrix<ValueType, memorySpace> &&u) noexcept
    {
      d_data = std::move(u.d_data);

      d_nGlobalRows = std::move(u.d_nGlobalRows);
      d_nGlobalCols = std::move(u.d_nGlobalCols);
      d_nLocalRows  = std::move(u.d_nLocalRows);
      d_nLocalCols  = std::move(u.d_nLocalCols);
      d_blasQueue   = std::move(u.d_blasQueue);
      d_property    = std::move(u.d_property);
      d_uplo        = std::move(u.d_uplo);
      d_layout      = std::move(u.d_layout);

      //      VectorAttributes vectorAttributesSerial(
      //        VectorAttributes::Distribution::SERIAL);
      //      bool areCompatible =
      //        d_vectorAttributes.areDistributionCompatible(vectorAttributesSerial);
      //      utils::throwException<utils::LogicError>(
      //        areCompatible,
      //        "Trying to move assign from an incompatible Vector. One is a
      //        SerialVector and the " " other a DistributedVector.");
      return *this;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::SerialDenseMatrix(
      size_type                                         rows,
      size_type                                         cols,
      blasLapack::blasQueueType<memorySpace> &          blasQueueInput,
      ValueType                                         initVal,
      typename Matrix<ValueType, memorySpace>::Property property,
      typename Matrix<ValueType, memorySpace>::Uplo     uplo,
      typename Matrix<ValueType, memorySpace>::Layout   layout)
    {
      d_nGlobalRows = rows;
      d_nGlobalCols = cols;
      d_nLocalRows  = rows;
      d_nLocalCols  = cols;
      d_blasQueue   = blasQueueInput;
      d_property    = property;
      d_uplo        = uplo;
      d_layout      = layout;
      d_data        = std::make_shared<
        typename dftefe::utils::MemoryStorage<ValueType, memorySpace>>(
        d_nGlobalRows * d_nGlobalCols, initVal);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SerialDenseMatrix<ValueType, memorySpace>::~SerialDenseMatrix()
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    double
    SerialDenseMatrix<ValueType, memorySpace>::frobeniusNorm() const
    {
      double value = 0;
      value        = nrm2(d_nLocalCols * d_nLocalRows, this->data(), 1);

      return value;
    }

  } // namespace linearAlgebra
} // namespace dftefe
