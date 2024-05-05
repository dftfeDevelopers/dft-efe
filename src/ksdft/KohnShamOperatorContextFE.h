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
 * @author Avirup Sircar
 */

#ifndef dftefeKohnShamOperatorContextFE_h
#define dftefeKohnShamOperatorContextFE_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <basis/FEBasisManager.h>
#include <basis/FEBasisDataStorage.h>
#include <quadrature/QuadratureAttributes.h>
#include <memory>
#include <variant>

namespace dftefe
{
  namespace ksdft
  {
    /**
     *@brief A derived class of linearAlgebra::OperatorContext to encapsulate
     * the action of a discrete Kohn-Sham operator on vectors.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNED, etc.) in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class KohnShamOperatorContextFE
      : public linearAlgebra::
          OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
    {
    public:
      /**
       * @brief define ValueType as the superior (bigger set) of the
       * ValueTypeOperator and ValueTypeOperand
       * (e.g., between double and complex<double>, complex<double>
       * is the bigger set)
       */
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

      using Storage = dftefe::utils::MemoryStorage<ValueType, memorySpace>;

      using HamiltonianPtrVariant =
        std::variant<Hamiltonian<float, memorySpace> *,
                     Hamiltonian<double, memorySpace> *,
                     Hamiltonian<std::complex<float>, memorySpace> *,
                     Hamiltonian<std::complex<double>, memorySpace> *>;

    public:
      /**
       * @brief Constructor
       */
      KohnShamOperatorContextFE(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeBasisData, memorySpace, dim>
            &                                    feBasisManager,
        std::vector<const HamiltonianPtrVariant> hamiltonianVec,
        const size_type                          maxCellTimesNumVecs);

      void
      apply(
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
        linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const override;

    private:
      const basis::
        FEBasisManager<ValueTypeOperand, ValueTypeBasisData, memorySpace, dim>
          *           d_feBasisManager;
      Storage         d_hamiltonianInAllCells;
      const size_type d_maxCellTimesNumVecs;
      const size_type d_cellWiseDataSize;
    }; // end of class KohnShamOperatorContextFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/KohnShamOperatorContextFE.t.cpp>
#endif // dftefeKohnShamOperatorContextFE_h
