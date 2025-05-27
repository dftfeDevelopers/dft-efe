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
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <basis/FEBasisManager.h>
#include <ksdft/Hamiltonian.h>
#include <memory>
#include <variant>
#include <type_traits>

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

      using Storage = utils::MemoryStorage<ValueTypeOperator, memorySpace>;

      using HamiltonianPtrVariant = std::variant<
        std::shared_ptr<Hamiltonian<float, memorySpace>>,
        std::shared_ptr<Hamiltonian<double, memorySpace>>,
        std::shared_ptr<Hamiltonian<std::complex<float>, memorySpace>>,
        std::shared_ptr<Hamiltonian<std::complex<double>, memorySpace>>>;

    public:
      /**
       * @brief Constructor
       */
      KohnShamOperatorContextFE(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeBasisData, memorySpace, dim>
            &                                        feBasisManager,
        const std::vector<HamiltonianPtrVariant> &   hamiltonianComponentsVec,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContext,
        const size_type                              maxCellBlock,
        const size_type                              maxWaveFnBatch);

      ~KohnShamOperatorContextFE() = default;

      void
      reinit(const basis::FEBasisManager<ValueTypeOperand,
                                         ValueTypeBasisData,
                                         memorySpace,
                                         dim> &        feBasisManager,
             const std::vector<HamiltonianPtrVariant> &hamiltonianVec);

      void
      apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
            linearAlgebra::MultiVector<ValueType, memorySpace> &       Y,
            bool updateGhostX = false,
            bool updateGhostY = false) const override;

    private:
      const basis::
        FEBasisManager<ValueTypeOperand, ValueTypeBasisData, memorySpace, dim>
          *                                       d_feBasisManager;
      Storage                                     d_hamiltonianInAllCells;
      const size_type                             d_maxCellBlock;
      const size_type                             d_maxWaveFnBatch;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>> d_linAlgOpContext;
      std::vector<HamiltonianPtrVariant>          d_hamiltonianComponentsVec;

      mutable linearAlgebra::MultiVector<ValueTypeOperator, memorySpace> d_scratchNonLocPSPApply;

    }; // end of class KohnShamOperatorContextFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/KohnShamOperatorContextFE.t.cpp>
#endif // dftefeKohnShamOperatorContextFE_h
