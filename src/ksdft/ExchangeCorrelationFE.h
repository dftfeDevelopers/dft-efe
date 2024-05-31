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

#ifndef dftefeExchangeCorrelationFE_h
#define dftefeExchangeCorrelationFE_h

#include <linearAlgebra/MultiVector.h>
#include <ksdft/Hamiltonian.h>
#include <ksdft/Energy.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/FEBasisDofHandler.h>
#include <basis/FEBasisOperations.h>
#include <ksdft/Defaults.h>
#include <xc.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class ExchangeCorrelationFE
      : public Hamiltonian<ValueTypeBasisData, memorySpace>,
        public Energy<linearAlgebra::blasLapack::real_type<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff>>>
    {
    public:
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                               ValueTypeBasisCoeff>;

      using RealType = linearAlgebra::blasLapack::real_type<ValueType>;

      using Storage = utils::MemoryStorage<ValueType, memorySpace>;

    public:
      /**
       * @brief Constructor
       */
      ExchangeCorrelationFE(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize);

      ~ExchangeCorrelationFE();

      void
      reinitBasis(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBasisDataStorage);

      void
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity);

      void
      getLocal(Storage &cellWiseStorage) const override;

      void
      evalEnergy(const utils::mpi::MPIComm &comm);

      RealType
      getEnergy() const override;

    private:
      std::shared_ptr<
        quadrature::QuadratureValuesContainer<RealType, memorySpace>>
        d_xcPotentialQuad;
      const quadrature::QuadratureValuesContainer<RealType, memorySpace>
        *d_electronChargeDensity;
      std::shared_ptr<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>
        d_feBasisDofHandler;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        d_feBasisDataStorage;
      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>>
                      d_feBasisOp;
      RealType        d_energy;
      const size_type d_cellBlockSize;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;

      xc_func_type *d_funcX;
      xc_func_type *d_funcC;

      utils::MemoryStorage<RealType, utils::MemorySpace::HOST> *d_rho;

    }; // end of class ExchangeCorrelationFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/ExchangeCorrelationFE.t.cpp>
#endif // dftefeExchangeCorrelationFE_h
