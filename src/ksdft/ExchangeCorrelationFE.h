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
#include <ksdft/RDM1.h>
#include <ksdft/ExcManager.h>
#include <atoms/AtomSevereFunction.h>
#include <utils/Point.h>

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
      // No NLCC.
      ExchangeCorrelationFE(
        const std::string                                                  xcType,
        RDM1<ValueType, memorySpace>                                     &rdm1,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>      linAlgOpContext,
        const size_type                                                   cellBlockSize);

      // With NLCC. Core correction is added internally before every libxc call.
      ExchangeCorrelationFE(
        const std::string                                                  xcType,
        RDM1<ValueType, memorySpace>                                     &rdm1,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>      linAlgOpContext,
        const size_type                                                   cellBlockSize,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>          atomSphericalDataContainerPSP,
        const std::vector<std::string>                                   &atomSymbolVec,
        const std::vector<utils::Point>                                  &atomCoordinates);
      
      ~ExchangeCorrelationFE();

      void
      reinitBasis(RDM1<ValueType, memorySpace> &rdm1);

      void
      reinitField(RDM1<ValueType, memorySpace> &rdm1);

      void
      getLocal(Storage &cellWiseStorage) const override;

      void
      evalEnergy(RDM1<ValueType, memorySpace> &rdm1,
                 const utils::mpi::MPIComm    &comm);

      RealType
      getEnergy() const override;

      const quadrature::QuadratureValuesContainer<ValueType, memorySpace> &
      getFunctionalDerivative() const;

      void
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const override;

      bool
      hasLocalComponent() const override;

      bool
      hasNonLocalComponent() const override;

      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>>
      getHamiltonianFEBasisOperations() const;

      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
      getLinAlgOpContext() const;

    private:
      std::shared_ptr<
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>
        d_xcPotentialQuad;
      std::shared_ptr<
        quadrature::QuadratureValuesContainer<RealType, memorySpace>>
        d_xcPotentialQuadMemspace;
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

      ExcManager<memorySpace> d_excManager;

      std::shared_ptr<
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>
        d_coreCorrectionUPF;

    }; // end of class ExchangeCorrelationFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/ExchangeCorrelationFE.t.cpp>
#endif // dftefeExchangeCorrelationFE_h
