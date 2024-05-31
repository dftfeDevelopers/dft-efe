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

#include <utils/DataTypeOverloads.h>
#include <xc.h>

namespace dftefe
{
  namespace ksdft
  {
    namespace ExchangeCorrelationFEInternal
    {
      template <typename ValueTypeBasisData,
                typename ValueTypeBasisCoeff,
                utils::MemorySpace memorySpace,
                size_type          dim>
      std::vector<typename ExchangeCorrelationFE<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff,
                                                 memorySpace,
                                                 dim>::RealType>
      getIntegralFieldTimesRho(
        const quadrature::QuadratureValuesContainer<
          typename ExchangeCorrelationFE<ValueTypeBasisData,
                                         ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::RealType,
          memorySpace> &field,
        const quadrature::QuadratureValuesContainer<
          typename ExchangeCorrelationFE<ValueTypeBasisData,
                                         ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::RealType,
          memorySpace> &                                             rho,
        const utils::MemoryStorage<ValueTypeBasisData, memorySpace> &jxwStorage,
        size_type numLocallyOwnedCells,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm)
      {
        using RealType = typename ExchangeCorrelationFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpace,
                                                        dim>::RealType;

        quadrature::QuadratureValuesContainer<RealType, memorySpace> fieldxrho(
          field);
        quadrature::QuadratureValuesContainer<RealType, memorySpace>
          fieldxrhoxJxW(field);

        linearAlgebra::blasLapack::
          hadamardProduct<RealType, RealType, memorySpace>(
            field.nEntries(),
            field.begin(),
            rho.begin(),
            linearAlgebra::blasLapack::ScalarOp::Identity,
            linearAlgebra::blasLapack::ScalarOp::Identity,
            fieldxrho.begin(),
            *linAlgOpContext);

        std::vector<linearAlgebra::blasLapack::ScalarOp> scalarOpA(
          numLocallyOwnedCells, linearAlgebra::blasLapack::ScalarOp::Identity);
        std::vector<linearAlgebra::blasLapack::ScalarOp> scalarOpB(
          numLocallyOwnedCells, linearAlgebra::blasLapack::ScalarOp::Identity);
        std::vector<size_type> mTmp(numLocallyOwnedCells, 0);
        std::vector<size_type> nTmp(numLocallyOwnedCells, 0);
        std::vector<size_type> kTmp(numLocallyOwnedCells, 0);
        std::vector<size_type> stATmp(numLocallyOwnedCells, 0);
        std::vector<size_type> stBTmp(numLocallyOwnedCells, 0);
        std::vector<size_type> stCTmp(numLocallyOwnedCells, 0);

        for (size_type iCell = 0; iCell < numLocallyOwnedCells; iCell++)
          {
            mTmp[iCell] = 1;
            nTmp[iCell] = fieldxrho.getNumberComponents();
            kTmp[iCell] =
              fieldxrho.getQuadratureRuleContainer()->nCellQuadraturePoints(
                iCell);
            stATmp[iCell] = mTmp[iCell] * kTmp[iCell];
            stBTmp[iCell] = nTmp[iCell] * kTmp[iCell];
            stCTmp[iCell] = mTmp[iCell] * nTmp[iCell] * kTmp[iCell];
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
          memoryTransferHost;

        utils::MemoryStorage<size_type, memorySpace> mSize(
          numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> nSize(
          numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> kSize(
          numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> stA(numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> stB(numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> stC(numLocallyOwnedCells);
        memoryTransferHost.copy(numLocallyOwnedCells,
                                mSize.data(),
                                mTmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                nSize.data(),
                                nTmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                kSize.data(),
                                kTmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                stA.data(),
                                stATmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                stB.data(),
                                stBTmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                stC.data(),
                                stCTmp.data());

        linearAlgebra::blasLapack::
          scaleStridedVarBatched<ValueTypeBasisData, RealType, memorySpace>(
            numLocallyOwnedCells,
            scalarOpA.data(),
            scalarOpB.data(),
            stA.data(),
            stB.data(),
            stC.data(),
            mSize.data(),
            nSize.data(),
            kSize.data(),
            jxwStorage.data(),
            fieldxrho.begin(),
            fieldxrhoxJxW.begin(),
            *linAlgOpContext);

        std::vector<RealType> value(fieldxrho.getNumberComponents());

        for (size_type iCell = 0; iCell < fieldxrhoxJxW.nCells(); iCell++)
          {
            for (size_type iComp = 0; iComp < fieldxrho.getNumberComponents();
                 iComp++)
              {
                std::vector<RealType> a(
                  fieldxrho.getQuadratureRuleContainer()->nCellQuadraturePoints(
                    iCell));
                fieldxrhoxJxW
                  .template getCellQuadValues<utils::MemorySpace::HOST>(
                    iCell, iComp, a.data());
                value[iComp] +=
                  std::accumulate(a.begin(), a.end(), (RealType)0);
              }
          }

        int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          value.data(),
          value.size(),
          utils::mpi::Types<RealType>::getMPIDatatype(),
          utils::mpi::MPISum,
          comm);

        return value;
      }
    } // namespace ExchangeCorrelationFEInternal

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::
      ExchangeCorrelationFE(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize)
      : d_cellBlockSize(cellBlockSize)
      , d_linAlgOpContext(linAlgOpContext)
    {
      reinit(electronChargeDensity, feBasisDataStorage);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::
      reinit(const quadrature::QuadratureValuesContainer<RealType, memorySpace>
               &electronChargeDensity,
             std::shared_ptr<
               const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
               feBasisDataStorage)
    {
      d_feBasisDataStorage    = feBasisDataStorage;
      d_electronChargeDensity = &electronChargeDensity;
      d_xcPotentialQuad       = std::make_shared<
        quadrature::QuadratureValuesContainer<RealType, memorySpace>>(
        electronChargeDensity.getQuadratureRuleContainer(),
        electronChargeDensity.getNumberComponents());
      d_feBasisOp =
        std::make_shared<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>>(
          feBasisDataStorage, d_cellBlockSize);

      std::shared_ptr<const basis::BasisDofHandler> basisDofHandlerData =
        feBasisDataStorage->getBasisDofHandler();
      d_feBasisDofHandler = std::dynamic_pointer_cast<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
        basisDofHandlerData);
      utils::throwException(
        d_feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input Field to FEBasisDofHandler "
        "in FEBasisOperations ExchangeCorrelationFE()");

      // compute v_xc = (4/3)*C*\rho^1/3 where C = -3/4 * (3/pi)^1/3

      for (size_type iCell = 0; iCell < electronChargeDensity.nCells(); iCell++)
        {
          for (size_type iComp = 0;
               iComp < electronChargeDensity.getNumberComponents();
               iComp++)
            {
              std::vector<RealType> a(
                electronChargeDensity.getQuadratureRuleContainer()
                  ->nCellQuadraturePoints(iCell));
              electronChargeDensity
                .template getCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                      iComp,
                                                                      a.data());
              for (auto &i : a)
                {
                  i = (RealType)((4.0 / 3) *
                                 Constants::LDA_EXCHANGE_ENERGY_CONST *
                                 std::pow(i, (1.0 / 3)));
                }
              d_xcPotentialQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(
                  iCell, iComp, a.data());
            }
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getLocal(Storage &cellWiseStorage) const
    {
      d_feBasisOp->computeFEMatrices(basis::realspace::LinearLocalOp::IDENTITY,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::LinearLocalOp::IDENTITY,
                                     *d_xcPotentialQuad,
                                     cellWiseStorage,
                                     *d_linAlgOpContext);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::evalEnergy(const utils::mpi::MPIComm &comm)
    {
      // compute e_xc = C*\rho^4/3 where C = -3/4 * (3/pi)^1/3

      auto jxwStorage = d_feBasisDataStorage->getJxWInAllCells();

      const size_type numLocallyOwnedCells =
        d_feBasisDofHandler->nLocallyOwnedCells();

      std::vector<RealType> totalEnergyVec =
        ExchangeCorrelationFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          memorySpace,
          dim>(*d_xcPotentialQuad,
               *d_electronChargeDensity,
               jxwStorage,
               numLocallyOwnedCells,
               d_linAlgOpContext,
               comm);
      d_energy = (3.0 / 4) * totalEnergyVec[0];
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename ExchangeCorrelationFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   memorySpace,
                                   dim>::RealType
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getEnergy() const
    {
      return d_energy;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
