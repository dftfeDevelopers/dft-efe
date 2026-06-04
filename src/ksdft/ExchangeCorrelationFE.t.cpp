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
#include <ksdft/Defaults.h>
#include <ksdft/RDM1FE.h>
#include <ksdft/RDM1Mixing.h>
#include <quadrature/QuadratureValuesContainer.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::
      ExchangeCorrelationFE(
        const std::string                                                  xcType,
        RDM1<ValueType, memorySpace>                                     &rdm1,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize)
      : d_cellBlockSize(cellBlockSize)
      , d_linAlgOpContext(linAlgOpContext)
    {
      d_excManager.init(xcType);
      reinitBasis(rdm1);
      reinitField(rdm1);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::
      ExchangeCorrelationFE(
        const std::string                                                  xcType,
        RDM1<ValueType, memorySpace>                                     &rdm1,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                        atomSphericalDataContainerPSP,
        const std::vector<std::string> &atomSymbolVec,
        const std::vector<utils::Point> &atomCoordinates)
      : d_cellBlockSize(cellBlockSize)
      , d_linAlgOpContext(linAlgOpContext)
    {
      d_excManager.init(xcType);
      reinitBasis(rdm1);

      auto quadRuleContainer =
        d_feBasisDataStorage->getQuadratureRuleContainer();

      d_coreCorrectionUPF = std::make_shared<
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>(
        quadRuleContainer, 1, (RealType)0.0);

      const atoms::AtomSevereFunction<memorySpace>
        rhoCoreCorrection(atomSphericalDataContainerPSP,
                          atomSymbolVec,
                          atomCoordinates,
                          "nlcc",
                          0,
                          1,
                          1,
                          linAlgOpContext.get());

      utils::MemoryStorage<RealType, memorySpace>
        coreCorrectionUPFMemspace(quadRuleContainer->nQuadraturePoints());

      rhoCoreCorrection.template eval<memorySpace>(
        quadRuleContainer->nQuadraturePoints(),
        quadRuleContainer->template getRealPointsPtr<memorySpace>(),
        coreCorrectionUPFMemspace.data());

      utils::MemoryTransfer<memorySpaceHost, memorySpace> memTrans;
      memTrans.copy(coreCorrectionUPFMemspace.size(),
                    d_coreCorrectionUPF->begin(),
                    coreCorrectionUPFMemspace.data());

      reinitField(rdm1);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::~ExchangeCorrelationFE()
    {}

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::reinitBasis(RDM1<ValueType, memorySpace> &rdm1)
    {
      using RDM1FEType = RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>;

      const RDM1FEType *rdm1FEPtr = dynamic_cast<const RDM1FEType *>(&rdm1);

      if (rdm1FEPtr == nullptr)
        {
          const auto *mixPtr =
            dynamic_cast<const RDM1Mixing<ValueType, memorySpace> *>(&rdm1);
          if (mixPtr != nullptr)
            rdm1FEPtr =
              dynamic_cast<const RDM1FEType *>(&mixPtr->getRDM1());
        }

      utils::throwException(
        rdm1FEPtr != nullptr,
        "ExchangeCorrelationFE::reinitBasis: could not resolve RDM1 to RDM1FE. "
        "Pass an RDM1FE or an RDM1Mixing wrapping one.");

      auto feBasisDataStorage = rdm1FEPtr->getFEBasisDataStorage();
      d_feBasisDataStorage    = feBasisDataStorage;
      d_xcPotentialQuad    = std::make_shared<
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>(
        feBasisDataStorage->getQuadratureRuleContainer(), 1);
      d_xcPotentialQuadMemspace = std::make_shared<
        quadrature::QuadratureValuesContainer<RealType, memorySpace>>(
        feBasisDataStorage->getQuadratureRuleContainer(), 1);
      d_feBasisOp =
        std::make_shared<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>>(
          feBasisDataStorage,
          d_cellBlockSize * d_xcPotentialQuad->getNumberComponents());

      std::shared_ptr<const basis::BasisDofHandler> basisDofHandlerData =
        feBasisDataStorage->getBasisDofHandler();
      d_feBasisDofHandler = std::dynamic_pointer_cast<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
        basisDofHandlerData);
      utils::throwException(
        d_feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input Field to FEBasisDofHandler "
        "in ExchangeCorrelationFE::reinitBasis()");
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::reinitField(RDM1<ValueType, memorySpace> &rdm1)
    {
      using RDM1AttrStorage = typename RDM1<ValueType, memorySpace>::AttrStorage;
      using ExcAttrStorage  = std::vector<utils::MemoryStorage<double, memorySpaceHost>>;
      using RDM1FEType =
        RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>;

      const RDM1FEType *p = dynamic_cast<const RDM1FEType *>(&rdm1);
      if (p == nullptr)
        {
          const auto *mix =
            dynamic_cast<const RDM1Mixing<ValueType, memorySpace> *>(&rdm1);
          if (mix != nullptr)
            p = dynamic_cast<const RDM1FEType *>(&mix->getRDM1());
        }
      utils::throwException(
        p != nullptr &&
          p->getFEBasisDataStorage() == d_feBasisDataStorage,
        "ExchangeCorrelationFE::reinitField: RDM1 uses a different "
        "FEBasisDataStorage than the one set in reinitBasis(). "
        "Call reinitBasis(rdm1) whenever the basis is reinitialised.");

      const ExcFamilyType excFamily =
        d_excManager.getExcSSDFunctionalObj()->getExcFamilyType();
      utils::throwException(
        excFamily == ExcFamilyType::LDA,
        "ExchangeCorrelationFE::reinitField: only LDA is currently supported. "
        "GGA (extend with DensityDescrAttr::Grad) and MGGA (extend with "
        "WfcDescrAttr::Tau) are not yet implemented.");

      std::unordered_map<DensityDescrAttr, RDM1AttrStorage> densAttr;
      std::unordered_map<WfcDescrAttr, RDM1AttrStorage>     wfcAttr;
      rdm1.getDescriptors({DensityDescrAttr::Val}, {}, densAttr, wfcAttr);
      const auto &rhoTotalQuad = densAttr.at(DensityDescrAttr::Val)[0];
      utils::throwException(
        rhoTotalQuad.getQuadratureRuleContainer() ==
          d_feBasisDataStorage->getQuadratureRuleContainer(),
        "ExchangeCorrelationFE::reinitField: density Quad from"
        "RDM1::getDescriptors has a different QuadratureRuleContainer "
        "than d_feBasisDataStorage.");

      utils::throwException(
        !rdm1.isNonCollinear(),
        "ExchangeCorrelationFE::reinitField: non-collinear spin is not yet "
        "supported (requires a B_xc vector field and 2x2 spin Hamiltonian).");

      const size_type nQuads = d_xcPotentialQuad->getQuadratureRuleContainer()->nQuadraturePoints();

      ExcAttrStorage spinVals(2);
      spinVals[0].resize(nQuads);
      spinVals[1].resize(nQuads);
      utils::MemoryTransfer<memorySpaceHost, memorySpaceHost> memTrans;
      if (!rdm1.isSpinPolarized())
        {
          // Unpolarized: [0] = ρ_total; ρ↑ = ρ↓ = (ρ_total + ρ_core (from PSP)) / 2
          memTrans.copy(nQuads, spinVals[0].data(), rhoTotalQuad.begin());
          if (d_coreCorrectionUPF != nullptr)
            {
              const RealType *coreCorrectionPtr = d_coreCorrectionUPF->begin();
              for (size_type i = 0; i < nQuads; ++i)
                spinVals[0][i] += static_cast<double>(coreCorrectionPtr[i]);
            }
          for (size_type i = 0; i < nQuads; ++i)
            {
              spinVals[0][i] *= 0.5;
              spinVals[1][i]  = spinVals[0][i];
            }
        }
      else
        {
          // Collinear: [0] = ρ_total, [1] = Mz
          // ρ↑ = (ρ_total + ρ_core + Mz) / 2
          // ρ↓ = (ρ_total + ρ_core - Mz) / 2
          const auto &MzQuad = densAttr.at(DensityDescrAttr::Val)[1];
          utils::throwException(
            MzQuad.getQuadratureRuleContainer() ==
              d_feBasisDataStorage->getQuadratureRuleContainer(),
            "ExchangeCorrelationFE::reinitField: Mz Quad has a different "
            "QuadratureRuleContainer than d_feBasisDataStorage.");
          utils::MemoryStorage<double, memorySpaceHost> rhoTotal(nQuads),
            Mz(nQuads);
          memTrans.copy(nQuads, rhoTotal.data(), rhoTotalQuad.begin());
          memTrans.copy(nQuads, Mz.data(), MzQuad.begin());
          if (d_coreCorrectionUPF != nullptr)
            {
              const RealType *coreCorrectionPtr = d_coreCorrectionUPF->begin();
              for (size_type i = 0; i < nQuads; ++i)
                rhoTotal[i] += static_cast<double>(coreCorrectionPtr[i]);
            }
          for (size_type i = 0; i < nQuads; ++i)
            {
              spinVals[0][i] = 0.5 * (rhoTotal[i] + Mz[i]);
              spinVals[1][i] = 0.5 * (rhoTotal[i] - Mz[i]);
            }
        }
      // GGA extension: also populate ExcAttrStorage for DensityDescrAttr::Grad
      // MGGA extension: also populate ExcAttrStorage for WfcDescrAttr::Tau

      std::unordered_map<DensityDescrAttr, ExcAttrStorage> densAttrSSD;
      densAttrSSD[DensityDescrAttr::Val] = std::move(spinVals);
      std::unordered_map<WfcDescrAttr, ExcAttrStorage> wfcAttrSSD;

      std::unordered_map<xcRemainderOutputDataAttributes,
                         utils::MemoryStorage<double, memorySpaceHost>>
        xDataOut, cDataOut;
      xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
      cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];

      d_excManager.getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
        densAttrSSD, wfcAttrSSD, xDataOut, cDataOut);

      // V_xc = ∂Ex/∂ρ↑ + ∂Ec/∂ρ↑  (equals total V_xc for unpolarized S=1)
      const auto &vxSpinUp =
        xDataOut.at(xcRemainderOutputDataAttributes::pdeDensitySpinUp);
      const auto &vcSpinUp =
        cDataOut.at(xcRemainderOutputDataAttributes::pdeDensitySpinUp);

      size_type count = 0;
      for (size_type iCell = 0; iCell < d_xcPotentialQuad->nCells(); ++iCell)
        {
          const size_type nCellQuads =
            d_xcPotentialQuad->getQuadratureRuleContainer()
              ->nCellQuadraturePoints(iCell);
          std::vector<RealType> cellVxc(nCellQuads);
          for (size_type q = 0; q < nCellQuads; ++q, ++count)
            cellVxc[q] =
              static_cast<RealType>(vxSpinUp[count] + vcSpinUp[count]);
          d_xcPotentialQuad->template setCellValues<memorySpaceHost>(iCell,
                                                                     cellVxc.data());
        }

      utils::MemoryTransfer<memorySpace, memorySpaceHost> memTransH2M;
      memTransH2M.copy(d_xcPotentialQuad->nEntries(),
                       d_xcPotentialQuadMemspace->data(),
                       d_xcPotentialQuad->data());
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
                                     *d_xcPotentialQuadMemspace,
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
                          dim>::evalEnergy(RDM1<ValueType, memorySpace> &rdm1,
                                           const utils::mpi::MPIComm    &comm)
    {
      using RDM1AttrStorage = typename RDM1<ValueType, memorySpace>::AttrStorage;
      using ExcAttrStorage  = std::vector<utils::MemoryStorage<double, memorySpaceHost>>;
      using RDM1FEType =
        RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>;
      const RDM1FEType *p = dynamic_cast<const RDM1FEType *>(&rdm1);
      if (p == nullptr)
        {
          const auto *mix =
            dynamic_cast<const RDM1Mixing<ValueType, memorySpace> *>(&rdm1);
          if (mix != nullptr)
            p = dynamic_cast<const RDM1FEType *>(&mix->getRDM1());
        }
      utils::throwException(
        p != nullptr &&
          p->getFEBasisDataStorage() == d_feBasisDataStorage,
        "ExchangeCorrelationFE::evalEnergy: RDM1 uses a different "
        "FEBasisDataStorage than the one set in reinitBasis(). "
        "Call reinitBasis(rdm1) whenever the basis is reinitialised.");

      const ExcFamilyType excFamily =
        d_excManager.getExcSSDFunctionalObj()->getExcFamilyType();
      utils::throwException(
        excFamily == ExcFamilyType::LDA,
        "ExchangeCorrelationFE::evalEnergy: only LDA is currently supported.");

      std::unordered_map<DensityDescrAttr, RDM1AttrStorage> densAttr;
      std::unordered_map<WfcDescrAttr, RDM1AttrStorage>     wfcAttr;
      rdm1.getDescriptors({DensityDescrAttr::Val}, {}, densAttr, wfcAttr);
      const auto &rhoTotalQuad = densAttr.at(DensityDescrAttr::Val)[0];
      utils::throwException(
        rhoTotalQuad.getQuadratureRuleContainer() ==
          d_feBasisDataStorage->getQuadratureRuleContainer(),
        "ExchangeCorrelationFE::evalEnergy: density Quad from"
        "RDM1::getDescriptors has a different QuadratureRuleContainer "
        "than d_feBasisDataStorage.");

      utils::throwException(
        !rdm1.isNonCollinear(),
        "ExchangeCorrelationFE::evalEnergy: non-collinear spin is not yet "
        "supported (requires a B_xc vector field and 2x2 spin Hamiltonian).");

      const size_type nQuads =
        d_xcPotentialQuad->getQuadratureRuleContainer()->nQuadraturePoints();

      ExcAttrStorage spinVals(2);
      spinVals[0].resize(nQuads);
      spinVals[1].resize(nQuads);
      {
        utils::MemoryTransfer<memorySpaceHost, memorySpaceHost> memTrans;
        if (!rdm1.isSpinPolarized())
          {
            // Unpolarized: [0] = ρ_total; ρ↑ = ρ↓ = (ρ_total + ρ_core) / 2
            memTrans.copy(nQuads, spinVals[0].data(), rhoTotalQuad.begin());
            if (d_coreCorrectionUPF != nullptr)
              {
                const RealType *coreCorrectionPtr = d_coreCorrectionUPF->begin();
                for (size_type i = 0; i < nQuads; ++i)
                  spinVals[0][i] += static_cast<double>(coreCorrectionPtr[i]);
              }
            for (size_type i = 0; i < nQuads; ++i)
              {
                spinVals[0][i] *= 0.5;
                spinVals[1][i]  = spinVals[0][i];
              }
          }
        else
          {
            // Collinear: [0] = ρ_total, [1] = Mz
            // ρ↑ = (ρ_total + ρ_core + Mz) / 2
            // ρ↓ = (ρ_total + ρ_core - Mz) / 2
            const auto &MzQuad = densAttr.at(DensityDescrAttr::Val)[1];
            utils::throwException(
              MzQuad.getQuadratureRuleContainer() ==
                d_feBasisDataStorage->getQuadratureRuleContainer(),
              "ExchangeCorrelationFE::evalEnergy: Mz Quad has a different "
              "QuadratureRuleContainer than d_feBasisDataStorage.");
            utils::MemoryStorage<double, memorySpaceHost> rhoTotal(nQuads),
              Mz(nQuads);
            memTrans.copy(nQuads, rhoTotal.data(), rhoTotalQuad.begin());
            memTrans.copy(nQuads, Mz.data(), MzQuad.begin());
            if (d_coreCorrectionUPF != nullptr)
              {
                const RealType *coreCorrectionPtr = d_coreCorrectionUPF->begin();
                for (size_type i = 0; i < nQuads; ++i)
                  rhoTotal[i] += static_cast<double>(coreCorrectionPtr[i]);
              }
            for (size_type i = 0; i < nQuads; ++i)
              {
                spinVals[0][i] = 0.5 * (rhoTotal[i] + Mz[i]);
                spinVals[1][i] = 0.5 * (rhoTotal[i] - Mz[i]);
              }
          }
      }

      std::unordered_map<DensityDescrAttr, ExcAttrStorage> densAttrSSD;
      densAttrSSD[DensityDescrAttr::Val] = std::move(spinVals);
      std::unordered_map<WfcDescrAttr, ExcAttrStorage> wfcAttrSSD;

      std::unordered_map<xcRemainderOutputDataAttributes,
                         utils::MemoryStorage<double, memorySpaceHost>>
        xDataOut, cDataOut;
      xDataOut[xcRemainderOutputDataAttributes::e];
      cDataOut[xcRemainderOutputDataAttributes::e];

      d_excManager.getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
        densAttrSSD, wfcAttrSSD, xDataOut, cDataOut);

      // exVals[i] = εx * ρ_total[i] (already multiplied by ρ in ExcDensityLDAClass)
      const auto &exVals = xDataOut.at(xcRemainderOutputDataAttributes::e);
      const auto &ecVals = cDataOut.at(xcRemainderOutputDataAttributes::e);
      const auto &jxw =
        d_feBasisDataStorage->getQuadratureRuleContainer()->getJxW();

      RealType totalEnergy = (RealType)0;
      for (size_type i = 0; i < nQuads; ++i)
        totalEnergy += static_cast<RealType>(exVals[i] + ecVals[i]) *
                       static_cast<RealType>(jxw[i]);

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &totalEnergy,
        1,
        utils::mpi::Types<RealType>::getMPIDatatype(),
        utils::mpi::MPISum,
        comm);

      d_energy = totalEnergy;
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

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const quadrature::QuadratureValuesContainer<
      typename ExchangeCorrelationFE<ValueTypeBasisData,
                                     ValueTypeBasisCoeff,
                                     memorySpace,
                                     dim>::ValueType,
      memorySpace> &
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getFunctionalDerivative() const
    {
      return *d_xcPotentialQuadMemspace;
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
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const
    {
      utils::throwException(
        false,
        "Non-Local component not present to call in ExchangeCorrelationFE.h");
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::hasLocalComponent() const
    {
      return true;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::hasNonLocalComponent() const
    {
      return false;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                   ValueTypeBasisData,
                                                   memorySpace,
                                                   dim>>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getHamiltonianFEBasisOperations() const
    {
      return d_feBasisOp;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getLinAlgOpContext() const
    {
      return d_linAlgOpContext;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
