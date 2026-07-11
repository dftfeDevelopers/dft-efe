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
        const std::string             xcType,
        RDM1<ValueType, memorySpace> &rdm1,
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
        const std::string             xcType,
        RDM1<ValueType, memorySpace> &rdm1,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainerPSP,
        const std::vector<std::string> & atomSymbolVec,
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

      const atoms::AtomSuperpositionFunction<memorySpace> rhoCoreCorrection(
        atomSphericalDataContainerPSP,
        atomSymbolVec,
        atomCoordinates,
        "nlcc",
        linAlgOpContext.get());

      utils::MemoryStorage<RealType, memorySpace> coreCorrectionUPFMemspace(
        quadRuleContainer->nQuadraturePoints());

      rhoCoreCorrection.evaluate(
        quadRuleContainer->nQuadraturePoints(),
        atoms::AtomSuperpositionFuncType::Identity,
        quadRuleContainer->template getRealPointsPtr<memorySpace>(),
        coreCorrectionUPFMemspace.data());

      utils::MemoryTransfer<memorySpaceHost, memorySpace> memTrans;
      memTrans.copy(coreCorrectionUPFMemspace.size(),
                    d_coreCorrectionUPF->begin(),
                    coreCorrectionUPFMemspace.data());

      // GGA NLCC: gradient of core correction — output is nQuads*dim values.
      if (d_excManager.getExcSSDFunctionalObj()->getExcFamilyType() ==
          ExcFamilyType::GGA)
        {
          const atoms::AtomSuperpositionFunction<memorySpace> rhoCoreGrad(
            atomSphericalDataContainerPSP,
            atomSymbolVec,
            atomCoordinates,
            "nlcc",
            linAlgOpContext.get());

          d_coreCorrectionGradUPF = std::make_shared<
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>(
            quadRuleContainer, dim, (RealType)0.0);

          utils::MemoryStorage<RealType, memorySpace> coreCorrGradMemspace(
            quadRuleContainer->nQuadraturePoints() * dim);

          rhoCoreGrad.evaluate(
            quadRuleContainer->nQuadraturePoints(),
            atoms::AtomSuperpositionFuncType::Grad,
            quadRuleContainer->template getRealPointsPtr<memorySpace>(),
            coreCorrGradMemspace.data());

          memTrans.copy(coreCorrGradMemspace.size(),
                        d_coreCorrectionGradUPF->data(),
                        coreCorrGradMemspace.data());
        }

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
      using RDM1FEType =
        RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>;

      const RDM1FEType *rdm1FEPtr = dynamic_cast<const RDM1FEType *>(&rdm1);

      if (rdm1FEPtr == nullptr)
        {
          const auto *mixPtr =
            dynamic_cast<const RDM1Mixing<ValueType, memorySpace> *>(&rdm1);
          if (mixPtr != nullptr)
            rdm1FEPtr = dynamic_cast<const RDM1FEType *>(&mixPtr->getRDM1());
        }

      utils::throwException(
        rdm1FEPtr != nullptr,
        "ExchangeCorrelationFE::reinitBasis: could not resolve RDM1 to RDM1FE. "
        "Pass an RDM1FE or an RDM1Mixing wrapping one.");

      auto feBasisDataStorage = rdm1FEPtr->getFEBasisDataStorage();
      d_feBasisDataStorage    = feBasisDataStorage;

      d_spinMode = rdm1.spinMode();
      d_S = (d_spinMode == SpinMode::Unpolarized) ? 1 : 2;

      // K = number of non-zero XC spin blocks in hamiltonain:
      //   unpolarized   → K=1
      //   collinear     → K=2  (V_xc^↑, V_xc^↓)
      //   non-collinear → K=4  (V_xc^↑↑, V_xc^↑↓, V_xc^↓↑, V_xc^↓↓)
      const size_type K = (d_spinMode == SpinMode::NonCollinear) ? 4 :
                          (d_spinMode == SpinMode::Collinear)    ? 2 : 1;

      if (d_spinMode == SpinMode::NonCollinear)
        {
          d_layout       = SpinStorageLayout::SpinFastest;
          d_spinIdsFilled = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        }
      else
        {
          d_layout = SpinStorageLayout::DofFastest;
          d_spinIdsFilled =
            (d_spinMode == SpinMode::Collinear)
              ? std::vector<std::pair<size_type, size_type>>{{0, 0}, {1, 1}}
              : std::vector<std::pair<size_type, size_type>>{{0, 0}};
        }

      d_xcPotentialQuadMemspace.resize(K);
      for (size_type k = 0; k < K; ++k)
        d_xcPotentialQuadMemspace[k] =
          quadrature::QuadratureValuesContainer<RealType, memorySpace>(
            feBasisDataStorage->getQuadratureRuleContainer(), 1);

      const bool isGGA = (d_excManager.getExcSSDFunctionalObj()
                            ->getExcFamilyType() == ExcFamilyType::GGA);
      if (isGGA)
        {
          d_derExcWithSigmaTimesGradRhoQuadMemspace.resize(K);
          for (size_type k = 0; k < K; ++k)
            d_derExcWithSigmaTimesGradRhoQuadMemspace[k] =
              quadrature::QuadratureValuesContainer<RealType, memorySpace>(
                feBasisDataStorage->getQuadratureRuleContainer(), dim);
        }
      d_feBasisOp =
        std::make_shared<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>>(
          feBasisDataStorage,
          d_cellBlockSize,
          K);

      std::shared_ptr<const basis::BasisDofHandler> basisDofHandlerData =
        feBasisDataStorage->getBasisDofHandler();
      d_feBasisDofHandler = std::dynamic_pointer_cast<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
        basisDofHandlerData);
      utils::throwException(
        d_feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input Field to FEBasisDofHandler "
        "in ExchangeCorrelationFE::reinitBasis()");

      const size_type nLocallyOwnedCells =
        d_feBasisDofHandler->nLocallyOwnedCells();
      d_numCellDofs.resize(nLocallyOwnedCells);
      d_basisOverlapSize = 0;
      for (size_type c = 0; c < nLocallyOwnedCells; ++c)
        {
          d_numCellDofs[c]    = d_feBasisDofHandler->nCellDofs(c);
          d_basisOverlapSize += d_numCellDofs[c] * d_numCellDofs[c];
        }
      d_xcCellWiseTemp.resize(K * d_basisOverlapSize, ValueType(0));
      if (isGGA)
        d_sigmaGradRhoCellStorage.resize(K * d_basisOverlapSize, ValueType(0));
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
      using RDM1AttrStorage =
        typename RDM1<ValueType, memorySpace>::AttrStorage;
      using ExcAttrStorage =
        std::vector<utils::MemoryStorage<double, memorySpaceHost>>;
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
        p != nullptr && p->getFEBasisDataStorage() == d_feBasisDataStorage,
        "ExchangeCorrelationFE::reinitField: RDM1 uses a different "
        "FEBasisDataStorage than the one set in reinitBasis(). "
        "Call reinitBasis(rdm1) whenever the basis is reinitialised.");

      const ExcFamilyType excFamily =
        d_excManager.getExcSSDFunctionalObj()->getExcFamilyType();
      utils::throwException(
        excFamily == ExcFamilyType::LDA || excFamily == ExcFamilyType::GGA,
        "ExchangeCorrelationFE::reinitField: only LDA and GGA are currently "
        "supported. MGGA (extend with WfcDescrAttr::Tau) is not yet "
        "implemented.");

      utils::throwException(
        d_spinMode != SpinMode::NonCollinear,
        "ExchangeCorrelationFE::reinitField: non-collinear XC not yet "
        "implemented.");

      const bool      isGGA       = (excFamily == ExcFamilyType::GGA);
      const bool      isCollinear = (d_spinMode == SpinMode::Collinear);
      const size_type K           = d_xcPotentialQuadMemspace.size();

      std::set<DensityDescrAttr> descrSet = {DensityDescrAttr::Val};
      if (isGGA)
        descrSet.insert(DensityDescrAttr::Grad);

      std::unordered_map<DensityDescrAttr, RDM1AttrStorage> densAttr;
      std::unordered_map<WfcDescrAttr, RDM1AttrStorage>     wfcAttr;
      rdm1.getDescriptors(descrSet, {}, densAttr, wfcAttr);
      const auto &rhoTotalQuad = densAttr.at(DensityDescrAttr::Val)[0];
      utils::throwException(
        rhoTotalQuad.getQuadratureRuleContainer() ==
          d_feBasisDataStorage->getQuadratureRuleContainer(),
        "ExchangeCorrelationFE::reinitField: density Quad from "
        "RDM1::getDescriptors has a different QuadratureRuleContainer "
        "than d_feBasisDataStorage.");

      const size_type nQuads =
        d_xcPotentialQuadMemspace[0].getQuadratureRuleContainer()
          ->nQuadraturePoints();

      std::vector<quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>
        xcPotentialQuad(K,
          quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
            d_feBasisDataStorage->getQuadratureRuleContainer(), 1));

      ExcAttrStorage spinVals(2);
      spinVals[0].resize(nQuads);
      spinVals[1].resize(nQuads);
      utils::MemoryTransfer<memorySpaceHost, memorySpaceHost> memTrans;
      utils::MemoryStorage<double, memorySpaceHost>           rhoTotal(nQuads);
      memTrans.copy(nQuads, rhoTotal.data(), rhoTotalQuad.begin());
      if (d_coreCorrectionUPF != nullptr)
        {
          const RealType *corePtr = d_coreCorrectionUPF->begin();
          for (size_type i = 0; i < nQuads; ++i)
            rhoTotal[i] += static_cast<double>(corePtr[i]);
        }

      if (isCollinear)
        {
          // val[1]=Mz
          const auto &MzQuad = densAttr.at(DensityDescrAttr::Val)[1];
          utils::throwException(
            MzQuad.getQuadratureRuleContainer() ==
              d_feBasisDataStorage->getQuadratureRuleContainer(),
            "ExchangeCorrelationFE::reinitField: Mz Quad has a different "
            "QuadratureRuleContainer than d_feBasisDataStorage.");
          utils::MemoryStorage<double, memorySpaceHost> Mz(nQuads);
          memTrans.copy(nQuads, Mz.data(), MzQuad.begin());
          for (size_type i = 0; i < nQuads; ++i)
            {
              spinVals[0][i] = 0.5 * (rhoTotal[i] + Mz[i]);
              spinVals[1][i] = 0.5 * (rhoTotal[i] - Mz[i]);
            }
        }
      else
        {
          for (size_type i = 0; i < nQuads; ++i)
            {
              spinVals[0][i] = 0.5 * rhoTotal[i];
              spinVals[1][i] = 0.5 * rhoTotal[i];
            }
        }

      // --- spin gradient density (GGA only) ---
      // grad[0]=∇ρ_tot, [1]=∇Mz (collinear); each Quad has dim components.
      ExcAttrStorage spinGradVals(2);
      if (isGGA)
        {
          utils::MemoryStorage<double, memorySpaceHost> gTot(nQuads * dim);
          const double *gTotPtr = densAttr.at(DensityDescrAttr::Grad)[0].data();
          for (size_type i = 0; i < nQuads * dim; ++i)
            gTot.data()[i] = gTotPtr[i];
          if (d_coreCorrectionGradUPF != nullptr)
            {
              const RealType *coreGradPtr = d_coreCorrectionGradUPF->data();
              for (size_type i = 0; i < nQuads * dim; ++i)
                gTot.data()[i] += static_cast<double>(coreGradPtr[i]);
            }

          spinGradVals[0].resize(nQuads * dim);
          spinGradVals[1].resize(nQuads * dim);

          if (isCollinear)
            {
              // grad[1]=∇Mz
              const double *gMzPtr =
                densAttr.at(DensityDescrAttr::Grad)[1].data();
              for (size_type i = 0; i < nQuads * dim; ++i)
                {
                  spinGradVals[0][i] = 0.5 * (gTot.data()[i] + gMzPtr[i]);
                  spinGradVals[1][i] = 0.5 * (gTot.data()[i] - gMzPtr[i]);
                }
            }
          else
            {
              for (size_type i = 0; i < nQuads * dim; ++i)
                {
                  spinGradVals[0][i] = 0.5 * gTot.data()[i];
                  spinGradVals[1][i] = 0.5 * gTot.data()[i];
                }
            }
        }

      // --- assemble input/output maps for libxc ---
      std::unordered_map<DensityDescrAttr, ExcAttrStorage> densAttrSSD;
      densAttrSSD[DensityDescrAttr::Val] = std::move(spinVals);
      if (isGGA)
        {
          densAttrSSD[DensityDescrAttr::Grad] = std::move(spinGradVals);
        }
      std::unordered_map<WfcDescrAttr, ExcAttrStorage> wfcAttrSSD;

      std::unordered_map<xcRemainderOutputDataAttributes,
                         utils::MemoryStorage<double, memorySpaceHost>>
        xDataOut, cDataOut;
      xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
      cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinUp];
      if (isCollinear)
        {
          xDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];
          cDataOut[xcRemainderOutputDataAttributes::pdeDensitySpinDown];
        }
      if (isGGA)
        {
          xDataOut[xcRemainderOutputDataAttributes::pdeSigma];
          cDataOut[xcRemainderOutputDataAttributes::pdeSigma];
        }

      d_excManager.getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
        densAttrSSD, wfcAttrSSD, xDataOut, cDataOut);

      // --- fill xcPotentialQuad: K components per quad, layout [q*K + k] ---
      // unpolarized (K=1): k=0 → V_xc
      // collinear   (K=2): k=0 → V_xc^↑,  k=1 → V_xc^↓
      const auto &vxSpinUp =
        xDataOut.at(xcRemainderOutputDataAttributes::pdeDensitySpinUp);
      const auto &vcSpinUp =
        cDataOut.at(xcRemainderOutputDataAttributes::pdeDensitySpinUp);
      const utils::MemoryStorage<double, memorySpaceHost> *vxSpinDownPtr =
        nullptr;
      const utils::MemoryStorage<double, memorySpaceHost> *vcSpinDownPtr =
        nullptr;
      if (isCollinear)
        {
          vxSpinDownPtr =
            &xDataOut.at(xcRemainderOutputDataAttributes::pdeDensitySpinDown);
          vcSpinDownPtr =
            &cDataOut.at(xcRemainderOutputDataAttributes::pdeDensitySpinDown);
        }

      size_type count = 0;
      for (size_type iCell = 0; iCell < xcPotentialQuad[0].nCells(); ++iCell)
        {
          const size_type nCellQuads =
            xcPotentialQuad[0].getQuadratureRuleContainer()
              ->nCellQuadraturePoints(iCell);
          std::vector<RealType> cellVxc(nCellQuads * K);
          for (size_type q = 0; q < nCellQuads; ++q, ++count)
            {
              cellVxc[q] =
                static_cast<RealType>(vxSpinUp[count] + vcSpinUp[count]);
              if (isCollinear)
                cellVxc[nCellQuads + q] = static_cast<RealType>(
                  (*vxSpinDownPtr)[count] + (*vcSpinDownPtr)[count]);
            }
          xcPotentialQuad[0].template setCellValues<memorySpaceHost>(
            iCell, cellVxc.data());
          if (isCollinear)
            xcPotentialQuad[1].template setCellValues<memorySpaceHost>(
              iCell, cellVxc.data() + nCellQuads);
        }

      utils::MemoryTransfer<memorySpace, memorySpaceHost> memTransH2M;
      for (size_type k = 0; k < K; ++k)
        memTransH2M.copy(xcPotentialQuad[k].nEntries(),
                         d_xcPotentialQuadMemspace[k].data(),
                         xcPotentialQuad[k].data());

      // --- fill d_derExcWithSigmaTimesGradRhoQuadMemspace (GGA only) ---
      // K*dim components per quad, layout [q*K*dim + s*dim + d]
      // s=0: f^↑_d = 2*(dEx/dσ_αα+dEc/dσ_αα)*∇ρ↑_d +
      // (dEx/dσ_αβ+dEc/dσ_αβ)*∇ρ↓_d s=1: f^↓_d = 2*(dEx/dσ_ββ+dEc/dσ_ββ)*∇ρ↓_d
      // + (dEx/dσ_αβ+dEc/dσ_αβ)*∇ρ↑_d
      if (isGGA)
        {
          const size_type nSpins = isCollinear ? 2 : 1;

          // pdexSigma/pdecSigma layout: [total_quads][3 sigma components]
          //   flat index: 3*globalQuadIdx + i
          //   i=0: σ_αα, i=1: σ_αβ (cross), i=2: σ_ββ
          const auto &pdexSigma =
            xDataOut.at(xcRemainderOutputDataAttributes::pdeSigma);
          const auto &pdecSigma =
            cDataOut.at(xcRemainderOutputDataAttributes::pdeSigma);
          // gUpPtr/gDownPtr layout: [total_quads][dim]
          //   flat index: globalQuadIdx*dim + d
          const double *gUpPtr =
            densAttrSSD.at(DensityDescrAttr::Grad)[0].data();
          const double *gDownPtr =
            densAttrSSD.at(DensityDescrAttr::Grad)[1].data();

          std::vector<quadrature::QuadratureValuesContainer<RealType,
                                                            memorySpaceHost>>
            derExcSigmaGradRhoHost(K,
              quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
                d_derExcWithSigmaTimesGradRhoQuadMemspace[0]
                  .getQuadratureRuleContainer(),
                dim));

          // cellSigmaGradField layout: [K (spin)][nQuadPts][dim]
          //   flat index: s*nQuadPts*dim + q*dim + d
          size_type globalQuadIdx = 0;
          for (size_type iCell = 0; iCell < derExcSigmaGradRhoHost[0].nCells();
               ++iCell)
            {
              const size_type nQuadPts =
                derExcSigmaGradRhoHost[0].getQuadratureRuleContainer()
                  ->nCellQuadraturePoints(iCell);
              std::vector<RealType> cellSigmaGradField(nQuadPts * K * dim);
              for (size_type q = 0; q < nQuadPts; ++q, ++globalQuadIdx)
                {
                  const double dExcdSigmaCross =
                    pdexSigma[3 * globalQuadIdx + 1] +
                    pdecSigma[3 * globalQuadIdx + 1];
                  for (size_type s = 0; s < nSpins; ++s)
                    {
                      // diagonal sigma index: 2*s  (s=0 → σ_αα, s=1 → σ_ββ)
                      const double twoDExcdSigmaDiag =
                        2.0 * (pdexSigma[3 * globalQuadIdx + 2 * s] +
                               pdecSigma[3 * globalQuadIdx + 2 * s]);
                      const double *gradRhoSameSpin =
                        isCollinear ? (s == 0 ? gUpPtr : gDownPtr) : gUpPtr;
                      const double *gradRhoOtherSpin =
                        isCollinear ? (s == 0 ? gDownPtr : gUpPtr) : gUpPtr;
                      for (size_type d = 0; d < dim; ++d)
                        {
                          cellSigmaGradField[s * nQuadPts * dim + q * dim + d] =
                            static_cast<RealType>(
                              twoDExcdSigmaDiag *
                                gradRhoSameSpin[globalQuadIdx * dim + d] +
                              dExcdSigmaCross *
                                gradRhoOtherSpin[globalQuadIdx * dim + d]);
                        }
                    }
                }
              for (size_type s = 0; s < nSpins; ++s)
                derExcSigmaGradRhoHost[s].template setCellValues<memorySpaceHost>(
                  iCell, cellSigmaGradField.data() + s * nQuadPts * dim);
            }

          for (size_type k = 0; k < K; ++k)
            memTransH2M.copy(derExcSigmaGradRhoHost[k].nEntries(),
                             d_derExcWithSigmaTimesGradRhoQuadMemspace[k].data(),
                             derExcSigmaGradRhoHost[k].data());
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
      // LDA: ∫ V_xc · N_i N_j dV — K blocks into d_xcCellWiseTemp
      d_feBasisOp->computeFEMatrices(basis::realspace::LinearLocalOp::IDENTITY,
                                     basis::realspace::VectorMathOp::MULT,
                                     d_xcPotentialQuadMemspace,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::LinearLocalOp::IDENTITY,
                                     d_xcCellWiseTemp,
                                     *d_linAlgOpContext);

      // GGA: ∫ f · ∇(N_i N_j) dV — accumulated into d_xcCellWiseTemp via axpy
      if (!d_derExcWithSigmaTimesGradRhoQuadMemspace.empty())
        {
          d_feBasisOp->computeFEMatrices(
            d_derExcWithSigmaTimesGradRhoQuadMemspace,
            basis::realspace::VectorMathOp::DOT,
            basis::realspace::LinearLocalOp::GRAD,
            d_sigmaGradRhoCellStorage,
            *d_linAlgOpContext);

          linearAlgebra::blasLapack::axpy<ValueType, ValueType, memorySpace>(
            d_sigmaGradRhoCellStorage.size(),
            ValueType(1.0),
            d_sigmaGradRhoCellStorage.data(),
            1,
            d_xcCellWiseTemp.data(),
            1,
            *d_linAlgOpContext);
        }

      // Zero-init cellWiseStorage: size = Σ_c (S·d_c)² = S² · d_basisOverlapSize
      cellWiseStorage.resize(d_S * d_S * d_basisOverlapSize, ValueType(0));

      // Block-copy d_xcCellWiseTemp → cellWiseStorage
      HamiltonianSpinBlockCopyKernels<ValueType, memorySpace>::copyIntoBlock(
        d_xcCellWiseTemp,
        cellWiseStorage,
        d_S,
        d_layout,
        d_spinIdsFilled,
        d_numCellDofs,
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
                                           const utils::mpi::MPIComm &   comm)
    {
      utils::throwException(
        d_spinMode != SpinMode::NonCollinear,
        "ExchangeCorrelationFE::evalEnergy: non-collinear XC not yet "
        "implemented.");

      using RDM1AttrStorage =
        typename RDM1<ValueType, memorySpace>::AttrStorage;
      using ExcAttrStorage =
        std::vector<utils::MemoryStorage<double, memorySpaceHost>>;
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
        p != nullptr && p->getFEBasisDataStorage() == d_feBasisDataStorage,
        "ExchangeCorrelationFE::evalEnergy: RDM1 uses a different "
        "FEBasisDataStorage than the one set in reinitBasis(). "
        "Call reinitBasis(rdm1) whenever the basis is reinitialised.");

      const ExcFamilyType excFamily =
        d_excManager.getExcSSDFunctionalObj()->getExcFamilyType();
      utils::throwException(
        excFamily == ExcFamilyType::LDA || excFamily == ExcFamilyType::GGA,
        "ExchangeCorrelationFE::evalEnergy: only LDA and GGA are currently "
        "supported.");

      const bool isGGA        = (excFamily == ExcFamilyType::GGA);
      const bool isCollinear  = (d_spinMode == SpinMode::Collinear);

      std::set<DensityDescrAttr> descrSet = {DensityDescrAttr::Val};
      if (isGGA)
        descrSet.insert(DensityDescrAttr::Grad);

      std::unordered_map<DensityDescrAttr, RDM1AttrStorage> densAttr;
      std::unordered_map<WfcDescrAttr, RDM1AttrStorage>     wfcAttr;
      rdm1.getDescriptors(descrSet, {}, densAttr, wfcAttr);
      const auto &rhoTotalQuad = densAttr.at(DensityDescrAttr::Val)[0];
      utils::throwException(
        rhoTotalQuad.getQuadratureRuleContainer() ==
          d_feBasisDataStorage->getQuadratureRuleContainer(),
        "ExchangeCorrelationFE::evalEnergy: density Quad from "
        "RDM1::getDescriptors has a different QuadratureRuleContainer "
        "than d_feBasisDataStorage.");

      const size_type nQuads =
        d_xcPotentialQuadMemspace[0].getQuadratureRuleContainer()
          ->nQuadraturePoints();

      // --- spin density ---
      ExcAttrStorage spinVals(2);
      spinVals[0].resize(nQuads);
      spinVals[1].resize(nQuads);

      utils::MemoryTransfer<memorySpaceHost, memorySpaceHost> memTrans;
      utils::MemoryStorage<double, memorySpaceHost>           rhoTotal(nQuads);
      memTrans.copy(nQuads, rhoTotal.data(), rhoTotalQuad.begin());
      if (d_coreCorrectionUPF != nullptr)
        {
          const RealType *corePtr = d_coreCorrectionUPF->begin();
          for (size_type i = 0; i < nQuads; ++i)
            rhoTotal[i] += static_cast<double>(corePtr[i]);
        }

      if (isCollinear)
        {
          // val[1]=Mz
          const auto &MzQuad = densAttr.at(DensityDescrAttr::Val)[1];
          utils::throwException(
            MzQuad.getQuadratureRuleContainer() ==
              d_feBasisDataStorage->getQuadratureRuleContainer(),
            "ExchangeCorrelationFE::evalEnergy: Mz Quad has a different "
            "QuadratureRuleContainer than d_feBasisDataStorage.");
          utils::MemoryStorage<double, memorySpaceHost> Mz(nQuads);
          memTrans.copy(nQuads, Mz.data(), MzQuad.begin());
          for (size_type i = 0; i < nQuads; ++i)
            {
              spinVals[0][i] = 0.5 * (rhoTotal[i] + Mz[i]);
              spinVals[1][i] = 0.5 * (rhoTotal[i] - Mz[i]);
            }
        }
      else
        {
          for (size_type i = 0; i < nQuads; ++i)
            {
              spinVals[0][i] = 0.5 * rhoTotal[i];
              spinVals[1][i] = 0.5 * rhoTotal[i];
            }
        }

      // --- GGA: spin gradient density ---
      std::unordered_map<DensityDescrAttr, ExcAttrStorage> densAttrSSD;
      densAttrSSD[DensityDescrAttr::Val] = std::move(spinVals);
      if (isGGA)
        {
          utils::MemoryStorage<double, memorySpaceHost> gTot(nQuads * dim);
          const double *gTotPtr = densAttr.at(DensityDescrAttr::Grad)[0].data();
          for (size_type i = 0; i < nQuads * dim; ++i)
            gTot.data()[i] = gTotPtr[i];
          if (d_coreCorrectionGradUPF != nullptr)
            {
              const RealType *coreGradPtr = d_coreCorrectionGradUPF->data();
              for (size_type i = 0; i < nQuads * dim; ++i)
                gTot.data()[i] += static_cast<double>(coreGradPtr[i]);
            }

          ExcAttrStorage spinGradVals(2);
          spinGradVals[0].resize(nQuads * dim);
          spinGradVals[1].resize(nQuads * dim);
          if (isCollinear)
            {
              const double *gMzPtr =
                densAttr.at(DensityDescrAttr::Grad)[1].data();
              for (size_type i = 0; i < nQuads * dim; ++i)
                {
                  spinGradVals[0][i] = 0.5 * (gTot.data()[i] + gMzPtr[i]);
                  spinGradVals[1][i] = 0.5 * (gTot.data()[i] - gMzPtr[i]);
                }
            }
          else
            {
              for (size_type i = 0; i < nQuads * dim; ++i)
                {
                  spinGradVals[0][i] = 0.5 * gTot.data()[i];
                  spinGradVals[1][i] = 0.5 * gTot.data()[i];
                }
            }
          densAttrSSD[DensityDescrAttr::Grad] = std::move(spinGradVals);
        }
      std::unordered_map<WfcDescrAttr, ExcAttrStorage> wfcAttrSSD;

      std::unordered_map<xcRemainderOutputDataAttributes,
                         utils::MemoryStorage<double, memorySpaceHost>>
        xDataOut, cDataOut;
      xDataOut[xcRemainderOutputDataAttributes::e];
      cDataOut[xcRemainderOutputDataAttributes::e];

      d_excManager.getExcSSDFunctionalObj()->computeRhoTauDependentXCData(
        densAttrSSD, wfcAttrSSD, xDataOut, cDataOut);

      // exVals[i] = εx * ρ_total[i] (already multiplied by ρ in
      // ExcDensityLDAClass)
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
    const std::vector<quadrature::QuadratureValuesContainer<
      typename ExchangeCorrelationFE<ValueTypeBasisData,
                                     ValueTypeBasisCoeff,
                                     memorySpace,
                                     dim>::RealType,
      memorySpace>> &
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getFunctionalDerivative() const
    {
      return d_xcPotentialQuadMemspace;
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

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExcFamilyType
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getExcFamilyType() const
    {
      return d_excManager.getExcSSDFunctionalObj()->getExcFamilyType();
    }

  } // end of namespace ksdft
} // end of namespace dftefe
