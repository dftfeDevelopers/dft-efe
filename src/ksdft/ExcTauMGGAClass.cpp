// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Srinibas Nandi, Vishal Subramanian, Sambit Das
//

#include <ksdft/ExcTauMGGAClass.h>
#include <utils/Exceptions.h>
#include <ksdft/ExcManagerKernels.h>
#include <utils/DeviceAPICalls.h>

namespace dftefe
{
  namespace ksdft
  {
    template <dftefe::utils::MemorySpace memorySpace>
    ExcTauMGGAClass<memorySpace>::ExcTauMGGAClass(
      std::shared_ptr<xc_func_type> &funcXPtr,
      std::shared_ptr<xc_func_type> &funcCPtr,
      std::string                    XCType)
      : ExcSSDFunctionalBaseClass<memorySpace>(
          ExcFamilyType::TauMGGA,
          densityFamilyType::GGA,
          std::set<DensityDescrAttr>{DensityDescrAttr::Val,
                                     DensityDescrAttr::Grad},
          std::set<WfcDescrAttr>{WfcDescrAttr::Tau})
    {
      d_funcXPtr = funcXPtr;
      d_funcCPtr = funcCPtr;
      d_XCType   = XCType;
    }

    template <dftefe::utils::MemorySpace memorySpace>
    ExcTauMGGAClass<memorySpace>::~ExcTauMGGAClass()
    {}

    template <dftefe::utils::MemorySpace memorySpace>
    void
    ExcTauMGGAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const
    {
      const std::vector<xcRemainderOutputDataAttributes>
        allowedOutputDataAttributes = {
          xcRemainderOutputDataAttributes::e,
          xcRemainderOutputDataAttributes::pdeDensitySpinUp,
          xcRemainderOutputDataAttributes::pdeDensitySpinDown,
          xcRemainderOutputDataAttributes::pdeSigma,
          xcRemainderOutputDataAttributes::pdeTauSpinUp,
          xcRemainderOutputDataAttributes::pdeTauSpinDown};

      for (size_t i = 0; i < outputDataAttributes.size(); i++)
        {
          bool isFound = false;
          for (size_t j = 0; j < allowedOutputDataAttributes.size(); j++)
            {
              if (outputDataAttributes[i] == allowedOutputDataAttributes[j])
                isFound = true;
            }

          std::string errMsg =
            "xcRemainderOutputDataAttributes do not match the allowed choices for the family type.";
          dftefe::utils::throwException(isFound, errMsg);
        }
    }

    template <dftefe::utils::MemorySpace memorySpace>
    void
    ExcTauMGGAClass<memorySpace>::computeRhoTauDependentXCData(
      const std::unordered_map<
        DensityDescrAttr,
        typename ExcSSDFunctionalBaseClass<memorySpace>::AttrStorage>
        &densityAttrVals,
      const std::unordered_map<
        WfcDescrAttr,
        typename ExcSSDFunctionalBaseClass<memorySpace>::AttrStorage>
        &wfcAttrVals,
      std::unordered_map<
        xcRemainderOutputDataAttributes,
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>
        &xDataOut,
      std::unordered_map<
        xcRemainderOutputDataAttributes,
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>
        &cDataOut) const
    {
      const double rhoThresholdMgga   = 1e-12;
      const double sigmaThresholdMgga = 1e-20;
      const double tauThresholdMgga   = 1e-10;

      const auto &densityValuesSpinUp =
        densityAttrVals.at(DensityDescrAttr::Val)[0];
      const auto &densityValuesSpinDown =
        densityAttrVals.at(DensityDescrAttr::Val)[1];
      const auto &gradValuesSpinUp =
        densityAttrVals.at(DensityDescrAttr::Grad)[0];
      const auto &gradValuesSpinDown =
        densityAttrVals.at(DensityDescrAttr::Grad)[1];
      const auto &tauValuesSpinUp   = wfcAttrVals.at(WfcDescrAttr::Tau)[0];
      const auto &tauValuesSpinDown = wfcAttrVals.at(WfcDescrAttr::Tau)[1];

      const size_type nquad = densityValuesSpinUp.size();

      std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;
      for (const auto &element : xDataOut)
        outputDataAttributes.push_back(element.first);

      checkInputOutputDataAttributesConsistency(outputDataAttributes);

      if (this->s_densityValues.size() != 2 * nquad)
        this->s_densityValues.resize(2 * nquad);
      if (this->s_sigmaValues.size() != 3 * nquad)
        this->s_sigmaValues.resize(3 * nquad);
      if (this->s_tauValues.size() != 2 * nquad)
        this->s_tauValues.resize(2 * nquad);

      auto &densityValues = this->s_densityValues;
      auto &sigmaValues   = this->s_sigmaValues;
      auto &tauValues     = this->s_tauValues;
      sigmaValues.setValue(0.0);

      if (this->s_pdexDensityValues.size() != 2 * nquad)
        this->s_pdexDensityValues.resize(2 * nquad);
      if (this->s_pdecDensityValues.size() != 2 * nquad)
        this->s_pdecDensityValues.resize(2 * nquad);

      auto &pdexDensityValues = this->s_pdexDensityValues;
      auto &pdecDensityValues = this->s_pdecDensityValues;

      if (this->s_pdexTauValues.size() != 2 * nquad)
        this->s_pdexTauValues.resize(2 * nquad);
      if (this->s_pdecTauValues.size() != 2 * nquad)
        this->s_pdecTauValues.resize(2 * nquad);

      auto &pdexTauValues = this->s_pdexTauValues;
      auto &pdecTauValues = this->s_pdecTauValues;

      auto &exValues =
        (xDataOut.find(xcRemainderOutputDataAttributes::e) != xDataOut.end()) ?
          xDataOut.find(xcRemainderOutputDataAttributes::e)->second :
          this->s_exValues;
      auto &ecValues =
        (cDataOut.find(xcRemainderOutputDataAttributes::e) != cDataOut.end()) ?
          cDataOut.find(xcRemainderOutputDataAttributes::e)->second :
          this->s_ecValues;

      auto &pdexDensitySpinUpValues =
        (xDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinUp) !=
         xDataOut.end()) ?
          xDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinUp)
            ->second :
          this->s_pdexDensitySpinUpValues;
      auto &pdexDensitySpinDownValues =
        (xDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinDown) !=
         xDataOut.end()) ?
          xDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinDown)
            ->second :
          this->s_pdexDensitySpinDownValues;
      auto &pdecDensitySpinUpValues =
        (cDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinUp) !=
         cDataOut.end()) ?
          cDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinUp)
            ->second :
          this->s_pdecDensitySpinUpValues;
      auto &pdecDensitySpinDownValues =
        (cDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinDown) !=
         cDataOut.end()) ?
          cDataOut.find(xcRemainderOutputDataAttributes::pdeDensitySpinDown)
            ->second :
          this->s_pdecDensitySpinDownValues;

      auto &pdexSigmaValues =
        (xDataOut.find(xcRemainderOutputDataAttributes::pdeSigma) !=
         xDataOut.end()) ?
          xDataOut.find(xcRemainderOutputDataAttributes::pdeSigma)->second :
          this->s_pdexSigmaValues;
      auto &pdecSigmaValues =
        (cDataOut.find(xcRemainderOutputDataAttributes::pdeSigma) !=
         cDataOut.end()) ?
          cDataOut.find(xcRemainderOutputDataAttributes::pdeSigma)->second :
          this->s_pdecSigmaValues;

      auto &pdexTauSpinUpValues =
        (xDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinUp) !=
         xDataOut.end()) ?
          xDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinUp)->second :
          this->s_pdexTauSpinUpValues;
      auto &pdexTauSpinDownValues =
        (xDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinDown) !=
         xDataOut.end()) ?
          xDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinDown)
            ->second :
          this->s_pdexTauSpinDownValues;
      auto &pdecTauSpinUpValues =
        (cDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinUp) !=
         cDataOut.end()) ?
          cDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinUp)->second :
          this->s_pdecTauSpinUpValues;
      auto &pdecTauSpinDownValues =
        (cDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinDown) !=
         cDataOut.end()) ?
          cDataOut.find(xcRemainderOutputDataAttributes::pdeTauSpinDown)
            ->second :
          this->s_pdecTauSpinDownValues;

      if (exValues.size() != nquad)
        exValues.resize(nquad);
      if (ecValues.size() != nquad)
        ecValues.resize(nquad);
      if (pdexDensitySpinUpValues.size() != nquad)
        pdexDensitySpinUpValues.resize(nquad);
      if (pdexDensitySpinDownValues.size() != nquad)
        pdexDensitySpinDownValues.resize(nquad);
      if (pdecDensitySpinUpValues.size() != nquad)
        pdecDensitySpinUpValues.resize(nquad);
      if (pdecDensitySpinDownValues.size() != nquad)
        pdecDensitySpinDownValues.resize(nquad);
      if (pdexSigmaValues.size() != 3 * nquad)
        pdexSigmaValues.resize(3 * nquad);
      if (pdecSigmaValues.size() != 3 * nquad)
        pdecSigmaValues.resize(3 * nquad);
      if (pdexTauSpinUpValues.size() != nquad)
        pdexTauSpinUpValues.resize(nquad);
      if (pdexTauSpinDownValues.size() != nquad)
        pdexTauSpinDownValues.resize(nquad);
      if (pdecTauSpinUpValues.size() != nquad)
        pdecTauSpinUpValues.resize(nquad);
      if (pdecTauSpinDownValues.size() != nquad)
        pdecTauSpinDownValues.resize(nquad);

      internal::fillRhoSigmaTauVector(nquad,
                                      densityValuesSpinUp,
                                      densityValuesSpinDown,
                                      gradValuesSpinUp,
                                      gradValuesSpinDown,
                                      tauValuesSpinUp,
                                      tauValuesSpinDown,
                                      densityValues,
                                      sigmaValues,
                                      tauValues,
                                      rhoThresholdMgga,
                                      sigmaThresholdMgga,
                                      tauThresholdMgga);

      dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>
        laplacianValues(2 * nquad, 0.0);
      dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>
        pdexLaplacianValues(2 * nquad, 0.0);
      dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>
        pdecLaplacianValues(2 * nquad, 0.0);

      exValues.setValue(0.0);
      ecValues.setValue(0.0);
      pdexDensityValues.setValue(0.0);
      pdecDensityValues.setValue(0.0);
      pdexSigmaValues.setValue(0.0);
      pdecSigmaValues.setValue(0.0);
      pdexTauValues.setValue(0.0);
      pdecTauValues.setValue(0.0);

      xc_mgga_exc_vxc(d_funcXPtr.get(),
                      nquad,
                      &densityValues[0],
                      &sigmaValues[0],
                      &laplacianValues[0],
                      &tauValues[0],
                      &exValues[0],
                      &pdexDensityValues[0],
                      &pdexSigmaValues[0],
                      &pdexLaplacianValues[0],
                      &pdexTauValues[0]);
      xc_mgga_exc_vxc(d_funcCPtr.get(),
                      nquad,
                      &densityValues[0],
                      &sigmaValues[0],
                      &laplacianValues[0],
                      &tauValues[0],
                      &ecValues[0],
                      &pdecDensityValues[0],
                      &pdecSigmaValues[0],
                      &pdecLaplacianValues[0],
                      &pdecTauValues[0]);

      for (size_t i = 0; i < nquad; i++)
        {
          if (std::abs(densityValues[2 * i + 0] + densityValues[2 * i + 1]) <=
                rhoThresholdMgga ||
              std::abs(tauValues[2 * i + 0] + tauValues[2 * i + 1]) <=
                tauThresholdMgga)
            {
              exValues[i]                  = 0.0;
              pdexDensityValues[2 * i + 0] = 0.0;
              pdexSigmaValues[3 * i + 0]   = 0.0;
              pdexTauValues[2 * i + 0]     = 0.0;

              pdexDensityValues[2 * i + 1] = 0.0;
              pdexSigmaValues[3 * i + 1]   = 0.0;
              pdexSigmaValues[3 * i + 2]   = 0.0;
              pdexTauValues[2 * i + 1]     = 0.0;

              ecValues[i]                  = 0.0;
              pdecDensityValues[2 * i + 0] = 0.0;
              pdecSigmaValues[3 * i + 0]   = 0.0;
              pdecTauValues[2 * i + 0]     = 0.0;

              pdecDensityValues[2 * i + 1] = 0.0;
              pdecSigmaValues[3 * i + 1]   = 0.0;
              pdecSigmaValues[3 * i + 2]   = 0.0;
              pdecTauValues[2 * i + 1]     = 0.0;
            }

          exValues[i] =
            exValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
          ecValues[i] =
            ecValues[i] * (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
          pdexDensitySpinUpValues[i]   = pdexDensityValues[2 * i + 0];
          pdexDensitySpinDownValues[i] = pdexDensityValues[2 * i + 1];
          pdecDensitySpinUpValues[i]   = pdecDensityValues[2 * i + 0];
          pdecDensitySpinDownValues[i] = pdecDensityValues[2 * i + 1];
          pdexTauSpinUpValues[i]       = pdexTauValues[2 * i + 0];
          pdexTauSpinDownValues[i]     = pdexTauValues[2 * i + 1];
          pdecTauSpinUpValues[i]       = pdecTauValues[2 * i + 0];
          pdecTauSpinDownValues[i]     = pdecTauValues[2 * i + 1];
        }
    }

    template class ExcTauMGGAClass<dftefe::utils::MemorySpace::HOST>;
#if defined(DFTEFE_WITH_DEVICE)
    template class ExcTauMGGAClass<dftefe::utils::MemorySpace::DEVICE>;
#endif

  } // namespace ksdft
} // namespace dftefe
