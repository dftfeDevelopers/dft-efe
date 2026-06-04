// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Vishal Subramanian, Sambit Das
//

#include <ksdft/ExcDensityGGAClass.h>
#include <utils/Exceptions.h>
#include <ksdft/ExcManagerKernels.h>
#include <utils/DeviceAPICalls.h>

namespace dftefe
{
  namespace ksdft
  {
    template <dftefe::utils::MemorySpace memorySpace>
    ExcDensityGGAClass<memorySpace>::ExcDensityGGAClass(
      std::shared_ptr<xc_func_type> &funcXPtr,
      std::shared_ptr<xc_func_type> &funcCPtr,
      std::string                    XCType)
      : ExcSSDFunctionalBaseClass<memorySpace>(
          ExcFamilyType::GGA,
          densityFamilyType::GGA,
          std::set<dftefe::ksdft::DensityDescrAttr>{
            dftefe::ksdft::DensityDescrAttr::Val,
            dftefe::ksdft::DensityDescrAttr::Grad})
    {
      d_funcXPtr = funcXPtr;
      d_funcCPtr = funcCPtr;
      d_XCType   = XCType;
    }

    template <dftefe::utils::MemorySpace memorySpace>
    ExcDensityGGAClass<memorySpace>::~ExcDensityGGAClass()
    {}

    template <dftefe::utils::MemorySpace memorySpace>
    void
    ExcDensityGGAClass<memorySpace>::checkInputOutputDataAttributesConsistency(
      const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
      const
    {
      const std::vector<xcRemainderOutputDataAttributes>
        allowedOutputDataAttributes = {
          xcRemainderOutputDataAttributes::e,
          xcRemainderOutputDataAttributes::pdeDensitySpinUp,
          xcRemainderOutputDataAttributes::pdeDensitySpinDown,
          xcRemainderOutputDataAttributes::pdeSigma};

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
    ExcDensityGGAClass<memorySpace>::computeRhoTauDependentXCData(
      const std::unordered_map<
        dftefe::ksdft::DensityDescrAttr,
        std::vector<dftefe::utils::MemoryStorage<
          double,
          dftefe::utils::MemorySpace::HOST>>> &densityAttrVals,
      const std::unordered_map<
        dftefe::ksdft::WfcDescrAttr,
        std::vector<dftefe::utils::MemoryStorage<
          double,
          dftefe::utils::MemorySpace::HOST>>> &wfcAttrVals,
      std::unordered_map<
        xcRemainderOutputDataAttributes,
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>
        &xDataOut,
      std::unordered_map<
        xcRemainderOutputDataAttributes,
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>
        &cDataOut) const
    {
      const auto &densityValuesSpinUp =
        densityAttrVals.at(dftefe::ksdft::DensityDescrAttr::Val)[0];
      const auto &densityValuesSpinDown =
        densityAttrVals.at(dftefe::ksdft::DensityDescrAttr::Val)[1];
      const auto &gradValuesSpinUp =
        densityAttrVals.at(dftefe::ksdft::DensityDescrAttr::Grad)[0];
      const auto &gradValuesSpinDown =
        densityAttrVals.at(dftefe::ksdft::DensityDescrAttr::Grad)[1];
      const size_type nquad = densityValuesSpinUp.size();

      std::vector<xcRemainderOutputDataAttributes> outputDataAttributes;
      for (const auto &element : xDataOut)
        outputDataAttributes.push_back(element.first);

      checkInputOutputDataAttributesConsistency(outputDataAttributes);

      if (this->s_densityValues.size() != 2 * nquad)
        this->s_densityValues.resize(2 * nquad);
      if (this->s_sigmaValues.size() != 3 * nquad)
        this->s_sigmaValues.resize(3 * nquad);

      auto &densityValues = this->s_densityValues;
      auto &sigmaValues   = this->s_sigmaValues;
      sigmaValues.setValue(0.0);

      if (this->s_pdexDensityValues.size() != 2 * nquad)
        this->s_pdexDensityValues.resize(2 * nquad);
      if (this->s_pdecDensityValues.size() != 2 * nquad)
        this->s_pdecDensityValues.resize(2 * nquad);

      auto &pdexDensityValues = this->s_pdexDensityValues;
      auto &pdecDensityValues = this->s_pdecDensityValues;

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

      internal::fillRhoSigmaVector(nquad,
                                           densityValuesSpinUp,
                                           densityValuesSpinDown,
                                           gradValuesSpinUp,
                                           gradValuesSpinDown,
                                           densityValues,
                                           sigmaValues);

      exValues.setValue(0.0);
      ecValues.setValue(0.0);
      pdexDensityValues.setValue(0.0);
      pdecDensityValues.setValue(0.0);
      pdexSigmaValues.setValue(0.0);
      pdecSigmaValues.setValue(0.0);

      xc_gga_exc_vxc(d_funcXPtr.get(),
                     nquad,
                     densityValues.data(),
                     sigmaValues.data(),
                     exValues.data(),
                     pdexDensityValues.data(),
                     pdexSigmaValues.data());
      xc_gga_exc_vxc(d_funcCPtr.get(),
                     nquad,
                     densityValues.data(),
                     sigmaValues.data(),
                     ecValues.data(),
                     pdecDensityValues.data(),
                     pdecSigmaValues.data());

      for (size_t i = 0; i < nquad; i++)
        {
          exValues[i] = exValues[i] *
                        (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
          ecValues[i] = ecValues[i] *
                        (densityValues[2 * i + 0] + densityValues[2 * i + 1]);
          pdexDensitySpinUpValues[i]   = pdexDensityValues[2 * i + 0];
          pdexDensitySpinDownValues[i] = pdexDensityValues[2 * i + 1];
          pdecDensitySpinUpValues[i]   = pdecDensityValues[2 * i + 0];
          pdecDensitySpinDownValues[i] = pdecDensityValues[2 * i + 1];
        }
    }

    template class ExcDensityGGAClass<dftefe::utils::MemorySpace::HOST>;
#ifdef DFTEFE_WITH_DEVICE
    template class ExcDensityGGAClass<dftefe::utils::MemorySpace::DEVICE>;
#endif

  } // namespace ksdft
} // namespace dftefe
