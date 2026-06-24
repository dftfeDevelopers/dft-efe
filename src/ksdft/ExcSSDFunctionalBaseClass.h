// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#ifndef DFTEFE_EXCSSDFUNCTIONALBASECLASS_H
#define DFTEFE_EXCSSDFUNCTIONALBASECLASS_H

#include <ksdft/RDM1.h>
#include <linearAlgebra/MultiVector.h>
#include <utils/Exceptions.h>
#include <vector>
#include <set>
#include <unordered_map>
#include <fstream>
#include <iostream>

namespace dftefe
{
  namespace ksdft
  {
    enum class ExcFamilyType
    {
      /*
      LLMGGA: Includes only Laplacian of the electron-density
      TauMGGA: Includes only kinetic energy density
      MGGA: Includes both the Laplacian of the electron-density and kinetic
      energy density
      */
      LDA,
      GGA,
      LLMGGA,
      HYBRID,
      DFTPlusU,
      MGGA,
      TauMGGA,
    };

    enum class densityFamilyType
    {
      LDA,
      GGA,
      LLMGGA,
    };

    /*
     * XC attributes for the derivatives for the remainder functional
     *
     */
    enum class xcRemainderOutputDataAttributes
    {
      e,         // energy density per unit volume for the remainder functional
      vSpinUp,   // the local multiplicative potential for spin up arising from
                 // remainder functional
      vSpinDown, // the local multiplicative potential for spin down arising
                 // from remainder functional
      pdeDensitySpinUp, // partial derivative of remainder functional energy
                        // density wrt spin-up electron density: d(e)/d(rho_up)
      pdeDensitySpinDown, // partial derivative of remainder functional energy
                          // density wrt spin-down electron density:
                          // d(e)/d(rho_down)
      pdeSigma,
      pdeLaplacianSpinUp,
      pdeLaplacianSpinDown,
      pdeTauSpinUp,
      pdeTauSpinDown
    };


    /**
     * @brief This class provides the structure for all
     * Exc functionals that can be written as a combination of
     * functional of Single Slater determinant that results in a
     * non-multiplicative potential plus a remainder functional
     * dependent on density and Tau.
     *
     * Exc = S{\phi} + R [\rho, \tau]
     * @author Vishal Subramanian, Sambit Das
     */
    template <dftefe::utils::MemorySpace memorySpace>
    class ExcSSDFunctionalBaseClass
    {
    public:
      using AttrStorage = std::vector<
        dftefe::utils::MemoryStorage<double, dftefe::utils::MemorySpace::HOST>>;

      ExcSSDFunctionalBaseClass(
        const ExcFamilyType                              excFamType,
        const densityFamilyType                          densityFamType,
        const std::set<dftefe::ksdft::DensityDescrAttr> &densityDescrAttrs);

      ExcSSDFunctionalBaseClass(
        const ExcFamilyType                              excFamType,
        const densityFamilyType                          densityFamType,
        const std::set<dftefe::ksdft::DensityDescrAttr> &densityDescrAttrs,
        const std::set<dftefe::ksdft::WfcDescrAttr> &    wfcDescrAttrs);

      virtual ~ExcSSDFunctionalBaseClass();

      const std::set<dftefe::ksdft::DensityDescrAttr> &
      getDensityDescriptorAttributesList() const;

      densityFamilyType
      getDensityBasedFamilyType() const;

      /**
       * @brief Pure virtual. Call directly when you have pre-built density maps
       * (e.g. from non-spin-polarized RDM1 split as ρ↑ = ρ↓ = ρ/2).
       */
      virtual void
      computeRhoTauDependentXCData(
        const std::unordered_map<dftefe::ksdft::DensityDescrAttr, AttrStorage>
          &densityAttrVals,
        const std::unordered_map<dftefe::ksdft::WfcDescrAttr, AttrStorage>
          &wfcAttrVals,
        std::unordered_map<
          xcRemainderOutputDataAttributes,
          dftefe::utils::MemoryStorage<double,
                                       dftefe::utils::MemorySpace::HOST>>
          &xDataOut,
        std::unordered_map<
          xcRemainderOutputDataAttributes,
          dftefe::utils::MemoryStorage<double,
                                       dftefe::utils::MemorySpace::HOST>>
          &cDataOut) const = 0;

      ExcFamilyType
      getExcFamilyType() const;

      virtual void
      checkInputOutputDataAttributesConsistency(
        const std::vector<xcRemainderOutputDataAttributes>
          &outputDataAttributes) const = 0;

    protected:
      const std::set<dftefe::ksdft::DensityDescrAttr> d_densityDescrAttrs;
      const std::set<dftefe::ksdft::WfcDescrAttr>     d_wfcDescrAttrs;

      ExcFamilyType     d_ExcFamilyType;
      densityFamilyType d_densityFamilyType;

      mutable dftefe::utils::MemoryStorage<double,
                                           dftefe::utils::MemorySpace::HOST>
        s_densityValues, s_sigmaValues, s_tauValues;

      mutable dftefe::utils::MemoryStorage<double,
                                           dftefe::utils::MemorySpace::HOST>
        s_pdexDensityValues, s_pdecDensityValues, s_pdexTauValues,
        s_pdecTauValues;

      mutable dftefe::utils::MemoryStorage<double,
                                           dftefe::utils::MemorySpace::HOST>
        s_exValues, s_ecValues, s_pdexDensitySpinUpValues,
        s_pdexDensitySpinDownValues, s_pdecDensitySpinUpValues,
        s_pdecDensitySpinDownValues, s_pdexSigmaValues, s_pdecSigmaValues,
        s_pdexTauSpinUpValues, s_pdexTauSpinDownValues, s_pdecTauSpinUpValues,
        s_pdecTauSpinDownValues;
    };

  } // namespace ksdft
} // namespace dftefe

#include "ExcSSDFunctionalBaseClass.t.cpp"
#endif // DFTEFE_EXCSSDFUNCTIONALBASECLASS_H
