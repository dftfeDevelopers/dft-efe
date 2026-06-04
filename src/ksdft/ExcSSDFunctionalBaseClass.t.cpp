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
// @author Vishal Subramanian
//

#include <ksdft/ExcSSDFunctionalBaseClass.h>

namespace dftefe
{
  namespace ksdft
  {
    template <dftefe::utils::MemorySpace memorySpace>
    ExcSSDFunctionalBaseClass<memorySpace>::ExcSSDFunctionalBaseClass(
      const ExcFamilyType                              excFamType,
      const densityFamilyType                          densityFamType,
      const std::set<dftefe::ksdft::DensityDescrAttr> &densityDescrAttrs)
      : d_ExcFamilyType(excFamType)
      , d_densityFamilyType(densityFamType)
      , d_densityDescrAttrs(densityDescrAttrs)
    {}

    template <dftefe::utils::MemorySpace memorySpace>
    ExcSSDFunctionalBaseClass<memorySpace>::ExcSSDFunctionalBaseClass(
      const ExcFamilyType                              excFamType,
      const densityFamilyType                          densityFamType,
      const std::set<dftefe::ksdft::DensityDescrAttr> &densityDescrAttrs,
      const std::set<dftefe::ksdft::WfcDescrAttr> &    wfcDescrAttrs)
      : d_ExcFamilyType(excFamType)
      , d_densityFamilyType(densityFamType)
      , d_densityDescrAttrs(densityDescrAttrs)
      , d_wfcDescrAttrs(wfcDescrAttrs)
    {}

    template <dftefe::utils::MemorySpace memorySpace>
    ExcSSDFunctionalBaseClass<memorySpace>::~ExcSSDFunctionalBaseClass()
    {}

    template <dftefe::utils::MemorySpace memorySpace>
    ExcFamilyType
    ExcSSDFunctionalBaseClass<memorySpace>::getExcFamilyType() const
    {
      return d_ExcFamilyType;
    }

    template <dftefe::utils::MemorySpace memorySpace>
    densityFamilyType
    ExcSSDFunctionalBaseClass<memorySpace>::getDensityBasedFamilyType() const
    {
      return d_densityFamilyType;
    }

    template <dftefe::utils::MemorySpace memorySpace>
    const std::set<dftefe::ksdft::DensityDescrAttr> &
    ExcSSDFunctionalBaseClass<memorySpace>::getDensityDescriptorAttributesList()
      const
    {
      return d_densityDescrAttrs;
    }

  } // namespace ksdft
} // namespace dftefe
