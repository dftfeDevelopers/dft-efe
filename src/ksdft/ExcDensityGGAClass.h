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
#ifndef DFTEFE_EXCDENSITYGGACLASS_H
#define DFTEFE_EXCDENSITYGGACLASS_H

#include <xc.h>
#include <ksdft/ExcSSDFunctionalBaseClass.h>

namespace dftefe
{
  namespace ksdft
  {
    template <dftefe::utils::MemorySpace memorySpace>
    class ExcDensityGGAClass : public ExcSSDFunctionalBaseClass<memorySpace>
    {
    public:
      ExcDensityGGAClass(std::shared_ptr<xc_func_type> &funcXPtr,
                         std::shared_ptr<xc_func_type> &funcCPtr,
                         std::string                    XCType);

      ~ExcDensityGGAClass();

      void
      checkInputOutputDataAttributesConsistency(
        const std::vector<xcRemainderOutputDataAttributes> &outputDataAttributes)
        const override;

    private:
      void
      computeRhoTauDependentXCData(
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
          &cDataout) const override;

      std::shared_ptr<xc_func_type> d_funcXPtr;
      std::shared_ptr<xc_func_type> d_funcCPtr;
      std::string                   d_XCType;
    };

  } // namespace ksdft
} // namespace dftefe
#endif // DFTEFE_EXCDENSITYGGACLASS_H
