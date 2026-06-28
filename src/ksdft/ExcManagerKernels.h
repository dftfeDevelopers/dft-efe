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

#ifndef DFTEFE_EXCMANAGERDEVICEKERNELS_H
#define DFTEFE_EXCMANAGERDEVICEKERNELS_H

#include <utils/DataTypeOverloads.h>
#include <utils/DeviceAPICalls.h>
#include <utils/DeviceDataTypeOverloads.h>
#include <utils/DeviceTypeConfig.h>
#include <utils/DeviceKernelLauncherHelpers.h>
#include <utils/MemoryStorage.h>
#include <memory>

namespace dftefe
{
  namespace ksdft
  {
    namespace internal
    {
      template <dftefe::utils::MemorySpace memorySpace>
      void
      fillRhoVector(
        const size_type                                          numQuadPoints,
        const dftefe::utils::MemoryStorage<double, memorySpace> &densitySpinUp,
        const dftefe::utils::MemoryStorage<double, memorySpace>
          &                                                densitySpinDown,
        dftefe::utils::MemoryStorage<double, memorySpace> &rhoVector);

      template <dftefe::utils::MemorySpace memorySpace>
      void
      fillRhoSigmaVector(
        const size_type                                          numQuadPoints,
        const dftefe::utils::MemoryStorage<double, memorySpace> &densitySpinUp,
        const dftefe::utils::MemoryStorage<double, memorySpace>
          &densitySpinDown,
        const dftefe::utils::MemoryStorage<double, memorySpace>
          &gradDensitySpinUp,
        const dftefe::utils::MemoryStorage<double, memorySpace>
          &                                                gradDensitySpinDown,
        dftefe::utils::MemoryStorage<double, memorySpace> &rhoVector,
        dftefe::utils::MemoryStorage<double, memorySpace> &sigmaVector);

      template <dftefe::utils::MemorySpace memorySpace>
      void
      fillRhoSigmaTauVector(
        const size_type                                          numQuadPoints,
        const dftefe::utils::MemoryStorage<double, memorySpace> &densitySpinUp,
        const dftefe::utils::MemoryStorage<double, memorySpace>
          &densitySpinDown,
        const dftefe::utils::MemoryStorage<double, memorySpace>
          &gradDensitySpinUp,
        const dftefe::utils::MemoryStorage<double, memorySpace>
          &gradDensitySpinDown,
        const dftefe::utils::MemoryStorage<double, memorySpace> &tauSpinUp,
        const dftefe::utils::MemoryStorage<double, memorySpace> &tauSpinDown,
        dftefe::utils::MemoryStorage<double, memorySpace> &      rhoVector,
        dftefe::utils::MemoryStorage<double, memorySpace> &      sigmaVector,
        dftefe::utils::MemoryStorage<double, memorySpace> &      tauVector,
        const double                                             rhoThreshold,
        const double                                             sigmaThreshold,
        const double                                             tauThreshold);

    } // namespace internal
  }   // namespace ksdft
} // namespace dftefe

#endif // DFTEFE_EXCMANAGERDEVICEKERNELS_H
