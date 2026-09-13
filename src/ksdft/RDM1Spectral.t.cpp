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
 * @author Bikash Kanungo
 */

#include <utils/Exceptions.h>

namespace dftefe
{
  namespace ksdft
  {
    //--------------------------------------------------------------------------
    // RDM1Spectral
    //--------------------------------------------------------------------------

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    RDM1Spectral<ValueType, memorySpace>::RDM1Spectral()
      : d_spectral(nullptr)
      , d_evalFlag(false)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Spectral<ValueType, memorySpace>::setSpectral(
      std::unique_ptr<SpectralRep<ValueType, memorySpace>> spectral)
    {
      d_spectral = std::move(spectral);
      d_evalFlag = true;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const SpectralRep<ValueType, memorySpace> &
    RDM1Spectral<ValueType, memorySpace>::getSpectral() const
    {
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        d_spectral != nullptr,
        "RDM1Spectral::getSpectral() called but spectral data is not set.");
      return *d_spectral;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    RDM1Access<ValueType, memorySpace>
    RDM1Spectral<ValueType, memorySpace>::getAccess()
    {
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        d_spectral != nullptr,
        "RDM1Spectral::getAccess() called but spectral data is not set "
        "(or access is already held by another RDM1Access).");
      return RDM1Access<ValueType, memorySpace>(this, std::move(d_spectral));
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    bool
    RDM1Spectral<ValueType, memorySpace>::isSpectralSet() const
    {
      return d_spectral != nullptr;
    }

    //--------------------------------------------------------------------------
    // RDM1Access
    //--------------------------------------------------------------------------

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    RDM1Access<ValueType, memorySpace>::RDM1Access(
      RDM1Spectral<ValueType, memorySpace> *               owner,
      std::unique_ptr<SpectralRep<ValueType, memorySpace>> spectral)
      : d_owner(owner)
      , d_spectral(std::move(spectral))
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    RDM1Access<ValueType, memorySpace>::RDM1Access(
      RDM1Access<ValueType, memorySpace> &&other) noexcept
      : d_owner(other.d_owner)
      , d_spectral(std::move(other.d_spectral))
    {
      other.d_owner = nullptr;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    RDM1Access<ValueType, memorySpace>::~RDM1Access()
    {
      if (d_owner != nullptr)
        d_owner->setSpectral(std::move(d_spectral));
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Access<ValueType, memorySpace>::returnBack()
    {
      if (d_owner != nullptr)
        {
          d_owner->setSpectral(std::move(d_spectral));
          d_owner = nullptr;
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SpectralRep<ValueType, memorySpace> &
    RDM1Access<ValueType, memorySpace>::getSpectral()
    {
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        d_spectral != nullptr,
        "RDM1Access::getSpectral() called but spectral data is null.");
      return *d_spectral;
    }

  } // namespace ksdft
} // namespace dftefe
