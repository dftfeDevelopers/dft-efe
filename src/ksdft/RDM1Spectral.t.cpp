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

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    RDM1Spectral<ValueType, memorySpace>::RDM1Spectral()
      : d_nKSOrbs(0)
      , d_ksSetFlag(false)
      , d_evalFlag(false)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Spectral<ValueType, memorySpace>::setSpectral(
      std::unique_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>>
                                                   ksOrbitals,
      const std::vector<std::vector<double>> &     occupancies,
      const size_type                              nKSOrbs)
    {
      d_ksOrbs      = std::move(ksOrbitals);
      d_occupancies = occupancies;
      d_nKSOrbs     = nKSOrbs;
      d_evalFlag    = true;
      d_ksSetFlag   = true;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Spectral<ValueType, memorySpace>::getSpectral(
      std::unique_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>>
        &                                      ksOrbitals,
      std::vector<std::vector<double>> &       occupancies,
      size_type &                              nKSOrbs)
    {
      ksOrbitals  = std::move(d_ksOrbs);
      occupancies = d_occupancies;
      nKSOrbs     = d_nKSOrbs;
      d_ksSetFlag = false;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    bool
    RDM1Spectral<ValueType, memorySpace>::getKSSetFlag() const
    {
      return d_ksSetFlag;
    }

  } // namespace ksdft
} // namespace dftefe
