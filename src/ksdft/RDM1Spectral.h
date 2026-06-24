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

#ifndef dftefe_RDM1Spectral_h
#define dftefe_RDM1Spectral_h

#include <vector>
#include <memory>
#include <ksdft/RDM1.h>
#include <linearAlgebra/MultiVector.h>

namespace dftefe
{
  namespace ksdft
  {
    /**
     * @brief Intermediate abstract class that stores the spectral (eigen)
     * decomposition of the one-particle reduced density matrix — KS orbitals,
     * occupancies, and orbital count. Concrete subclasses (e.g. RDM1FE)
     * implement getDescriptors() and getDensityObs() using this stored data.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class RDM1Spectral : public RDM1<ValueType, memorySpace>
    {
    public:
      using AttrStorage = typename RDM1<ValueType, memorySpace>::AttrStorage;

    public:
      RDM1Spectral();
      virtual ~RDM1Spectral() = default;

      void
      setSpectral(
        std::unique_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>>
                                                ksOrbitals,
        const std::vector<std::vector<double>> &occupancies,
        const size_type                         nKSOrbs);

      void
      getSpectral(
        std::unique_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>>
          &                               ksOrbitals,
        std::vector<std::vector<double>> &occupancies,
        size_type &                       nKSOrbs);

      bool
      getKSSetFlag() const;

    protected:
      std::unique_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>>
                                       d_ksOrbs;
      size_type                        d_nKSOrbs;
      std::vector<std::vector<double>> d_occupancies;
      bool                             d_ksSetFlag;
      bool                             d_evalFlag;
    };

  } // namespace ksdft
} // namespace dftefe
#include "RDM1Spectral.t.cpp"
#endif // dftefe_RDM1Spectral_h
