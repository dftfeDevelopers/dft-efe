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
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class RDM1Access;

    /**
     * @brief Plain data bundle holding the KS spectral decomposition:
     * orbitals, occupancies (one vector per k-point/spin), and orbital count.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    struct SpectralRep
    {
      std::unique_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>>
                          ksOrbs;
      std::vector<double> occupancies;
      size_type           nKSOrbs;
    };

    /**
     * @brief Intermediate abstract class that stores the spectral (eigen)
     * decomposition of the one-particle reduced density matrix — KS orbitals,
     * occupancies, and orbital count via a SpectralRep bundle.
     * Concrete subclasses (e.g. RDM1FE) implement getDescriptors() and
     * getDensityObs() using this stored data.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class RDM1Spectral : public RDM1<ValueType, memorySpace>
    {
    public:
      using AttrStorage = typename RDM1<ValueType, memorySpace>::AttrStorage;

    public:
      RDM1Spectral();
      virtual ~RDM1Spectral() = default;

      /**
       * @brief Store the spectral representation (takes ownership).
       *        Sets d_evalFlag = true.
       */
      void
      setSpectral(
        std::unique_ptr<SpectralRep<ValueType, memorySpace>> spectral);

      /**
       * @brief Read-only reference to the spectral data — does not move
       *        ownership.  Use for Ts / Exc / NLPSP read-only access.
       */
      const SpectralRep<ValueType, memorySpace> &
      getSpectral() const;

      /**
       * @brief Resource Acquisition Is Initialization (RAII) access object
       * that moves ownership out for the
       *        eigensolver (write access).  Returns ownership on destruction
       *        or explicit returnBack().
       */
      RDM1Access<ValueType, memorySpace>
      getAccess();

      bool
      isSpectralSet() const;

    protected:
      std::unique_ptr<SpectralRep<ValueType, memorySpace>> d_spectral;
      bool                                                 d_evalFlag;
    };

    /**
     * @brief Resource Acquisition Is Initialization (RAII) wrapper that
     * temporarily holds the SpectralRep moved out of
     *        an RDM1Spectral.  On destruction (or returnBack()) the rep is
     *        returned via setSpectral().
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class RDM1Access
    {
      RDM1Spectral<ValueType, memorySpace> *               d_owner;
      std::unique_ptr<SpectralRep<ValueType, memorySpace>> d_spectral;

    public:
      RDM1Access(RDM1Spectral<ValueType, memorySpace> *               owner,
                 std::unique_ptr<SpectralRep<ValueType, memorySpace>> spectral);

      RDM1Access(const RDM1Access &) = delete;
      RDM1Access &
      operator=(const RDM1Access &) = delete;

      RDM1Access(RDM1Access &&other) noexcept;

      ~RDM1Access();

      /** Explicit early return of ownership before scope ends. */
      void
      returnBack();

      /** Mutable reference for eigensolver write access. */
      SpectralRep<ValueType, memorySpace> &
      getSpectral();
    };

  } // namespace ksdft
} // namespace dftefe
#include "RDM1Spectral.t.cpp"
#endif // dftefe_RDM1Spectral_h
