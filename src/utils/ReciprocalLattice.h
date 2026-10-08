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
 * @author Avirup Sircar
 */

#ifndef dftefeReciprocalLattice_h
#define dftefeReciprocalLattice_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <vector>

namespace dftefe
{
  namespace utils
  {
    /**
     * @brief Reciprocal lattice of a real-space cell, and the Monkhorst-Pack
     * k-point grids built on it. The k-points are returned in cartesian
     * (bohr^-1) coordinates, which is what the Bloch phase e^{-ik.x} consumes.
     */
    class ReciprocalLattice
    {
    public:
      /**
       * @param[in] domainBoundingVectors the real-space lattice vectors, one
       * per dimension. Must be linearly independent.
       */
      ReciprocalLattice(
        const std::vector<utils::Point> &domainBoundingVectors);

      ~ReciprocalLattice() = default;

      /**
       * @brief The reciprocal vectors b_i, satisfying a_i . b_j = 2 pi delta_ij.
       */
      const std::vector<utils::Point> &
      getReciprocalVectors() const;

      //
      // Signed volume of the real-space cell, a_0 . (a_1 x a_2)
      //
      double
      getCellVolume() const;

      /**
       * @brief Monkhorst-Pack grid in cartesian coordinates.
       *
       * A non-periodic direction is collapsed to its single k = 0 point
       * whatever @p nkGrid asks for, so the same call serves periodic,
       * semi-periodic and non-periodic cells.
       *
       * @param[in] nkGrid                    grid subdivisions per dimension
       * @param[in] isPeriodicFlags           periodicity per dimension
       * @param[in] isShifted                 half-grid shift per dimension
       * @param[in] useTimeReversalSymmetry   fold k and -k onto one point
       * @param[out] kPointCoordinates        nK cartesian points
       * @param[out] kPointWeights            nK, summing to one
       */
      void
      generateMonkhorstPackGrid(const std::vector<size_type> &nkGrid,
                                const std::vector<bool> &isPeriodicFlags,
                                const std::vector<bool> &isShifted,
                                const bool           useTimeReversalSymmetry,
                                std::vector<utils::Point> &kPointCoordinates,
                                std::vector<double> &      kPointWeights) const;

    private:
      std::vector<utils::Point> d_reciprocalVectors;
      double                    d_cellVolume;
      size_type                 d_dim;
    };

  } // end of namespace utils
} // end of namespace dftefe
#endif // dftefeReciprocalLattice_h
