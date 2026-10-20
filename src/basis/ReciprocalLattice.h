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
#include <map>
#include <vector>

namespace dftefe
{
  namespace basis
  {
    namespace kpoint
    {
    /**
     * @brief A k-point's identity: its index among the points this lattice
     * keeps, and the GLOBAL id every consumer refers to it by.
     *
     * Why a distinct type and not a plain size_type. Two different integers
     * describe a k-point, and they are not interchangeable:
     *
     *   GLOBAL id     which k this is, in the grid generated here. Stable,
     *                 and NOT necessarily contiguous -- time reversal already
     *                 folds k and -k onto one representative, and a later
     *                 symmetry reduction or k-point pool would hand a given
     *                 process an arbitrary subset.
     *   LOCAL index   where that k sits in a particular object's per-k
     *                 containers, always 0 .. nKLocal-1 and dense.
     *
     * They coincide only in the case we happen to run today: the full grid,
     * no reduction, every k on every rank. The moment that stops being true
     * they diverge, and a global id used to subscript a per-k vector is
     * out of bounds or, worse, silently the wrong k.
     *
     * As plain size_type both are the same type, so mixing them compiles
     * without a word and the bug surfaces as wrong numbers much later. A
     * scoped enum has no implicit conversion to size_type and no arithmetic,
     * so a global id CANNOT be used as a subscript: the mistake is a compile
     * error. That is the whole reason for the type -- it costs nothing at
     * runtime and makes the distinction unforgettable rather than
     * documented. Local indices stay plain size_type.
     */
    enum class KPointId : size_type
    {
    };

    /**
     * @brief One k-point: where it is and what it is worth. Kept together so
     * a coordinate can never be paired with the wrong weight.
     */
    struct KPointData
    {
      utils::Point coord;  // cartesian, bohr^-1, what e^{-ik.x} consumes
      double       weight; // BZ weight; the weights of a set sum to one
    };

    /**
     * @brief Give every k-point the same object @p v: how a caller hands a
     * single basis (non-periodic, or CFE at any k) to a per-k interface.
     */
    template <typename T>
    std::map<KPointId, T>
    sameForEveryKPoint(const std::map<KPointId, KPointData> &kPoints,
                       const T &                             v)
    {
      std::map<KPointId, T> m;
      for (const auto &kv : kPoints)
        m.emplace(kv.first, v);
      return m;
    }

    /**
     * @brief @p kPoints unchanged, or a single Gamma point of weight one if
     * it is empty.
     *
     * No k-point set means non-periodic. Normalizing it to one Gamma slot
     * here keeps nKLocal >= 1 with no branch downstream, and makes an empty
     * map and a one-entry Gamma map identical internally.
     */
    std::map<KPointId, KPointData>
    gammaIfEmpty(const std::map<KPointId, KPointData> &kPoints,
                 const size_type                       dimension);
    } // end of namespace kpoint

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
      ReciprocalLattice(const std::vector<utils::Point> &domainBoundingVectors);

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
       * Fractional coordinates are i / n, Gamma-centred for every n, plus
       * half a spacing where shifted, folded into [-1/2, 1/2): the points and
       * their order are dftfe's for the same SAMPLING POINTS / SHIFT.
       *
       * A non-periodic direction is collapsed to its single k = 0 point
       * whatever @p nkGrid asks for, so the same call serves periodic,
       * semi-periodic and non-periodic cells.
       *
       * @param[in] nkGrid                    grid subdivisions per dimension
       * @param[in] isPeriodicFlags           periodicity per dimension
       * @param[in] isShifted                 half-grid shift per dimension
       * @param[in] useTimeReversalSymmetry   fold k and -k onto one point
       * @param[out] kPoints                  the grid, keyed by KPointId.
       * The key is the point's index in this grid and is the GLOBAL id every
       * consumer refers to; the weights sum to one. Returned as a map rather
       * than parallel arrays so that a coordinate cannot be paired with the
       * wrong weight, and so that a later reduced set (time reversal already
       * reduces; symmetry or k-pools would too) can be an irregular subset of
       * the ids without the consumer inferring anything from position.
       */
      void
      generateMonkhorstPackGrid(const std::vector<size_type> &nkGrid,
                                const std::vector<bool> &     isPeriodicFlags,
                                const std::vector<bool> &     isShifted,
                                const bool useTimeReversalSymmetry,
                                std::map<kpoint::KPointId, kpoint::KPointData> &kPoints) const;

    private:
      std::vector<utils::Point> d_reciprocalVectors;
      double                    d_cellVolume;
      size_type                 d_dim;
    };

  } // end of namespace basis
} // end of namespace dftefe
#endif // dftefeReciprocalLattice_h
