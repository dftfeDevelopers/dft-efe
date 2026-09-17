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

#ifndef dftefePeriodicImageAtomGenerator_h
#define dftefePeriodicImageAtomGenerator_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <basis/Defaults.h>
#include <vector>

namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Enumerates the periodic images of each atom that lie close enough
     * to the simulation cell to contribute to it, following dftfe's
     * generateImageCharges (src/dft/generateImageCharges.cc:341-712).
     *
     * Two lists are produced, differing only in the cutoff envelope they use.
     * Both are geometric envelopes rather than physical cutoffs: they bound
     * which lattice translations are candidates, and every consumer then
     * applies its own physics cutoff to the candidates it is handed. With the
     * default cutoffs the two lists coincide; they are kept separate so the
     * electrostatic envelope can be widened without disturbing the others.
     *
     * An atom's own (0,0,0) translation is never an image, so a non-periodic
     * system produces empty lists and every accessor returns an empty vector.
     */
    class PeriodicImageAtomGenerator
    {
    public:
      /**
       * @param[in] atomCoordinates       positions of the master atoms
       * @param[in] atomCharges           charge of each master atom
       * @param[in] domainBoundingVectors the three lattice vectors
       * @param[in] isPeriodicFlags       which directions are periodic
       * @param[in] cutOff             envelope for the full list
       * @param[in] cutOffTrunc        envelope for the truncated list;
       *                                  must be grown by the caller to cover
       *                                  the largest enrichment reach when
       *                                  that exceeds the default
       */
      PeriodicImageAtomGenerator(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<utils::Point> &domainBoundingVectors,
        const std::vector<bool> &        isPeriodicFlags,
        const double                     cutOff =
          PeriodicImageAtomGeneratorDefaults::CUTOFF,
        const double cutOffTrunc =
          PeriodicImageAtomGeneratorDefaults::CUTOFF_TRUNC);

      ~PeriodicImageAtomGenerator() = default;

      //
      // Full list, consumed by the electrostatics
      //
      const std::vector<size_type> &
      getImageIds() const;

      const std::vector<double> &
      getImageCharges() const;

      const std::vector<utils::Point> &
      getImagePositions() const;

      /**
       * @brief Indices into the full list of the images belonging to
       * @p masterAtomId. Empty when that atom has none within the envelope.
       * The master's own index is not included, being implicit.
       */
      const std::vector<size_type> &
      getImageIdsForMaster(const size_type masterAtomId) const;

      //
      // Truncated list, consumed by the nonlocal projectors, the densities,
      // mesh generation and the enrichment functions
      //
      const std::vector<size_type> &
      getImageIdsTrunc() const;

      const std::vector<double> &
      getImageChargesTrunc() const;

      const std::vector<utils::Point> &
      getImagePositionsTrunc() const;

      const std::vector<size_type> &
      getImageIdsForMasterTrunc(const size_type masterAtomId) const;

      double
      getCutOff() const;

      double
      getCutOffTrunc() const;

      size_type
      nMasterAtoms() const;

      /**
       * @brief The master atom positions this generator was built from.
       * Consumers that index images by master atom id must have been built
       * from the same list, in the same order.
       */
      const std::vector<utils::Point> &
      getAtomCoordinates() const;

    private:
      void
      generateImageCharges(const double               cutOff,
                           std::vector<size_type> &   imageIds,
                           std::vector<double> &      imageCharges,
                           std::vector<utils::Point> &imagePositions) const;

      static void
      buildImageIdsPerMaster(
        const size_type                      nMasterAtoms,
        const std::vector<size_type> &       imageIds,
        std::vector<std::vector<size_type>> &imageIdsPerMaster);

      std::vector<utils::Point> d_atomCoordinates;
      std::vector<double>       d_atomCharges;
      std::vector<utils::Point> d_domainBoundingVectors;
      std::vector<bool>         d_isPeriodicFlags;
      double                    d_cutOff;
      double                    d_cutOffTrunc;
      size_type                 d_nMasterAtoms;

      std::vector<size_type>    d_imageIds;
      std::vector<double>       d_imageCharges;
      std::vector<utils::Point> d_imagePositions;

      std::vector<size_type>    d_imageIdsTrunc;
      std::vector<double>       d_imageChargesTrunc;
      std::vector<utils::Point> d_imagePositionsTrunc;

      std::vector<std::vector<size_type>> d_imageIdsPerMaster;
      std::vector<std::vector<size_type>> d_imageIdsPerMasterTrunc;

      // Returned by the per-master accessors for an atom with no images, so
      // that they can hand back a reference.
      std::vector<size_type> d_emptyImageIds;

    }; // end of class PeriodicImageAtomGenerator

  } // end of namespace basis
} // end of namespace dftefe
#endif // dftefePeriodicImageAtomGenerator_h
