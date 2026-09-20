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

#include <basis/PeriodicImageAtomGenerator.h>
#include <utils/Exceptions.h>
#include <algorithm>
#include <cmath>

namespace dftefe
{
  namespace basis
  {
    namespace PeriodicImageAtomGeneratorInternal
    {
      const size_type dim = 3;

      void
      cross(const double *v1, const double *v2, double *result)
      {
        result[0] = v1[1] * v2[2] - v1[2] * v2[1];
        result[1] = -v1[0] * v2[2] + v2[0] * v1[2];
        result[2] = v1[0] * v2[1] - v2[0] * v1[1];
      }

      //
      // The lattice matrix and the centred-frame to cell-frame translation.
      // GenerateMesh centres the mesh on the origin by -0.5*sum(a_i), so atom
      // coordinates arrive centred and must be shifted into the cell frame
      // before realToFrac and back after fracToReal (dftfe's "shift").
      //
      void
      buildCellFrame(const std::vector<utils::Point> &domainBoundingVectors,
                     std::vector<double> &            latticeVectors,
                     double *                         shift)
      {
        latticeVectors.resize(dim * dim, 0.0);
        for (size_type i = 0; i < dim; ++i)
          for (size_type j = 0; j < dim; ++j)
            latticeVectors[dim * i + j] = domainBoundingVectors[i][j];

        for (size_type i = 0; i < dim; ++i)
          {
            shift[i] = 0.0;
            for (size_type j = 0; j < dim; ++j)
              shift[i] += domainBoundingVectors[j][i] / 2.0;
          }
      }

      //
      // Clamps a fractional coordinate into the domain, which is what turns
      // the nearest point on an unbounded plane into the nearest point on the
      // bounded face of the domain.
      //
      double
      roundToDomain(const double frac)
      {
        if (frac < 0.0)
          return 0.0;
        else if (frac <= 1.0)
          return frac;
        else
          return 1.0;
      }

      //
      // latticeVectors is row major with row i holding lattice vector i, so
      // real_i = sum_j frac_j * a_j[i].
      //
      void
      fracToReal(const std::vector<double> &latticeVectors,
                 const double *             frac,
                 double *                   real)
      {
        for (size_type i = 0; i < dim; ++i)
          {
            real[i] = 0.0;
            for (size_type j = 0; j < dim; ++j)
              real[i] += latticeVectors[dim * j + i] * frac[j];
          }
      }

      //
      // Inverse of the above. dftfe solves the same 3x3 system with dgesv;
      // inverting by cofactors avoids pulling LAPACK in for a 3x3.
      //
      void
      realToFrac(const std::vector<double> &latticeVectors,
                 const double *             real,
                 double *                   frac)
      {
        // m is the transpose of latticeVectors, i.e. the matrix taking frac to
        // real above
        double m[dim][dim];
        for (size_type i = 0; i < dim; ++i)
          for (size_type j = 0; j < dim; ++j)
            m[i][j] = latticeVectors[dim * j + i];

        const double det =
          m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
          m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
          m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        utils::throwException<utils::InvalidArgument>(
          std::abs(det) > 1e-14,
          "PeriodicImageAtomGenerator: the domain bounding vectors are "
          "linearly dependent, so no fractional coordinates exist.");

        double inv[dim][dim];
        inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
        inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
        inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
        inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
        inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
        inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
        inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
        inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
        inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;

        for (size_type i = 0; i < dim; ++i)
          {
            frac[i] = 0.0;
            for (size_type j = 0; j < dim; ++j)
              frac[i] += inv[i][j] * real[j];
          }
      }

      //
      // Fractional coordinates of the point on the domain face through xred2
      // with the given normal that is closest to xred1.
      //
      void
      getNearestPointOnGivenSurface(const std::vector<double> &latticeVectors,
                                    const double *             xred1,
                                    const double *             xred2,
                                    const double *             surfaceNormal,
                                    double *                   nearestFrac)
      {
        double p[dim], q[dim], r[dim];
        fracToReal(latticeVectors, xred1, p);
        fracToReal(latticeVectors, xred2, q);
        for (size_type i = 0; i < dim; ++i)
          r[i] = q[i] - p[i];

        double num   = 0.0;
        double denom = 0.0;
        for (size_type i = 0; i < dim; ++i)
          {
            num += r[i] * surfaceNormal[i];
            denom += surfaceNormal[i] * surfaceNormal[i];
          }
        const double t = num / denom;

        double nearestReal[dim];
        for (size_type i = 0; i < dim; ++i)
          nearestReal[i] = p[i] + t * surfaceNormal[i];

        realToFrac(latticeVectors, nearestReal, nearestFrac);
        for (size_type i = 0; i < dim; ++i)
          nearestFrac[i] = roundToDomain(nearestFrac[i]);
      }

      //
      // Shortest distance to the periodic domain -- the parallelepiped spanned
      // by the lattice vectors, NOT a triangulation cell. Zero when inside.
      //
      double
      getMinDistanceFromImageToDomain(const std::vector<double> &latticeVectors,
                                      const double *             xreduced)
      {
        bool isInside = true;
        for (size_type i = 0; i < dim; ++i)
          if (xreduced[i] < 0.0 || xreduced[i] > 1.0)
            isInside = false;
        if (isInside)
          return 0.0;

        const double *a = &latticeVectors[0];
        const double *b = &latticeVectors[dim];
        const double *c = &latticeVectors[2 * dim];

        double surfaceNormal[dim][dim];
        cross(b, c, surfaceNormal[0]);
        cross(c, a, surfaceNormal[1]);
        cross(a, b, surfaceNormal[2]);

        double minDistance = 0.0;
        bool   isFirst     = true;

        // The six faces are frac_d = 0 and frac_d = 1 for each direction d,
        // each carrying the normal of that direction.
        for (size_type d = 0; d < dim; ++d)
          {
            for (size_type iSide = 0; iSide < 2; ++iSide)
              {
                double surfacePoint[dim];
                for (size_type i = 0; i < dim; ++i)
                  surfacePoint[i] = xreduced[i];
                surfacePoint[d] = (double)iSide;

                double nearestFrac[dim];
                getNearestPointOnGivenSurface(latticeVectors,
                                              xreduced,
                                              surfacePoint,
                                              surfaceNormal[d],
                                              nearestFrac);

                double dFrac[dim], dReal[dim];
                for (size_type i = 0; i < dim; ++i)
                  dFrac[i] = xreduced[i] - nearestFrac[i];
                fracToReal(latticeVectors, dFrac, dReal);

                double distance = 0.0;
                for (size_type i = 0; i < dim; ++i)
                  distance += dReal[i] * dReal[i];
                distance = std::sqrt(distance);

                if (isFirst || distance < minDistance)
                  {
                    minDistance = distance;
                    isFirst     = false;
                  }
              }
          }

        return minDistance;
      }
    } // namespace PeriodicImageAtomGeneratorInternal

    PeriodicImageAtomGenerator::PeriodicImageAtomGenerator(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<double> &      atomCharges,
      const std::vector<utils::Point> &domainBoundingVectors,
      const std::vector<bool> &        isPeriodicFlags,
      const double                     cutOff,
      const double                     cutOffTrunc)
      : d_atomCoordinates(atomCoordinates)
      , d_atomCharges(atomCharges)
      , d_domainBoundingVectors(domainBoundingVectors)
      , d_isPeriodicFlags(isPeriodicFlags)
      , d_cutOff(cutOff)
      , d_cutOffTrunc(cutOffTrunc)
      , d_nMasterAtoms(atomCoordinates.size())
      , d_emptyImageIds(0)
    {
      utils::throwException<utils::InvalidArgument>(
        atomCharges.size() == d_nMasterAtoms,
        "PeriodicImageAtomGenerator: atomCoordinates and atomCharges differ "
        "in size.");
      utils::throwException<utils::InvalidArgument>(
        domainBoundingVectors.size() ==
            PeriodicImageAtomGeneratorInternal::dim &&
          isPeriodicFlags.size() == PeriodicImageAtomGeneratorInternal::dim,
        "PeriodicImageAtomGenerator is implemented only for dim = 3.");

      throwIfAtomsOutsideCell();

      generateImageCharges(d_cutOff,
                           d_imageIds,
                           d_imageCharges,
                           d_imagePositions);
      generateImageCharges(d_cutOffTrunc,
                           d_imageIdsTrunc,
                           d_imageChargesTrunc,
                           d_imagePositionsTrunc);

      buildImageIdsPerMaster(d_nMasterAtoms, d_imageIds, d_imageIdsPerMaster);
      buildImageIdsPerMaster(d_nMasterAtoms,
                             d_imageIdsTrunc,
                             d_imageIdsPerMasterTrunc);
    }

    void
    PeriodicImageAtomGenerator::generateImageCharges(
      const double               cutOff,
      std::vector<size_type> &   imageIds,
      std::vector<double> &      imageCharges,
      std::vector<utils::Point> &imagePositions) const
    {
      using namespace PeriodicImageAtomGeneratorInternal;

      imageIds.clear();
      imageCharges.clear();
      imagePositions.clear();

      const int periodic[dim] = {d_isPeriodicFlags[0] ? 1 : 0,
                                 d_isPeriodicFlags[1] ? 1 : 0,
                                 d_isPeriodicFlags[2] ? 1 : 0};
      if (periodic[0] == 0 && periodic[1] == 0 && periodic[2] == 0)
        return;

      std::vector<double> latticeVectors(0);
      double              shift[dim];
      buildCellFrame(d_domainBoundingVectors, latticeVectors, shift);

      double minMagnitude = 0.0;
      for (size_type i = 0; i < dim; ++i)
        {
          double magnitude = 0.0;
          for (size_type j = 0; j < dim; ++j)
            magnitude += d_domainBoundingVectors[i][j] *
                         d_domainBoundingVectors[i][j];
          magnitude = std::sqrt(magnitude);
          if (i == 0 || magnitude < minMagnitude)
            minMagnitude = magnitude;
        }

      // Enough shells that the cutoff sphere is covered even along the
      // shortest lattice direction, with dftfe's factor of two of slack.
      const int numberLayers = (int)std::ceil(2.0 * cutOff / minMagnitude);

      // Two passes over the same lattice translations: the first only counts
      // how many survive the cutoff so the three lists can be sized exactly,
      // the second fills them by index.
      size_type nImages = 0;
      for (size_type iPass = 0; iPass < 2; ++iPass)
        {
      if (iPass == 1)
        {
          imageIds.resize(nImages, 0);
          imageCharges.resize(nImages, 0.0);
          imagePositions.resize(nImages, utils::Point((size_type)dim, 0.0));
          nImages = 0;
        }

      for (size_type iAtom = 0; iAtom < d_nMasterAtoms; ++iAtom)
        {
          double atomReal[dim], atomFrac[dim];
          for (size_type i = 0; i < dim; ++i)
            atomReal[i] = d_atomCoordinates[iAtom][i] + shift[i];
          realToFrac(latticeVectors, atomReal, atomFrac);

          for (int iz = -numberLayers; iz <= numberLayers; ++iz)
            {
              if (periodic[2] == 0 && iz != 0)
                continue;
              for (int iy = -numberLayers; iy <= numberLayers; ++iy)
                {
                  if (periodic[1] == 0 && iy != 0)
                    continue;
                  for (int ix = -numberLayers; ix <= numberLayers; ++ix)
                    {
                      if (periodic[0] == 0 && ix != 0)
                        continue;

                      // the zero translation is the master atom itself
                      if (ix == 0 && iy == 0 && iz == 0)
                        continue;

                      const double imageFrac[dim] = {atomFrac[0] + ix,
                                                     atomFrac[1] + iy,
                                                     atomFrac[2] + iz};

                      if (getMinDistanceFromImageToDomain(latticeVectors,
                                                        imageFrac) >= cutOff)
                        continue;

                      double imageReal[dim];
                      fracToReal(latticeVectors, imageFrac, imageReal);
                      for (size_type i = 0; i < dim; ++i)
                        imageReal[i] -= shift[i];

                      if (iPass == 1)
                        {
                          imageIds[nImages]     = iAtom;
                          imageCharges[nImages] = d_atomCharges[iAtom];
                          for (size_type i = 0; i < dim; ++i)
                            imagePositions[nImages][i] = imageReal[i];
                        }
                      nImages++;
                    }
                }
            }
        }
        }
    }

    void
    PeriodicImageAtomGenerator::buildImageIdsPerMaster(
      const size_type                      nMasterAtoms,
      const std::vector<size_type> &       imageIds,
      std::vector<std::vector<size_type>> &imageIdsPerMaster)
    {
      imageIdsPerMaster.clear();
      imageIdsPerMaster.resize(nMasterAtoms);

      std::vector<size_type> nImagesPerMaster(nMasterAtoms, 0);
      for (size_type iImage = 0; iImage < imageIds.size(); ++iImage)
        nImagesPerMaster[imageIds[iImage]]++;
      for (size_type iAtom = 0; iAtom < nMasterAtoms; ++iAtom)
        imageIdsPerMaster[iAtom].resize(nImagesPerMaster[iAtom], 0);

      std::vector<size_type> nFilledPerMaster(nMasterAtoms, 0);
      for (size_type iImage = 0; iImage < imageIds.size(); ++iImage)
        {
          const size_type iAtom = imageIds[iImage];
          imageIdsPerMaster[iAtom][nFilledPerMaster[iAtom]] = iImage;
          nFilledPerMaster[iAtom]++;
        }
    }

    const std::vector<size_type> &
    PeriodicImageAtomGenerator::getImageIds() const
    {
      return d_imageIds;
    }

    const std::vector<double> &
    PeriodicImageAtomGenerator::getImageCharges() const
    {
      return d_imageCharges;
    }

    const std::vector<utils::Point> &
    PeriodicImageAtomGenerator::getImagePositions() const
    {
      return d_imagePositions;
    }

    const std::vector<size_type> &
    PeriodicImageAtomGenerator::getImageIdsForMaster(
      const size_type masterAtomId) const
    {
      if (masterAtomId >= d_imageIdsPerMaster.size())
        return d_emptyImageIds;
      return d_imageIdsPerMaster[masterAtomId];
    }

    const std::vector<size_type> &
    PeriodicImageAtomGenerator::getImageIdsTrunc() const
    {
      return d_imageIdsTrunc;
    }

    const std::vector<double> &
    PeriodicImageAtomGenerator::getImageChargesTrunc() const
    {
      return d_imageChargesTrunc;
    }

    const std::vector<utils::Point> &
    PeriodicImageAtomGenerator::getImagePositionsTrunc() const
    {
      return d_imagePositionsTrunc;
    }

    const std::vector<size_type> &
    PeriodicImageAtomGenerator::getImageIdsForMasterTrunc(
      const size_type masterAtomId) const
    {
      if (masterAtomId >= d_imageIdsPerMasterTrunc.size())
        return d_emptyImageIds;
      return d_imageIdsPerMasterTrunc[masterAtomId];
    }

    void
    PeriodicImageAtomGenerator::throwIfAtomsOutsideCell() const
    {
      using namespace PeriodicImageAtomGeneratorInternal;

      // Nothing periodic means no cell frame is implied by the input, so there
      // is nothing to be outside of.
      if (!std::any_of(d_isPeriodicFlags.begin(),
                       d_isPeriodicFlags.end(),
                       [](bool v) { return v; }))
        return;

      std::vector<double> latticeVectors(0);
      double              shift[dim];
      buildCellFrame(d_domainBoundingVectors, latticeVectors, shift);

      const double tol = 1e-6;
      for (size_type iAtom = 0; iAtom < d_nMasterAtoms; ++iAtom)
        {
          double atomReal[dim], atomFrac[dim];
          for (size_type i = 0; i < dim; ++i)
            atomReal[i] = d_atomCoordinates[iAtom][i] + shift[i];
          realToFrac(latticeVectors, atomReal, atomFrac);

          for (size_type i = 0; i < dim; ++i)
            {
              const bool inside =
                d_isPeriodicFlags[i] ?
                  (atomFrac[i] > -tol && atomFrac[i] < 1.0 + tol) :
                  (atomFrac[i] > tol && atomFrac[i] < 1.0 - tol);

              utils::throwException<utils::InvalidArgument>(
                inside,
                "PeriodicImageAtomGenerator: atom " + std::to_string(iAtom) +
                  " lies outside the simulation domain along direction " +
                  std::to_string(i) + ". Centred Cartesian position (" +
                  std::to_string(d_atomCoordinates[iAtom][0]) + ", " +
                  std::to_string(d_atomCoordinates[iAtom][1]) + ", " +
                  std::to_string(d_atomCoordinates[iAtom][2]) +
                  "), fractional (" + std::to_string(atomFrac[0]) + ", " +
                  std::to_string(atomFrac[1]) + ", " +
                  std::to_string(atomFrac[2]) +
                  "). A periodic direction admits [0,1]; a non periodic one "
                  "requires a strict interior.");
            }
        }
    }

    void
    PeriodicImageAtomGenerator::throwIfAtomCoordinatesDiffer(
      const std::vector<utils::Point> &atomCoordinates) const
    {
      using namespace PeriodicImageAtomGeneratorInternal;

      utils::throwException<utils::InvalidArgument>(
        atomCoordinates.size() == d_nMasterAtoms,
        "PeriodicImageAtomGenerator: this generator was built from " +
          std::to_string(d_nMasterAtoms) + " master atoms but is being used "
          "with " + std::to_string(atomCoordinates.size()) +
          ", so its images belong to a different system.");

      for (size_type iAtom = 0; iAtom < d_nMasterAtoms; ++iAtom)
        for (size_type j = 0; j < (size_type)dim; ++j)
          utils::throwException<utils::InvalidArgument>(
            std::abs(d_atomCoordinates[iAtom][j] - atomCoordinates[iAtom][j]) <
              1e-12,
            "PeriodicImageAtomGenerator: the master atoms have moved since "
            "this generator was built, so its images are at stale positions. "
            "Rebuild the generator alongside the new coordinates.");
    }

    void
    PeriodicImageAtomGenerator::fillExtendedAtoms(
      const std::vector<utils::Point> &imagePositions,
      const std::vector<double> &      imageCharges,
      std::vector<utils::Point> &      coordinates,
      std::vector<double> &            charges) const
    {
      using namespace PeriodicImageAtomGeneratorInternal;

      coordinates.resize(d_nMasterAtoms + imagePositions.size(),
                         utils::Point((size_type)dim, 0.0));
      charges.resize(d_nMasterAtoms + imageCharges.size(), 0.0);

      std::copy(d_atomCoordinates.begin(),
                d_atomCoordinates.end(),
                coordinates.begin());
      std::copy(d_atomCharges.begin(), d_atomCharges.end(), charges.begin());

      std::copy(imagePositions.begin(),
                imagePositions.end(),
                coordinates.begin() + d_nMasterAtoms);
      std::copy(imageCharges.begin(),
                imageCharges.end(),
                charges.begin() + d_nMasterAtoms);
    }

    void
    PeriodicImageAtomGenerator::fillExtendedAtomsForMaster(
      const size_type                  masterAtomId,
      const std::vector<size_type> &   imageIds,
      const std::vector<utils::Point> &imagePositions,
      const std::vector<double> &      imageCharges,
      std::vector<utils::Point> &      coordinates,
      std::vector<double> &            charges) const
    {
      using namespace PeriodicImageAtomGeneratorInternal;

      utils::throwException<utils::InvalidArgument>(
        masterAtomId < d_nMasterAtoms,
        "PeriodicImageAtomGenerator: master atom id out of range.");

      coordinates.resize(1 + imageIds.size(),
                         utils::Point((size_type)dim, 0.0));
      charges.resize(1 + imageIds.size(), 0.0);

      coordinates[0] = d_atomCoordinates[masterAtomId];
      charges[0]     = d_atomCharges[masterAtomId];

      for (size_type i = 0; i < imageIds.size(); ++i)
        {
          coordinates[1 + i] = imagePositions[imageIds[i]];
          charges[1 + i]     = imageCharges[imageIds[i]];
        }
    }

    void
    PeriodicImageAtomGenerator::getExtendedAtoms(
      const std::vector<utils::Point> &atomCoordinates,
      std::vector<utils::Point> &      coordinates,
      std::vector<double> &            charges) const
    {
      throwIfAtomCoordinatesDiffer(atomCoordinates);
      fillExtendedAtoms(d_imagePositions, d_imageCharges, coordinates, charges);
    }

    void
    PeriodicImageAtomGenerator::getExtendedAtomsTrunc(
      const std::vector<utils::Point> &atomCoordinates,
      std::vector<utils::Point> &      coordinates,
      std::vector<double> &            charges) const
    {
      throwIfAtomCoordinatesDiffer(atomCoordinates);
      fillExtendedAtoms(d_imagePositionsTrunc,
                        d_imageChargesTrunc,
                        coordinates,
                        charges);
    }

    void
    PeriodicImageAtomGenerator::getExtendedAtomsForMaster(
      const std::vector<utils::Point> &atomCoordinates,
      const size_type                  masterAtomId,
      std::vector<utils::Point> &      coordinates,
      std::vector<double> &            charges) const
    {
      throwIfAtomCoordinatesDiffer(atomCoordinates);
      fillExtendedAtomsForMaster(masterAtomId,
                                 getImageIdsForMaster(masterAtomId),
                                 d_imagePositions,
                                 d_imageCharges,
                                 coordinates,
                                 charges);
    }

    void
    PeriodicImageAtomGenerator::getExtendedAtomsForMasterTrunc(
      const std::vector<utils::Point> &atomCoordinates,
      const size_type                  masterAtomId,
      std::vector<utils::Point> &      coordinates,
      std::vector<double> &            charges) const
    {
      throwIfAtomCoordinatesDiffer(atomCoordinates);
      fillExtendedAtomsForMaster(masterAtomId,
                                 getImageIdsForMasterTrunc(masterAtomId),
                                 d_imagePositionsTrunc,
                                 d_imageChargesTrunc,
                                 coordinates,
                                 charges);
    }

    double
    PeriodicImageAtomGenerator::getCutOff() const
    {
      return d_cutOff;
    }

    double
    PeriodicImageAtomGenerator::getCutOffTrunc() const
    {
      return d_cutOffTrunc;
    }

    size_type
    PeriodicImageAtomGenerator::nMasterAtoms() const
    {
      return d_nMasterAtoms;
    }

    const std::vector<utils::Point> &
    PeriodicImageAtomGenerator::getAtomCoordinates() const
    {
      return d_atomCoordinates;
    }

  } // namespace basis
} // namespace dftefe
