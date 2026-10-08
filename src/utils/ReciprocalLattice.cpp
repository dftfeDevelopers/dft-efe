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

#include <utils/ReciprocalLattice.h>
#include <utils/Exceptions.h>
#include <utils/MathConstants.h>
#include <cmath>

namespace dftefe
{
  namespace utils
  {
    namespace
    {
      //
      // Fold a fractional coordinate into [-0.5, 0.5), so that two k-points
      // differing by a reciprocal lattice vector compare equal.
      //
      double
      wrapToFirstCell(const double x)
      {
        return x - std::floor(x + 0.5);
      }
    } // namespace

    ReciprocalLattice::ReciprocalLattice(
      const std::vector<utils::Point> &domainBoundingVectors)
      : d_dim(domainBoundingVectors.size())
    {
      utils::throwException<utils::InvalidArgument>(
        d_dim == 3,
        "ReciprocalLattice is defined for three dimensions only.");

      for (size_type iDim = 0; iDim < d_dim; iDim++)
        utils::throwException<utils::InvalidArgument>(
          domainBoundingVectors[iDim].size() == d_dim,
          "ReciprocalLattice: each domain bounding vector must have as many "
          "components as there are dimensions.");

      // b_i is built from the cross product of the other two vectors, so the
      // three cross products are formed once here and scaled below
      std::vector<utils::Point> crossProducts(d_dim, utils::Point(d_dim, 0.0));
      for (size_type iDim = 0; iDim < d_dim; iDim++)
        {
          const utils::Point &u = domainBoundingVectors[(iDim + 1) % d_dim];
          const utils::Point &v = domainBoundingVectors[(iDim + 2) % d_dim];
          crossProducts[iDim][0] = u[1] * v[2] - u[2] * v[1];
          crossProducts[iDim][1] = u[2] * v[0] - u[0] * v[2];
          crossProducts[iDim][2] = u[0] * v[1] - u[1] * v[0];
        }

      d_cellVolume = 0.0;
      for (size_type iDim = 0; iDim < d_dim; iDim++)
        d_cellVolume += domainBoundingVectors[0][iDim] * crossProducts[0][iDim];

      utils::throwException<utils::InvalidArgument>(
        std::abs(d_cellVolume) > 1e-12,
        "ReciprocalLattice: the domain bounding vectors are linearly "
        "dependent, so the cell has no volume.");

      d_reciprocalVectors.resize(d_dim, utils::Point(d_dim, 0.0));
      const double scale = 2.0 * utils::mathConstants::pi / d_cellVolume;
      for (size_type iDim = 0; iDim < d_dim; iDim++)
        for (size_type jDim = 0; jDim < d_dim; jDim++)
          d_reciprocalVectors[iDim][jDim] = scale * crossProducts[iDim][jDim];
    }

    const std::vector<utils::Point> &
    ReciprocalLattice::getReciprocalVectors() const
    {
      return d_reciprocalVectors;
    }

    double
    ReciprocalLattice::getCellVolume() const
    {
      return d_cellVolume;
    }

    void
    ReciprocalLattice::generateMonkhorstPackGrid(
      const std::vector<size_type> &nkGrid,
      const std::vector<bool> &     isPeriodicFlags,
      const std::vector<bool> &     isShifted,
      const bool                    useTimeReversalSymmetry,
      std::vector<utils::Point> &   kPointCoordinates,
      std::vector<double> &         kPointWeights) const
    {
      utils::throwException<utils::InvalidArgument>(
        nkGrid.size() == d_dim && isPeriodicFlags.size() == d_dim &&
          isShifted.size() == d_dim,
        "ReciprocalLattice::generateMonkhorstPackGrid: nkGrid, "
        "isPeriodicFlags and isShifted must all have one entry per "
        "dimension.");

      // A non-periodic direction carries no Bloch phase, so it contributes
      // its single k = 0 point whatever subdivision was asked for.
      std::vector<size_type> nkEffective(d_dim, 1);
      size_type              nkTotal = 1;
      for (size_type iDim = 0; iDim < d_dim; iDim++)
        {
          utils::throwException<utils::InvalidArgument>(
            nkGrid[iDim] > 0,
            "ReciprocalLattice::generateMonkhorstPackGrid: each grid "
            "subdivision must be at least one.");
          nkEffective[iDim] = isPeriodicFlags[iDim] ? nkGrid[iDim] : 1;
          nkTotal *= nkEffective[iDim];
        }

      //
      // Full grid in fractional coordinates, Monkhorst-Pack offsets
      // u_i = (2i - n + 1) / (2n), optionally shifted by half a spacing.
      //
      std::vector<double> fractionalCoords(nkTotal * d_dim, 0.0);
      size_type           iPoint = 0;
      for (size_type i0 = 0; i0 < nkEffective[0]; i0++)
        for (size_type i1 = 0; i1 < nkEffective[1]; i1++)
          for (size_type i2 = 0; i2 < nkEffective[2]; i2++)
            {
              const size_type gridIndex[3] = {i0, i1, i2};
              for (size_type iDim = 0; iDim < d_dim; iDim++)
                {
                  const double n = (double)nkEffective[iDim];
                  double       u =
                    (2.0 * (double)gridIndex[iDim] - n + 1.0) / (2.0 * n);
                  if (isShifted[iDim] && isPeriodicFlags[iDim])
                    u += 1.0 / (2.0 * n);
                  fractionalCoords[iPoint * d_dim + iDim] = wrapToFirstCell(u);
                }
              iPoint++;
            }

      //
      // Fold k and -k when time reversal is used. representative[i] is the
      // index of the point that carries i's weight; it is i for a point kept.
      //
      std::vector<size_type> representative(nkTotal, 0);
      for (size_type i = 0; i < nkTotal; i++)
        representative[i] = i;

      if (useTimeReversalSymmetry)
        {
          const double tol = 1e-10;
          for (size_type i = 0; i < nkTotal; i++)
            {
              if (representative[i] != i)
                continue;
              for (size_type j = i + 1; j < nkTotal; j++)
                {
                  if (representative[j] != j)
                    continue;
                  bool isNegative = true;
                  for (size_type iDim = 0; iDim < d_dim; iDim++)
                    {
                      const double sum =
                        wrapToFirstCell(fractionalCoords[i * d_dim + iDim] +
                                        fractionalCoords[j * d_dim + iDim]);
                      if (std::abs(sum) > tol)
                        {
                          isNegative = false;
                          break;
                        }
                    }
                  if (isNegative)
                    representative[j] = i;
                }
            }
        }

      size_type nkIrreducible = 0;
      for (size_type i = 0; i < nkTotal; i++)
        if (representative[i] == i)
          nkIrreducible++;

      kPointCoordinates.assign(nkIrreducible, utils::Point(d_dim, 0.0));
      kPointWeights.resize(nkIrreducible, 0.0);

      // compactIndex[i] is where the representative i landed in the output
      std::vector<size_type> compactIndex(nkTotal, 0);
      size_type              iKept = 0;
      for (size_type i = 0; i < nkTotal; i++)
        {
          if (representative[i] != i)
            continue;
          compactIndex[i] = iKept;
          for (size_type iDim = 0; iDim < d_dim; iDim++)
            {
              double kCartesian = 0.0;
              for (size_type jDim = 0; jDim < d_dim; jDim++)
                kCartesian += fractionalCoords[i * d_dim + jDim] *
                              d_reciprocalVectors[jDim][iDim];
              kPointCoordinates[iKept][iDim] = kCartesian;
            }
          iKept++;
        }

      const double weightPerGridPoint = 1.0 / (double)nkTotal;
      for (size_type i = 0; i < nkTotal; i++)
        kPointWeights[compactIndex[representative[i]]] += weightPerGridPoint;
    }

  } // end of namespace utils
} // end of namespace dftefe
