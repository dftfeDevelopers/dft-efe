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

#include <cmath>

namespace dftefe
{
  namespace atoms
  {
    template <utils::MemorySpace memorySpace>
    AtomSuperpositionFunction<memorySpace>::AtomSuperpositionFunction(
      std::shared_ptr<const AtomSphericalDataContainer>
                                                   atomSphericalDataContainer,
      const std::vector<std::string> &             atomSymbol,
      const std::vector<utils::Point> &            atomCoordinates,
      const std::string                            fieldName,
      linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbol)
      , d_atomCoordinatesVec(atomCoordinates)
      , d_fieldName(fieldName)
      , d_numAtoms(atomCoordinates.size())
      , d_dim(atomCoordinates[0].size())
      , d_numEnrichmentFuncTotal(0)
      , d_linAlgOpContext(linAlgOpContext)
    {
      std::vector<double> originsFlat(0);
      for (size_type atomId = 0; atomId < d_numAtoms; atomId++)
        {
          auto vec = d_atomSphericalDataContainer->getSphericalData(
            d_atomSymbolVec[atomId], d_fieldName);
          for (auto &enrichmentObjId : vec)
            {
              d_sphericalDataVecAll.push_back(enrichmentObjId);
              for (size_type iDim = 0; iDim < d_dim; iDim++)
                originsFlat.push_back(d_atomCoordinatesVec[atomId][iDim]);
              d_numEnrichmentFuncTotal++;
            }
        }
      d_originsFlat.resize(originsFlat.size());
      d_originsFlat.copyFrom(originsFlat);
    }

    template <utils::MemorySpace memorySpace>
    void
    AtomSuperpositionFunction<memorySpace>::evalHost(
      size_type                       numPoints,
      const AtomSuperpositionFuncType atomSupType,
      const double                    constant,
      const double *                  t,
      double *                        q) const
    {
      utils::Point              p(d_dim);
      std::vector<utils::Point> points(numPoints, p);
      for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
        for (size_type iDim = 0; iDim < d_dim; ++iDim)
          points[iPoint][iDim] = t[iPoint * d_dim + iDim];

      if (atomSupType == AtomSuperpositionFuncType::Identity)
        {
          for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
            q[iPoint] = 0.0;
          for (size_type atomId = 0; atomId < d_numAtoms; atomId++)
            {
              auto vec = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName);
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              for (auto &enrichmentObjId : vec)
                {
                  std::vector<double> val =
                    enrichmentObjId->getValue(points, origin);
                  for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
                    q[iPoint] += val[iPoint];
                }
            }
          for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
            q[iPoint] *= constant;
        }
      else if (atomSupType == AtomSuperpositionFuncType::IdentitySq)
        {
          for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
            q[iPoint] = 0.0;
          for (size_type atomId = 0; atomId < d_numAtoms; atomId++)
            {
              auto vec = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName);
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              for (auto &enrichmentObjId : vec)
                {
                  std::vector<double> val =
                    enrichmentObjId->getValue(points, origin);
                  for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
                    q[iPoint] += val[iPoint] * val[iPoint];
                }
            }
          for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
            q[iPoint] *= constant;
        }
      else if (atomSupType == AtomSuperpositionFuncType::GradDotGradSq)
        {
          for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
            q[iPoint] = 0.0;
          for (size_type atomId = 0; atomId < d_numAtoms; atomId++)
            {
              auto vec = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName);
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              for (auto &enrichmentObjId : vec)
                {
                  std::vector<double> val =
                    enrichmentObjId->getGradientValue(points, origin);
                  for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
                    for (size_type iDim = 0; iDim < d_dim; ++iDim)
                      q[iPoint] +=
                        val[iPoint * d_dim + iDim] * val[iPoint * d_dim + iDim];
                }
            }
          for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
            q[iPoint] *= constant;
        }
      else if (atomSupType == AtomSuperpositionFuncType::Grad)
        {
          for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
            for (size_type iDim = 0; iDim < d_dim; ++iDim)
              q[iPoint * d_dim + iDim] = 0.0;
          for (size_type atomId = 0; atomId < d_numAtoms; atomId++)
            {
              auto vec = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName);
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              for (auto &enrichmentObjId : vec)
                {
                  std::vector<double> val =
                    enrichmentObjId->getGradientValue(points, origin);
                  for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
                    for (size_type iDim = 0; iDim < d_dim; ++iDim)
                      q[iPoint * d_dim + iDim] += val[iPoint * d_dim + iDim];
                }
            }
          for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
            for (size_type iDim = 0; iDim < d_dim; ++iDim)
              q[iPoint * d_dim + iDim] *= constant;
        }
    }

  } // namespace atoms
} // namespace dftefe
