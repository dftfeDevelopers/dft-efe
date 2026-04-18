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
    AtomSevereFunction<memorySpace>::AtomSevereFunction(
      std::shared_ptr<const AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      const size_type                  derivativeType,
      const size_type                  sphericalValPower,
      const double                     constant,
      linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(atomCoordinatesVec)
      , d_fieldName(fieldName)
      , d_derivativeType(derivativeType)
      , d_sphericalValPower(sphericalValPower)
      , d_constant(constant)
      , d_numAtoms(atomCoordinatesVec.size())
      , d_dim(atomCoordinatesVec[0].size())
      , d_numEnrichmentFuncTotal(0)
      , d_linAlgOpContext(linAlgOpContext)
    {
      utils::throwException(derivativeType == 0 || derivativeType == 1,
                            "The derivative type can only be 0 or 1");

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
    double
    AtomSevereFunction<memorySpace>::operator()(
      const utils::Point &point) const
    {
      double retValue = 0;
      if (d_derivativeType == 0)
        {
          for (size_type atomId = 0; atomId < d_atomCoordinatesVec.size();
               atomId++)
            {
              auto vec = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName);
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              for (auto &enrichmentObjId : vec)
                {
                  double val = enrichmentObjId->getValue(point, origin);
                  retValue   = retValue + pow(val, d_sphericalValPower);
                }
            }
        }
      if (d_derivativeType == 1)
        {
          for (size_type atomId = 0; atomId < d_atomCoordinatesVec.size();
               atomId++)
            {
              auto vec = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName);
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              for (auto &enrichmentObjId : vec)
                {
                  std::vector<double> val =
                    enrichmentObjId->getGradientValue(point, origin);
                  for (size_type iDim = 0; iDim < d_dim; iDim++)
                    {
                      retValue =
                        retValue + pow(val[iDim], d_sphericalValPower);
                    }
                }
            }
        }
      return retValue * d_constant;
    }

    template <utils::MemorySpace memorySpace>
    std::vector<double>
    AtomSevereFunction<memorySpace>::operator()(
      const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> retValue(N, 0.0);
      if (d_derivativeType == 0)
        {
          for (size_type atomId = 0; atomId < d_atomCoordinatesVec.size();
               atomId++)
            {
              auto vec = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName);
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              for (auto &enrichmentObjId : vec)
                {
                  std::vector<double> val =
                    enrichmentObjId->getValue(points, origin);
                  for (unsigned int iPoint = 0; iPoint < N; ++iPoint)
                    retValue[iPoint] =
                      retValue[iPoint] + pow(val[iPoint], d_sphericalValPower);
                }
            }
        }
      if (d_derivativeType == 1)
        {
          for (size_type atomId = 0; atomId < d_atomCoordinatesVec.size();
               atomId++)
            {
              auto vec = d_atomSphericalDataContainer->getSphericalData(
                d_atomSymbolVec[atomId], d_fieldName);
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              for (auto &enrichmentObjId : vec)
                {
                  std::vector<double> val =
                    enrichmentObjId->getGradientValue(points, origin);
                  for (unsigned int iPoint = 0; iPoint < N; ++iPoint)
                    for (size_type iDim = 0; iDim < d_dim; iDim++)
                      retValue[iPoint] =
                        retValue[iPoint] +
                        pow(val[iPoint * d_dim + iDim], d_sphericalValPower);
                }
            }
        }
      for (int i = 0; i < retValue.size(); i++)
        retValue[i] *= d_constant;
      return retValue;
    }

    template <utils::MemorySpace memorySpace>
    void
    AtomSevereFunction<memorySpace>::evalHost(size_type     numPoints,
                                              const double *t,
                                              double *      q) const
    {
      utils::Point              p(d_dim);
      std::vector<utils::Point> points(numPoints, p);
      for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
        for (size_type iDim = 0; iDim < d_dim; ++iDim)
          points[iPoint][iDim] = t[iPoint * d_dim + iDim];
      std::vector<double> retValue = (*this)(points);
      std::copy(retValue.begin(), retValue.end(), q);
    }

  } // namespace atoms
} // namespace dftefe
