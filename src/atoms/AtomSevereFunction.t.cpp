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

namespace dftefe
{
  namespace atoms
  {
    template <unsigned int dim>
    AtomSevereFunction<dim>::AtomSevereFunction(
      std::shared_ptr<const AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      const size_type                  derivativeType)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(atomCoordinatesVec)
      , d_fieldName(fieldName)
      , d_derivativeType(derivativeType)
    {
      utils::throwException(derivativeType == 0 || derivativeType == 1,
                            "The derivative type can only be 0 or 1");
    }

    template <unsigned int dim>
    double
    AtomSevereFunction<dim>::operator()(const utils::Point &point) const
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
                  retValue   = retValue + val * val;
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
                  for (size_type iDim = 0; iDim < dim; iDim++)
                    {
                      retValue = retValue + val[iDim] * val[iDim];
                    }
                }
            }
        }
      return retValue;
    }

    // check for r!=0 gradient.
    template <unsigned int dim>
    std::vector<double>
    AtomSevereFunction<dim>::operator()(
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
              for (unsigned int iPoint = 0; iPoint < N; ++iPoint)
                {
                  for (auto &enrichmentObjId : vec)
                    {
                      double val =
                        enrichmentObjId->getValue(points[iPoint], origin);
                      retValue[iPoint] = retValue[iPoint] + val * val;
                    }
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
              for (unsigned int iPoint = 0; iPoint < N; ++iPoint)
                {
                  for (auto &enrichmentObjId : vec)
                    {
                      std::vector<double> val =
                        enrichmentObjId->getGradientValue(points[iPoint],
                                                          origin);
                      for (size_type iDim = 0; iDim < dim; iDim++)
                        {
                          retValue[iPoint] =
                            retValue[iPoint] + val[iDim] * val[iDim];
                        }
                    }
                }
            }
        }
      return retValue;
    }
  } // namespace atoms
} // namespace dftefe


/**
    template <unsigned int dim>
    double
    AtomSevereFunction<dim>::operator()(const utils::Point &point) const
    {
      double                                        retValue = 0;
      std::pair<global_size_type, global_size_type> locallyOwnedEnrichemntIds =
        d_enrichmentIdsPartition->locallyOwnedEnrichmentIds();
      std::vector<global_size_type> ghostEnrichmentIds =
        d_enrichmentIdsPartition->ghostEnrichmentIds();
      if (d_derivativeType == 0)
        {
          for (global_size_type i = locallyOwnedEnrichemntIds.first;
               i < locallyOwnedEnrichemntIds.second;
               i++)
            {
              size_type atomId = d_enrichmentIdsPartition->getAtomId(i);
              size_type qNumberId =
                (d_enrichmentIdsPartition->getEnrichmentIdAttribute(i))
                  .localIdInAtom;
              std::string  atomSymbol = d_atomSymbolVec[atomId];
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              std::vector<std::vector<int>> qNumbers(0);
              qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                   d_fieldName);
              auto sphericalData =
                d_atomSphericalDataContainer->getSphericalData(
                  atomSymbol, d_fieldName, qNumbers[qNumberId]);
              retValue = retValue + sphericalData->getValue(point, origin) *
                                      sphericalData->getValue(point, origin);
            }
          for (auto i : ghostEnrichmentIds)
            {
              size_type atomId = d_enrichmentIdsPartition->getAtomId(i);
              size_type qNumberId =
                (d_enrichmentIdsPartition->getEnrichmentIdAttribute(i))
                  .localIdInAtom;
              std::string  atomSymbol = d_atomSymbolVec[atomId];
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              std::vector<std::vector<int>> qNumbers(0);
              qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                   d_fieldName);
              auto sphericalData =
                d_atomSphericalDataContainer->getSphericalData(
                  atomSymbol, d_fieldName, qNumbers[qNumberId]);
              retValue = retValue + sphericalData->getValue(point, origin) *
                                      sphericalData->getValue(point, origin);
            }
        }
      if (d_derivativeType == 1)
        {
          for (global_size_type i = locallyOwnedEnrichemntIds.first;
               i < locallyOwnedEnrichemntIds.second;
               i++)
            {
              size_type atomId = d_enrichmentIdsPartition->getAtomId(i);
              size_type qNumberId =
                (d_enrichmentIdsPartition->getEnrichmentIdAttribute(i))
                  .localIdInAtom;
              std::string  atomSymbol = d_atomSymbolVec[atomId];
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              std::vector<std::vector<int>> qNumbers(0);
              qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                   d_fieldName);
              auto sphericalData =
                d_atomSphericalDataContainer->getSphericalData(
                  atomSymbol, d_fieldName, qNumbers[qNumberId]);
              for (size_type j = 0; j < dim; j++)
                {
                  retValue =
                    retValue +
                    sphericalData->getGradientValue(point, origin)[j] *
                      sphericalData->getGradientValue(point, origin)[j];
                }
            }
          for (auto i : ghostEnrichmentIds)
            {
              size_type atomId = d_enrichmentIdsPartition->getAtomId(i);
              size_type qNumberId =
                (d_enrichmentIdsPartition->getEnrichmentIdAttribute(i))
                  .localIdInAtom;
              std::string  atomSymbol = d_atomSymbolVec[atomId];
              utils::Point origin(d_atomCoordinatesVec[atomId]);
              std::vector<std::vector<int>> qNumbers(0);
              qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                   d_fieldName);
              auto sphericalData =
                d_atomSphericalDataContainer->getSphericalData(
                  atomSymbol, d_fieldName, qNumbers[qNumberId]);
              for (size_type j = 0; j < dim; j++)
                {
                  retValue =
                    retValue +
                    sphericalData->getGradientValue(point, origin)[j] *
                      sphericalData->getGradientValue(point, origin)[j];
                }
            }
        }
      return retValue;
    }
**/
