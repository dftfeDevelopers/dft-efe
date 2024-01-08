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
      std::shared_ptr<const basis::EnrichmentIdsPartition<dim>>
        enrichmentIdsPartition,
      std::shared_ptr<const AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      const size_type                  derivativeType)
      : d_enrichmentIdsPartition(enrichmentIdsPartition)
      , d_atomSphericalDataContainer(atomSphericalDataContainer)
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

    template <unsigned int dim>
    std::vector<double>
    AtomSevereFunction<dim>::operator()(
      const std::vector<utils::Point> &points) const
    {
      const size_type                               N = points.size();
      std::vector<double>                           retValue(N, 0.0);
      std::pair<global_size_type, global_size_type> locallyOwnedEnrichemntIds =
        d_enrichmentIdsPartition->locallyOwnedEnrichmentIds();
      std::vector<global_size_type> ghostEnrichmentIds =
        d_enrichmentIdsPartition->ghostEnrichmentIds();
      for (unsigned int j = 0; j < N; ++j)
        {
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
                  qNumbers =
                    d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                              d_fieldName);
                  auto sphericalData =
                    d_atomSphericalDataContainer->getSphericalData(
                      atomSymbol, d_fieldName, qNumbers[qNumberId]);
                  retValue[j] =
                    retValue[j] + sphericalData->getValue(points[j], origin) *
                                    sphericalData->getValue(points[j], origin);
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
                  qNumbers =
                    d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                              d_fieldName);
                  auto sphericalData =
                    d_atomSphericalDataContainer->getSphericalData(
                      atomSymbol, d_fieldName, qNumbers[qNumberId]);
                  retValue[j] =
                    retValue[j] + sphericalData->getValue(points[j], origin) *
                                    sphericalData->getValue(points[j], origin);
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
                  qNumbers =
                    d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                              d_fieldName);
                  auto sphericalData =
                    d_atomSphericalDataContainer->getSphericalData(
                      atomSymbol, d_fieldName, qNumbers[qNumberId]);
                  for (size_type k = 0; k < dim; k++)
                    {
                      retValue[j] =
                        retValue[j] +
                        sphericalData->getGradientValue(points[j], origin)[k] *
                          sphericalData->getGradientValue(points[j], origin)[k];
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
                  qNumbers =
                    d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                              d_fieldName);
                  auto sphericalData =
                    d_atomSphericalDataContainer->getSphericalData(
                      atomSymbol, d_fieldName, qNumbers[qNumberId]);
                  for (size_type k = 0; k < dim; k++)
                    {
                      retValue[j] =
                        retValue[j] +
                        sphericalData->getGradientValue(points[j], origin)[k] *
                          sphericalData->getGradientValue(points[j], origin)[k];
                    }
                }
            }
        }
      return retValue;
    }
  } // namespace atoms
} // namespace dftefe
