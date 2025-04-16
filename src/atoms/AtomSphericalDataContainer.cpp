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

#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <map>
#include <string>
#include <atoms/AtomSphericalDataEnrichment.h>
#include <atoms/AtomSphericalDataPSP.h>
#include <atoms/SphericalData.h>
#include <atoms/AtomSphericalDataContainer.h>
namespace dftefe
{
  namespace atoms
  {
    // Constructor
    AtomSphericalDataContainer::AtomSphericalDataContainer(
      const AtomSphericalDataType &             atomSphericalDataType,
      const std::map<std::string, std::string> &atomSymbolToFilename,
      const std::vector<std::string> &          fieldNames,
      const std::vector<std::string> &          metadataNames,
      const bool                                isAssocLegendreSplineEval)
      : d_atomSymbolToFilename(atomSymbolToFilename)
      , d_fieldNames(fieldNames)
      , d_metadataNames(metadataNames)
      , d_isAssocLegendreSplineEval(isAssocLegendreSplineEval)
    {
      d_SphericalHarmonicFunctions =
        std::make_shared<const SphericalHarmonicFunctions>(false);
      auto iter = d_atomSymbolToFilename.begin();

      if (atomSphericalDataType == AtomSphericalDataType::ENRICHMENT)
        {
          for (; iter != d_atomSymbolToFilename.end(); iter++)
            {
              d_mapAtomSymbolToAtomSphericalData.insert(
                {iter->first,
                 std::make_shared<AtomSphericalDataEnrichment>(
                   iter->second,
                   d_fieldNames,
                   d_metadataNames,
                   *d_SphericalHarmonicFunctions)});
            }
        }
      else if (atomSphericalDataType == AtomSphericalDataType::PSEUDOPOTENTIAL)
        {
          for (; iter != d_atomSymbolToFilename.end(); iter++)
            {
              d_mapAtomSymbolToAtomSphericalData.insert(
                {iter->first,
                 std::make_shared<AtomSphericalDataPSP>(
                   iter->second,
                   d_fieldNames,
                   d_metadataNames,
                   *d_SphericalHarmonicFunctions)});
            }
        }
      else
        utils::throwException(
          false,
          "AtomSphericalDataType can only be of types ENRICHMENT and PSEUDOPOTENTIAL.");
    }

    void
    AtomSphericalDataContainer::addFieldName(
      const std::string fieldName)
    {
      for(auto &pair : d_mapAtomSymbolToAtomSphericalData)
      {
        pair.second->addFieldName(fieldName);
      }
    }

    const std::vector<std::shared_ptr<SphericalData>> &
    AtomSphericalDataContainer::getSphericalData(
      std::string       atomSymbol,
      const std::string fieldName) const
    {
      auto it = d_mapAtomSymbolToAtomSphericalData.find(atomSymbol);
      DFTEFE_AssertWithMsg(
        it != d_mapAtomSymbolToAtomSphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataContainer::getSphericalData");
      return (it->second)->getSphericalData(fieldName);
    }

    const std::shared_ptr<SphericalData>
    AtomSphericalDataContainer::getSphericalData(
      std::string             atomSymbol,
      const std::string       fieldName,
      const std::vector<int> &qNumbers) const
    {
      auto it = d_mapAtomSymbolToAtomSphericalData.find(atomSymbol);
      DFTEFE_AssertWithMsg(
        it != d_mapAtomSymbolToAtomSphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataContainer::getSphericalData");
      return (it->second)->getSphericalData(fieldName, qNumbers);
    }

    std::string
    AtomSphericalDataContainer::getMetadata(std::string atomSymbol,
                                            std::string metadataName) const
    {
      auto it = d_mapAtomSymbolToAtomSphericalData.find(atomSymbol);
      utils::throwException<utils::InvalidArgument>(
        it != d_mapAtomSymbolToAtomSphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataContainer::getMetadata");
      return (it->second)->getMetadata(metadataName);
    }

    std::vector<std::vector<int>>
    AtomSphericalDataContainer::getQNumbers(std::string       atomSymbol,
                                            const std::string fieldName) const
    {
      auto it = d_mapAtomSymbolToAtomSphericalData.find(atomSymbol);
      DFTEFE_AssertWithMsg(
        it != d_mapAtomSymbolToAtomSphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataContainer::getQNumbers");
      std::vector<std::shared_ptr<SphericalData>> sphericalDataVec =
        (it->second)->getSphericalData(fieldName);
      std::vector<std::vector<int>> qNumberVec;
      for (auto i : sphericalDataVec)
        qNumberVec.push_back(i->getQNumbers());
      return qNumberVec;
    }

    size_type
    AtomSphericalDataContainer::getQNumberID(
      std::string             atomSymbol,
      const std::string       fieldName,
      const std::vector<int> &qNumbers) const
    {
      auto it = d_mapAtomSymbolToAtomSphericalData.find(atomSymbol);
      utils::throwException<utils::InvalidArgument>(
        it != d_mapAtomSymbolToAtomSphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataContainer::getQNumberID");
      return (it->second)->getQNumberID(fieldName, qNumbers);
    }

    size_type
    AtomSphericalDataContainer::nSphericalData(
      std::string       atomSymbol,
      const std::string fieldName) const
    {
      auto it = d_mapAtomSymbolToAtomSphericalData.find(atomSymbol);
      utils::throwException<utils::InvalidArgument>(
        it != d_mapAtomSymbolToAtomSphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataContainer::nSphericalDataContainer");
      return (it->second)->nSphericalData(fieldName);
    }

    std::map<std::string, std::string>
    AtomSphericalDataContainer::atomSymbolToFileMap() const
    {
      return d_atomSymbolToFilename;
    }
  } // end of namespace atoms
} // end of namespace dftefe
