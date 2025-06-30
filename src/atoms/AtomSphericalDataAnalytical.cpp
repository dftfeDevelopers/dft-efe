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

#include <atoms/AtomSphericalDataAnalytical.h>
#include <utils/Exceptions.h>
#include <utils/StringOperations.h>
#include <sstream>
#include <vector>
#include <iterator>
#include <iomanip>
#include <cstring>
#include <string>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      void
      storeQNumbersToDataIdMap(
        const std::vector<std::shared_ptr<SphericalData>> &sphericalDataVec,
        std::map<std::vector<int>, size_type> &            qNumbersToDataIdMap)
      {
        size_type N = sphericalDataVec.size();
        for (size_type i = 0; i < N; ++i)
          {
            qNumbersToDataIdMap[sphericalDataVec[i]->getQNumbers()] = i;
          }
      }
    } // namespace

    AtomSphericalDataAnalytical::AtomSphericalDataAnalytical(
        const std::map<std::string , std::vector<std::vector<int>>> &fieldToQuantumNumbersVec,
        const std::map<std::string , std::vector<std::shared_ptr<utils::ScalarSpatialFunctionReal>>> &fieldToScalarSpatialFnRealVec,
        const std::vector<std::string> &  fieldNames,
        const SphericalHarmonicFunctions &sphericalHarmonicFunc)
      : d_fieldNames(fieldNames)
      , d_sphericalHarmonicFunc(sphericalHarmonicFunc)
      , d_radialPointMax(100.0)
    {
      //
      // store field spherical data
      //
      for (size_type iField = 0; iField < fieldNames.size(); ++iField)
        {
          const std::string fieldName = fieldNames[iField];
          std::vector<std::shared_ptr<SphericalData>> sphericalDataVec(0);
          std::map<std::vector<int>, size_type>       qNumbersToIdMap;
          getSphericalDataFromSpatialFn(sphericalDataVec,
                                      fieldToQuantumNumbersVec,
                                      fieldToScalarSpatialFnRealVec,
                                      fieldName,
                                      sphericalHarmonicFunc);
          storeQNumbersToDataIdMap(sphericalDataVec, qNumbersToIdMap);
          d_sphericalData[fieldName]   = sphericalDataVec;
          d_qNumbersToIdMap[fieldName] = qNumbersToIdMap;
        }
    }

    void
    AtomSphericalDataAnalytical::getSphericalDataFromSpatialFn(
      std::vector<std::shared_ptr<SphericalData>> &sphericalDataVec,
      const std::map<std::string , std::vector<std::vector<int>>> &fieldToQuantumNumbersVec,
      const std::map<std::string , std::vector<std::shared_ptr<utils::ScalarSpatialFunctionReal>>> &fieldToScalarSpatialFnRealVec,
      const std::string &                          fieldName,
      const SphericalHarmonicFunctions &           sphericalHarmonicFunc)
    {
      size_type N;

      sphericalDataVec.resize(0);

      std::vector<std::vector<int>> qNumbersVec(0);
     
      // get quantum Numbers vec
      std::vector<int> nodeIndex(0);

      double cutoff , smoothness;

      auto itQNum = fieldToQuantumNumbersVec.find(fieldName);
      auto itFun = fieldToScalarSpatialFnRealVec.find(fieldName);

      utils::throwException((itQNum != fieldToQuantumNumbersVec.end() && itFun != fieldToScalarSpatialFnRealVec.end()),
                          "Could not find field name in the input maps in AtomSphericalDataAnalytical");

      utils::throwException((itQNum->second.size() == itFun->second.size()),
                              "The sizes of Quantum numbers and functions vector are not the same in AtomSphericalDataAnalytical");

      N     = itQNum->second.size();
      for (size_type iQNum = 0; iQNum < N; ++iQNum)
        {
          qNumbersVec.push_back(itQNum->second[iQNum]);
          if(std::abs((*itFun->second[iQNum])(utils::Point({d_radialPointMax,0.,0.}))) > 1e-10)
          {
            cutoff = 1e6;
            smoothness = 1e10;
          }
          else
          {
            for(double iPoint = d_radialPointMax ; iPoint > 0 ; iPoint-=1e-2)
            {
              if(std::abs((*itFun->second[iQNum])(utils::Point({iPoint,0.,0.}))) > 1e-10)
              {
                cutoff = iPoint;
                smoothness = 1e10;
                break;
              }
            }
          }
          sphericalDataVec.push_back(
            std::make_shared<SphericalDataAnalytical>(
              itQNum->second[iQNum],
              *itFun->second[iQNum],
              cutoff,
              smoothness,
              sphericalHarmonicFunc));
        }
    }

    void
    AtomSphericalDataAnalytical::addFieldName(const std::string fieldName)
    {
      utils::throwException(false,
                            "addFiledName() function cannot be called in AtomSphDataAnalytical class.");
    }

    std::string
    AtomSphericalDataAnalytical::getFileName() const
    {
      utils::throwException(false,
                            "getFileName() function cannot be called in AtomSphDataAnalytical class.");
      return std::string();                      
    }

    std::vector<std::string>
    AtomSphericalDataAnalytical::getFieldNames() const
    {
      return d_fieldNames;
    }

    std::vector<std::string>
    AtomSphericalDataAnalytical::getMetadataNames() const
    {
      utils::throwException(false,
                            "getMetadataNames() function cannot be called in AtomSphDataAnalytical class.");
      return std::vector<std::string>(0);
    }

    const std::vector<std::shared_ptr<SphericalData>> &
    AtomSphericalDataAnalytical::getSphericalData(const std::string fieldName) const
    {
      auto it = d_sphericalData.find(fieldName);
      DFTEFE_AssertWithMsg(it != d_sphericalData.end(),
                           "FieldName " + fieldName +
                            " not found in AtomSphDataAnalytical");
      return it->second;
    }

    const std::shared_ptr<SphericalData>
    AtomSphericalDataAnalytical::getSphericalData(
      const std::string       fieldName,
      const std::vector<int> &qNumbers) const
    {
      auto it = d_sphericalData.find(fieldName);
      DFTEFE_AssertWithMsg(it != d_sphericalData.end(),
                           ("Unable to find the field " + fieldName +
                            " not found in AtomSphDataAnalytical"));
      auto iter = d_qNumbersToIdMap.find(fieldName);
      DFTEFE_AssertWithMsg(iter != d_qNumbersToIdMap.end(),
                           ("Unable to find the field " + fieldName +
                            " not found in AtomSphDataAnalytical"));
      auto iterQNumberToId = (iter->second).find(qNumbers);
      if (iterQNumberToId != (iter->second).end())
        return *((it->second).begin() + iterQNumberToId->second);
      else
        {
          std::string s = "";
          for (size_type i = 0; i < qNumbers.size(); i++)
            s += std::to_string(qNumbers[i]) + " ";

          DFTEFE_AssertWithMsg(false,
                               ("Unable to find the qNumbers " + s + " for " +
                                " the field " + fieldName +
                                " not found in AtomSphDataAnalytical"));
          return *((it->second).begin() + iterQNumberToId->second);
        }
    }

    std::string
    AtomSphericalDataAnalytical::getMetadata(const std::string metadataName) const
    {
      utils::throwException(false,
                      "getMetadata() function cannot be called in AtomSphDataAnalytical class.");
      return std::string();
    }

    size_type
    AtomSphericalDataAnalytical::getQNumberID(const std::string       fieldName,
                                       const std::vector<int> &qNumbers) const
    {
      auto it = d_qNumbersToIdMap.find(fieldName);
      utils::throwException<utils::InvalidArgument>(
        it != d_qNumbersToIdMap.end(),
        "Cannot find the fieldName provided to AtomSphericalDataAnalytical::getQNumberID");
      auto it1 = (it->second).find(qNumbers);
      utils::throwException<utils::InvalidArgument>(
        it1 != (it->second).end(),
        "Cannot find the qnumbers provided to AtomSphericalDataAnalytical::getQNumberID");
      return (it1)->second;
    }

    size_type
    AtomSphericalDataAnalytical::nSphericalData(std::string fieldName) const
    {
      auto it = d_sphericalData.find(fieldName);
      utils::throwException<utils::InvalidArgument>(
        it != d_sphericalData.end(),
        "Cannot find the fieldName provided to AtomSphericalDataAnalytical::nSphericalData");
      return (it->second).size();
    }

  } // end of namespace atoms
} // end of namespace dftefe
