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

#include <atoms/AtomSphericalDataEnrichmentJSON.h>
#include <utils/Exceptions.h>
#include <utils/StringOperations.h>
#include <sstream>
#include <algorithm> 
#include <set>
#include "simdjson.h"
namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      void
      getSphericalDataFromJSON(
        std::vector<std::shared_ptr<SphericalData>> &         sphericalDataVec,
        const std::vector<double> &                           radialPoints,
        const std::string         &                           fieldName,
        const std::string         &                           fileName,
        std::vector<std::pair<int, int>> &        nlPairs,
        const SphericalHarmonicFunctions &                    sphericalHarmonicFunc)
      {
        sphericalDataVec.clear();
        auto json_res = simdjson::padded_string::load(fileName);
        if (json_res.error()) {
            utils::throwException(false ,  "Error loading JSON: " + json_res.error());
        }
        simdjson::padded_string json = std::move(json_res).value();

        simdjson::dom::parser parser;
        simdjson::dom::element doc = parser.parse(json);

        for(int j = 0 ; j < nlPairs.size() ; j++)
        {
          for(int m = -nlPairs[j].second ; m <= nlPairs[j].second ; m++)
          {
            std::vector<double> radialValues(0);
            std::vector<int> qNumbers(0);
            for(int i = 0 ; i < radialPoints.size() ; i++)
            {
              if(fieldName == "orbital")
              {
                simdjson::dom::array arr = doc["eigVecsQuad"].at(i).get_array();
                radialValues.push_back(double(arr.at(0).get_array().at(nlPairs[j].second).get_array().at(nlPairs[j].first)));
              }
              else if(fieldName == "vhartree")
              {
                radialValues.push_back(double(doc["vhartreeQuad"].at(i)));
              }
              else if(fieldName == "density")
              {
                radialValues.push_back(double(doc["rhoQuad"].at(0).get_array().at(i)));
              }
              else
              {
                utils::throwException(false,
                                      "Incorrect fieldname given.");
              }                
            }

            qNumbers = {nlPairs[j].first , nlPairs[j].second , m};

            double cutoff     = 0;
            double smoothness = 0;

            sphericalDataVec.push_back(
              std::make_shared<SphericalDataNumerical>(qNumbers,
                                                       radialPoints,
                                                       radialValues,
                                                       cutoff,
                                                       smoothness,
                                                       sphericalHarmonicFunc));                                  
          }
        }
      }

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

    AtomSphericalDataEnrichmentJSON::AtomSphericalDataEnrichmentJSON(
      const std::string                 fileName,            
      const std::vector<std::string> &  fieldNames,
      const std::vector<std::string> &  metadataNames,
      const SphericalHarmonicFunctions &sphericalHarmonicFunc)
      : d_fileName(fileName)
      , d_fieldNames(fieldNames)
      , d_metadataNames(metadataNames)
    {
        auto json_res = simdjson::padded_string::load(d_fileName);
        if (json_res.error()) {
            utils::throwException(false ,  "Error loading JSON: " + json_res.error());
        }
        simdjson::padded_string json = std::move(json_res).value();

        simdjson::dom::parser parser;
        simdjson::dom::element doc = parser.parse(json);

      double nspin = doc["nspin"].get_double();
      
      utils::throwException(nspin == 1, "The number of spin channels in the JSON data should be one.");

      std::string_view type;
      //
      // storing meta data
      //
      for (size_type iMeta = 0; iMeta < metadataNames.size(); ++iMeta)
      {
        const std::string &metadataName = metadataNames[iMeta];
        simdjson::simdjson_result<simdjson::dom::element>  typeRes;

        if(metadataName == "symbol" || metadataName == "Z")
            typeRes = doc["vext"][metadataName]; 
        else if(metadataName != "NR")
            typeRes = doc[metadataName];
        else
          continue;

        // Check the error code:
        if (typeRes.error()) {
            utils::throwException(false, metadataName + " metadataName type not found.");
        }

        simdjson::dom::element elem = typeRes.value();

        std::string value;
          switch (elem.type()) {
          case simdjson::dom::element_type::INT64:
              value = std::to_string(int64_t(elem));
              break;
          case simdjson::dom::element_type::UINT64:
              value = std::to_string(uint64_t(elem));
              break;
          case simdjson::dom::element_type::DOUBLE:
              value = std::to_string(double(elem));
              break;
          case simdjson::dom::element_type::STRING:
              value = std::string(std::string_view(elem));
              break;
          default:
              value = "";
              break;
          }
        d_metadata[metadataName] = value;
      }

      d_occupancies.clear();
      d_eigenValues.clear();

      auto it = std::find(fieldNames.begin(), fieldNames.end(), "orbital");
      if (it != fieldNames.end()) 
      {
        simdjson::simdjson_result<simdjson::dom::element>  typeRes;
         typeRes = doc["eigVals"]; 
        // Check the error code:
        if (typeRes.error()) {
            utils::throwException(false, "eigVals type not found.");}
        simdjson::dom::array rows = typeRes.at(0); // the inner 2D array

        for (simdjson::dom::element row : rows) 
        {
            std::vector<double> values;
            for (simdjson::dom::element val : row.get_array()) 
            {
                values.push_back(double(val));
            }
            d_eigenValues.push_back(std::move(values));
        }
        
        utils::throwException(!d_eigenValues.empty(),
                              "Not found eigVals in JSON file.");

         typeRes = doc["occupancies"]; 
        // Check the error code:
        if (typeRes.error()) {
            utils::throwException(false, "occupancies type not found.");}

        rows = doc["occupancies"].at(0); // the inner 2D array

        for (simdjson::dom::element row : rows) 
        {
            std::vector<double> values;
            for (simdjson::dom::element val : row.get_array()) 
            {
                values.push_back(double(val));
            }
            d_occupancies.push_back(std::move(values));
        }
        utils::throwException(!d_occupancies.empty(),
                              "Not found occupancies in JSON file.");
      }

      // ---------------------This class constructor is only designed for
      // SphericalDataNumerical for now , hence radial grid hard-coded------------------------

      std::vector<double>       radialPoints(0);
      for (double v : doc["rQPts"])
        {
          radialPoints.push_back(v);
        }
      if (radialPoints.empty())
        {
          utils::throwException(false,
                                "Not found rQpts grid in JSON file.");
        }

      d_metadata["NR"] = radialPoints.size();

      //
      // store field spherical data
      //
      for (size_type iField = 0; iField < fieldNames.size(); ++iField)
        {
          const std::string fieldName = fieldNames[iField];
          std::vector<std::shared_ptr<SphericalData>> sphericalDataVec(0);
          std::map<std::vector<int>, size_type>       qNumbersToIdMap;

          std::vector<std::pair<int, int>> nlPairs(0);
          size_type numUnoccupiedOrbitalsTaken = 0;
          if(fieldName == "orbital")
          {
            double d_homoEigenVal = -1e6;
            for(int l = 0 ; l < d_occupancies.size() ; l++)
            {
              for(int n = 0 ; n < d_occupancies[l].size() ; n++)
              {
                if(d_occupancies[l][n] > 1e-3)
                {
                  nlPairs.push_back({n , l});
                  if(d_homoEigenVal < d_eigenValues[l][n])
                  {
                    d_homoEigenVal = d_eigenValues[l][n];
                  }
                }
              }
            }
            if(numUnoccupiedOrbitalsTaken > 0)
            {
              std::vector<double> unOccEig(0);
              std::vector<std::pair<int , int>> nlPairUnOcc(0); 
              int count = 0;
              for(int l = 0 ; l < d_eigenValues.size() ; l++)
              {
                for(int n = 0 ; n < d_eigenValues[l].size() ; n++)
                {
                  if(d_eigenValues[l][n] > d_homoEigenVal && d_eigenValues[l][n] < 0 && count < numUnoccupiedOrbitalsTaken)
                  {
                    nlPairUnOcc.push_back({n , l});
                    unOccEig.push_back(d_eigenValues[l][n]);
                    count += 1;
                  }
                }
              }
              std::vector<size_type> idx(unOccEig.size());
              for (size_type i = 0; i < idx.size(); ++i) idx[i] = i;
              std::sort(idx.begin(), idx.end(),
                        [&](size_type i1, size_type i2) {
                            return unOccEig[i1] < unOccEig[i2];
                        });
              std::vector<double> values_sorted(unOccEig.size());
              std::vector<std::pair<int , int>> other_sorted(nlPairUnOcc.size());
              for (size_type i = 0; i < idx.size(); ++i) 
              {
                  values_sorted[i] = unOccEig[idx[i]];
                  other_sorted[i] = nlPairUnOcc[idx[i]];
              }
              unOccEig = std::move(values_sorted);
              nlPairUnOcc = std::move(other_sorted);
              nlPairs.insert(nlPairs.end(), nlPairUnOcc.begin(), nlPairUnOcc.end());
            }

            for(auto &i : nlPairs)
            {
              std::cout << i.first << "," << i.second << "\n";
            }
          }
          else
          {
            nlPairs.push_back({0 , 0});
          }
          getSphericalDataFromJSON(sphericalDataVec,
                                  radialPoints,
                                  fieldName,
                                  fileName,
                                  nlPairs,
                                  sphericalHarmonicFunc);
          storeQNumbersToDataIdMap(sphericalDataVec, qNumbersToIdMap);
          d_sphericalData[fieldName]   = sphericalDataVec;
          d_qNumbersToIdMap[fieldName] = qNumbersToIdMap;
        }
    }

    void
    AtomSphericalDataEnrichmentJSON::addFieldName(const std::string fieldName)
    {
      utils::throwException(
        false,
        "addFieldName() not yet implemented in AtomSphericalDataEnrichmentJSON class.");
    }

    std::string
    AtomSphericalDataEnrichmentJSON::getFileName() const
    {
      return d_fileName;
    }

    std::vector<std::string>
    AtomSphericalDataEnrichmentJSON::getFieldNames() const
    {
      return d_fieldNames;
    }

    std::vector<std::string>
    AtomSphericalDataEnrichmentJSON::getMetadataNames() const
    {
      return d_metadataNames;
    }

    const std::vector<std::shared_ptr<SphericalData>> &
    AtomSphericalDataEnrichmentJSON::getSphericalData(
      const std::string fieldName) const
    {
      auto it = d_sphericalData.find(fieldName);
      DFTEFE_AssertWithMsg(it != d_sphericalData.end(),
                           ("FieldName " + fieldName +
                            " not while parsing the JSON file:" + d_fileName)
                             .c_str());
      return it->second;
    }


    const std::shared_ptr<SphericalData>
    AtomSphericalDataEnrichmentJSON::getSphericalData(
      const std::string       fieldName,
      const std::vector<int> &qNumbers) const
    {
      auto it = d_sphericalData.find(fieldName);
      DFTEFE_AssertWithMsg(it != d_sphericalData.end(),
                           ("Unable to find the field " + fieldName +
                            " while parsing the JSON file " + d_fileName)
                             .c_str());
      auto iter = d_qNumbersToIdMap.find(fieldName);
      DFTEFE_AssertWithMsg(iter != d_qNumbersToIdMap.end(),
                           ("Unable to find the field " + fieldName +
                            " while parsing the JSON file " + d_fileName)
                             .c_str());
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
                                " while parsing the JSON file " + d_fileName)
                                 .c_str());
          return *((it->second).begin() + iterQNumberToId->second);
        }
    }

    std::string
    AtomSphericalDataEnrichmentJSON::getMetadata(
      const std::string metadataName) const
    {
      auto it = d_metadata.find(metadataName);
      utils::throwException(it != d_metadata.end(),
                            "Unable to find the metadata " + metadataName +
                              " while parsing the JSON file " + d_fileName);

      return it->second;
    }


    size_type
    AtomSphericalDataEnrichmentJSON::getQNumberID(
      const std::string       fieldName,
      const std::vector<int> &qNumbers) const
    {
      auto it = d_qNumbersToIdMap.find(fieldName);
      utils::throwException<utils::InvalidArgument>(
        it != d_qNumbersToIdMap.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataEnrichmentJSON::getQNumberID");
      auto it1 = (it->second).find(qNumbers);
      utils::throwException<utils::InvalidArgument>(
        it1 != (it->second).end(),
        "Cannot find the qnumbers provided to AtomSphericalDataEnrichmentJSON::getQNumberID");
      return (it1)->second;
    }

    size_type
    AtomSphericalDataEnrichmentJSON::nSphericalData(std::string fieldName) const
    {
      auto it = d_sphericalData.find(fieldName);
      utils::throwException<utils::InvalidArgument>(
        it != d_sphericalData.end(),
        "Cannot find the atom symbol provided to AtomSphericalDataEnrichmentJSON::nSphericalData");
      return (it->second).size();
    }

  } // end of namespace atoms
} // end of namespace dftefe
