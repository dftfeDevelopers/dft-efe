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
#include <utils/SmearChargePotentialFunction.h>
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
      int
      findLastExtremumIndex(const std::vector<double> &f)
      {
        if (f.size() < 3)
          utils::throwException(false, "Vector f must have at least length 3.");

        const double eps =
          10 * std::numeric_limits<double>::epsilon(); // robust tolerance
        int lastIndex = -1;

        for (size_t i = 1; i + 1 < f.size(); ++i)
          {
            double slope1 = f[i] - f[i - 1];
            double slope2 = f[i + 1] - f[i];

            // Normalize tolerance relative to local magnitudes
            double scale = std::max(
              {std::fabs(f[i - 1]), std::fabs(f[i]), std::fabs(f[i + 1]), 1.0});
            double tol = eps * scale;

            // Skip near-zero slopes — treat them as flat
            if (std::fabs(slope1) < tol || std::fabs(slope2) < tol)
              continue;

            // Detect robust sign change (avoid false flips near zero)
            if (slope1 * slope2 < -tol * tol)
              lastIndex = static_cast<int>(i);
          }

        if (lastIndex == -1)
          lastIndex = static_cast<int>(f.size() - 1);

        return lastIndex;
      }

      void
      derivativef(const std::vector<double> &x,
                  const std::vector<double> &f,
                  std::vector<double> &      res)
      {
        res.clear();
        for (size_type j = 0; j < f.size() - 1; j++)
          {
            res.push_back((f[j + 1] - f[j]) / (x[j + 1] - x[j]));
          }
        res.push_back(res[f.size() - 2]);
      }

      void
      integralxSqfSq(const std::vector<double> &x,
                     const std::vector<double> &f,
                     std::vector<double> &      res)
      {
        double sum = 0;
        res.clear();
        res.push_back(0);
        for (size_type j = 0; j < f.size() - 1; j++)
          {
            sum += 0.5 *
                   (f[j + 1] * f[j + 1] * x[j + 1] * x[j + 1] +
                    f[j] * f[j] * x[j] * x[j]) *
                   (x[j + 1] - x[j]);
            res.push_back(sum);
          }
      }

      void
      createSplineFromSphericalData(
        std::vector<std::shared_ptr<SphericalData>> &sphericalDataVec,
        const std::vector<double> &                  radialPoints,
        std::vector<std::vector<double>> &           radialValuesVec,
        std::vector<std::vector<int>> &              qNumVec,
        std::vector<std::pair<double, double>> &     cutOffInfoVec,
        const SphericalHarmonicFunctions &           sphericalHarmonicFunc)
      {
        for (int i = 0; i < radialValuesVec.size(); i++)
          {
            double cutoff     = cutOffInfoVec[i].first;
            double smoothness = cutOffInfoVec[i].second;

            sphericalDataVec.push_back(
              std::make_shared<SphericalDataNumerical>(qNumVec[i],
                                                       radialPoints,
                                                       radialValuesVec[i],
                                                       cutoff,
                                                       smoothness,
                                                       sphericalHarmonicFunc));
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
      const std::string                        fileName,
      const std::vector<std::string> &         fieldNames,
      const std::vector<std::string> &         metadataNames,
      const SphericalHarmonicFunctions &       sphericalHarmonicFunc,
      const std::map<std::string, std::string> additionalParams)
      : d_fileName(fileName)
      , d_fieldNames(fieldNames)
      , d_metadataNames(metadataNames)
    {
      auto json_res = simdjson::padded_string::load(d_fileName);
      if (json_res.error())
        {
          std::string err_msg = simdjson::error_message(json_res.error());
          utils::throwException(false, "Error loading JSON: " + err_msg);
        }
      simdjson::padded_string json = std::move(json_res).value();

      simdjson::dom::parser  parser;
      simdjson::dom::element doc = parser.parse(json);

      double nspin = doc["nspin"].get_double();

      utils::throwException(
        nspin == 1,
        "The number of spin channels in the JSON data should be one.");

      if (std::find(d_metadataNames.begin(), d_metadataNames.end(), "Z") !=
          d_metadataNames.end())
        {
          d_metadataNames.push_back("Z");
        }

      std::string_view type;
      //
      // storing meta data
      //
      for (size_type iMeta = 0; iMeta < d_metadataNames.size(); ++iMeta)
        {
          const std::string &metadataName = d_metadataNames[iMeta];
          simdjson::simdjson_result<simdjson::dom::element> typeRes;

          if (metadataName == "symbol" || metadataName == "Z")
            typeRes = doc["vext"][metadataName];
          else if (metadataName != "NR")
            typeRes = doc[metadataName];
          else
            continue;

          // Check the error code:
          if (typeRes.error())
            {
              utils::throwException(false,
                                    metadataName +
                                      " metadataName type not found for " +
                                      fileName + ".");
            }

          simdjson::dom::element elem = typeRes.value();

          std::string value;
          switch (elem.type())
            {
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

      if (std::find(fieldNames.begin(), fieldNames.end(), "vtotal") !=
          fieldNames.end())
        {
          d_atomCharge = std::stod(d_metadata["Z"]);
          auto iter    = additionalParams.find("rcsmear");
          if (iter != additionalParams.end())
            {
              d_smearedCharge = std::stod(iter->second);
            }
          else
            {
              utils::throwException(
                false,
                "rcsmear not found in additionalParams. Required for vtotal field");
            }
        }

      d_occupancies.clear();
      d_eigenValues.clear();

      if (std::find(fieldNames.begin(), fieldNames.end(), "orbital") !=
          fieldNames.end())
        {
          auto iter = additionalParams.find("PSP/AE");
          if (iter != additionalParams.end())
            {
              d_PSPorAE = iter->second;
            }
          else
            {
              utils::throwException(
                false,
                "PSP/AE not found in additionalParams. Required for orbital field");
            }
          simdjson::simdjson_result<simdjson::dom::element> typeRes;
          typeRes = doc["eigVals"];
          // Check the error code:
          if (typeRes.error())
            {
              utils::throwException(false, "eigVals type not found.");
            }
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
          if (typeRes.error())
            {
              utils::throwException(false, "occupancies type not found.");
            }

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
      // SphericalDataNumerical for now , hence radial grid
      // hard-coded------------------------

      std::vector<double> radialPoints(0);
      for (double v : doc["rQPts"])
        {
          radialPoints.push_back(v);
        }
      if (radialPoints.empty())
        {
          utils::throwException(false, "Not found rQpts grid in JSON file.");
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
          size_type                        numUnoccupiedOrbitalsTaken = 0;
          if (fieldName == "orbital")
            {
              double d_homoEigenVal = -1e6;
              for (int l = 0; l < d_occupancies.size(); l++)
                {
                  for (int n = 0; n < d_occupancies[l].size(); n++)
                    {
                      if (d_occupancies[l][n] >
                          1e-3) // Change here to control number of eid
                        {
                          nlPairs.push_back({n, l});
                          if (d_homoEigenVal < d_eigenValues[l][n])
                            {
                              d_homoEigenVal = d_eigenValues[l][n];
                            }
                        }
                    }
                }
              if (numUnoccupiedOrbitalsTaken > 0)
                {
                  std::vector<double>              unOccEig(0);
                  std::vector<std::pair<int, int>> nlPairUnOcc(0);
                  int                              count = 0;
                  for (int l = 0; l < d_eigenValues.size(); l++)
                    {
                      for (int n = 0; n < d_eigenValues[l].size(); n++)
                        {
                          if (d_eigenValues[l][n] > d_homoEigenVal &&
                              d_eigenValues[l][n] < 0 &&
                              count < numUnoccupiedOrbitalsTaken)
                            {
                              nlPairUnOcc.push_back({n, l});
                              unOccEig.push_back(d_eigenValues[l][n]);
                              count += 1;
                            }
                        }
                    }
                  std::vector<size_type> idx(unOccEig.size());
                  for (size_type i = 0; i < idx.size(); ++i)
                    idx[i] = i;
                  std::sort(idx.begin(),
                            idx.end(),
                            [&](size_type i1, size_type i2) {
                              return unOccEig[i1] < unOccEig[i2];
                            });
                  std::vector<double> values_sorted(unOccEig.size());
                  std::vector<std::pair<int, int>> other_sorted(
                    nlPairUnOcc.size());
                  for (size_type i = 0; i < idx.size(); ++i)
                    {
                      values_sorted[i] = unOccEig[idx[i]];
                      other_sorted[i]  = nlPairUnOcc[idx[i]];
                    }
                  unOccEig    = std::move(values_sorted);
                  nlPairUnOcc = std::move(other_sorted);
                  nlPairs.insert(nlPairs.end(),
                                 nlPairUnOcc.begin(),
                                 nlPairUnOcc.end());
                }
            }
          else
            {
              nlPairs.push_back({0, 0});
            }
          std::vector<std::vector<double>>       radialValuesVec;
          std::vector<std::vector<int>>          qNumVec;
          std::vector<std::pair<double, double>> cutOffInfoVec;
          getSphericalDataFromJSON(radialValuesVec,
                                   qNumVec,
                                   radialPoints,
                                   fieldName,
                                   fileName,
                                   nlPairs);
          getCutoffs(cutOffInfoVec,
                     radialValuesVec,
                     qNumVec,
                     radialPoints,
                     fieldName,
                     fileName,
                     nlPairs);
          createSplineFromSphericalData(sphericalDataVec,
                                        radialPoints,
                                        radialValuesVec,
                                        qNumVec,
                                        cutOffInfoVec,
                                        sphericalHarmonicFunc);
          storeQNumbersToDataIdMap(sphericalDataVec, qNumbersToIdMap);
          d_sphericalData[fieldName]   = sphericalDataVec;
          d_qNumbersToIdMap[fieldName] = qNumbersToIdMap;
        }
    }


    void
    AtomSphericalDataEnrichmentJSON::getSphericalDataFromJSON(
      std::vector<std::vector<double>>
        &radialValuesVec, // returns n,l,m pairs where m = -l to l
      std::vector<std::vector<int>> &   qNumVec,
      const std::vector<double> &       radialPoints,
      const std::string &               fieldName,
      const std::string &               fileName,
      std::vector<std::pair<int, int>> &nlPairs)
    {
      radialValuesVec.clear();
      auto        json_res = simdjson::padded_string::load(fileName);
      std::string err_msg  = simdjson::error_message(json_res.error());
      if (json_res.error())
        {
          utils::throwException(false, "Error loading JSON: " + err_msg);
        }
      simdjson::padded_string json = std::move(json_res).value();

      simdjson::dom::parser  parser;
      simdjson::dom::element doc = parser.parse(json);

      for (int j = 0; j < nlPairs.size(); j++)
        {
          for (int m = -nlPairs[j].second; m <= nlPairs[j].second; m++)
            {
              std::vector<double> radialValues(0);
              std::vector<int>    qNumbers(0);
              for (int i = 0; i < radialPoints.size(); i++)
                {
                  if (fieldName == "orbital")
                    {
                      simdjson::dom::array arr =
                        doc["eigVecsQuad"].at(i).get_array();
                      radialValues.push_back(double(arr.at(0)
                                                      .get_array()
                                                      .at(nlPairs[j].second)
                                                      .get_array()
                                                      .at(nlPairs[j].first)));
                    }
                  else if (fieldName == "vhartree" || fieldName == "vtotal")
                    {
                      radialValues.push_back(double(doc["vhartreeQuad"].at(i)));
                    }
                  else if (fieldName == "density")
                    {
                      radialValues.push_back(
                        double(doc["rhoQuad"].at(0).get_array().at(i)));
                    }
                  else
                    {
                      utils::throwException(false,
                                            "Incorrect fieldname given.");
                    }
                }
              if (fieldName == "vtotal")
                {
                  std::vector<double> nuclearChargePot(radialPoints.size());

                  const utils::SmearChargePotentialFunction smfuncPot(
                    {utils::Point({0, 0, 0})},
                    -1.0 * std::abs(d_atomCharge),
                    d_smearedCharge);

                  for (int i = 0; i < radialPoints.size(); i++)
                    {
                      radialValues[i] +=
                        smfuncPot(utils::Point({radialPoints[i], 0, 0}));
                    }
                }
              qNumbers = {nlPairs[j].first, nlPairs[j].second, m};
              radialValuesVec.push_back(radialValues);
              qNumVec.push_back(qNumbers);
            }
        }
    }

    void
    AtomSphericalDataEnrichmentJSON::getCutoffs(
      std::vector<std::pair<double, double>> &cutOffInfoVec,
      std::vector<std::vector<double>> &      radialValuesVec,
      std::vector<std::vector<int>> &         qNumVec,
      const std::vector<double> &             radialPoints,
      const std::string &                     fieldName,
      const std::string &                     fileName,
      std::vector<std::pair<int, int>> &      nlPairs)
    {
      cutOffInfoVec.clear();
      cutOffInfoVec.resize(qNumVec.size());
      if (fieldName == "vhartree")
        {
          for (int i = 0; i < qNumVec.size(); i++)
            {
              cutOffInfoVec[i] = {1e6, 1e6};
            }
        }
      if (fieldName == "vtotal" || fieldName == "density")
        {
          for (int i = 0; i < qNumVec.size(); i++)
            {
              for (int j = radialPoints.size() - 1; j > 0; j--)
                {
                  if (std::abs(radialValuesVec[i][j]) > 1e-10)
                    {
                      cutOffInfoVec[i].first  = radialPoints[j];
                      cutOffInfoVec[i].second = 1e6;
                      break;
                    }
                }
            }
        }
      if (fieldName == "orbital")
        {
          bool        useHeurestics = false;
          auto        json_res      = simdjson::padded_string::load(fileName);
          std::string err_msg       = simdjson::error_message(json_res.error());
          if (json_res.error())
            {
              utils::throwException(false, "Error loading JSON: " + err_msg);
            }
          simdjson::padded_string json = std::move(json_res).value();

          simdjson::dom::parser  parser;
          simdjson::dom::element doc = parser.parse(json);

          simdjson::simdjson_result<simdjson::dom::element> typeRes;
          typeRes = doc["eigVecsCutoff"];
          // Check the error code:
          if (typeRes.error())
            {
              useHeurestics = true;
            }
          if (!useHeurestics)
            {
              std::vector<std::vector<double>> eigVecCutoff;
              std::vector<std::vector<double>> eigVecSmoothness;
              simdjson::dom::array rows = typeRes.at(0); // the inner 2D array

              for (simdjson::dom::element row : rows)
                {
                  std::vector<double> values;
                  for (simdjson::dom::element val : row.get_array())
                    {
                      values.push_back(double(val));
                    }
                  eigVecCutoff.push_back(std::move(values));
                }

              typeRes = doc["eigVecsSmoothness"];
              rows    = typeRes.at(0); // the inner 2D array

              for (simdjson::dom::element row : rows)
                {
                  std::vector<double> values;
                  for (simdjson::dom::element val : row.get_array())
                    {
                      values.push_back(double(val));
                    }
                  eigVecSmoothness.push_back(std::move(values));
                }

              // use from file
              for (int i = 0; i < qNumVec.size(); i++)
                {
                  cutOffInfoVec[i].first =
                    eigVecCutoff[qNumVec[i][1]][qNumVec[i][0]];
                  cutOffInfoVec[i].second =
                    eigVecSmoothness[qNumVec[i][1]][qNumVec[i][0]];
                }
            }
          else
            {
              for (int i = 0; i < qNumVec.size(); i++)
                {
                  std::vector<double> h(radialPoints.size());
                  for (int j = 0; j < h.size(); j++)
                    {
                      h[j] = radialPoints[j] * radialPoints[j] *
                             radialValuesVec[i][j] * radialValuesVec[i][j];
                    }
                  int lastTurningPtId =
                    std::min(findLastExtremumIndex(radialValuesVec[i]),
                             findLastExtremumIndex(h));
                  std::vector<double> der(0), intgl(0);
                  derivativef(radialPoints, radialValuesVec[i], der);
                  integralxSqfSq(radialPoints, radialValuesVec[i], intgl);
                  int cutoffId  = lastTurningPtId;
                  int cutoffId1 = lastTurningPtId;
                  for (int j = radialPoints.size() - 2; j > lastTurningPtId;
                       j--)
                    {
                      if (std::abs(der[j]) > intgl[j] * 5.e-3)
                        {
                          cutoffId = j;
                          break;
                        }
                      if (1 - intgl[j] > 5.e-3)
                        {
                          cutoffId1 = j;
                          break;
                        }
                    }
                  // if (std::abs(der[findLastExtremumIndex(der)]) > 5e-1 &&
                  //     radialPoints[std::max(cutoffId, cutoffId1)] < 8)
                  //   {
                  //     for (int j = radialPoints.size() - 2; j >
                  //     lastTurningPtId;
                  //          j--)
                  //       {
                  //         if (std::abs(der[j]) > intgl[j] * 1.e-3)
                  //           {
                  //             cutoffId = j;
                  //             break;
                  //           }
                  //       }
                  //   }
                  if (d_PSPorAE == "PSP")
                    {
                      double smoothness = 1. / 3;
                      if (radialPoints[std::max(cutoffId, cutoffId1)] > 10.0)
                        {
                          cutOffInfoVec[i].first =
                            10 * (smoothness / (1 + smoothness));
                        }
                      else
                        {
                          cutOffInfoVec[i].first =
                            radialPoints[std::max(cutoffId, cutoffId1)] *
                            (smoothness / (1 + smoothness));
                        }
                      cutOffInfoVec[i].second = smoothness;
                    }
                  else if (d_PSPorAE == "AE")
                    {
                      double smoothness = 1.01;
                      if (radialPoints[std::max(cutoffId, cutoffId1)] > 10.0)
                        {
                          cutOffInfoVec[i].first =
                            10 * (smoothness / (1 + smoothness));
                        }
                      cutOffInfoVec[i].first =
                        radialPoints[std::max(cutoffId, cutoffId1)] *
                        (smoothness / (1 + smoothness));
                      cutOffInfoVec[i].second = smoothness;
                    }
                  else
                    {
                      utils::throwException(
                        false,
                        "Heuristic cutoffs only defined for PSP/AE type calculation.");
                    }
                }
            }
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
