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
#include <utils/Exceptions.h>
#include "AtomTCIASpline.h"
#include "simdjson.h"
#include <unordered_set>
namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      void
      printObject(simdjson::ondemand::object obj)
      {
        std::cout << "Fields in object:\n";
        for (auto field : obj)
          {
            std::string_view          key = field.unescaped_key();
            simdjson::ondemand::value val = field.value();

            std::cout << "  " << key << ": ";

            // Print based on type
            auto t = val.type().value();
            switch (t)
              {
                case simdjson::ondemand::json_type::string:
                  std::cout << val.get_string().value();
                  break;
                case simdjson::ondemand::json_type::number:
                  std::cout << double(val); // auto converts int/float
                  break;
                case simdjson::ondemand::json_type::boolean:
                  std::cout << (bool(val) ? "true" : "false");
                  break;
                case simdjson::ondemand::json_type::array:
                  std::cout << "[array]";
                  break;
                case simdjson::ondemand::json_type::object:
                  std::cout << "{object}";
                  break;
                case simdjson::ondemand::json_type::null:
                  std::cout << "null";
                  break;
              }
            std::cout << "\n";
          }
      }
      double
      linf_error(const std::vector<double> &a, const std::vector<double> &b)
      {
        if (a.size() != b.size())
          {
            throw std::runtime_error("Vectors must have the same size!");
          }

        double max_err = 0.0;
        for (size_t i = 0; i < a.size(); i++)
          {
            max_err = std::max(max_err, std::fabs(a[i] - b[i]));
          }
        return max_err;
      }
    } // namespace
    void
    AtomTCIASpline::touch(const std::string &key)
    {
      auto it = d_keyToIter.find(key);
      if (it != d_keyToIter.end())
        {
          d_lruList.erase(it->second);
        }
      d_lruList.push_front(key);
      d_keyToIter[key] = d_lruList.begin();
    }

    void
    AtomTCIASpline::evictIfNeeded()
    {
      if (d_cache.size() > d_maxSize)
        {
          std::string victim = d_lruList.back();
          d_lruList.pop_back();
          d_keyToIter.erase(victim);
          d_cache.erase(victim);
        }
    }

    bool
    AtomTCIASpline::loadAtomCombination(
      const std::string &               atomCombination,
      std::vector<std::vector<double>> &values)
    {
      simdjson::ondemand::parser parser;
      std::string                jsonfile = d_tciaparams.folderName + "/" +
                             d_tciaparams.outFilePrefix + "." +
                             atomCombination + ".json";
      simdjson::padded_string json = simdjson::padded_string::load(jsonfile);
      simdjson::ondemand::document doc = parser.iterate(json);

      simdjson::ondemand::object fieldObj = doc[d_fieldName];
      values.clear();
      bool tciTypeFound = false;
      for (auto &i : d_tciTypes)
        {
          std::vector<double>       valuesId(0);
          simdjson::ondemand::array arr = fieldObj[i];
          for (double v : arr)
            {
              valuesId.push_back(v);
            }
          if (!valuesId.empty())
            {
              tciTypeFound = true;
            }
          else
            {
              tciTypeFound = false;
              break;
            }
          values.push_back(valuesId);
        }
      return tciTypeFound;
    }

    AtomTCIASpline::AtomTCIASpline(const std::string &            fieldName,
                                   const TCIADataParams &         params,
                                   const std::vector<std::string> atomSymbols,
                                   const std::vector<std::string> tciTypes,
                                   const size_type                maxPairs)
      : d_dgrid(0)
      , d_maxSize(maxPairs)
      , d_tciaparams(params)
      , d_fieldName(fieldName)
      , d_tciTypes(tciTypes)
    {
      // Create an unordered_set from the vector, which automatically handles uniqueness
      std::unordered_set<std::string> uniqueSymbolsSet(atomSymbols.begin(), atomSymbols.end());
      
      // Create a new vector from the set
      std::vector<std::string> uniqueAtomsymbols(uniqueSymbolsSet.begin(), uniqueSymbolsSet.end());
      
      for (int i = 0; i < d_tciTypes.size(); i++)
        {
          if (d_tciTypes[i] == "S")
            d_tciTypeToIndex["S"] = i;
          else if (d_tciTypes[i] == "Sprime")
            d_tciTypeToIndex["Sprime"] = i;
          else
            utils::throwException(false,
                                  "tciTypes can be just be S or Sprime.");
        }
      if (!(fieldName == "rhoAtom-vlocCorrection" ||
            fieldName == "rhoAtom-phiAtom" || fieldName == "bSmear-phiAtom" ||
            fieldName == "sumBZZCorrBSmear-diffVZZCorrVSmear"))
        {
          utils::throwException(false,
                                "FieldName '" + fieldName +
                                  "' not found in JSON");
        }
      // Load rgrid once (top-level "d")
      simdjson::ondemand::parser parser;
      std::vector<std::string>   atomComb(0);
      for (int i = 0; i < uniqueAtomsymbols.size(); i++)
        {
          for (int j = 0; j < uniqueAtomsymbols.size(); j++)
            {
              if (fieldName == "rhoAtom-vlocCorrection" ||
                  fieldName == "rhoAtom-phiAtom")
                {
                  atomComb.push_back(
                    std::string(uniqueAtomsymbols[i] + "-" + uniqueAtomsymbols[j]));
                }
            }
          if (fieldName == "bSmear-phiAtom")
            {
              atomComb.push_back(std::string(uniqueAtomsymbols[i]));
            }
        }
        if (fieldName == "sumBZZCorrBSmear-diffVZZCorrVSmear")
          {
            atomComb.push_back(std::string("DefaultAtom"));
          }

      std::vector<double> dGridTmp(0);

      for (auto &comb : atomComb)
        {
          std::string jsonfile = d_tciaparams.folderName + "/" +
                                 d_tciaparams.outFilePrefix + "." + comb +
                                 ".json";
          auto jsonRes = simdjson::padded_string::load(jsonfile);

          if (jsonRes.error()) 
          {
            utils::throwException(false, "Failed to load JSON file: " + jsonfile + 
                                          " (" + simdjson::error_message(jsonRes.error()) + ") from the input JSON data folder.");
          }
          simdjson::ondemand::document doc = parser.iterate(jsonRes);

          if(fieldName == "bSmear-phiAtom")
          {
            std::string_view type;
            auto typeRes = doc["vlocInfo"]["type"].get(type);
            if (typeRes) 
            {
              utils::throwException(false, "vlocInfo.type not found");
            }
            if (type == "psp") 
            {
              std::string_view z_str , tot_pspen , psp_type;

              auto p1 = doc["vlocInfo"]["params"][comb]["z_valence"].get(z_str);
              auto p2 = doc["vlocInfo"]["params"][comb]["total_psenergy"].get(tot_pspen);
              auto p3 = doc["vlocInfo"]["params"][comb]["pseudo_type"].get(psp_type);

              if (p1) 
              {
                utils::throwException(false, "z_valence not found");
              }
              if (p2) 
              {
                utils::throwException(false, "total_psenergy not found");
              }
              if (p3) 
              {
                utils::throwException(false, "pseudo_type not found");
              }

              // convert string to int
              d_vLocParams[comb]["z_valence"] = std::string(z_str);
              d_vLocParams[comb]["total_psenergy"] = std::string(tot_pspen);
              d_vLocParams[comb]["pseudo_type"] = std::string(psp_type);
            }
          }

          // Check fieldName exists
          simdjson::ondemand::object fieldObj;
          auto                       fieldRes = doc[fieldName].get(fieldObj);
          if (fieldRes)
            {
              utils::throwException(false,
                                    "FieldName '" + fieldName +
                                      "' not found in JSON");
            }

          // Check which exists inside fieldName
          simdjson::ondemand::array whichArr;
          for (auto &i : d_tciTypes)
            {
              auto whichRes = fieldObj[i].get(whichArr);
              if (whichRes)
                {
                  utils::throwException(false,
                                        i + "' not found under field '" +
                                          fieldName + "'");
                }
            }

          d_dgrid.clear();

          for (double v : doc["d"].get_array())
            {
              d_dgrid.push_back(v);
            }

          dGridTmp = d_dgrid;

          if (linf_error(dGridTmp, d_dgrid) > 1e-12)
            {
              utils::throwException(
                false,
                "d Grid is different for different files in the folder.");
            }

          simdjson::simdjson_result<double> fieldRes1 = doc["rcSmear"].get_double();

          if (fieldRes1.error()) 
          {
            utils::throwException(false, "rcSmear not found in JSON file.");
          } 
          else 
          {
            d_rcSmear = fieldRes1.value();
          }

          fieldRes1 = doc["rcZZCorr"].get_double();

          if (fieldRes1.error()) 
          {
            d_rcSmearZZCorr = d_rcSmear;
          } 
          else 
          {
            d_rcSmearZZCorr = fieldRes1.value();
          }

        }
    }

    utils::Spline *
    AtomTCIASpline::getSpline(const std::string &atomCombination,
                              const std::string  tciType)
    {
      // Using std::find to get an iterator
      auto iter = std::find(d_tciTypes.begin(), d_tciTypes.end(), tciType);
      if (iter != d_tciTypes.end())
        int index = std::distance(d_tciTypes.begin(), iter);
      else
        utils::throwException(false,
                              "TCI Type '" + tciType + "' not found in JSON");
      auto it = d_cache.find(atomCombination);
      if (it != d_cache.end())
        {
          touch(atomCombination);
          return it->second[d_tciTypeToIndex[tciType]].get();
        }
      std::vector<std::vector<double>> values(0);
      if (!loadAtomCombination(atomCombination, values))
        {
          utils::throwException(false,
                                "Not found tcia data for the atom " +
                                  atomCombination);
        }

      d_cache[atomCombination].clear();
      for (int i = 0; i < d_tciTypes.size(); i++)
        {
          d_cache[atomCombination].push_back(std::make_unique<utils::Spline>(
            d_dgrid,
            values[i],
            false,
            utils::Spline::spline_type::cspline,
            false,
            utils::Spline::bd_type::second_deriv,
            0.0,
            utils::Spline::bd_type::second_deriv,
            0.0));
        }
      touch(atomCombination);
      evictIfNeeded();

      return d_cache[atomCombination][d_tciTypeToIndex[tciType]].get();
    }

    double
    AtomTCIASpline::maxRadialGrid()
    {
      return d_dgrid.back();
    }

    double
    AtomTCIASpline::smearedChargeRadius()
    {
      return d_rcSmear;
    }

    double
    AtomTCIASpline::smearedChargeRadiusZZCorr()
    {
      return d_rcSmearZZCorr;
    }

    std::string
    AtomTCIASpline::getVLocInfo(std::string atomSymbol, std::string paramName)
    {
      if(d_fieldName == "bSmear-phiAtom")
      {
        auto iter = d_vLocParams.find(atomSymbol);
        if (iter != d_vLocParams.end())
        {
          auto iter1 = iter->second.find(paramName);
          if (iter1 != iter->second.end())
          {
            return iter1->second;
          }
          else
          {
            utils::throwException(false,
                                  "paramName '" + paramName + "' not found in JSON folder files.");
            return "";
          }
        }
        else
          utils::throwException(false,
                                "atomSymbol Type '" + atomSymbol + "' not found in JSON folder.");
        return "";
      }
      else
      {
        utils::throwException(false,
                              "getVLocInfo only avalibale for bSmear-phiAtom fieldname.");
        return "";
      }
    }

  } // end of namespace atoms
} // end of namespace dftefe
