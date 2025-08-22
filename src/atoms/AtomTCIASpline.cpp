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
namespace dftefe
{
  namespace atoms
  {
    void AtomTCIASpline::touch(const std::string &key) 
    {
      auto it = d_keyToIter.find(key);
      if (it != d_keyToIter.end()) {
          d_lruList.erase(it->second);
      }
      d_lruList.push_front(key);
      d_keyToIter[key] = d_lruList.begin();
    }

    void AtomTCIASpline::evictIfNeeded() 
    {
      if (d_cache.size() > d_maxSize) 
      {
        std::string victim = d_lruList.back();
        d_lruList.pop_back();
        d_keyToIter.erase(victim);
        d_cache.erase(victim);
      }
    }

    bool AtomTCIASpline::loadAtomPair(const std::string &atomPair,
                      std::vector<double> &values) 
    {
      simdjson::ondemand::parser parser;
      simdjson::padded_string json = simdjson::padded_string::load(d_jsonFile);
      simdjson::ondemand::document doc = parser.iterate(json);

      simdjson::ondemand::object fieldObj = doc[d_fieldName];
      simdjson::ondemand::object subObj   = fieldObj[d_SorSprime];
      simdjson::ondemand::array arr       = subObj[atomPair];

      values.clear();
      for (double v : arr) values.push_back(v);
      return !values.empty();
    }

    AtomTCIASpline::AtomTCIASpline(const std::string &fieldName,
                    const std::string  &tciaDataFileName,
                    const int derivativeOrder,
                    const size_type maxPairs)
      : d_rgrid(0),
        d_maxSize(maxPairs), d_jsonFile(tciaDataFileName),
        d_fieldName(fieldName), 
        d_SorSprime(derivativeOrder == 0 ? "S" : "Sprime")
    {
      if(derivativeOrder > 1)
      {
        utils::throwException(false , "Derivative can be just be 0 or 1.");
      }
      // Load rgrid once (top-level "d")
      simdjson::ondemand::parser parser;
      simdjson::padded_string json = simdjson::padded_string::load(d_jsonFile);
      simdjson::ondemand::document doc = parser.iterate(json);

      // Check fieldName exists
      simdjson::ondemand::object fieldObj;
      auto fieldRes = doc[fieldName].get(fieldObj);
      if (fieldRes) 
      {
        utils::throwException(false , "FieldName '" + fieldName + "' not found in JSON");
      }

      // Check which exists inside fieldName
      simdjson::ondemand::object whichObj;
      auto whichRes = fieldObj[d_SorSprime].get(whichObj);
      if (whichRes) 
      {
        utils::throwException(false , d_SorSprime + "' not found under field '" + fieldName + "'");
      }

      for (double v : doc["d"].get_array()) 
      {
        d_rgrid.push_back(v);
      }
    }

    utils::Spline* AtomTCIASpline::getSpline(const std::string &atomPair) 
    {
      auto it = d_cache.find(atomPair);
      if (it != d_cache.end()) 
      {
        touch(atomPair);
        return it->second.get();
      }
      std::vector<double> values;
      if (!loadAtomPair(atomPair, values)) 
      {
        utils::throwException(false , "Not found: " + d_fieldName + "/" + d_SorSprime + "/" + atomPair);
      }

      d_cache[atomPair] = std::make_unique<utils::Spline>(d_rgrid, values);
      touch(atomPair);
      evictIfNeeded();

      return d_cache[atomPair].get();
    }

    double
    AtomTCIASpline::maxRadialGrid()
    {
      return d_rgrid.back();
    }

  } // end of namespace atoms
} // end of namespace dftefe
