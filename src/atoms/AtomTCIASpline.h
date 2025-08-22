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

#ifndef dftefeAtomTCIASpline_h
#define dftefeAtomTCIASpline_h

#include <string>
#include <unordered_map>
#include <utils/Spline.h>
#include <unordered_map>
#include <list>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <utils/TypeConfig.h>
namespace dftefe
{
  namespace atoms
  {
    class AtomTCIASpline
    {
    public:
      AtomTCIASpline(const std::string &fieldName,
                    const std::string  &tciaDataFileName,
                    const int derivativeOrder = 0,
                    const size_type maxPairs = 1000);

      /**
       * @brief Destructor
       */
      ~AtomTCIASpline() = default;

      utils::Spline*
      getSpline(const std::string &atomPair);

      double
      maxRadialGrid();

    private:
      bool loadAtomPair(const std::string &atomPair,
                        std::vector<double> &values);
      void evictIfNeeded();
      void touch(const std::string &key);

      std::unordered_map<std::string, std::unique_ptr<utils::Spline>> d_cache;
      std::string d_jsonFile;
      std::vector<double> d_rgrid;
      size_type d_maxSize;
      std::string d_SorSprime , d_fieldName;

    // LRU bookkeeping
    std::list<std::string> d_lruList; // front = most recent, back = least
    std::unordered_map<std::string, std::list<std::string>::iterator> d_keyToIter;

    }; // end of class AtomTCIASpline
  }    // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeAtomTCIASpline_h
