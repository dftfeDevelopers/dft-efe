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

#include <utils/TypeConfig.h>
#include <map>
#include <string>
#include <atoms/AtomSphericalDataContainer.h>
#include <atoms/SphericalData.h>
#include <atoms/OverlappingAtomIds.h>
namespace dftefe
{
  namespace atoms
  {
    template <unsigned int dim>
    OverlappingAtomIds::OverlappingAtomIds(
      const std::map<std::string, utils::Point> &mapAtomSymbolToCoordinates,
      const AtomSphericalDataContainer &         AtomSphericalDataContainer,
      const std::string                          fieldName,
      const utils::Point &                       minbound,
      const utils::Point &                       maxbound)
      : d_mapAtomSymbolToCoordinates(mapAtomSymbolToCoordinates)
      , d_fieldname(fieldname)
      , d_minbound(minbound)
      , d_maxbound(maxbound)
    {}

    template <unsigned int dim>
    std::vector<size_type>
    getOverlappingAtomIdsWithinBox() const
    {
      size_type              Id = 0;
      std::vector<size_type> getAtomIds;
      std::vector<double>    cutoff;
      boolean                flag;

      for (auto it : d_mapAtomSymbolToCoordinates)
        {
          cutoff.resize(0, 0.);
          flag = false;
          for (auto i :
               AtomSphericalDataContainer.getSphericalData(it.first,
                                                           d_fieldName))
            cutoff.push_back(i.cutoff + i.cutoff / i.smoothness);
          double maxcutoff = std::max_element(cutoff.begin(), cutoff.end());
          for (unsigned int i = 0; i < dim; i++)
            if (d_maxbound[i] - it.second[i] >= maxcutoff &&
                it.second[i] - d_minbound[i] >= maxcutoff)
              flag = true;
          if (flag)
            std::vector<size_type> getAtomIds.push_back(Id);
          Id++;
        }

      return getAtomIds;
    }
  } // end of namespace atoms
} // end of namespace dftefe