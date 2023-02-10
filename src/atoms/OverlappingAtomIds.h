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

#ifndef dftefeOverlappingAtomIds_h
#define dftefeOverlappingAtomIds_h

#include <utils/TypeConfig.h>
#include <map>
#include <string>
#include <atoms/AtomSphericalDataContainer.h>
#include <atoms/SphericalData.h>
namespace dftefe
{
  namespace atoms
  {
    /**
     * @brief Class to get the Ids of the Atoms overlapping a particular Processor
     * for a particular field. Each Processor stores info of all the atoms.
     */
    template <unsigned int dim>
    class OverlappingAtomIds
    {
    public:
      /**
       * @brief Constructor
       *
       * @param[in] mapAtomSymbolToCoordinates Map from atomic symbol to input
       * coordinates
       * @param[in] fieldname String defining the field that needs to be read
       * from the atom's XML file
       * @param[in] AtomSphericalDataContainer take an object of the
       * atomsphericaldatacontainer class
       * @param[in] minbound minimum limit of the processor boundary given
       * @param[in] maxbound maximum limit of the processor boundary given
       * @return getOverlappingAtomIdsWithinBox() returns the vectors of the overlappig atom Ids in a processor
       * consIdering it a cuboId box with upper and lower bounds
       */
      OverlappingAtomIds(const std::vector<std::string, utils::Point>
                           &                 mapAtomSymbolToCoordinates,
                         const std::string   fieldName,
                         const utils::Point &minbound,
                         const utils::Point &maxbound);

      /**
       * @brief Destructor
       */
      ~OverlappingAtomIds() = default;

      /**
       * @brief Function to return the vector
       */
      std::vector<size_type>
      getOverlappingAtomIdsWithinBox() const;

    private:
      std::map<std::string, utils::Point> d_mapAtomSymbolToCoordinates;
      std::string                         d_fieldName;
      utils::Point                        d_minbound;
      utils::Point                        d_maxbound;
    }; // end of class OverlappingAtomIds
  }    // end of namespace atoms
} // end of namespace dftefe
#include <atoms/OverlappingAtomIds.t.cpp>
#endif // dftefeOverlappingAtomIds_h