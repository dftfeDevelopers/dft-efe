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

#ifndef dftefeAtomSphericalDataContainer_h
#define dftefeAtomSphericalDataContainer_h

#include <utils/TypeConfig.h>
#include <map>
#include <string>
#include "AtomSphericalData.h"
#include "SphericalData.h"
namespace dftefe
{
  namespace atoms
  {
    /**
     * @brief Class to store a field specific data for a given atomic species.
     * It <b> assumes the atomic field data to be spherical in nature</b>, i.e.,
     * the field can be written as a product of a radial and an angular part,
     * given as \f{equation*}{ N(\boldsymbol{\textbf{r}}) = f_{nl}(r)
     * Y_{lm}(\theta,\phi) \f}
     *
     * where \f$r\f$ is the distance from origin, \f$\theta\f$ is the polar
     * angle, and \f$\phi\f$ is the azimuthal angle. \f$n,l,m\f$ denote the
     * principal, angular, and magnetic quantum numbers, respectively.
     * $\fY_{lm}\f$ denotes a spherical harmonic of degree \f$l\f$ and order
     * \f$m\f$. See https://en.wikipedia.org/wiki/Spherical_harmonics for more
     * details on spherical harmonics.
     */
    class AtomSphericalDataContainer
    {
    public:
      
      /**
       * @brief Constructor 
       *    
       * @param[in] atomSymbolToFilename Map from atomic symbol to the XML file containing 
       *   the atom's spherical field data
       * @param[in] fieldname String defining the field that needs to be read from the atom's XML file
       * 
       */
      AtomSphericalDataContainer(
        const std::map<std::string, std::string> &atomSymbolToFilename,
        std::string                               fieldname);
      
      /**
       * @brief Destructor
       */
      ~AtomSphericalDataContainer();

      /**
       * @brief Returns the speherical data for a given atom and quantum numbers
       * 
       * @param[in] atomSymbol String defining the atom
       * @param[in] qNumbers Vector of integers defining the quantum numbers 
       *  (e.g., n,l,m quantum numbers) for which the SphericalData is required
       * @return SphericalData object for the given atom and quantum numbers
       */
      SphericalData
      getSphericalData(std::string             atomSymbol,
                       const std::vector<int> &qNumbers) const;

      size_type
      nSphericalData(std::string atomSymbol) const;

      std::vector<std::vector<int>>
      getQNumbers(std::string atomSymbol) const;

      std::string
      getFieldName() const;

    private:
      std::map<std::string, std:: : string>    d_atomSymbolToFilename;
      std::string                              d_fieldName;
      std::vector<AtomSphericalElectronicData> d_atomSphericalElectronicData;

    }; // end of class AtomSphericalDataContainer
  }    // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeAtomSphericalDataContainer_h
