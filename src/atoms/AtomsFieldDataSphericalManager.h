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

#ifndef dftefeAtomsFieldDataSphericalManager_h
#define dftefeAtomsFieldDataSphericalManager_h

#include <utils/TypeConfig.h>
#include <atoms/AtomFieldDataSpherical.h>
#include <memory>
#include <map>
#include <vector>
#include <string>
namespace dftefe
{
    namespace atoms
    {

    /**
     * @brief Class to store a field specific data for all the atomic species in the system.
     * It <b> assumes the atomic field data to be spherical in nature</b>, i.e., the field can be written
     * as a product of a radial and an angular part, given as
     * \f{equation*}{
     *  N(\boldsymbol{\textbf{r}}) = f_{nl}(r) Y_{lm}(\theta,\phi)
     * \f}
     *
     * where \f$r\f$ is the distance from origin, \f$\theta\f$ is the polar angle, and \f$\phi\f$ is the azimuthal angle.
     * \f$n,l,m\f$ denote the principal, angular, and magnetic quantum numbers, respectively.
     * $\fY_{lm}\f$ denotes a spherical harmonic of degree \f$l\f$ and order \f$m\f$. 
     * See https://en.wikipedia.org/wiki/Spherical_harmonics for more details on spherical harmonics.  
     */
      class AtomFieldDataSphericalManager {

	public:
	  /**
	   * @brief Constructor
	   *
	   * @param[in] atomSymbolToAtomFileMap map from the atomic symbol to the filename containing the atomic data
	   * @param[in] atomFieldName name of the atomic field that needs to be stored
	   */
	 AtomsFieldDataSphericalManager(const std::map<std::string, std::string> & atomSymbolToAtomFileMap,
	     const std::string atomFieldName);

	  /**
	   * @brief Destructor 
	   */
	 ~AtomsFieldDataSphericalManager() = default;

	 /**
	  * @brief Return all the quantum number triplets for a given atomSymbol
	  *
	  * @param[in] atomSymbol symbol of the atomic species
	  *
	  * @return 2D vector containing all the quantum number triplets for the given atomSymbol.
	  * Each element in the 2D vector contains a vector of size 3 containing the \f$n,l,m\f$ 
	  * quantum numbers (see top for definitions), in that sequence.
	  */ 
	 std::vector<std::vector<int>> 
	 getAtomQuantumNumbers(const std::string atomSymbol) const;
	
	 /**
	  * @brief Get the vector containing the radial grid points for a given atom symbol and quantum numbers (\f$n,l,m\f$ , see top for definition) 
	  * @param[in] atomSymbol symbol of the atomic species
	  * @param[in] quantumNumbers vector containing the \f$n,l,m\f$ quantum numbers
	  *
	  * @return vector containing the radial grid points for the given atom symbol and quantum numbers  
	  */ 
	 std::vector<double>
	 getRadialGridPoints(const std::string atomSymbol,
	     const std::vector<int> quantumNumbers) const;
	 
	 /**
	  * @brief Get the radial function for a given atom symbol and quantum numbers (\f$n,l,m\f$ , see top for definition) 
	  * @param[in] atomSymbol symbol of the atomic species
	  * @param[in] quantumNumbers vector containing the \f$n,l,m\f$ quantum numbers
	  *
	  * @return a pair of vectors where the first vector contains the radial grid points and the second vector contains the values 
	  * of the radial function at the grid points.
	  */ 
	 std::pair<std::vector<double>, std::vector<double>>
	 getRadialFunction(const std::string atomSymbol,
	     const std::vector<int> quantumNumbers) const;

	private:
	 std::map<std::string, std::shared_ptr<AtomFieldDataSpherical>> d_atomSymbolToAtomFieldDataSphericalMap;

      };

  } // end of namespace atoms
} // end of namespace dftefe

#endif // dftefeAtomsFieldDataSphericalManager_h
