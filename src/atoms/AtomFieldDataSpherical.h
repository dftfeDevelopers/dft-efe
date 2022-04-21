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

#ifndef dftefeAtomFieldDataSpherical_h
#define dftefeAtomFieldDataSpherical_h

#include <utils/TypeConfig.h>
#include <memory>
#include <map>
#include <vector>
#include <string>
namespace dftefe
{
    namespace atoms
    {

    /**
     * @brief Class to store a field specific data for a given atomic species.
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
      class AtomFieldDataSpherical{

	public:
	  /**
	   * @brief Constructor
	   *
	   * @param[in] filename name of the file containing the atomic data
	   * @param[in] atomFieldName name of the atomic field that needs to be stored
	   */
	 AtomFieldDataSpherical(const std::string filename,
	     const std::string atomFieldname);

	  /**
	   * @brief Destructor 
	   */
	 ~AtomFieldDataSpherical() = default;

	 /**
	  * @brief Return all the quantum number triplets for the given atom
	  *
	  * @return 2D vector containing all the quantum number triplets.
	  * Each element in the 2D vector contains a vector of size 3 containing the \f$n,l,m\f$ 
	  * quantum numbers (see top for definitions), in that sequence.
	  */ 
	 std::vector<std::vector<int>> 
	 getQuantumNumbers() const;
	
	 /**
	  * @brief Get the vector containing the radial grid points for a given quantum numbers (\f$n,l,m\f$ , see top for definition) 
	  * @param[in] quantumNumbers vector containing the \f$n,l,m\f$ quantum numbers
	  *
	  * @return vector containing the radial grid points for the given atom symbol and quantum numbers  
	  */ 
	 std::vector<double>
	 getRadialGridPoints(const std::vector<int> & quantumNumbers) const;
	 
	 /**
	  * @brief Get the radial function for a given triplet of quantum numbers (\f$n,l,m\f$ , see top for definition) 
	  * @param[in] quantumNumbers vector containing the \f$n,l,m\f$ quantum numbers
	  *
	  * @return a pair of vectors where the first vector contains the radial grid points and the second vector contains the values 
	  * of the radial function at the grid points.
	  */ 
	 std::pair<std::vector<double>, std::vector<double>>
	 getRadialFunction(const std::vector<int> & quantumNumbers) const;
	 
	 /**
	  * @brief Get the cutoff radius and smoothness parameter for the radial part of the field corresponding to a triplet of quantum numbers 
	  * (\f$n,l,m\f$ , see top for definition) 
	  * @param[in] quantumNumbers vector containing the \f$n,l,m\f$ quantum numbers
	  *
	  * @return a pair of numbers where the first one is the cutoff radius and the second one is the smoothness parameter 
	  */ 
	 std::pair<double, double>
	 getCutOffAndSmoothness(const std::vector<int> & quantumNumbers) const;

	private:
	 std::string d_filename;
	 std::string d_atomFieldname;
	 std::vector<std::vector<int>> d_quantumNumbers;
	 std::vector<std::vector<double>> d_radialGridPoints;
	 std::vector<std::vector<double>> d_radialFunctionValues;
	 std::map<std::vector<int>, std::pair<size_type, size_type>> d_quantumNumbersToRadialGridAndFunctionIdMap;
	 std::map<std::vector<int>, std::pair<double, double>> d_quantumNumbersToCutoffRadiusAndSmoothnessMap;

      };
  } // end of namespace atoms
} // end of namespace dftefe

#endif // dftefeAtomFieldDataSphericalManager_h
