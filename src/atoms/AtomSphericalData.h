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

#ifndef dftefeAtomSphericalData_h
#define dftefeAtomSphericalData_h

#include <utils/TypeConfig.h>
#include <atoms/SphericalData.h>
#include <memory>
#include <map>
#include <vector>
#include <string>
namespace dftefe
{
    namespace atoms
    {

    /**
     * @brief Class to spherical data for a given atomic species.
     * It <b> assumes the atomic data to be provided in a file to be in XML format</b>
     * It <b> assumes the atomic data to be spherical in nature</b>, i.e., the field can be written
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
      class AtomSphericalData{

	public:
	  AtomSphericalData(const std::string filename);
	  ~AtomSphericalData() = default;
	  double getAtomicNumber() const;
	  double getCharge() const;
	  std::string getSymbol() const;
	  std::vector<double> getRadialPoints() const;
	  std::vector<double> getRadialDensity() const;
	  std::vector<double> getRadialVNuclear(const std::vector<int> qNumbers) const;
	  const std::vector<SphericalData> & getHartreeData() const;
	  std::vector<double> getRadialVTotal(const std::vector<int> qNumbers) const;
	  std::vector<double> getRadialOrbital(const std::vector<int> qNumbers) const;

       protected:
	  std::string d_filename;
	  std::string d_symbol;
	 double d_Z;
	 double d_charge;
	 size_type d_numRadialPoints;
	 std::vector<double> d_radialPoints;
	 std::vector<SphericalData> d_density;
	 std::vector<SphericalData> d_hartree;
	 //std::map<std::vector<int>, std::vector<double>> d_qNumbersToVselfMap;
	 //std::map<std::vector<int>, std::vector<double>> d_qNumbersToVHartreeMap;
	 //std::map<std::vector<int>, std::vector<double>> d_qNumbersToVNuclearMap;
	 //std::map<std::vector<int>, std::vector<double>> d_qNumbersToVTotalMap;
	 //std::map<std::vector<int>, std::vector<double>> d_qNumbersToRadialOrbitalMap;
	 //std::map<std::vector<int>, std::pair<double,double>> d_qNumbersToRadialOrbitalCutoffParamsMap;
      };
  } // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeAtomSphericalData_h

