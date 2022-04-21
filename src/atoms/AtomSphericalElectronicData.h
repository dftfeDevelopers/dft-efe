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

#ifndef dftefeAtomSphericalElectronicData_h
#define dftefeAtomSphericalElectronicData_h

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
      class AtomSphericalElectronicData{

	public:
	  AtomSphericalElectronicData(const std::string filename);
	  ~AtomSphericalElectronicData() = default;
	  double getAtomicNumber() const;
	  double getCharge() const;
	  std::string getSymbol() const;
	  std::vector<double> getRadialPoints() const;
	  const std::vector<SphericalData> & getDensityData() const;
	  const std::vector<SphericalData> & getVHartreeData() const;
	  const std::vector<SphericalData> & getVNuclearData() const;
	  const std::vector<SphericalData> & getVTotalData() const;
	  const std::vector<SphericalData> & getOrbitalData() const;

       protected:
	  std::string d_filename;
	  std::string d_symbol;
	 double d_Z;
	 double d_charge;
	 size_type d_numRadialPoints;
	 std::vector<double> d_radialPoints;
	 std::vector<SphericalData> d_densityData;
	 std::vector<SphericalData> d_vHartreeData;
	 std::vector<SphericalData> d_vNuclearData;
	 std::vector<SphericalData> d_vTotalData;
	 std::vector<SphericalData> d_orbitalData;
	 std::map<std::vector<int>, size_type> d_qNumbersToDensityDataIdMap;
	 std::map<std::vector<int>, size_type> d_qNumbersToVHartreeDataIdMap;
	 std::map<std::vector<int>, size_type> d_qNumbersToVNuclearDataIdMap;
	 std::map<std::vector<int>, size_type> d_qNumbersToVTotalDataIdMap;
	 std::map<std::vector<int>, size_type> d_qNumbersToOrbitalDataIdMap;
      };
  } // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeAtomSphericalElectronicData_h

