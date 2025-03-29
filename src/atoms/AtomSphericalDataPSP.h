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

#ifndef dftefeAtomSphericalData_h
#define dftefeAtomSphericalData_h

#include <utils/TypeConfig.h>
#include <atoms/SphericalData.h>
#include <atoms/SphericalDataNumerical.h>
#include <memory>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
namespace dftefe
{
  namespace atoms
  {
    /**
     * @brief Class to spherical data for a given atomic species.
     * It <b> assumes the atomic data to be provided in a file to be in XML
     * format</b> It <b> assumes the atomic data to be spherical in nature</b>,
     * i.e., the field can be written as a product of a radial and an angular
     * part, given as \f{equation*}{ N(\boldsymbol{\textbf{r}}) = f_{nl}(r)
     * Y_{lm}(\theta,\phi) \f}
     *
     * where \f$r\f$ is the distance from origin, \f$\theta\f$ is the polar
     * angle, and \f$\phi\f$ is the azimuthal angle. \f$n,l,m\f$ denote the
     * principal, angular, and magnetic quantum numbers, respectively.
     * $\fY_{lm}\f$ denotes a spherical harmonic of degree \f$l\f$ and order
     * \f$m\f$. See https://en.wikipedia.org/wiki/Spherical_harmonics for more
     * details on spherical harmonics.
     */
    class AtomSphericalData
    {
    public:
      AtomSphericalData(
        const std::string                 fileName,
        const std::vector<std::string> &  fieldNames,
        const std::vector<std::string> &  metadataNames,
        const SphericalHarmonicFunctions &sphericalHarmonicFunc);

      ~AtomSphericalData() = default;

      std::string
      getFileName() const;

      std::vector<std::string>
      getFieldNames() const;

      std::vector<std::string>
      getMetadataNames() const;

      const std::vector<std::shared_ptr<SphericalData>> &
      getSphericalData(const std::string fieldName) const;

      const std::shared_ptr<SphericalData>
      getSphericalData(const std::string       fieldName,
                       const std::vector<int> &qNumbers) const;

      std::string
      getAttributeFieldData(const std::string fieldName,
                       const std::string attributeName) const;

      std::string
      getAttributeMetaData(const std::string metaDataName,
                       const std::string attributeName) const;

      std::string
      getMetadata(const std::string metadataName) const;

      size_type
      getQNumberID(const std::string       fieldName,
                   const std::vector<int> &qNumbers) const;

      size_type
      nSphericalData(std::string fieldName) const;

    private:
      std::string              d_fileName;
      std::vector<std::string> d_fieldNames;
      std::vector<std::string> d_metadataNames;
      std::unordered_map<std::string,
                         std::vector<std::shared_ptr<SphericalData>>>
        d_sphericalData;
      std::unordered_map<std::string, std::map<std::vector<int>, size_type>>
                                                   d_qNumbersToIdMap;
      std::unordered_map<std::string, std::string> d_metadata;
      
      std::unordered_map<std::string, std::pair<std::string, std::string>>
        d_attributeMetaData;
      std::unordered_map<std::string, std::pair<std::string, std::string>>
        d_attributeFieldData;

    };
  } // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeAtomSphericalData_h
