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

#ifndef dftefeAtomSphericalDataPSP_h
#define dftefeAtomSphericalDataPSP_h

#include <utils/TypeConfig.h>
#include <atoms/SphericalData.h>
#include <atoms/AtomSphericalData.h>
#include <atoms/SphericalDataNumerical.h>
#include <atoms/SphericalDataMixed.h>
#include <utils/PointChargePotentialFunction.h>
#include <memory>
#include <unordered_map>
#include <map>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <libxml/tree.h>
#include <libxml/xpathInternals.h>
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
    class AtomSphericalDataPSP : public AtomSphericalData
    {
    public:
      struct XPathInfo
      {
        xmlDocPtr   doc;
        std::string fileName;
        std::string xpath;
        std::string ns;
        std::string nsHRef;
      };

    public:
      AtomSphericalDataPSP(
        const std::string                 fileName,
        const std::vector<std::string> &  fieldNames,
        const std::vector<std::string> &  metadataNames,
        const SphericalHarmonicFunctions &sphericalHarmonicFunc);

      ~AtomSphericalDataPSP() = default;

      void
      addFieldName(const std::string fieldName) override;

      std::string
      getFileName() const override;

      std::vector<std::string>
      getFieldNames() const override;

      std::vector<std::string>
      getMetadataNames() const override;

      const std::vector<std::shared_ptr<SphericalData>> &
      getSphericalData(const std::string fieldName) const override;

      const std::shared_ptr<SphericalData>
      getSphericalData(const std::string       fieldName,
                       const std::vector<int> &qNumbers) const override;

      std::string
      getMetadata(const std::string metadataName) const override;

      size_type
      getQNumberID(const std::string       fieldName,
                   const std::vector<int> &qNumbers) const override;

      size_type
      nSphericalData(std::string fieldName) const override;

    private:
      void
      getSphericalDataFromXMLNode(
        std::vector<std::shared_ptr<SphericalData>> &sphericalDataVec,
        const std::vector<double> &                  radialPoints,
        XPathInfo &                                  xPathInfo,
        const std::string &                          fieldName,
        const SphericalHarmonicFunctions &           sphericalHarmonicFunc);

      std::string              d_fileName, d_rootElementName;
      std::vector<std::string> d_fieldNames;
      std::vector<std::string> d_metadataNames;
      std::unordered_map<std::string,
                         std::vector<std::shared_ptr<SphericalData>>>
        d_sphericalData;
      std::unordered_map<std::string, std::map<std::vector<int>, size_type>>
                                                   d_qNumbersToIdMap;
      std::unordered_map<std::string, std::string> d_metadata;
      double                                       d_zvalance;
      const double                                 d_PSPVLocalCutoff;
      int d_lmax;
      std::shared_ptr<utils::ScalarSpatialFunctionReal>
        d_scalarSpatialFnAfterRadialGrid;

      std::vector<double> d_radialPoints;
      const SphericalHarmonicFunctions &d_sphericalHarmonicFunc;
    };
  } // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeAtomSphericalDataPSP_h
