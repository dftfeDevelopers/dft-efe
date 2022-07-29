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

#ifndef dftefeEnrichmentManagerAtomSphericalNumerical_h
#define dftefeEnrichmentManagerAtomSphericalNumerical_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <physics/atoms/AtomsData.h>
#include <memory>
#include <map>
#include <vector>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief A class which provides access to spherical atom-centered enrichment functions, with the
     * radial part given numerically on a grid.
     * This specifically assumes the <b> dimensionality of the problem to be
     * 3</b>. Thus, an enrichment function \f$
     * N^{\boldsymbol{\textbf{R}}}(\boldsymbol{\textbf{r}})\f$, centered on a
     * point \f$\boldsymbol{\textbf{R}}\f$ can be written as \f{equation*}{
     *  N^{\boldsymbol{\textbf{r}}}(\boldsymbol{\textbf{r}}) = f(r)
     * Y_{lm}(\theta,\phi) \f} where \f$r\f$ is the distance of
     * \f$\boldsymbol{\textbf{r}}\f$ from \f$\boldsymbol{\textbf{R}}\f$;
     * \f$f(r)\f$ is the numerical radial part which is provided on a 1D grid.
     * \f$Y_{lm}(\theta,\phi)\f$ is a spherical harmonic of degree \f$l\f$ and
     * order \f$m\f$, and \f$\theta\f$  and \f$\phi\f$ are the azimuthal and
     * polar angles for the point \f$\boldsymbol{\textbf{r}}\f$ with the origin
     * at \f$\boldsymbol{\textbf{R}}\f$. See
     * https://en.wikipedia.org/wiki/Spherical_harmonics for more details on
     * spherical harmonics.
     *
     * @tparam ValueType the primitive data for the enrichment function (e.g., double, float, complex<double>, complex<float>, etc.).
     * @tparam dim dimension of the enrichment (e.g., 1D, 2D, 3D,...).
     */
    template <typename ValueType, size_type dim>
    class EnrichmentFunctionManagerAtomCenteredNumerical
    {
    public:
      /**
       * @brief Constructor
       *
       * @param[in] atomSymbols vector containing the symbols of all the atoms
       * in the system
       * @param[in] atomCoordinates vector containing the coordinates of all the
       * atoms in the system
       * @param[in] atomsData AtomsData object containing the radial and angular
       * data for the atomic species in the system
       */
      EnrichmentManagerAtomSphericalNumerical(
        const std::vector<std::string> > &atomSymbols,
        const std::vector<utils::Point> &atomCoordinates,
        const AtomsData &                atomsData);

      /**
       * @brief Destructor.
       */
      ~EnrichmentManagerAtomSphericalNumerical() = default;

      /**
       * @brief Get number of enrichement functions.
       *
       * @return Number of enrichment functions.
       */
      virtual int
      nFunctions() const = 0;

      /**
       * @brief Get the value of an enrichment function at a point
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] point coordinate of the point
       *
       * @return value of the enrichment function at the point
       */
      virtual ValueType
      getValue(const size_type functionId, const utils::Point &point) const = 0;

      /**
       * @brief Get the value of an enrichment function at a point
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] point coordinate of the point
       *
       * @return value of the enrichment function at the point
       */
      virtual ValueType
      getValue(const size_type            functionId,
               const std::vector<double> &point) const = 0;


      /**
       * @brief Get the value of an enriched function for a set of points
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] points vector of Point containing the coordinates of all the
       * input points
       *
       * @return values of the enrichment function on all the input points
       */
      virtual std::vector<ValueType>
      getValues(const int                        functionId,
                const std::vector<utils::Point> &points) const = 0;

      /**
       * @brief Get the value of an enriched function for a set of points
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] points 2D vector containing the coordinates of all the input
       * points. The expected format is such that points[i][j] should correspond
       * to the j-th coordinate of the i-th point.
       *
       * @return values of the enrichment function on all the input points
       */
      virtual std::vector<ValueType>
      getValues(const int                               functionId,
                const std::vector<std::vector<double>> &points) const = 0;

      /**
       * @brief Get the derivative of an enrichment function at a point
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] point coordinate of the point
       * @param[in] derivativeOrder order of derivative
       *
       * @return vector containing the derivative of the enrichment function at the point.
       * The i-th value in the value denotes the derivative with respect to the
       * i-th coordinate of the point.
       */
      virtual std::vector<ValueType>
      getDerivativeValue(const size_type     functionId,
                         const utils::Point &point,
                         const size_type     derivativeOrder) const = 0;

      /**
       * @brief Get the derivative of an enrichment function at a point
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] point coordinate of the point
       * @param[in] derivativeOrder order of derivative
       *
       * @return vector containing the derivative of the enrichment function at the point.
       * The i-th value in the vector denotes the derivative with respect to the
       * i-th coordinate of the point.
       */
      virtual ValueType
      getDerivativeValue(const size_type            functionId,
                         const std::vector<double> &point,
                         const size_type            derivativeOrder) const = 0;

      /**
       * @brief Get the derivative of an enriched function for a set of points
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] points vector of Point containing the coordinates of all the
       * input points
       * @param[in] derivativeOrder order of derivative
       *
       * @return 2D vector containing the derivative of the enrichment function at all the input points.
       * The (i,j)-th value in the 2D vector denotes the derivative with respect
       * to the j-th coordinate of the i-th point.
       */
      virtual std::vector<std::vector<ValueType>>
      getDerivativeValues(const int                        functionId,
                          const std::vector<utils::Point> &points,
                          const size_type derivativeOrder) const = 0;

      /**
       * @brief Get the derivative of an enriched function for a set of points
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] points vector of Point containing the coordinates of all the
       * input points
       * @param[in] derivativeOrder order of derivative
       *
       * @return 2D vector containing the derivative of the enrichment function at all the input points.
       * The (i,j)-th value in the 2D vector denotes the derivative with respect
       * to the j-th coordinate of the i-th point.
       */
      virtual std::vector<std::vector<ValueType>>
      getDerivativeValues(const int                               functionId,
                          const std::vector<std::vector<double>> &points,
                          const size_type derivativeOrder) const = 0;
    };
  } // end of namespace basis
} // end of namespace dftefe

#endif // dftefeEnrichmentManagerAtomSphericalNumerical_h
