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

#ifndef dftefeEnrichmentManager_h
#define dftefeEnrichmentManager_h

#include <utils/TypeConfig.h>
#include <memory>
#include <map>
#include <vector>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Base class which provides access to the enrichment functions
     *
     * @tparam ValueTypeBasisData the primitive data for the enrichment function (e.g., double, float, complex<double>, complex<float>, etc.).
     * @tparam dim dimension of the enrichment (e.g., 1D, 2D, 3D,...).
     *
     */
    template <typename ValueTypeBasisData, size_type dim>
    class EnrichmentManager
    {
      //
      // types
      //
    public:
      /**
       * @brief Destructor.
       */
      virtual ~EnrichmentManager() = default;

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
      virtual ValueTypeBasisData
      getValue(const size_type functionId, const utils::Point &point) const = 0;

      /**
       * @brief Get the value of an enrichment function at a point
       *
       * @param[in] functionId index of the enrichment function
       * @param[in] point coordinate of the point
       *
       * @return value of the enrichment function at the point
       */
      virtual ValueTypeBasisData
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
      virtual std::vector<ValueTypeBasisData>
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
      virtual std::vector<ValueTypeBasisData>
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
      virtual std::vector<ValueTypeBasisData>
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
      virtual ValueTypeBasisData
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
      virtual std::vector<std::vector<ValueTypeBasisData>>
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
      virtual std::vector<std::vector<ValueTypeBasisData>>
      getDerivativeValues(const int                               functionId,
                          const std::vector<std::vector<double>> &points,
                          const size_type derivativeOrder) const = 0;
    };

  } // end of namespace basis
} // end of namespace dftefe

#endif // dftefeEnrichmentManager_h
