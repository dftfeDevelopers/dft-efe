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

#ifndef dftefeQuadratureAttributes_h
#define dftefeQuadratureAttributes_h

#include <utils/TypeConfig.h>
#include <map>
namespace dftefe
{
  namespace quadrature
  {
    enum class QuadratureFamily
    {
      GAUSS, // Uniform Gauss quadrature rule across all cells in the domain
      GLL, // Uniform Gauss-Legendre-Lobatto quadrature rule across all cells in
           // the domain
      GAUSS_VARIABLE, // Variable Gauss quadrature rule (i.e., different cells
                      // have different Gauss quadrature)
      GLL_VARIABLE,   // Variable Gauss-Legendre-Lobatto quadrature rule (i.e.,
                      // different cells have different Gauss-Legendre-Lobatto
                      // quadrature)
      ADAPTIVE        // Adaptive quadrature rule
    };

    enum class QuadratureRuleType
    {

      // Gauss quadrature rules
      GAUSS_1,
      GAUSS_2,
      GAUSS_3,
      GAUSS_4,
      GAUSS_5,
      GAUSS_6,
      GAUSS_7,
      GAUSS_8,
      GAUSS_9,
      GAUSS_10,
      GAUSS_11,
      GAUSS_12,
      GAUSS_VARIABLE,

      // Gauss-Legendre-Lobatta quadrature rules
      GLL_1,
      GLL_2,
      GLL_3,
      GLL_4,
      GLL_5,
      GLL_6,
      GLL_7,
      GLL_8,
      GLL_9,
      GLL_10,
      GLL_11,
      GLL_12,
      GLL_VARIABLE,

      // Adaptive quadrature rule
      ADAPTIVE
    };

extern std::map<QuadratureRuleType, size_type>
      _dftefe_quadrature_rule_to_1d_num_points_map_ ;

    extern  std::map<QuadratureRuleType, QuadratureFamily>
	    _dftefe_quadrature_rule_to_quad_family_ ;

    class QuadratureRuleAttributes
    {
    public:
      QuadratureRuleAttributes();
      QuadratureRuleAttributes(const QuadratureFamily quadratureFamily,
                               const bool      isCartesianTensorStructured,
                               const size_type num1DPoints = 0);
      ~QuadratureRuleAttributes() = default;
      QuadratureFamily
      getQuadratureFamily() const;
      bool
      isCartesianTensorStructured() const;
      size_type
      getNum1DPoints() const;
      bool
      operator<(const QuadratureRuleAttributes &quadratureRuleAttributes) const;

      bool
      operator==(
        const QuadratureRuleAttributes &quadratureRuleAttributes) const;

    private:
      QuadratureFamily d_quadratureFamily;
      bool             d_isCartesianTensorStructured;
      size_type        d_num1DPoints;
    }; // end of QuadratureRuleAttributes

    /**
     * @brief Class to store the attributes of a quad point, such as
     * the cell Id it belongs, the quadPointId within the cell it belongs to,
     * and the quadrature rule (defined by quadratureRuleId) it is part of.
     */
    class QuadraturePointAttributes
    {
    public:
      QuadraturePointAttributes()
        : cellId(0)
        , quadratureRuleAttributesPtr(nullptr)
        , quadPointId(0){};

      QuadraturePointAttributes(
        const size_type                 inputCellId,
        const QuadratureRuleAttributes *inputQuadratureRuleAttributesPtr,
        const size_type                 inputQuadPointId)
        : cellId(inputCellId)
        , quadratureRuleAttributesPtr(inputQuadratureRuleAttributesPtr)
        , quadPointId(inputQuadPointId){};
      size_type                       cellId;
      const QuadratureRuleAttributes *quadratureRuleAttributesPtr;
      size_type                       quadPointId;
    }; // end of class QuadraturePointAttributes
  }    // end of namespace quadrature
} // end of namespace dftefe
#endif
