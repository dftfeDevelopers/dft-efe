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

#include <quadrature/QuadratureAttributes.h>
#include <utils/Exceptions.h>
namespace dftefe
{
  namespace quadrature
  {

	     std::map<QuadratureRuleType, size_type>
      _dftefe_quadrature_rule_to_1d_num_points_map_ = {
        {QuadratureRuleType::GAUSS_1, 1},   {QuadratureRuleType::GAUSS_2, 2},
        {QuadratureRuleType::GAUSS_3, 3},   {QuadratureRuleType::GAUSS_4, 4},
        {QuadratureRuleType::GAUSS_5, 5},   {QuadratureRuleType::GAUSS_6, 6},
        {QuadratureRuleType::GAUSS_7, 7},   {QuadratureRuleType::GAUSS_8, 8},
        {QuadratureRuleType::GAUSS_9, 9},   {QuadratureRuleType::GAUSS_10, 10},
        {QuadratureRuleType::GAUSS_11, 11}, {QuadratureRuleType::GAUSS_12, 12},
        {QuadratureRuleType::GLL_1, 1},     {QuadratureRuleType::GLL_2, 2},
        {QuadratureRuleType::GLL_3, 3},     {QuadratureRuleType::GLL_4, 4},
        {QuadratureRuleType::GLL_5, 5},     {QuadratureRuleType::GLL_6, 6},
        {QuadratureRuleType::GLL_7, 7},     {QuadratureRuleType::GLL_8, 8},
        {QuadratureRuleType::GLL_9, 9},     {QuadratureRuleType::GLL_10, 10},
        {QuadratureRuleType::GLL_11, 11},   {QuadratureRuleType::GLL_12, 12}};

	     std::map<QuadratureRuleType, QuadratureFamily>
	          _dftefe_quadrature_rule_to_quad_family_ = {
        {QuadratureRuleType::GAUSS_1, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_2, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_3, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_4, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_5, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_6, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_7, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_8, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_9, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_10, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_11, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GAUSS_12, QuadratureFamily::GAUSS},
        {QuadratureRuleType::GLL_1, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_2, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_3, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_4, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_5, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_6, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_7, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_8, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_9, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_10, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_11, QuadratureFamily::GLL},
        {QuadratureRuleType::GLL_12, QuadratureFamily::GLL}};


    QuadratureRuleAttributes::QuadratureRuleAttributes()
    {
      utils::throwException(
        false,
        "Cannot use default constructor of QuadratureRuleAttributes. Use "
        "QuadratureRuleAttributes(const quadrature::QuadratureFamily, bool isCartesianTensorStructured, const size_type num1DPoints) constructor.");
    }

    QuadratureRuleAttributes::QuadratureRuleAttributes(
      const QuadratureFamily quadratureFamily,
      const bool             isCartesianTensorStructured,
      const size_type        num1DPoints /*= 0*/)
      : d_quadratureFamily(quadratureFamily)
      , d_isCartesianTensorStructured(isCartesianTensorStructured)
      , d_num1DPoints(num1DPoints)
    {
      if (d_isCartesianTensorStructured == false)
        {
          utils::throwException<utils::LogicError>(
            d_num1DPoints == 0,
            "The use of non-zero number of points in 1D for a non-cartesian-tensored-structured quadrature is not allowed.");
        }

      if (d_quadratureFamily == QuadratureFamily::GAUSS_VARIABLE ||
          d_quadratureFamily == QuadratureFamily::GLL_VARIABLE)
        {
          utils::throwException<utils::LogicError>(
            d_num1DPoints == 0,
            "The use of non-zero number of points in 1D for a quadrature rule that is variable across cells is not allowed.");
        }
    }

    QuadratureFamily
    QuadratureRuleAttributes::getQuadratureFamily() const
    {
      return d_quadratureFamily;
    }

    bool
    QuadratureRuleAttributes::isCartesianTensorStructured() const
    {
      return d_isCartesianTensorStructured;
    }

    size_type
    QuadratureRuleAttributes::getNum1DPoints() const
    {
      return d_num1DPoints;
    }

    bool
    QuadratureRuleAttributes::operator<(
      const QuadratureRuleAttributes &quadratureRuleAttributes) const
    {
      if (d_quadratureFamily == quadratureRuleAttributes.d_quadratureFamily)
        {
          return d_num1DPoints < quadratureRuleAttributes.d_num1DPoints;
        }
      else
        return d_quadratureFamily < quadratureRuleAttributes.d_quadratureFamily;
    }

    bool
    QuadratureRuleAttributes::operator==(
      const QuadratureRuleAttributes &quadratureRuleAttributes) const
    {
      const bool flag =
        (d_quadratureFamily == quadratureRuleAttributes.d_quadratureFamily) &&
        (d_num1DPoints == quadratureRuleAttributes.d_num1DPoints) &&
        (d_isCartesianTensorStructured ==
         quadratureRuleAttributes.d_isCartesianTensorStructured);
      return flag;
    }
  } // end of namespace quadrature
} // end of namespace dftefe
