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
      const size_type        num1DPoints /*= 0*/,
      const std::string      tag /*= std::string()*/)
      : d_quadratureFamily(quadratureFamily)
      , d_isCartesianTensorStructured(isCartesianTensorStructured)
      , d_num1DPoints(num1DPoints)
      , d_tag(tag)
    {
      if (d_isCartesianTensorStructured == false)
        {
          utils::throwException<utils::LogicError>(
            d_num1DPoints == 0,
            "The use of non-zero number of points in 1D for a non-cartesian-tensored-structured quadrature is not allowed.");
        }

      if (d_quadratureFamily == QuadratureFamily::GAUSS_VARIABLE ||
          d_quadratureFamily == QuadratureFamily::GLL_VARIABLE ||
          d_quadratureFamily == QuadratureFamily::ADAPTIVE)
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

    std::string
    QuadratureRuleAttributes::getTag() const
    {
      return d_tag;
    }

    bool
    QuadratureRuleAttributes::operator==(
      const QuadratureRuleAttributes &quadratureRuleAttributes) const
    {
      const bool flag =
        (d_quadratureFamily == quadratureRuleAttributes.d_quadratureFamily) &&
        (d_num1DPoints == quadratureRuleAttributes.d_num1DPoints) &&
        (d_isCartesianTensorStructured ==
         quadratureRuleAttributes.d_isCartesianTensorStructured) &&
        (d_tag == quadratureRuleAttributes.d_tag);
      return flag;
    }
  } // end of namespace quadrature
} // end of namespace dftefe
