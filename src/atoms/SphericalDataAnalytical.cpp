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

#include <vector>
#include <utils/TypeConfig.h>
#include "SphericalHarmonicFunctions.h"
#include "SmoothCutoffFunctions.h"
#include <utils/Spline.h>
#include "BoostAutoDiff.h"
#include <cmath>
#include <atoms/SphericalDataAnalytical.h>

namespace dftefe
{
  namespace atoms
  {
    namespace SphericalDataAnalyticalInternal
    {
      void
      getValueAnalytical(
        const std::vector<utils::Point> &       point,
        const utils::Point &                    origin,
        const double                            cutoff,
        const double                            smoothness,
        const utils::ScalarSpatialFunctionReal &function,
        const SphericalHarmonicFunctions &      sphericalHarmonicFunc,
        const std::vector<int> &                qNumbers,
        const double                            polarAngleTolerance,
        std::vector<double> &                   value)
      {
        for (int i = 0; i < point.size(); i++)
          {
            double r, theta, phi;
            convertCartesianToSpherical(
              point[i] - origin, r, theta, phi, polarAngleTolerance);
            int  n = qNumbers[0], l = qNumbers[1], m = qNumbers[2];
            auto Ylm = Clm(l, m) * Dm(m) *
                       sphericalHarmonicFunc.Plm(l, std::abs(m), theta) *
                       Qm(m, phi);
            value[i] = function(point[i] - origin) * Ylm;
          }
      }
    } // namespace SphericalDataAnalyticalInternal

    SphericalDataAnalytical::SphericalDataAnalytical(
      const std::vector<int>                  qNumbers,
      const utils::ScalarSpatialFunctionReal &function,
      const double                            cutoff,
      const double                            smoothness,
      const SphericalHarmonicFunctions &      sphericalHarmonicFunc,
      const double                            polarAngleTolerance,
      const size_type                         dim)
      : d_qNumbers(qNumbers)
      , d_polarAngleTolerance(polarAngleTolerance)
      , d_func(function)
      , d_dim(dim)
      , d_cutoff(cutoff)
      , d_smoothness(smoothness)
      , d_sphericalHarmonicFunc(sphericalHarmonicFunc)
    {
      utils::throwException<utils::InvalidArgument>(d_dim == 3,
                                                    "Dimension has to be 3.");
    }

    std::vector<double>
    SphericalDataAnalytical::getValue(const std::vector<utils::Point> &point,
                                 const utils::Point &             origin)
    {
      std::vector<double> value(point.size(), 0.);
      DFTEFE_AssertWithMsg(point[0].size() == d_dim && origin.size() == d_dim,
                           "getValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataAnalyticalInternal::getValueAnalytical(point,
                                                     origin,
                                                     d_cutoff,
                                                     d_smoothness,
                                                     d_func,
                                                     d_sphericalHarmonicFunc,
                                                     d_qNumbers,
                                                     d_polarAngleTolerance,
                                                     value);
      return value;
    }

    std::vector<double>
    SphericalDataAnalytical::getGradientValue(const std::vector<utils::Point> &point,
                                         const utils::Point &origin)
    {
      std::vector<double> gradient(d_dim * point.size(), 0.);
      utils::throwException(
        false,
        "getGradientValue() function in SphericalDataAnalytical is not yet implemented.");
      return gradient;
    }

    std::vector<double>
    SphericalDataAnalytical::getHessianValue(const std::vector<utils::Point> &point,
                                        const utils::Point &             origin)
    {
      std::vector<double> hessian(d_dim * d_dim, 0.),
        ret(d_dim * d_dim * point.size(), 0.);
      utils::throwException(
        false,
        "getHessianValue() function in SphericalDataAnalytical is not yet implemented.");
      return ret;
    }

    double
    SphericalDataAnalytical::getValue(const utils::Point &point,
                                 const utils::Point &origin)
    {
      std::vector<double> value(1, 0);
      DFTEFE_AssertWithMsg(point.size() == d_dim && origin.size() == d_dim,
                           "getValue() has a dimension mismatch");
      DFTEFE_AssertWithMsg(d_qNumbers.size() == 3,
                           "All quantum numbers not given");
      SphericalDataAnalyticalInternal::getValueAnalytical(
        std::vector<utils::Point>{point},
        origin,
        d_cutoff,
        d_smoothness,
        d_func,
        d_sphericalHarmonicFunc,
        d_qNumbers,
        d_polarAngleTolerance,
        value);

      return value[0];
    }

    std::vector<double>
    SphericalDataAnalytical::getGradientValue(const utils::Point &point,
                                         const utils::Point &origin)
    {
      std::vector<double> gradient(d_dim, 0.);
      utils::throwException(
        false,
        "getGradientValue() function in SphericalDataAnalytical is not yet implemented.");
      return gradient;
    }

    std::vector<double>
    SphericalDataAnalytical::getHessianValue(const utils::Point &point,
                                        const utils::Point &origin)
    {
      std::vector<double> hessian(d_dim * d_dim, 0.);
      utils::throwException(
        false,
        "hessian() function in SphericalDataAnalytical is not yet implemented.");
      return hessian;
    }

    std::vector<double>
    SphericalDataAnalytical::getRadialValue(const std::vector<double> &r)
    {
      std::vector<double> retVal(r.size(), 0.);
      utils::throwException(
        false,
        "getRadialValue() function in SphericalDataAnalytical is not yet implemented.");
      return retVal;
    }

    std::vector<double>
    SphericalDataAnalytical::getAngularValue(const std::vector<double> &r,
                                        const std::vector<double> &theta,
                                        const std::vector<double> &phi)
    {
      std::vector<double> retVal(r.size(), 0.);
      utils::throwException(
        false,
        "getRadialDerivative() function in SphericalDataAnalytical is not yet implemented.");
      return retVal;
    }

    std::vector<double>
    SphericalDataAnalytical::getRadialDerivative(const std::vector<double> &r)
    {
      std::vector<double> retVal(r.size(), 0.);
      utils::throwException(
        false,
        "getRadialDerivative() function in SphericalDataAnalytical is not yet implemented.");
      return retVal;
    }

    std::vector<std::vector<double>>
    SphericalDataAnalytical::getAngularDerivative(
      const std::vector<double> &r,
      const std::vector<double> &thetaVec,
      const std::vector<double> &phiVec)
    {
      std::vector<std::vector<double>> retVal(2,
                                              std::vector<double>(r.size(),
                                                                  0.));
      utils::throwException(
        false,
        "getAngularDerivative() function in SphericalDataAnalytical is not yet implemented.");
      return retVal;
    }

    std::vector<int>
    SphericalDataAnalytical::getQNumbers() const
    {
      return d_qNumbers;
    }

    double
    SphericalDataAnalytical::getCutoff() const
    {
      return d_cutoff;
    }

    double
    SphericalDataAnalytical::getSmoothness() const
    {
      return d_smoothness;
    }

  } // namespace atoms
} // namespace dftefe
