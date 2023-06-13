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
#include <utils/Point.h>
#include "BoostAutoDiff.h"
#include <boost/math/differentiation/autodiff.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/complex/atan.hpp>
#include <iostream>
#include <cmath>

namespace dftefe
{
  namespace atoms
  {
    namespace BoostAutoDiffInternal
    {
      double
      Dm(const int m)
      {
        if (m == 0)
          return 1.0 / sqrt(2 * M_PI);
        else
          return 1.0 / sqrt(M_PI);
      }

      double
      Clm(const int l, const int m)
      {
        assert(std::abs(m) <= l);
        return sqrt(
          ((2.0 * l + 1) * boost::math::factorial<double>(l - std::abs(m))) /
          (2.0 * boost::math::factorial<double>(l + std::abs(m))));
      }

      template <typename X, typename Y, typename Z>
      boost::math::differentiation::promote<X, Y, Z>
      radVal(const X &                  x,
             const Y &                  y,
             const Z &                  z,
             const std::vector<double> &coeffVec)
      {
        auto r        = sqrt(x * x + y * y + z * z);
        auto radValue = coeffVec[1] + coeffVec[2] * (r - coeffVec[0]) +
                        coeffVec[3] * pow((r - coeffVec[0]), 2) +
                        coeffVec[4] * pow((r - coeffVec[0]), 3);
        return radValue;
      }

      template <typename X, typename Y, typename Z>
      boost::math::differentiation::promote<X, Y, Z>
      cutOff(const X &    x,
             const Y &    y,
             const Z &    z,
             const double smoothness,
             const double cutoff)
      {
        auto r = sqrt(x * x + y * y + z * z);
        auto smoothCutoff =
          (exp(-1.0 / (1 - smoothness * (r - cutoff) / cutoff)) /
           (exp(-1.0 / (1 - smoothness * (r - cutoff) / cutoff)) +
            exp(-1.0 / (1 - (1 - smoothness * (r - cutoff) / cutoff)))));
        return smoothCutoff;
      }

      template <typename X, typename Y, typename Z>
      boost::math::differentiation::promote<X, Y, Z>
      legendre(const X &x, const Y &y, const Z &z, const int l, const int m)
      {
        auto r  = sqrt(x * x + y * y + z * z);
        auto z1 = z / r;
        boost::math::differentiation::promote<X, Y, Z> cxM     = 1.0;
        boost::math::differentiation::promote<X, Y, Z> cxMplus = 0.0;
        auto somx2 = sqrt(1.0 - z1 * z1);
        Z    fact  = 1.0;
        for (auto i = 0; i < m; i++)
          {
            cxM  = -cxM * fact * somx2;
            fact = fact + 2.0;
          }
        auto cx = cxM;
        if (m != l)
          {
            auto cxMPlus1 = z1 * (2 * m + 1) * cxM;
            cx            = cxMPlus1;

            auto cxPrev     = cxMPlus1;
            auto cxPrevPrev = cxM;
            for (auto i = m + 2; i < l + 1; i++)
              {
                cx = ((2 * i - 1) * z1 * cxPrev + (-i - m + 1) * cxPrevPrev) /
                     (i - m);
                cxPrevPrev = cxPrev;
                cxPrev     = cx;
              }
          }
        return pow((-1.0), m) * cx;
      }

      template <typename X, typename Y, typename Z>
      boost::math::differentiation::promote<X, Y, Z>
      f(const X &                  x,
        const Y &                  y,
        const Z &                  z,
        const std::vector<double> &coeffVec,
        const double               smoothness,
        const double               cutoff,
        const int                  l,
        const int                  m,
        const double               polarAngleTolerance)
      {
        auto r            = sqrt(x * x + y * y + z * z);
        auto radValue     = radVal(x, y, z, coeffVec);
        auto smoothCutoff = cutOff(x, y, z, smoothness, cutoff);
        auto lege         = legendre(x, y, z, l, m);
        auto theta        = acos(z / r);
        auto phi          = atan2(y, x);
        if ((double)x == 0)
          phi = acos(x / sqrt(x * x + y * y));
        if ((double)y == 0)
          phi = asin(y / sqrt(x * x + y * y));

        //
        // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
        // If yes, assign phi = 0.0.
        // NOTE: In case theta = 0 or PI, phi is undetermined. The actual
        // value of phi doesn't matter in computing the enriched function
        // value or its gradient. We assign phi = 0.0 here just as a dummy
        // value
        //
        if (std::abs((double)(theta)-0.0) < polarAngleTolerance &&
            std::abs((double)(theta)-M_PI) < polarAngleTolerance)
          boost::math::differentiation::promote<X, Y> phi = 0.0;

        boost::math::differentiation::promote<X, Y, Z> retValue;
        auto cutoffmax = cutoff + cutoff / smoothness;
        if ((double)r <= cutoff)
          {
            if (m > 0)
              {
                retValue = radValue * Clm(l, m) * Dm(m) * lege * cos(m * phi);
              }
            if (m < 0)
              {
                retValue =
                  radValue * Clm(l, m) * Dm(m) * lege * sin(std::abs(m) * phi);
              }
            if (m == 0)
              {
                retValue = radValue * Clm(l, m) * Dm(m) * lege;
              }
          }
        else if ((double)r > cutoff && (double)r <= cutoffmax)
          {
            if (m > 0)
              {
                retValue = radValue * smoothCutoff * Clm(l, m) * Dm(m) * lege *
                           cos(m * phi);
              }
            if (m < 0)
              {
                retValue = radValue * smoothCutoff * Clm(l, m) * Dm(m) * lege *
                           sin(std::abs(m) * phi);
              }
            if (m == 0)
              {
                retValue = radValue * smoothCutoff * Clm(l, m) * Dm(m) * lege;
              }
          }
        else
          {
            retValue = 0;
          }
        return retValue;
      }
    } // namespace BoostAutoDiffInternal

    double
    getValueBoostAutoDiff(const utils::Point &       point,
                          const utils::Point &       origin,
                          const std::vector<double> &coeffVec,
                          const double               smoothness,
                          const double               cutoff,
                          const int                  l,
                          const int                  m,
                          const double               polarAngleTolerance)
    {
      constexpr unsigned Nx = 0; // Max order of derivative to calculate for x
      constexpr unsigned Ny = 0; // Max order of derivative to calculate for y
      constexpr unsigned Nz = 0; // Max order of derivative to calculate for z
      // Declare 4 independent variables together into a std::tuple.
      double     r1 = point[0] - origin[0];
      double     r2 = point[1] - origin[1];
      double     r3 = point[2] - origin[2];
      auto const variables =
        boost::math::differentiation::make_ftuple<double, Nx, Ny, Nz>(r1,
                                                                      r2,
                                                                      r3);
      auto const &x = std::get<0>(variables);
      auto const &y = std::get<1>(variables);
      auto const &z = std::get<2>(variables);
      auto const  v = BoostAutoDiffInternal::f(
        x, y, z, coeffVec, smoothness, cutoff, l, m, polarAngleTolerance);

      return v.derivative(Nx, Ny, Nz);
    }

    std::vector<double>
    getGradientValueBoostAutoDiff(const utils::Point &       point,
                                  const utils::Point &       origin,
                                  const std::vector<double> &coeffVec,
                                  const double               smoothness,
                                  const double               cutoff,
                                  const int                  l,
                                  const int                  m,
                                  const double polarAngleTolerance)
    {
      std::vector<double> retValue;
      retValue.resize(3);
      constexpr unsigned Nx = 1; // Max order of derivative to calculate for x
      constexpr unsigned Ny = 1; // Max order of derivative to calculate for y
      constexpr unsigned Nz = 1; // Max order of derivative to calculate for z
      // Declare 4 independent variables together into a std::tuple.
      double     r1 = point[0] - origin[0];
      double     r2 = point[1] - origin[1];
      double     r3 = point[2] - origin[2];
      auto const variables =
        boost::math::differentiation::make_ftuple<double, Nx, Ny, Nz>(r1,
                                                                      r2,
                                                                      r3);
      auto const &x = std::get<0>(variables);
      auto const &y = std::get<1>(variables);
      auto const &z = std::get<2>(variables);
      auto const  v = BoostAutoDiffInternal::f(
        x, y, z, coeffVec, smoothness, cutoff, l, m, polarAngleTolerance);
      retValue[0] = v.derivative(1, 0, 0); // df/dx1
      retValue[1] = v.derivative(0, 1, 0); // df/dx2
      retValue[2] = v.derivative(0, 0, 1); // df/dx3
      return retValue;
    }

    std::vector<double>
    getHessianValueBoostAutoDiff(const utils::Point &       point,
                                 const utils::Point &       origin,
                                 const std::vector<double> &coeffVec,
                                 const double               smoothness,
                                 const double               cutoff,
                                 const int                  l,
                                 const int                  m,
                                 const double               polarAngleTolerance)
    {
      std::vector<double> retValue;
      retValue.resize(9);
      constexpr unsigned Nx = 2; // Max order of derivative to calculate for x
      constexpr unsigned Ny = 2; // Max order of derivative to calculate for y
      constexpr unsigned Nz = 2; // Max order of derivative to calculate for z
      // Declare 4 independent variables together into a std::tuple.
      double     r1 = point[0] - origin[0];
      double     r2 = point[1] - origin[1];
      double     r3 = point[2] - origin[2];
      auto const variables =
        boost::math::differentiation::make_ftuple<double, Nx, Ny, Nz>(r1,
                                                                      r2,
                                                                      r3);
      auto const &x = std::get<0>(variables);
      auto const &y = std::get<1>(variables);
      auto const &z = std::get<2>(variables);
      auto const  v = BoostAutoDiffInternal::f(
        x, y, z, coeffVec, smoothness, cutoff, l, m, polarAngleTolerance);
      retValue[0] = v.derivative(2, 0, 0); // d^2f/dx1^2
      retValue[1] = v.derivative(1, 1, 0); // d^2f/dx1dx2
      retValue[2] = v.derivative(1, 0, 1); // d^2f/dx1dx3
      retValue[3] = retValue[1];           // d^2f/dx2dx1
      retValue[4] = v.derivative(0, 2, 0); // d^2f/dx2^2
      retValue[5] = v.derivative(0, 1, 1); // d^2f/dx2dx3
      retValue[6] = retValue[2];           // d^2f/dx3dx1
      retValue[7] = retValue[5];           // d^2f/dx3dx2
      retValue[8] = v.derivative(0, 0, 2); // d^2f/dx3^2
      return retValue;
    }
  } // namespace atoms
} // namespace dftefe
