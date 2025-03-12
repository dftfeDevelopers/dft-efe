/*
 * spline.h
 *
 * simple cubic spline interpolation library without external
 * dependencies
 *
 * ---------------------------------------------------------------------
 * Copyright (C) 2011, 2014, 2016, 2021 Tino Kluge (ttk448 at gmail.com)
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * ---------------------------------------------------------------------
 *
 */

#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <string>
#include "Spline.h"
#include <utils/Exceptions.h>

namespace dftefe
{
  namespace utils
  {
    namespace SplineInternal
    {
      double
      f(double x, const double alpha, const double p)
      {
        return (x - 1) / log(x) * log(p) + (x - 1) - (x * p - 1) * alpha;
      }

      double
      bisection(const double alpha, const double p)
      {
        double a = 1 + 1e-10;
        double b = 1e1;
        while (b - a > 1e-10)
          {
            double c = (a + b) / 2;
            if (std::abs(f(c, alpha, p)) < 1e-6)
              {
                return c;
              }
            else if (f(a, alpha, p) * f(c, alpha, p) < 0)
              {
                b = c;
              }
            else
              {
                a = c;
              }
          }
        return (a + b) / 2;
      }

      void
      getSubdivPowerLawGridParams(const std::vector<double> &X,
                                  double &                   a,
                                  double &                   r,
                                  unsigned int &             numSubDiv)
      {
        unsigned int N = X.size();
        if (N < 2)
          utils::throwException(
            false, "Number of points is < 2 for getSubdivPowerLawGridParams()");
        double p = (X[N - 1] - X[N - 2]) / (X[1] - X[0]);
        if (!(p >= 1.0 || std::abs(p - 1.0) < 1e-6))
          utils::throwException(
            false,
            "X grid should have GP with r>=1 in getSubdivPowerLawGridParams(), given r^(n-1) = " +
              std::to_string(p));
        if (std::abs(p - 1.0) < 1e-6)
          {
            r         = 1;
            a         = X[1] - X[0];
            numSubDiv = 1;
            return;
          }
        double       q     = X[1] - X[0];
        double       s     = X[N - 1] - X[0];
        const double alpha = q * (N - 1) / (s);
        r                  = bisection(alpha, p);
        unsigned int n     = std::round(alpha * (r * p - 1) / (r - 1));
        numSubDiv          = (N - 1) / n;
        r                  = std::pow(p * 1.0, 1.0 / (n - 1));
        a                  = numSubDiv * q;
      }
    } // namespace SplineInternal
    // spline implementation
    // -----------------------

    // default constructor: set boundary condition to be zero curvature
    // at both ends, i.e. natural splines
    Spline::Spline()
      : d_type(cspline)
      , d_left(second_deriv)
      , d_right(second_deriv)
      , d_left_value(0.0)
      , d_right_value(0.0)
      , d_made_monotonic(false)
      , d_isSubdivPowerLawGrid(false)
    {
      ;
    }
    Spline::Spline(const std::vector<double> &X,
                   const std::vector<double> &Y,
                   const bool                 isSubdivPowerLawGrid,
                   spline_type                type,
                   bool                       make_monotonic,
                   bd_type                    left,
                   double                     left_value,
                   bd_type                    right,
                   double                     right_value)
      : d_type(type)
      , d_left(left)
      , d_right(right)
      , d_left_value(left_value)
      , d_right_value(right_value)
      , d_made_monotonic(false) // false correct here: make_monotonic() sets it
      , d_isSubdivPowerLawGrid(isSubdivPowerLawGrid)
    {
      this->set_points(X, Y, d_type);
      if (d_made_monotonic)
        {
          this->make_monotonic();
        }
      if (d_isSubdivPowerLawGrid == true)
        {
          SplineInternal::getSubdivPowerLawGridParams(X, d_a, d_r, d_numSubDiv);
          // std::cout << d_a << "\t" << d_r << "\t" << d_numSubDiv << "\n";
        }
    }

    void
    Spline::set_boundary(Spline::bd_type left,
                         double          left_value,
                         Spline::bd_type right,
                         double          right_value)
    {
      assert(d_x.size() == 0); // set_points() must not have happened yet
      d_left        = left;
      d_right       = right;
      d_left_value  = left_value;
      d_right_value = right_value;
    }


    void
    Spline::set_coeffs_from_b()
    {
      assert(d_x.size() == d_y.size());
      assert(d_x.size() == d_b.size());
      assert(d_x.size() > 2);
      size_t n = d_b.size();
      if (d_c.size() != n)
        d_c.resize(n);
      if (d_d.size() != n)
        d_d.resize(n);

      for (size_t i = 0; i < n - 1; i++)
        {
          const double h = d_x[i + 1] - d_x[i];
          // from continuity and differentiability condition
          d_c[i] =
            (3.0 * (d_y[i + 1] - d_y[i]) / h - (2.0 * d_b[i] + d_b[i + 1])) / h;
          // from differentiability condition
          d_d[i] = ((d_b[i + 1] - d_b[i]) / (3.0 * h) - 2.0 / 3.0 * d_c[i]) / h;
        }

      // for left extrapolation coefficients
      d_c0 = (d_left == first_deriv) ? 0.0 : d_c[0];
    }

    void
    Spline::set_points(const std::vector<double> &x,
                       const std::vector<double> &y,
                       spline_type                type)
    {
      assert(x.size() == y.size());
      assert(x.size() > 2);
      d_type           = type;
      d_made_monotonic = false;
      d_x              = x;
      d_y              = y;
      int n            = (int)x.size();
      // check strict monotonicity of input vector x
      for (int i = 0; i < n - 1; i++)
        {
          assert(d_x[i] < d_x[i + 1]);
        }


      if (type == linear)
        {
          // linear interpolation
          d_d.resize(n);
          d_c.resize(n);
          d_b.resize(n);
          for (int i = 0; i < n - 1; i++)
            {
              d_d[i] = 0.0;
              d_c[i] = 0.0;
              d_b[i] = (d_y[i + 1] - d_y[i]) / (d_x[i + 1] - d_x[i]);
            }
          // ignore boundary conditions, set slope equal to the last segment
          d_b[n - 1] = d_b[n - 2];
          d_c[n - 1] = 0.0;
          d_d[n - 1] = 0.0;
        }
      else if (type == cspline)
        {
          // classical cubic splines which are C^2 (twice cont differentiable)
          // this requires solving an equation system

          // setting up the matrix and right hand side of the equation system
          // for the parameters b[]
          splineInternal::band_matrix A(n, 1, 1);
          std::vector<double>         rhs(n);
          for (int i = 1; i < n - 1; i++)
            {
              A(i, i - 1) = 1.0 / 3.0 * (x[i] - x[i - 1]);
              A(i, i)     = 2.0 / 3.0 * (x[i + 1] - x[i - 1]);
              A(i, i + 1) = 1.0 / 3.0 * (x[i + 1] - x[i]);
              rhs[i]      = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
                       (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            }
          // boundary conditions
          if (d_left == Spline::second_deriv)
            {
              // 2*c[0] = f''
              A(0, 0) = 2.0;
              A(0, 1) = 0.0;
              rhs[0]  = d_left_value;
            }
          else if (d_left == Spline::first_deriv)
            {
              // b[0] = f', needs to be re-expressed in terms of c:
              // (2c[0]+c[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
              A(0, 0) = 2.0 * (x[1] - x[0]);
              A(0, 1) = 1.0 * (x[1] - x[0]);
              rhs[0]  = 3.0 * ((y[1] - y[0]) / (x[1] - x[0]) - d_left_value);
            }
          else
            {
              assert(false);
            }
          if (d_right == Spline::second_deriv)
            {
              // 2*c[n-1] = f''
              A(n - 1, n - 1) = 2.0;
              A(n - 1, n - 2) = 0.0;
              rhs[n - 1]      = d_right_value;
            }
          else if (d_right == Spline::first_deriv)
            {
              // b[n-1] = f', needs to be re-expressed in terms of c:
              // (c[n-2]+2c[n-1])(x[n-1]-x[n-2])
              // = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
              A(n - 1, n - 1) = 2.0 * (x[n - 1] - x[n - 2]);
              A(n - 1, n - 2) = 1.0 * (x[n - 1] - x[n - 2]);
              rhs[n - 1]      = 3.0 * (d_right_value - (y[n - 1] - y[n - 2]) /
                                                    (x[n - 1] - x[n - 2]));
            }
          else
            {
              assert(false);
            }

          // solve the equation system to obtain the parameters c[]
          d_c = A.lu_solve(rhs);

          // calculate parameters b[] and d[] based on c[]
          d_d.resize(n);
          d_b.resize(n);
          for (int i = 0; i < n - 1; i++)
            {
              d_d[i] = 1.0 / 3.0 * (d_c[i + 1] - d_c[i]) / (x[i + 1] - x[i]);
              d_b[i] =
                (y[i + 1] - y[i]) / (x[i + 1] - x[i]) -
                1.0 / 3.0 * (2.0 * d_c[i] + d_c[i + 1]) * (x[i + 1] - x[i]);
            }
          // for the right extrapolation coefficients (zero cubic term)
          // f_{n-1}(x) = y_{n-1} + b*(x-x_{n-1}) + c*(x-x_{n-1})^2
          double h = x[n - 1] - x[n - 2];
          // d_c[n-1] is determined by the boundary condition
          d_d[n - 1] = 0.0;
          d_b[n - 1] = 3.0 * d_d[n - 2] * h * h + 2.0 * d_c[n - 2] * h +
                       d_b[n - 2]; // = f'_{n-2}(x_{n-1})
          if (d_right == first_deriv)
            d_c[n - 1] = 0.0; // force linear extrapolation
        }
      else if (type == cspline_hermite)
        {
          // hermite cubic splines which are C^1 (cont. differentiable)
          // and derivatives are specified on each grid point
          // (here we use 3-point finite differences)
          d_b.resize(n);
          d_c.resize(n);
          d_d.resize(n);
          // set b to match 1st order derivative finite difference
          for (int i = 1; i < n - 1; i++)
            {
              const double h  = d_x[i + 1] - d_x[i];
              const double hl = d_x[i] - d_x[i - 1];
              d_b[i]          = -h / (hl * (hl + h)) * d_y[i - 1] +
                       (h - hl) / (hl * h) * d_y[i] +
                       hl / (h * (hl + h)) * d_y[i + 1];
            }
          // boundary conditions determine b[0] and b[n-1]
          if (d_left == first_deriv)
            {
              d_b[0] = d_left_value;
            }
          else if (d_left == second_deriv)
            {
              const double h = d_x[1] - d_x[0];
              d_b[0]         = 0.5 * (-d_b[1] - 0.5 * d_left_value * h +
                              3.0 * (d_y[1] - d_y[0]) / h);
            }
          else
            {
              assert(false);
            }
          if (d_right == first_deriv)
            {
              d_b[n - 1] = d_right_value;
              d_c[n - 1] = 0.0;
            }
          else if (d_right == second_deriv)
            {
              const double h = d_x[n - 1] - d_x[n - 2];
              d_b[n - 1]     = 0.5 * (-d_b[n - 2] + 0.5 * d_right_value * h +
                                  3.0 * (d_y[n - 1] - d_y[n - 2]) / h);
              d_c[n - 1]     = 0.5 * d_right_value;
            }
          else
            {
              assert(false);
            }
          d_d[n - 1] = 0.0;

          // parameters c and d are determined by continuity and
          // differentiability
          set_coeffs_from_b();
        }
      else
        {
          assert(false);
        }

      // for left extrapolation coefficients
      d_c0 = (d_left == first_deriv) ? 0.0 : d_c[0];
    }

    bool
    Spline::make_monotonic()
    {
      assert(d_x.size() == d_y.size());
      assert(d_x.size() == d_b.size());
      assert(d_x.size() > 2);
      bool      modified = false;
      const int n        = (int)d_x.size();
      // make sure: input data monotonic increasing --> b_i>=0
      //            input data monotonic decreasing --> b_i<=0
      for (int i = 0; i < n; i++)
        {
          int im1 = std::max(i - 1, 0);
          int ip1 = std::min(i + 1, n - 1);
          if (((d_y[im1] <= d_y[i]) && (d_y[i] <= d_y[ip1]) && d_b[i] < 0.0) ||
              ((d_y[im1] >= d_y[i]) && (d_y[i] >= d_y[ip1]) && d_b[i] > 0.0))
            {
              modified = true;
              d_b[i]   = 0.0;
            }
        }
      // if input data is monotonic (b[i], b[i+1], avg have all the same sign)
      // ensure a sufficient criteria for monotonicity is satisfied:
      //     sqrt(b[i]^2+b[i+1]^2) <= 3 |avg|, with avg=(y[i+1]-y[i])/h,
      for (int i = 0; i < n - 1; i++)
        {
          double h   = d_x[i + 1] - d_x[i];
          double avg = (d_y[i + 1] - d_y[i]) / h;
          if (avg == 0.0 && (d_b[i] != 0.0 || d_b[i + 1] != 0.0))
            {
              modified   = true;
              d_b[i]     = 0.0;
              d_b[i + 1] = 0.0;
            }
          else if ((d_b[i] >= 0.0 && d_b[i + 1] >= 0.0 && avg > 0.0) ||
                   (d_b[i] <= 0.0 && d_b[i + 1] <= 0.0 && avg < 0.0))
            {
              // input data is monotonic
              double r = sqrt(d_b[i] * d_b[i] + d_b[i + 1] * d_b[i + 1]) /
                         std::fabs(avg);
              if (r > 3.0)
                {
                  // sufficient criteria for monotonicity: r<=3
                  // adjust b[i] and b[i+1]
                  modified = true;
                  d_b[i] *= (3.0 / r);
                  d_b[i + 1] *= (3.0 / r);
                }
            }
        }

      if (modified == true)
        {
          set_coeffs_from_b();
          d_made_monotonic = true;
        }

      return modified;
    }

    // return the closest idx so that d_x[idx] <= x (return 0 if x<d_x[0])
    size_t
    Spline::find_closest(double x) const
    {
      if (d_isSubdivPowerLawGrid == true)
        {
          size_t       idx = 0;
          unsigned int n = 0, subId = 0;
          if (x > d_x.back())
            {
              idx = d_x.size() - 1;
              return idx;
            }
          else if (x < d_x[0])
            {
              idx = 0;
              return idx;
            }
          else
            {
              n     = std::abs(d_r - 1.0) < 1e-6 ?
                        0 :
                        std::floor(
                      std::abs(log(x * (d_r - 1) / d_a + 1.0) / log(d_r)));
              subId = std::abs(d_r - 1.0) < 1e-6 ?
                        std::floor(x / d_a) :
                        std::floor(d_numSubDiv *
                                   (x - d_a * (std::pow(d_r, 1.0 * n) - 1.0) /
                                          (d_r - 1.0)) /
                                   (d_a * std::pow(d_r, 1.0 * n)));
              idx   = n * d_numSubDiv + subId;
              return idx;
            }

          /**
          std::vector<double>::const_iterator it;
          it          = std::upper_bound(d_x.begin(), d_x.end(), x); // *it > x
          size_t idx1 = std::max(int(it - d_x.begin()) - 1, 0); // d_x[idx] <= x

          utils::throwException(idx == idx1,
                                "idx is incorrect found = " +
                                  std::to_string(idx) +
                                  " n= " + std::to_string(n) +
                                  " subId= " + std::to_string(subId) +
                                  " actual = " + std::to_string(idx1) +
                                  "for x = " + std::to_string(x));
          **/
        }
      else
        {
          std::vector<double>::const_iterator it;
          it         = std::upper_bound(d_x.begin(), d_x.end(), x); // *it > x
          size_t idx = std::max(int(it - d_x.begin()) - 1, 0); // d_x[idx] <= x
          return idx;
        }
    }

    double
    Spline::operator()(double x) const
    {
      // polynomial evaluation using Horner's scheme
      // TODO: consider more numerically accurate algorithms, e.g.:
      //   - Clenshaw
      //   - Even-Odd method by A.C.R. Newbery
      //   - Compensated Horner Scheme
      size_t n   = d_x.size();
      size_t idx = find_closest(x);

      double h = x - d_x[idx];
      double interpol;
      if (x < d_x[0])
        {
          // extrapolation to the left
          interpol = (d_c0 * h + d_b[0]) * h + d_y[0];
        }
      else if (x > d_x[n - 1])
        {
          // extrapolation to the right
          interpol = (d_c[n - 1] * h + d_b[n - 1]) * h + d_y[n - 1];
        }
      else
        {
          // interpolation
          interpol = ((d_d[idx] * h + d_c[idx]) * h + d_b[idx]) * h + d_y[idx];
        }
      return interpol;
    }

    std::vector<double>
    Spline::coefficients(double x) const
    {
      std::vector<double> retValue;
      retValue.resize(5);
      size_t n   = d_x.size();
      size_t idx = find_closest(x);
      // interpolation parameters
      // f(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
      // where a_i = y_i, or else it won't go through grid points
      double closestX = d_x[idx];
      double interpol;
      if (x < d_x[0])
        {
          retValue[0] = closestX;
          retValue[1] = d_y[0];
          retValue[2] = d_b[0];
          retValue[3] = d_c0;
          retValue[4] = 0;
        }
      else if (x > d_x[n - 1])
        {
          retValue[0] = closestX;
          retValue[1] = d_y[n - 1];
          retValue[2] = d_b[n - 1];
          retValue[3] = d_c[n - 1];
          retValue[4] = 0;
        }
      else
        {
          retValue[0] = closestX;
          retValue[1] = d_y[idx];
          retValue[2] = d_b[idx];
          retValue[3] = d_c[idx];
          retValue[4] = d_d[idx];
        }
      return retValue;
    }

    double
    Spline::deriv(int order, double x) const
    {
      assert(order > 0);
      size_t n   = d_x.size();
      size_t idx = find_closest(x);

      double h = x - d_x[idx];
      double interpol;
      if (x < d_x[0])
        {
          // extrapolation to the left
          switch (order)
            {
              case 1:
                interpol = 2.0 * d_c0 * h + d_b[0];
                break;
              case 2:
                interpol = 2.0 * d_c0;
                break;
              default:
                interpol = 0.0;
                break;
            }
        }
      else if (x > d_x[n - 1])
        {
          // extrapolation to the right
          switch (order)
            {
              case 1:
                interpol = 2.0 * d_c[n - 1] * h + d_b[n - 1];
                break;
              case 2:
                interpol = 2.0 * d_c[n - 1];
                break;
              default:
                interpol = 0.0;
                break;
            }
        }
      else
        {
          // interpolation
          switch (order)
            {
              case 1:
                interpol = (3.0 * d_d[idx] * h + 2.0 * d_c[idx]) * h + d_b[idx];
                break;
              case 2:
                interpol = 6.0 * d_d[idx] * h + 2.0 * d_c[idx];
                break;
              case 3:
                interpol = 6.0 * d_d[idx];
                break;
              default:
                interpol = 0.0;
                break;
            }
        }
      return interpol;
    }

    std::string
    Spline::info() const
    {
      std::stringstream ss;
      ss << "type " << d_type << ", left boundary deriv " << d_left << " = ";
      ss << d_left_value << ", right boundary deriv " << d_right << " = ";
      ss << d_right_value << std::endl;
      if (d_made_monotonic)
        {
          ss << "(spline has been adjusted for piece-wise monotonicity)";
        }
      return ss.str();
    }


    namespace splineInternal
    {
      // band_matrix implementation
      // -------------------------

      band_matrix::band_matrix(int dim, int n_u, int n_l)
      {
        resize(dim, n_u, n_l);
      }
      void
      band_matrix::resize(int dim, int n_u, int n_l)
      {
        assert(dim > 0);
        assert(n_u >= 0);
        assert(n_l >= 0);
        d_upper.resize(n_u + 1);
        d_lower.resize(n_l + 1);
        for (size_t i = 0; i < d_upper.size(); i++)
          {
            d_upper[i].resize(dim);
          }
        for (size_t i = 0; i < d_lower.size(); i++)
          {
            d_lower[i].resize(dim);
          }
      }
      int
      band_matrix::dim() const
      {
        if (d_upper.size() > 0)
          {
            return d_upper[0].size();
          }
        else
          {
            return 0;
          }
      }


      // defines the new operator (), so that we can access the elements
      // by A(i,j), index going from i=0,...,dim()-1
      double &
      band_matrix::operator()(int i, int j)
      {
        int k = j - i; // what band is the entry
        assert((i >= 0) && (i < dim()) && (j >= 0) && (j < dim()));
        assert((-num_lower() <= k) && (k <= num_upper()));
        // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if (k >= 0)
          return d_upper[k][i];
        else
          return d_lower[-k][i];
      }
      double
      band_matrix::operator()(int i, int j) const
      {
        int k = j - i; // what band is the entry
        assert((i >= 0) && (i < dim()) && (j >= 0) && (j < dim()));
        assert((-num_lower() <= k) && (k <= num_upper()));
        // k=0 -> diagonal, k<0 lower left part, k>0 upper right part
        if (k >= 0)
          return d_upper[k][i];
        else
          return d_lower[-k][i];
      }
      // second diag (used in LU decomposition), saved in d_lower
      double
      band_matrix::saved_diag(int i) const
      {
        assert((i >= 0) && (i < dim()));
        return d_lower[0][i];
      }
      double &
      band_matrix::saved_diag(int i)
      {
        assert((i >= 0) && (i < dim()));
        return d_lower[0][i];
      }

      // LR-Decomposition of a band matrix
      void
      band_matrix::lu_decompose()
      {
        int    i_max, j_max;
        int    j_min;
        double x;

        // preconditioning
        // normalize column i so that a_ii=1
        for (int i = 0; i < this->dim(); i++)
          {
            assert(this->operator()(i, i) != 0.0);
            this->saved_diag(i) = 1.0 / this->operator()(i, i);
            j_min               = std::max(0, i - this->num_lower());
            j_max = std::min(this->dim() - 1, i + this->num_upper());
            for (int j = j_min; j <= j_max; j++)
              {
                this->operator()(i, j) *= this->saved_diag(i);
              }
            this->operator()(i, i) = 1.0; // prevents rounding errors
          }

        // Gauss LR-Decomposition
        for (int k = 0; k < this->dim(); k++)
          {
            i_max = std::min(this->dim() - 1,
                             k + this->num_lower()); // num_lower not a mistake!
            for (int i = k + 1; i <= i_max; i++)
              {
                assert(this->operator()(k, k) != 0.0);
                x = -this->operator()(i, k) / this->operator()(k, k);
                this->operator()(i, k) = -x; // assembly part of L
                j_max = std::min(this->dim() - 1, k + this->num_upper());
                for (int j = k + 1; j <= j_max; j++)
                  {
                    // assembly part of R
                    this->  operator()(i, j) =
                      this->operator()(i, j) + x * this->operator()(k, j);
                  }
              }
          }
      }
      // solves Ly=b
      std::vector<double>
      band_matrix::l_solve(const std::vector<double> &b) const
      {
        assert(this->dim() == (int)b.size());
        std::vector<double> x(this->dim());
        int                 j_start;
        double              sum;
        for (int i = 0; i < this->dim(); i++)
          {
            sum     = 0;
            j_start = std::max(0, i - this->num_lower());
            for (int j = j_start; j < i; j++)
              sum += this->operator()(i, j) * x[j];
            x[i] = (b[i] * this->saved_diag(i)) - sum;
          }
        return x;
      }
      // solves Rx=y
      std::vector<double>
      band_matrix::r_solve(const std::vector<double> &b) const
      {
        assert(this->dim() == (int)b.size());
        std::vector<double> x(this->dim());
        int                 j_stop;
        double              sum;
        for (int i = this->dim() - 1; i >= 0; i--)
          {
            sum    = 0;
            j_stop = std::min(this->dim() - 1, i + this->num_upper());
            for (int j = i + 1; j <= j_stop; j++)
              sum += this->operator()(i, j) * x[j];
            x[i] = (b[i] - sum) / this->operator()(i, i);
          }
        return x;
      }

      std::vector<double>
      band_matrix::lu_solve(const std::vector<double> &b, bool is_lu_decomposed)
      {
        assert(this->dim() == (int)b.size());
        std::vector<double> x, y;
        if (is_lu_decomposed == false)
          {
            this->lu_decompose();
          }
        y = this->l_solve(b);
        x = this->r_solve(y);
        return x;
      }

    } // namespace splineInternal


  } // namespace utils


} // namespace dftefe
