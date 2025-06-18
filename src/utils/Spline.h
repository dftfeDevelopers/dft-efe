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


#ifndef dftefeSpline_h
#define dftefeSpline_h

#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sstream>
#include <string>


// header file and we don't want to export symbols to the obj files
namespace dftefe
{
  namespace utils
  {
    // spline interpolation
    class Spline
    {
    public:
      // spline types
      enum spline_type
      {
        linear          = 10, // linear interpolation
        cspline         = 30, // cubic splines (classical C^2)
        cspline_hermite = 31  // cubic hermite splines (local, only C^1)
      };

      // boundary condition type for the spline end-points
      enum bd_type
      {
        first_deriv  = 1,
        second_deriv = 2
      };

    private:
      std::vector<double> d_x, d_y; // x,y coordinates of points
      // interpolation parameters
      // f(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
      // where a_i = y_i, or else it won't go through grid points
      std::vector<double> d_b, d_c, d_d; // spline coefficients
      double              d_c0;          // for left extrapolation
      spline_type         d_type;
      bd_type             d_left, d_right;
      double              d_left_value, d_right_value;
      bool                d_made_monotonic;
      bool                d_isSubdivPowerLawGrid;
      double              d_a, d_r;
      unsigned int        d_numSubDiv;
      void
      set_coeffs_from_b(); // calculate c_i, d_i from b_i
      size_t
      find_closest(double x) const; // closest idx so that d_x[idx]<=x

    public:
      // default constructor: set boundary condition to be zero curvature
      // at both ends, i.e. natural splines
      Spline();

      Spline(const std::vector<double> &X,
             const std::vector<double> &Y,
             const bool                 isSubdivPowerLawGrid = false,
             spline_type                type                 = cspline,
             bool                       make_monotonic       = false,
             bd_type                    left                 = first_deriv,
             double                     left_value           = 0.0,
             bd_type                    right                = first_deriv,
             double                     right_value          = 0.0);

      // modify boundary conditions: if called it must be before set_points()
      void
      set_boundary(bd_type left,
                   double  left_value,
                   bd_type right,
                   double  right_value);

      // set all data points (cubic_spline=false means linear interpolation)
      void
      set_points(const std::vector<double> &x,
                 const std::vector<double> &y,
                 spline_type                type = cspline);

      // adjust coefficients so that the spline becomes piecewise monotonic
      // where possible
      //   this is done by adjusting slopes at grid points by a non-negative
      //   factor and this will break C^2
      //   this can also break boundary conditions if adjustments need to
      //   be made at the boundary points
      // returns false if no adjustments have been made, true otherwise
      bool
      make_monotonic();

      // evaluates the spline at point x
      double
      operator()(double x) const;
      std::vector<double>
      coefficients(double x) const;
      double
      deriv(int order, double x) const;

      // returns the input data points
      std::vector<double>
      get_x() const
      {
        return d_x;
      }
      std::vector<double>
      get_y() const
      {
        return d_y;
      }
      double
      get_x_min() const
      {
        assert(!d_x.empty());
        return d_x.front();
      }
      double
      get_x_max() const
      {
        assert(!d_x.empty());
        return d_x.back();
      }

      // spline info string, i.e. spline type, boundary conditions etc.
      std::string
      info() const;
    };

    namespace splineInternal
    {
      // band matrix solver
      class band_matrix
      {
      private:
        std::vector<std::vector<double>> d_upper; // upper band
        std::vector<std::vector<double>> d_lower; // lower band
      public:
        band_matrix(){};                        // constructor
        band_matrix(int dim, int n_u, int n_l); // constructor
        ~band_matrix(){};                       // destructor
        void
        resize(int dim, int n_u, int n_l); // init with dim,n_u,n_l
        int
        dim() const; // matrix dimension
        int
        num_upper() const
        {
          return (int)d_upper.size() - 1;
        }
        int
        num_lower() const
        {
          return (int)d_lower.size() - 1;
        }
        // access operator
        double &
        operator()(int i, int j); // write
        double
        operator()(int i, int j) const; // read
        // we can store an additional diagonal (in d_lower)
        double &
        saved_diag(int i);
        double
        saved_diag(int i) const;
        void
        lu_decompose();
        std::vector<double>
        r_solve(const std::vector<double> &b) const;
        std::vector<double>
        l_solve(const std::vector<double> &b) const;
        std::vector<double>
        lu_solve(const std::vector<double> &b, bool is_lu_decomposed = false);
      };
    } // namespace splineInternal
  }   // namespace utils
} // namespace dftefe

#endif // dftefeSpline_h
