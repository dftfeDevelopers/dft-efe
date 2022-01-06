#ifndef dftefeCubicSpline1D_h
#define dftefeCubicSpline1D_h

#include "alglib/interpolation.h"
#include <vector>
#include <map>

namespace dftefe
{
  namespace utils
  {
    class CubicSpline1D
    {
    public:
      enum class BoundaryType
      {
        PERIODIC,
        PARABOLIC_TERMINATION,
        FIRST_DERIVATIVE,
        SECOND_DERIVATIVE
      };

      CubicSpline1D(
        const std::vector<double> &x,
        const std::vector<double> &y,
        const BoundaryType lBoundaryType  = BoundaryType::PARABOLIC_TERMINATION,
        const BoundaryType rBoundaryType  = BoundaryType::PARABOLIC_TERMINATION,
        const double       lBoundaryValue = 0.0,
        const double       rBoundaryValue = 0.0);

      ~CubicSpline1D() = default;

      double
      getValue(const double &x) const;
      std::vector<double>
      getValue(const std::vector<double> &x) const;

      double
      getFirstDerivativeValue(const double &x) const;
      std::vector<double>
      getFirstDerivativeValue(const std::vector<double> &x) const;

      double
      getSecondDerivativeValue(const double &x) const;
      std::vector<double>
      getSecondDerivativeValue(const std::vector<double> &x) const;

    private:
      static std::map<BoundaryType, alglib::ae_int_t>
                                  d_boundaryTypeToAlglibIntegerMap;
      alglib::spline1dinterpolant d_spline;
      std::vector<double>         d_x;
      std::vector<double>         d_y;
      unsigned int                d_numPoints;
      BoundaryType                d_lBoundaryType;
      BoundaryType                d_rBoundaryType;
      double                      d_lBoundaryValue;
      double                      d_rBoundaryValue;
    };

  } // end of namespace utils


} // end of namespace dftefe

#endif // dftefeCubicSpline1D_h
