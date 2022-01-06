#include "CubicSpline1D.h"

namespace dftefe
{
  namespace utils
  {
    std::map<CubicSpline1D::BoundaryType, alglib::ae_int_t>
      CubicSpline1D::d_boundaryTypeToAlglibIntegerMap = {
        {BoundaryType::PERIODIC, -1},
        {BoundaryType::PARABOLIC_TERMINATION, 0},
        {BoundaryType::FIRST_DERIVATIVE, 1},
        {BoundaryType::SECOND_DERIVATIVE, 2}};

    CubicSpline1D::CubicSpline1D(const std::vector<double> &x,
                                 const std::vector<double> &y,
                                 const BoundaryType         lBoundaryType /*=
                                         BoundaryType::PARABOLIC_TERMINATION*/
                                 ,
                                 const BoundaryType rBoundaryType /*=
                                 BoundaryType::PARABOLIC_TERMINATION*/
                                 ,
                                 const double lBoundaryValue /*= 0.0*/,
                                 const double rBoundaryValue /*= 0.0*/)
      : d_x(x)
      , d_y(y)
      , d_numPoints(x.size())
      , d_lBoundaryType(lBoundaryType)
      , d_rBoundaryType(rBoundaryType)
      , d_lBoundaryValue(lBoundaryValue)
      , d_rBoundaryValue(rBoundaryValue)
    {
      alglib::real_1d_array alglibX;
      alglib::real_1d_array alglibY;
      alglibX.setcontent(d_numPoints, &d_x[0]);
      alglibY.setcontent(d_numPoints, &d_y[0]);

      alglib::spline1dbuildcubic(
        alglibX,
        alglibY,
        d_numPoints,
        d_boundaryTypeToAlglibIntegerMap[lBoundaryType],
        lBoundaryValue,
        d_boundaryTypeToAlglibIntegerMap[rBoundaryType],
        rBoundaryValue,
        d_spline);
    }

    double
    CubicSpline1D::getValue(const double &x) const
    {
      return alglib::spline1dcalc(d_spline, x);
    }

    std::vector<double>
    CubicSpline1D::getValue(const std::vector<double> &x) const
    {
      const unsigned int  N = x.size();
      std::vector<double> returnValue(N, 0.0);
      for (unsigned int i = 0; i < N; ++i)
        returnValue[i] = alglib::spline1dcalc(d_spline, x[i]);

      return returnValue;
    }

    double
    CubicSpline1D::getFirstDerivativeValue(const double &x) const
    {
      double s, dsDx, d2sDx2;
      alglib::spline1ddiff(d_spline, x, s, dsDx, d2sDx2);
      return dsDx;
    }

  } // end of namespace utils


} // end of namespace dftefe
