#include <cmath>
#include <vector>

namespace dftefe
{
  namespace atoms
  {
    ///////////////////////////////////////////////////////////////////////////
    ///////////// START OF SMOOTH CUTOFF FUNCTION RELATED FUNCTIONS ///////////
    ///////////////////////////////////////////////////////////////////////////
    double
    f1(const double x)
    {
      if (x <= 0.0)
        return 0.0;
      else
        return exp(-1.0 / x);
    }

    double
    f1Der(const double x)
    {
      return f1(x) / (x * x);
    }

    double
    f2(const double x)
    {
      return (f1(x) / (f1(x) + f1(1 - x)));
    }

    double
    f2Der(const double x, const double tolerance)
    {
      if (fabs(x - 0.0) < tolerance || fabs(1 - x) < tolerance)
        return 0.0;
      else
        return ((f1Der(x) * f1(1 - x) + f1(x) * f1Der(1 - x)) /
                (pow(f1(x) + f1(1 - x), 2.0)));
    }

    double
    Y(const double x, const double r, const double d)
    {
      return (1 - d * (x - r) / r);
    }

    double
    YDer(const double x, const double r, const double d)
    {
      return (-d / r);
    }

    double
    smoothCutoffValue(const double x, const double r, const double d)
    {
      const double y = Y(x, r, d);
      return pow(f2(y), 1.0);
    }

    double
    smoothCutoffDerivative(const double x,
                           const double r,
                           const double d,
                           const double tolerance)
    {
      const double y = Y(x, r, d);
      return f2Der(y, tolerance) * YDer(x, r, d);
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////// END OF SMOOTH CUTOFF FUNCTION RELATED FUNCTIONS ///////////
    ///////////////////////////////////////////////////////////////////////////
  } // namespace atoms
} // namespace dftefe
