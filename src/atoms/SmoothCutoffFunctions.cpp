#include <cmath>
#include <vector>
#include <atoms/SmoothCutoffFunctions.h>

namespace dftefe
{
  namespace atoms
  {
    // double
    // f1(const double x)
    // {
    //   if (x <= 0.0)
    //     return 0.0;
    //   else
    //     return exp(-1.0 / x);
    // }

    // double
    // f1Der(const double x)
    // {
    //   return f1(x) / (x * x);
    // }

    // double
    // f2(const double x)
    // {
    //   return (f1(x) / (f1(x) + f1(1 - x)));
    // }

    // double
    // f2Der(const double x, const double tolerance)
    // {
    //   if (fabs(x - 0.0) < tolerance || fabs(1 - x) < tolerance)
    //     return 0.0;
    //   else
    //     return ((f1Der(x) * f1(1 - x) + f1(x) * f1Der(1 - x)) /
    //             (pow(f1(x) + f1(1 - x), 2.0)));
    // }

    // double
    // Y(const double x, const double r, const double d)
    // {
    //   return (1 - d * (x - r) / r);
    // }

    // double
    // YDer(const double x, const double r, const double d)
    // {
    //   return (-d / r);
    // }

    // double
    // smoothCutoffValue(const double x, const double r, const double d)
    // {
    //   const double y = Y(x, r, d);
    //   return pow(f2(y), 1.0);
    // }

    // double
    // smoothCutoffDerivative(const double x,
    //                        const double r,
    //                        const double d,
    //                        const double tolerance)
    // {
    //   const double y = Y(x, r, d);
    //   return f2Der(y, tolerance) * YDer(x, r, d);
    // }

    // double
    // smoothCutoffValue(const double x, const double r, const double d)
    // {
    //   // Y(x)
    //   const double y = (1.0 - d * (x - r) / r);
    //   // f1(y)
    //   const double f1_y =
    //     (y <= 0.0) ? 0.0 : std::exp(-1.0 / y);
    //   // f1(1 - y)
    //   const double one_minus_y = 1.0 - y;
    //   const double f1_1my =
    //     (one_minus_y <= 0.0) ? 0.0 : std::exp(-1.0 / one_minus_y);
    //   // f2(y)
    //   return f1_y / (f1_y + f1_1my);
    // }

    // double
    // smoothCutoffDerivative(const double x,
    //                        const double r,
    //                        const double d,
    //                        const double tolerance)
    // {
    //   // Y(x)
    //   const double y = (1.0 - d * (x - r) / r);
    //   // Boundary protection
    //   if (std::fabs(y) < tolerance || std::fabs(1.0 - y) < tolerance)
    //     return 0.0;
    //   // f1(y)
    //   const double f1_y = (y <= 0.0) ? 0.0 : std::exp(-1.0 / y);
    //   // f1(1 - y)
    //   const double one_minus_y = 1.0 - y;
    //   const double f1_1my =
    //     (one_minus_y <= 0.0) ? 0.0 : std::exp(-1.0 / one_minus_y);
    //   // f1Der(y) = f1(y) / y^2
    //   const double f1Der_y = f1_y / (y * y);
    //   // f1Der(1 - y)
    //   const double f1Der_1my =
    //     f1_1my / (one_minus_y * one_minus_y);
    //   // f2 derivative numerator & denominator
    //   const double denom = f1_y + f1_1my;
    //   const double f2Der =
    //     (f1Der_y * f1_1my + f1_y * f1Der_1my) /
    //     (denom * denom);
    //   // YDer = -d / r
    //   return f2Der * (-d / r);
    // }

    //=========================================================================
    // smoothCutoffValue<HOST>
    //=========================================================================
    template <>
    void
    smoothCutoffValue<utils::MemorySpace::HOST>(
      size_type             numPoints,
      const double *        x,
      const double          r,
      const double          d,
      double *              out,
      utils::deviceStream_t streamId)
    {
      for (size_type i = 0; i < numPoints; i++)
        out[i] = smoothCutoffValue(x[i], r, d);
    }

    //=========================================================================
    // smoothCutoffDerivative<HOST>
    //=========================================================================
    template <>
    void
    smoothCutoffDerivative<utils::MemorySpace::HOST>(
      size_type             numPoints,
      const double *        x,
      const double          r,
      const double          d,
      const double          tolerance,
      double *              out,
      utils::deviceStream_t streamId)
    {
      for (size_type i = 0; i < numPoints; i++)
        out[i] = smoothCutoffDerivative(x[i], r, d, tolerance);
    }

  } // namespace atoms
} // namespace dftefe
