#ifndef dftefeScalarSpatialFunction_h
#define dftefeScalarSpatialFunction_h
#include "Point.h"
#include "TypeConfig.h"
#include "Exceptions.h"
#include "MemorySpaceType.h"
#include <complex>
#include <vector>
namespace dftefe
{
  namespace utils
  {
    /**
     * @brief Abstract base for scalar functions of a spatial point.
     *
     * operator() provides the host interface via utils::Point.
     * eval<memorySpace>() is a non-virtual template that dispatches to
     * virtual protected evalHost() or evalDevice(). Derived classes
     * override evalHost/evalDevice; the base provides throwing defaults.
     */
    template <typename Q>
    class ScalarSpatialFunction
    {
    public:
      virtual ~ScalarSpatialFunction() = default;

      virtual Q
      operator()(const utils::Point &point) const = 0;

      virtual std::vector<Q>
      operator()(const std::vector<utils::Point> &points) const = 0;

      template <MemorySpace memorySpace>
      void
      eval(const size_type numPoints, const double *t, Q *q) const
      {
        if constexpr (memorySpace == MemorySpace::DEVICE)
          {
#ifdef DFTEFE_WITH_DEVICE
            evalDevice(numPoints, t, q);
#else
            utils::throwException(
              false,
              "eval<DEVICE>() called but DEVICE support not compiled.");
#endif
          }
        else
          {
            evalHost(numPoints, t, q);
          }
      }

    protected:
      virtual void
      evalHost(size_type numPoints, const double *t, Q *q) const
      {
        utils::throwException(
          false,
          "evalHost() not implemented for this ScalarSpatialFunction.");
      }

#ifdef DFTEFE_WITH_DEVICE
      virtual void
      evalDevice(size_type numPoints, const double *t, Q *q) const
      {
        utils::throwException(
          false,
          "evalDevice() not implemented for this ScalarSpatialFunction.");
      }
#endif
    };

    template <typename Q>
    using ScalarSpatialFunctionT = ScalarSpatialFunction<Q>;

    using ScalarSpatialFunctionReal = ScalarSpatialFunction<double>;

    using ScalarSpatialFunctionComplex =
      ScalarSpatialFunction<std::complex<double>>;

    /**
     * @brief Abstract base for scalar functions of a single real variable.
     */
    class ScalarSpatialFunctionReal1D
    {
    public:
      virtual ~ScalarSpatialFunctionReal1D() = default;

      virtual double
      operator()(double x) const = 0;

      virtual std::vector<double>
      operator()(const std::vector<double> &x) const = 0;

      template <MemorySpace memorySpace>
      void
      eval(const size_type numPoints, const double *t, double *q) const
      {
        if constexpr (memorySpace == MemorySpace::DEVICE)
          {
#ifdef DFTEFE_WITH_DEVICE
            evalDevice(numPoints, t, q);
#else
            utils::throwException(
              false,
              "eval<DEVICE>() called but DEVICE support not compiled.");
#endif
          }
        else
          {
            evalHost(numPoints, t, q);
          }
      }

    protected:
      virtual void
      evalHost(size_type numPoints, const double *t, double *q) const
      {
        utils::throwException(
          false,
          "evalHost() not implemented for this ScalarSpatialFunctionReal1D.");
      }

#ifdef DFTEFE_WITH_DEVICE
      virtual void
      evalDevice(size_type numPoints, const double *t, double *q) const
      {
        utils::throwException(
          false,
          "evalDevice() not implemented for this ScalarSpatialFunctionReal1D.");
      }
#endif
    };

  } // end of namespace utils
} // end of namespace dftefe
#endif // dftefeScalarSpatialFunction_h
