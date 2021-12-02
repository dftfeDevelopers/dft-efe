#include "ScalarFunction.h"
#include "TypeConfig.h"

namespace dftefe
{
  namespace utils
  {
    void
    ScalarFunction::getValue(const dftefe::utils::Point &realPoint,
                             double                      outputVal)
    {
      outputVal = 1.0;
    }

    void
    ScalarFunction::getValue(const std::vector<dftefe::utils::Point> &realPoint,
                             std::vector<double> &                    outputVal)
    {
      utils::throwException(
        outputVal.size() == realPoint.size(),
        "Size of output vector not equal to the number of points in getValAtPoint()");
      size_type nPoint = outputVal.size();
      for (size_type iPoint = 0; iPoint < nPoint; iPoint++)
        {
          outputVal[iPoint] = 1.0;
        }
    }

    void
    ScalarFunction::getValue(const dftefe::utils::Point &realPoint,
                             std::complex<double>        outputVal)
    {
      outputVal = 1.0;
    }

    void
    ScalarFunction::getValue(const std::vector<dftefe::utils::Point> &realPoint,
                             std::vector<std::complex<double>> &      outputVal)
    {
      utils::throwException(
        outputVal.size() == realPoint.size(),
        "Size of output vector not equal to the number of points in getValAtPoint()");
      size_type nPoint = outputVal.size();
      for (size_type iPoint = 0; iPoint < nPoint; iPoint++)
        {
          outputVal[iPoint] = 1.0;
        }
    }
  } // namespace utils

} // namespace dftefe
