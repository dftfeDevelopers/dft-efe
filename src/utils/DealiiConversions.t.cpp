#include "DealiiConversions.h"
#include "Exceptions.h"

namespace dftefe
{
  namespace utils
  {
    template <unsigned int dim>
    void
    convertToDealiiPoint(const utils::Point &        point,
                         dealii::Point<dim, double> &outputDealiiPoint)
    {
      DFTEFE_AssertWithMsg(
        dim == point.size(),
        "Mismatch of dimension for dealii and the dimension of the point");
      for (unsigned int i = 0; i < dim; ++i)
        outputDealiiPoint[i] = point[i];
    }

    template <unsigned int dim>
    void
    convertToDealiiPoint(const std::vector<double> & v,
                         dealii::Point<dim, double> &outputDealiiPoint)
    {
      DFTEFE_AssertWithMsg(
        dim == v.size(),
        "Mismatch of dimension for dealii and the dimension of the vector");
      for (unsigned int i = 0; i < dim; ++i)
        outputDealiiPoint[i] = v[i];
    }

    template <unsigned int dim>
    void
    convertToDftefePoint(const dealii::Point<dim, double> &dealiiPoint,
                         Point &                           outputDftefePoint)
    {
      outputDftefePoint = Point(dim);
      for (unsigned int i = 0; i < dim; ++i)
        outputDftefePoint[i] = dealiiPoint[i];
    }

  } // namespace utils
} // namespace dftefe
