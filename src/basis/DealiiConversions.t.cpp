#include <utils/Exceptions.h>

namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    void
    convertToDealiiPoint(const utils::Point &        point,
                         dealii::Point<dim, double> &dealiiPoint)
    {
      DFTEFE_AssertWithMsg(
        dim == point.size(),
        "Mismatch of dimension for dealii and the dimension of the point");
      std::copy(point.begin(), point.end(), dealiiPoint.begin_raw());
      // for (unsigned int i = 0; i < dim; ++i)
      //  outputDealiiPoint[i] = point[i];
    }

    template <unsigned int dim>
    void
    convertToDealiiPoint(const std::vector<utils::Point> &        points,
                         std::vector<dealii::Point<dim, double>> &dealiiPoints)
    {
      DFTEFE_AssertWithMsg(
        dim == points[0].size(),
        "Mismatch of dimension for dealii and the dimension of the point");
      const size_type numPoints = points.size();
      dealiiPoints.resize(numPoints);
      for (unsigned int j = 0; j < numPoints; j++)
        {
          std::copy(points[j].begin(),
                    points[j].end(),
                    dealiiPoints[j].begin_raw());
          // for (unsigned int i = 0; i < dim; ++i)
          //  dealiiPoints[j][i] = points[j][i];
        }
    }

    template <unsigned int dim>
    void
    convertToDealiiPoint(const std::vector<double> & v,
                         dealii::Point<dim, double> &dealiiPoint)
    {
      DFTEFE_AssertWithMsg(
        dim == v.size(),
        "Mismatch of dimension for dealii and the dimension of the vector");
      std::copy(v.begin(), v.end(), dealiiPoint.begin_raw());
      // for (unsigned int i = 0; i < dim; ++i)
      //  dealiiPoint[i] = v[i];
    }

    template <unsigned int dim>
    void
    convertToDftefePoint(const dealii::Point<dim, double> &dealiiPoint,
                         utils::Point &                    point)
    {
      point = utils::Point(dim);
      std::copy(dealiiPoint.begin_raw(), dealiiPoint.end_raw(), point.begin());
      // for (unsigned int i = 0; i < dim; ++i)
      //  point[i] = dealiiPoint[i];
    }

    template <unsigned int dim>
    void
    convertToDftefePoint(
      const std::vector<dealii::Point<dim, double>> &dealiiPoints,
      std::vector<utils::Point> &                    points)
    {
      const size_type numPoints = dealiiPoints.size();
      points.resize(numPoints, utils::Point(dim));
      for (unsigned int j = 0; j < numPoints; ++j)
        {
          std::copy(dealiiPoints[j].begin_raw(),
                    dealiiPoints[j].end_raw(),
                    points[j].begin());
        }
    }

        template <unsigned int dim>
    void
    convertToDftefePoint(
      const std::map<global_size_type, dealii::Point<dim, double>>
        &                                       dealiiPoints,
      std::map<global_size_type, utils::Point> &points)
    {
      auto         iter = dealiiPoints.begin();
      utils::Point pointTmp(dim);
      for (; iter != dealiiPoints.end(); iter++)
        {
          std::copy((iter->second).begin_raw(),
                    (iter->second).end_raw(),
                    pointTmp.begin());
          // points[iter->first] = pointTmp;
          points.insert({iter->first, pointTmp});
        }
    }

  } // namespace basis
} // namespace dftefe
