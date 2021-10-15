#include "Point.h"

namespace dftefe
{
  namespace utils
  {
    Point::Point(double x)
      : d_dim(1)
      , d_data(new double[1])
    {
      d_data[0] = x;
    }

    Point::Point(double x, double y)
      : d_dim(2)
      , d_data(new double[2])
    {
      d_data[0] = x;
      d_data[1] = y;
    }

    Point::Point(double x, double y, double z)
      : d_dim(3)
      , d_data(new double[3])
    {
      d_data[0] = x;
      d_data[1] = y;
      d_data[2] = z;
    }

    Point::Point(const std::vector<double> &x)
      : d_dim(x.size())
      , d_data(new double[x.size()])
    {
      std::copy(x.begin(), x.end(), &(d_data[0]));
    }

    /**
     * @brief Constructor for an N-dimensional point where N=1,2,3
     * @param[in] x is pointer to coordinates of the point
     * @param[in] N dimension of the point
     * @throws exception if N > 3
     */
    Point(const double *x, size_type N);

    /**
     * @brief Constructor for an N-dimensional point where N=1,2,3
     * @param[in] N dimension of the point
     * @throws exception if N > 3
     */
    Point(size_type N);

    /**
     * @brief Constructor for an N-dimensional point where N=1,2,3
     * @param[in] N dimension of the point
     * @param[out] init initial value to be assigned to all components of the
     * point
     * @throws exception if N > 3
     */
    Point(size_type N, double init);

    /**
     * @brief Copy  Constructor for a point
     * @param[in] p Point object to copy from
     */
    Point(const Point &p);

    /**
     * @brief Destructor
     */
    ~Point();

  } // end of namespace utils

} // end of namespace dftefe
