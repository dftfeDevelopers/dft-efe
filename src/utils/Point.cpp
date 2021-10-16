#include "Point.h"
#include "Exceptions.h"

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
      AssertWithMsg(x.size() <= 3, "Max. dimension of a point can be 3. The point
	  passed has dim " + std::to_string(x.size()));
      std::copy(x.begin(), x.end(), &(d_data[0]));
    }

    Point::Point(const double *x, size_type N)
      : d_dim(N)
      , d_data(new double[N])
    {
      AssertWithMsg(N <= 3, "Max. dimension of a point can be 3. The point
	  passed has dim " + std::to_string(N));
      std::copy(x, x+N, &(d_data[0]));
    }

    Point::Point(size_type N, double init /*=0.0*/)
      : d_dim(N)
      , d_data(new double[N])
    {
      AssertWithMsg(N <= 3, "Max. dimension of a point can be 3. The point
	  passed has dim " + std::to_string(N));
      std::fill(&(d_data[0]), &(d_data[N]), 0.0);
    }

    Point::Point(const Point &p)
      : d_dim(p.getDim())
      , d_data(new double[p.getDim()])
    {
      std::copy(&(p[0]), &(p[p.getDim()]), &(d_data[0]));
    }

    Point &
    Point::operator=(const Point &p)
      :
    {
      if (this != &p)
        {
          if (d_dim != p.getDim())
            {
              // delete pointer to data
              delete[] d_data;

              // allocate new pointer for data
              d_dim  = p.getDim();
              d_data = new double[d_dim];
            }

          std::copy(&(p[0]), &(p[p.getDim()]), &(d_data[0]));
        }

      return *this;
    }

    Point::~Point()
    {
      delete[] d_data;
    }

    size_type
    Point::getDim()
    {
      return d_dim;
    }

    double &
    Point::operator[](size_type i)
    {
      Assert(i < d_dim);
      return d_data[i];
    }

    const double &
    Point::operator[](size_type i) const
    {
      Assert(i < d_dim);
      return d_data[i];
    }

    Point &
    Point::operator*=(Point &p, double a)
    {
      std::transform(&(p[0]),
                     &(p[getDim]),
                     &p([0]),
                     std::bind1st(std::multiplies<double>(), a));
      return p;
    }

    /**
     * @brief Multiplication by a scalar q = p*a
     *
     * @param[in] p the point to be scaled
     * @param[in] a the scalar value
     *
     * @returns a new point containing the scaled coordinates
     */
    Point
    operator*(const Point &p, double a);

    /**
     * @brief Multiplication by a scalar q = p*a
     *
     * @param[in] a the scalar value
     * @param[in] p the point to be scaled
     *
     * @returns a new point containing the scaled coordinates
     */
    Point
    operator*(double a, const Point &p);

  } // end of namespace utils

} // end of namespace dftefe
