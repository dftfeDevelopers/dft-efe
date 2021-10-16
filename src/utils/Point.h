#ifndef dftefePoint_h
#define dftefePoint_h
#include <utils/TypeConfig.h>
#include <vector>
namespace dftefe
{
  namespace utils
  {
    class Point
    {
    public:
      /**
       * @brief Constructor for a 1D point
       * @param[in] x coordinate of the point
       */
      Point(double x);

      /**
       * @brief Constructor for a 2D point
       * @param[in] x, y  are the coordinates of the point
       */
      Point(double x, double y);

      /**
       * @brief Constructor for a 3D point
       * @param[in] x, y, z  are the coordinates of the point
       */
      Point(double x, double y, double z);

      /**
       * @brief Constructor for an N-dimensional point where N=1,2,3
       * @param[in] x is reference to std::vector contaning the coordinates of
       * the point
       * @throws exception if N > 3
       */
      Point(const std::vector<double> &x);

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
       * @param[out] init initial value to be assigned to all components of the
       * point
       * @throws exception if N > 3
       */
      Point(size_type N, double init = 0.0);

      /**
       * @brief Copy  Constructor for a point
       * @param[in] p Point object to copy from
       */
      Point(const Point &p);

      /**
       * @brief Destructor
       */
      ~Point();


      /**
       * @brief Returns the dimension of the point
       * @param[out] dimension of the point
       */
      size_type
      getDim() const;

      /**
       * @brief Returns pointer to const double that stores underlying data of the point
       * @param[out] pointer to const double for the underlying data of the
       * point
       */
      const double *
      data() const;

      /**
       * @brief Operator overload for assignment q=p
       * @param[in] p the rhs point from which to copy
       *
       * @returns reference to the lhs point
       */
      Point &
      operator=(const Point &p);

      /**
       * @brief Operator to get a reference to a component of the point
       * @param[in] i is the index to the component of the point
       * @returns reference to the component of the point
       * @throws exception if i >= dimension of the point
       */
      double &
      operator[](size_type i);

      /**
       * @brief Operator to get a const reference to a component of the point
       * @param[in] i is the index to the component of the point
       * @returns const reference to the component of the point
       * @throws exception if i >= dimension of the point
       */
      const double &
      operator[](size_type i) const;

      /**
       * @brief In-plance multiplication of the point by a scalar p->p*a
       *
       * @param[in] p the point to be scaled
       * @param[in] a the scalar value
       *
       * @returns reference to updated point \p p.
       */
      Point &
      operator*=(Point &p, double a);

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

      /**
       * @brief In-place addition of two pointts p-> p + q
       *
       * @param[in] p the first point (will be modified)
       * @param[in] q the second point
       *
       * @returns Reference to updated point \p p.
       * @throws exception if the two points have different dimensions
       */
      Point &
      operator+=(Point &p, const Point &q);

      /**
       * @brief Addition of two points r = p + q.
       *
       * @param[in] p the first point
       * @param[in] q the second point
       *
       * @returns Object containing p + q.
       * @throws exception if the two points have different dimensions
       */
      Point
      operator+(const Point &v, const Point &u);

      /**
       * @brief In-place subtraction of two points p -> p - q;
       *
       * @param[in] p the first point
       * @param[in] q the second point
       *
       * @returns reference to updated point \p p.
       * @throws exception if the two points have different dimensions
       */
      Point &
      operator-=(Point &p, const Point &q);

      /**
       * @brief Subtraction of two points = p - q.
       *
       * @param[in] p the first point
       * @param[in] q the second point
       *
       * @return Object containing p - q.
       * @throws exception if the two points have different dimensions
       */
      Point
      operator-(const Point &p, const Point &q);

    private:
      double *  d_data;
      size_type d_dim;
    };

  } // end of namespace utils
} // end of namespace dftefe

#endif // dftefePoint_h
