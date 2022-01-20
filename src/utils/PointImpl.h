#ifndef dftefePointImpl_h
#define dftefePointImpl_h
#include "TypeConfig.h"
#include <iostream>
#include <vector>

namespace dftefe
{
  namespace utils
  {
    template <typename T>
    class PointImpl
    {
      typedef T        value_type;
      typedef T *      pointer;
      typedef T &      reference;
      typedef const T &const_reference;
      typedef int      difference_type;
      typedef T *      iterator;
      typedef const T *const_iterator;

    public:
      /**
       * @brief Constructor for an N-dimensional point where N=1,2,3
       * @param[in] x is reference to std::vector contaning the coordinates of
       * the point
       * @throws exception if N > 3
       */
      PointImpl(const std::vector<T> &x);

      /**
       * @brief Constructor for an N-dimensional point where N=1,2,3
       * @param[in] x is pointer to coordinates of the point
       * @param[in] N dimension of the point
       * @throws exception if N > 3
       */
      PointImpl(const T *x, size_type N);

      /**
       * @brief Constructor for an N-dimensional point where N=1,2,3
       * @param[in] N dimension of the point
       * point
       * @throws exception if N > 3
       */
      PointImpl(size_type N);

      /**
       * @brief Constructor for an N-dimensional point where N=1,2,3
       * @param[in] N dimension of the point
       * @param[out] init initial value to be assigned to all components of the
       * point
       * @throws exception if N > 3
       */
      PointImpl(size_type N, const_reference init);

      /**
       * @brief Copy constructor for a point
       * @param[in] p PointImpl object to copy from
       */
      PointImpl(const PointImpl &p);

      /**
       * @brief Destructor
       */
      ~PointImpl();


      /**
       * @brief Returns the dimension of the point
       * @returns dimension of the point
       */
      size_type
      size() const;

      /**
       * @brief Return iterator pointing to the begining of point
       * data.
       *
       * @returns Iterator pointing to the begingin of PointImpl.
       */
      iterator
      begin();

      /**
       * @brief Return iterator pointing to the begining of PointImpl
       * data.
       *
       * @returns Constant iterator pointing to the begining of
       * PointImpl.
       */
      const_iterator
      begin() const;

      /**
       * @brief Return iterator pointing to the end of PointImpl data.
       *
       * @returns Iterator pointing to the end of PointImpl.
       */
      iterator
      end();

      /**
       * @brief Return iterator pointing to the end of PointImpl data.
       *
       * @returns Constant iterator pointing to the end of
       * PointImpl.
       */
      const_iterator
      end() const;

      /**
       * @brief Operator overload for assignment q=p
       * @param[in] p the rhs PointImpl from which to copy
       *
       * @returns reference to the lhs PointImpl
       */
      PointImpl &
      operator=(const PointImpl &p);

      /**
       * @brief Operator to get a reference to a component of the point
       * @param[in] i is the index to the component of the point
       * @returns reference to the component of the point
       * @throws exception if i >= dimension of the point
       */
      reference
      operator[](size_type i);

      /**
       * @brief Operator to get a const reference to a component of the point
       * @param[in] i is the index to the component of the point
       * @returns const reference to the component of the point
       * @throws exception if i >= dimension of the point
       */
      const_reference
      operator[](size_type i) const;


    private:
      pointer   d_data;
      size_type d_size;
    };

    //
    // A list of helper functions to operate on PointImpl
    //

    /**
     * @brief In-plance multiplication of the point by a scalar p->p*a
     * @tparam value type for the point (int, float, double, ...)
     * @param[in] p the point to be scaled
     * @param[in] a the scalar value
     *
     * @returns reference to updated point \p p.
     */
    template <typename T>
    PointImpl<T> &
    operator*=(PointImpl<T> &p, T a);

    /**
     * @brief Multiplication by a scalar q = p*a
     * @tparam value type for the point (int, float, double, ...)
     * @param[in] p the point to be scaled
     * @param[in] a the scalar value
     *
     * @returns a new point containing the scaled coordinates
     */
    template <typename T>
    PointImpl<T>
    operator*(const PointImpl<T> &p, T a);

    /**
     * @brief Multiplication by a scalar q = p*a
     *
     * @tparam value type for the point (int, float, double, ...)
     * @param[in] a the scalar value
     * @param[in] p the point to be scaled
     *
     * @returns a new point containing the scaled coordinates
     */
    template <typename T>
    PointImpl<T>
    operator*(T a, const PointImpl<T> &p);

    /**
     * @brief In-place addition of two pointts p-> p + q
     *
     * @tparam value type for the point (int, float, double, ...)
     * @param[in] p the first point (will be modified)
     * @param[in] q the second point
     *
     * @returns Reference to updated point \p p.
     * @throws exception if the two points have different dimensions
     */
    template <typename T>
    PointImpl<T> &
    operator+=(PointImpl<T> &p, const PointImpl<T> &q);

    /**
     * @brief Addition of two points r = p + q.
     *
     * @tparam value type for the point (int, float, double, ...)
     * @param[in] p the first point
     * @param[in] q the second point
     *
     * @returns Object containing p + q.
     * @throws exception if the two points have different dimensions
     */
    template <typename T>
    PointImpl<T>
    operator+(const PointImpl<T> &v, const PointImpl<T> &u);

    /**
     * @brief In-place subtraction of two points p -> p - q;
     *
     * @tparam value type for the point (int, float, double, ...)
     * @param[in] p the first point
     * @param[in] q the second point
     *
     * @returns reference to updated point \p p.
     * @throws exception if the two points have different dimensions
     */
    template <typename T>
    PointImpl<T> &
    operator-=(PointImpl<T> &p, const PointImpl<T> &q);

    /**
     * @brief Subtraction of two points = p - q.
     *
     * @tparam value type for the point (int, float, double, ...)
     * @param[in] p the first point
     * @param[in] q the second point
     *
     * @return Object containing p - q.
     * @throws exception if the two points have different dimensions
     */
    template <typename T>
    PointImpl<T>
    operator-(const PointImpl<T> &p, const PointImpl<T> &q);


    /**
     * @brief Output perator.
     *
     * @tparam value type for the point (int, float, double, ...)
     * @param outputStream Output stream to use for output.
     * @param p PointImpl object.
     *
     * @returns Updated output stream.
     */
    template <typename T>
    std::ostream &
    operator<<(std::ostream &outputStream, const PointImpl<T> &p);

  } // end of namespace utils
} // end of namespace dftefe

#include "PointImpl.t.cpp"

#endif // dftefePointImpl_h
