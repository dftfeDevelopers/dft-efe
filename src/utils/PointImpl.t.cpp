#include "Exceptions.h"
#include <algorithm>
namespace dftefe
{
  namespace utils
  {
    template <typename T>
    inline PointImpl<T>::PointImpl(const std::vector<T> &x)
      : d_size(x.size())
      , d_data(new T[x.size()])
    {
      DFTEFE_AssertWithMsg(x.size() <= 3,
                           "Max. dimension of a point can be 3.");
      std::copy(x.begin(), x.end(), &(d_data[0]));
    }

    template <typename T>
    inline PointImpl<T>::PointImpl(const T *x, size_type N)
      : d_size(N)
      , d_data(new T[N])
    {
      DFTEFE_AssertWithMsg(x.size() <= 3,
                           "Max. dimension of a point can be 3.");
      std::copy(x, x + N, &(d_data[0]));
    }

    template <typename T>
    inline PointImpl<T>::PointImpl(size_type N)
      : d_size(N)
      , d_data(new T[N])
    {
      DFTEFE_AssertWithMsg(N <= 3, "Max. dimension of a point can be 3.");
    }

    template <typename T>
    inline PointImpl<T>::PointImpl(size_type N, const_reference init)
      : d_size(N)
      , d_data(new T[N])
    {
      DFTEFE_AssertWithMsg(N <= 3, "Max. dimension of a point can be 3.");
      std::fill(&(d_data[0]), &(d_data[N]), init);
    }

    template <typename T>
    inline PointImpl<T>::PointImpl(const PointImpl<T> &p)
      : d_size(p.size())
      , d_data(new T[p.size()])
    {
      std::copy(p.begin(), p.end(), &(d_data[0]));
    }


    template <typename T>
    inline PointImpl<T> &
    PointImpl<T>::operator=(const PointImpl<T> &p)
    {
      if (this != &p)
        {
          if (d_size != p.size())
            {
              // delete pointer to data
              delete[] d_data;

              // allocate new pointer for data
              d_size = p.size();
              d_data = new T[d_size];
            }

          std::copy(p.begin(), p.end(), &(d_data[0]));
        }

      return *this;
    }

    template <typename T>
    inline PointImpl<T>::~PointImpl()
    {
      delete[] d_data;
    }

    //
    // Return iterator pointing to the begining of PointImpl
    // data.
    //
    template <typename T>
    inline typename PointImpl<T>::iterator
    PointImpl<T>::begin()
    {
      return &(d_data[0]);
    }

    //
    // Return const_iterator pointing to the begining of PointImpl
    // data.
    //
    template <typename T>
    inline typename PointImpl<T>::const_iterator
    PointImpl<T>::begin() const
    {
      return &(d_data[0]);
    }

    //
    // Return iterator pointing to the end of PointImpl data.
    //
    template <typename T>
    inline typename PointImpl<T>::iterator
    PointImpl<T>::end()
    {
      return &(d_data[d_size]);
    }

    //
    // Return const_iterator pointing to the end of PointImpl data.
    //
    template <typename T>
    inline typename PointImpl<T>::const_iterator
    PointImpl<T>::end() const
    {
      return &(d_data[d_size]);
    }

    template <typename T>
    inline size_type
    PointImpl<T>::size() const
    {
      return d_size;
    }

    template <typename T>
    inline typename PointImpl<T>::reference
    PointImpl<T>::operator[](size_type i)
    {
      DFTEFE_Assert(i < d_size);
      return d_data[i];
    }

    template <typename T>
    inline typename PointImpl<T>::const_reference
    PointImpl<T>::operator[](size_type i) const
    {
      DFTEFE_Assert(i < d_size);
      return d_data[i];
    }

    template <typename T>
    inline PointImpl<T> &
    operator*=(PointImpl<T> &p, T a)
    {
      std::transform(p.begin(),
                     p.end(),
                     p.begin(),
                     std::bind1st(std::multiplies<T>(), a));
      return p;
    }

    template <typename T>
    inline PointImpl<T>
    operator*(const PointImpl<T> &p, T a)
    {
      PointImpl<T> q(p);
      q *= a;
      return q;
    }

    template <typename T>
    inline PointImpl<T>
    operator*(T a, const PointImpl<T> &p)
    {
      PointImpl<T> q(p);
      q *= a;
      return q;
    }

    template <typename T>
    inline PointImpl<T> &
    operator+=(PointImpl<T> &p, const PointImpl<T> &q)
    {
      DFTEFE_Assert(p.size() == q.size());
      std::transform(p.begin(), p.end(), q.begin(), p.begin(), std::plus<T>());

      return p;
    }

    template <typename T>
    inline PointImpl<T>
    operator+(const PointImpl<T> &p, const PointImpl<T> &q)
    {
      PointImpl<T> r(p);
      r += q;
      return r;
    }

    template <typename T>
    inline PointImpl<T> &
    operator-=(PointImpl<T> &p, const PointImpl<T> &q)
    {
      DFTEFE_Assert(p.size() == q.size());
      std::transform(p.begin(), p.end(), q.begin(), p.begin(), std::minus<T>());

      return p;
    }

    template <typename T>
    inline PointImpl<T>
    operator-(const PointImpl<T> &p, const PointImpl<T> &q)
    {
      PointImpl<T> r(p);
      r -= q;

      return r;
    }

    template <typename T>
    inline std::ostream &
    operator<<(std::ostream &outputStream, const PointImpl<T> &p)
    {
      const size_type pSize = p.size();
      outputStream << "{ ";
      for (size_type i = 0; i < pSize - 1; ++i)
        outputStream << p[i] << ", ";

      outputStream << p[pSize - 1] << " }";
      return outputStream;
    }

  } // end of namespace utils

} // end of namespace dftefe
