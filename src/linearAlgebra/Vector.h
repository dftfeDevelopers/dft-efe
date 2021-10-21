#ifndef dftefeVector_h
#define dftefeVector_h

#include "MemoryManager.h"
#include "TypeConfig.h"
#include <vector>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename NumType, dftefe::utils::MemorySpace memorySpace>
    class Vector
    {
      typedef NumType        value_type;
      typedef NumType       *pointer;
      typedef NumType       &reference;
      typedef const NumType &const_reference;
      typedef NumType       *iterator;
      typedef const NumType *const_iterator;

    public:
      /**
       * @brief Constructor for Vector with size and initial value arguments
       * @param[in] size size of the Vector
       * @param[in] initVal initial value of elements of the Vector
       */
      explicit Vector(size_type size, NumType initVal = 0);

      /**
       * @brief Destructor
       */
      ~Vector();

      Vector() = default;

      /**
       * @brief Copy constructor for a Vector
       * @param[in] u Vector object to copy from
       */
      Vector(const Vector &u);

      /**
       * @brief Return iterator pointing to the beginning of point
       * data.
       *
       * @returns Iterator pointing to the beginning of Vector.
       */
      iterator
      begin();

      /**
       * @brief Return iterator pointing to the beginning of Vector
       * data.
       *
       * @returns Constant iterator pointing to the beginning of
       * Vector.
       */
      const_iterator
      begin() const;

      /**
       * @brief Return iterator pointing to the end of Vector data.
       *
       * @returns Iterator pointing to the end of Vector.
       */
      iterator
      end();

      /**
       * @brief Return iterator pointing to the end of Vector data.
       *
       * @returns Constant iterator pointing to the end of
       * Vector.
       */
      const_iterator
      end() const;


      /**
       * @brief Operator overload for assignment v=u
       * @param[in] rhs the rhs Vector from which to copy
       *
       * @returns reference to the lhs Vector
       */
      Vector &
      operator=(const Vector &rhs);

      /**
       * @brief Operator to get a reference to a element of the Vector
       * @param[in] i is the index to the element of the Vector
       * @returns reference to the element of the Vector
       * @throws exception if i >= size of the Vector
       */
      reference
      operator[](size_type i);

      /**
       * @brief Operator to get a const reference to a element of the Vector
       * @param[in] i is the index to the element of the Vector
       * @returns const reference to the element of the Vector
       * @throws exception if i >= size of the Vector
       */
      const_reference
      operator[](size_type i) const;

      /**
       * @brief Deallocates and then resizes Vector with new size
       * and initial value arguments
       * @param[in] size size of the Vector
       * @param[in] initVal initial value of elements of the Vector
       */
      void
      resize(size_type size, NumType initVal = 0);

      /**
       * @brief Compound addition for elementwise addition lhs += rhs
       * @param[in] rhs the vector to add
       * @return the original vector
       */
      Vector &
      operator+=(const Vector &rhs);

      // todo move semantic
      //      Vector<NumberType, memorySpace> operator=()


      /**
       * @brief Returns the dimension of the Vector
       * @returns size of the Vector
       */
      size_type
      size() const;

      /**
       * @brief Return the raw pointer to the Vector
       * @return pointer to data
       */
      NumType *
      data() noexcept;

      /**
       * @brief Return the raw pointer to the Vector without modifying the values
       * @return pointer to const data
       */
      const NumType *
      data() const noexcept;

    private:
      NumType  *d_data = nullptr;
      size_type d_size = 0;
    };

    // helper functions


    template <typename NumType, dftefe::utils::MemorySpace memorySpace>
    void
    add(NumType                             a,
        const Vector<NumType, memorySpace> &u,
        NumType                             b,
        const Vector<NumType, memorySpace> &v,
        Vector<NumType, memorySpace>       &w);

  } // namespace linearAlgebra
} // end of namespace dftefe

#include "Vector.t.cpp"

#endif
