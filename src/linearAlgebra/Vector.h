#ifndef dftefeVector_h
#define dftefeVector_h

#include "MemoryManager.h"
#include "TypeConfig.h"
#include <vector>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    class Vector
    {
      typedef NumberType        value_type;
      typedef NumberType *      pointer;
      typedef NumberType &      reference;
      typedef const NumberType &const_reference;
      typedef NumberType *      iterator;
      typedef const NumberType *const_iterator;

    public:
      /**
       * @brief Constructor for Vector with size and initial value arguments
       * @param[in] size size of the Vector
       * @param[in] initVal initial value of elements of the Vector
       */
      explicit Vector(size_type size, NumberType initVal = 0);

      /**
       * @brief Destructor
       */
      ~Vector();

      Vector() = default;

      /**
       * @brief Copy constructor for a Vector
       * @param[in] u Vector object to copy from
       */
      Vector(const Vector<NumberType, memorySpace> &u);

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
       * @param[in] u the rhs Vector from which to copy
       *
       * @returns reference to the lhs Vector
       */
      Vector &
      operator=(const Vector<NumberType, memorySpace> &u);

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
      resize(size_type size, NumberType initVal = 0);

      /**
       * @brief Returns the dimension of the Vector
       * @returns size of the Vector
       */
      size_type
      size() const;

    private:
      NumberType *d_data = nullptr;
      size_type   d_size = 0;
    };

    //
    // A list of helper functions to operate on Vector
    //


    /**
     * @brief In-place addition of two Vectors \f$v= v + u\f$
     *
     * @tparam NumberType
     * @tparam memorySpace
     * @param[in] v the first Vector (will be modified)
     * @param[in] u the second Vector
     *
     * @returns Reference to updated Vector \p v.
     * @throws exception if the two Vectors have different dimensions
     */
    template <typename NumberType, dftefe::utils::MemorySpace memorySpace>
    Vector<NumberType, memorySpace> &
    operator+(Vector<NumberType, memorySpace> &      v,
              const Vector<NumberType, memorySpace> &u);

  } // namespace linearAlgebra
} // end of namespace dftefe

#include "Vector.t.cpp"

#endif
