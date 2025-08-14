/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Sambit Das, Vishal Subramanian, Avirup Sircar
 */

#ifndef dftefeBlasWrapperTypedef_h
#define dftefeBlasWrapperTypedef_h

#include <utils/MemoryStorage.h>
#include <complex>
#include <cstdarg>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      namespace typeInternal
      {
        // Adapted from include/blas/util.hh
        // (https://github.com/icl-utk-edu/blaspp/blob/b5d0761a0c3a45ed1bad73b697125c2e46ac1617/include/blas/util.hh#L309)

        // -----------------------------------------------------------------------------
        // Based on C++14 common_type implementation from
        // http://www.cplusplus.com/reference/type_traits/common_type/
        // Adds promotion of complex types based on the common type of the
        // associated real types. This fixes various cases:
        //
        // std::std::common_type_t< double, complex<float> > is complex<float>
        // (wrong)
        //        scalar_type< double, complex<float> > is complex<double>
        //        (right)
        //
        // std::std::common_type_t< int, complex<long> > is not defined (compile
        // error)
        //        scalar_type< int, complex<long> > is complex<long> (right)

        // for zero types
        template <typename... Types>
        struct scalar_type_traits;

        // define scalar_type<> type alias
        template <typename... Types>
        using scalar_type = typename scalar_type_traits<Types...>::type;

        // for one type
        template <typename T>
        struct scalar_type_traits<T>
        {
          using type = std::decay_t<T>;
        };

        // for two types
        // relies on type of ?: operator being the common type of its two
        // arguments
        template <typename T1, typename T2>
        struct scalar_type_traits<T1, T2>
        {
          using type = std::decay_t<decltype(true ? std::declval<T1>() :
                                                    std::declval<T2>())>;
        };

        // for either or both complex,
        // find common type of associated real types, then add complex
        template <typename T1, typename T2>
        struct scalar_type_traits<std::complex<T1>, T2>
        {
          using type = std::complex<std::common_type_t<T1, T2>>;
        };

        template <typename T1, typename T2>
        struct scalar_type_traits<T1, std::complex<T2>>
        {
          using type = std::complex<std::common_type_t<T1, T2>>;
        };

        template <typename T1, typename T2>
        struct scalar_type_traits<std::complex<T1>, std::complex<T2>>
        {
          using type = std::complex<std::common_type_t<T1, T2>>;
        };

        // for three or more types
        template <typename T1, typename T2, typename... Types>
        struct scalar_type_traits<T1, T2, Types...>
        {
          using type = scalar_type<scalar_type<T1, T2>, Types...>;
        };

        // -----------------------------------------------------------------------------
        // for any combination of types, determine associated real, scalar,
        // and complex types.
        //
        // real_type< float >                               is float
        // real_type< float, double, complex<float> >       is double
        //
        // scalar_type< float >                             is float
        // scalar_type< float, complex<float> >             is complex<float>
        // scalar_type< float, double, complex<float> >     is complex<double>
        //
        // complex_type< float >                            is complex<float>
        // complex_type< float, double >                    is complex<double>
        // complex_type< float, double, complex<float> >    is complex<double>

        // for zero types
        template <typename... Types>
        struct real_type_traits;

        // define real_type<> type alias
        template <typename... Types>
        using real_type = typename real_type_traits<Types...>::real_t;

        // define complex_type<> type alias
        template <typename... Types>
        using complex_type = std::complex<real_type<Types...>>;

        // for one type
        template <typename T>
        struct real_type_traits<T>
        {
          using real_t = T;
        };

        // for one complex type, strip complex
        template <typename T>
        struct real_type_traits<std::complex<T>>
        {
          using real_t = T;
        };

        // for two or more types
        template <typename T1, typename... Types>
        struct real_type_traits<T1, Types...>
        {
          using real_t = scalar_type<real_type<T1>, real_type<Types...>>;
        };
      } // namespace typeInternal

      enum class Layout
      {
        ColMajor,
        RowMajor
      };

      using LapackInt = int64_t;

      enum class ScalarOp
      {
        Identity,
        Conj
      };

      // real_type< float >                               is float
      // real_type< float, double, complex<float> >       is double
      template <typename ValueType>
      using real_type = typeInternal::real_type<ValueType>;

      // scalar_type< float >                             is float
      // scalar_type< float, complex<float> >             is complex<float>
      // scalar_type< float, double, complex<float> >     is complex<double>
      template <typename ValueType1, typename ValueType2>
      using scalar_type = typeInternal::scalar_type<ValueType1, ValueType2>;

      template <dftefe::utils::MemorySpace memorySpace>
      struct BlasQueueTypedef
      {
        typedef void TYPE; //  default
      };

      // template specified mapping
      template <>
      struct BlasQueueTypedef<dftefe::utils::MemorySpace::HOST>
      {
        typedef int TYPE;
      };

      template <>
      struct BlasQueueTypedef<dftefe::utils::MemorySpace::HOST_PINNED>
      {
        typedef int TYPE;
      };

      template <>
      struct BlasQueueTypedef<dftefe::utils::MemorySpace::DEVICE>
      {
        typedef int TYPE;
      };

      template <dftefe::utils::MemorySpace memorySpace>
      using BlasQueue = typename BlasQueueTypedef<memorySpace>::TYPE;

      template <dftefe::utils::MemorySpace memorySpace>
      struct LapackQueueTypedef
      {
        typedef void LAPACKTYPE; //  default
      };

      // template specified mapping
      template <>
      struct LapackQueueTypedef<dftefe::utils::MemorySpace::HOST>
      {
        typedef int LAPACKTYPE;
      };

      template <>
      struct LapackQueueTypedef<dftefe::utils::MemorySpace::HOST_PINNED>
      {
        typedef int LAPACKTYPE;
      };

      template <>
      struct LapackQueueTypedef<dftefe::utils::MemorySpace::DEVICE>
      {
        typedef int LAPACKTYPE;
      };

      template <dftefe::utils::MemorySpace memorySpace>
      using LapackQueue = typename LapackQueueTypedef<memorySpace>::LAPACKTYPE;

    } // namespace blasLapack

  } // namespace linearAlgebra

} // namespace dftefe

#endif // define blasWrapperTypedef
