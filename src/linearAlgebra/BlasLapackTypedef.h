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

#include <blas.hh>
#define LAPACK_COMPLEX_CPP
#define HAVE_LAPACK_CONFIG_H
#include <lapack.hh>
#include <lapack/device.hh>
#include <utils/MemoryStorage.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasLapack
    {
      using Op     = blas::Op; // Op::NoTrans, Op::Trans, Op::ConjTrans
      using Diag   = blas::Diag;
      using Layout = blas::Layout;

      // lapack
      using Job   = lapack::Job;   // Job::Vec, Job::NoVec
      using Uplo  = lapack::Uplo;  // Uplo::Lower, Uplo::Upper
      using Diag  = lapack::Diag;  // Diag::NonUnit, Diag::Unit
      using Range = lapack::Range; // Range::All , Range::Value , Range::Index

      using LapackInt = int64_t;

      enum class ScalarOp
      {
        Identity,
        Conj
      };

      // real_type< float >                               is float
      // real_type< float, double, complex<float> >       is double
      template <typename ValueType>
      using real_type = blas::real_type<ValueType>;

      // scalar_type< float >                             is float
      // scalar_type< float, complex<float> >             is complex<float>
      // scalar_type< float, double, complex<float> >     is complex<double>
      template <typename ValueType1, typename ValueType2>
      using scalar_type = blas::scalar_type<ValueType1, ValueType2>;
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
        typedef blas::Queue TYPE;
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
        typedef lapack::Queue LAPACKTYPE;
      };

      template <dftefe::utils::MemorySpace memorySpace>
      using LapackQueue = typename LapackQueueTypedef<memorySpace>::LAPACKTYPE;

    } // namespace blasLapack

  } // namespace linearAlgebra

} // namespace dftefe

#endif // define blasWrapperTypedef
