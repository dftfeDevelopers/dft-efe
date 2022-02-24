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
 * @author Sambit Das, Vishal Subramanian
 */

#ifndef dftefeBlasWrapperTypedef_h
#define dftefeBlasWrapperTypedef_h

#include <blas.hh>
#include <utils/MemoryStorage.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace blasWrapper
    {
      typedef blas::Side    Side;
     typedef blas::Op      Op;
     typedef blas::Diag    Diag;
     typedef blas::Uplo    Uplo;
     typedef blas::Layout  Layout; 
     typedef blas::Queue   Queue;
     typedef blas::real_type real_type;
     typedef blas::scalar_type scalar_type;

     template<dftefe::utils::MemorySpace memorySpace >
     struct blasQueueTypedef
     {
       typedef void TYPE;  //  default
     };

     //template specified mapping
     template<>
     struct blasQueueTypedef<dftefe::utils::MemorySpace::HOST>
      {  typedef int TYPE;   };

      template<>
      struct blasQueueTypedef<dftefe::utils::MemorySpace::HOST_PINED>
      {  typedef int TYPE;   };

     template<>
     struct blasQueueTypedef<dftefe::utils::MemorySpace::DEVICE>
     {   typedef blas::Queue  TYPE;  };

     template<dftefe::utils::MemorySpace memorySpace >
     using blasQueueType = typename blasQueueTypedef<memorySpace>::TYPE;


    }// namespace blasWrapper

  } // namespace linearAlgebra

} // namespace dftefe

#endif // define blasWrapperTypedef
