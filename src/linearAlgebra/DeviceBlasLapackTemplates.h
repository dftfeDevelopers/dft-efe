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
 * @author Sambit Das.
 */

#ifdef DFTEFE_WITH_DEVICE
#  ifndef dftefeDeviceBlasLapackTemplates_h
#    define dftefeDeviceBlasLapackTemplates_h

#    include <linearAlgebra/DeviceLAContextsSingleton.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType>
    class DeviceBlasLapack
    {
    public:
      static void
      gemm(deviceBlasHandleType    handle,
           deviceBlasOperationType transa,
           deviceBlasOperationType transb,
           int                     m,
           int                     n,
           int                     k,
           const ValueType *       alpha,
           const ValueType *       A,
           int                     lda,
           const ValueType *       B,
           int                     ldb,
           const ValueType *       beta,
           ValueType *             C,
           int                     ldc);

      static void
      nrm2(deviceBlasHandleType handle,
           int                  n,
           const ValueType *    x,
           int                  incx,
           double *             result);

      static void
      iamax(deviceBlasHandleType handle,
            int                  n,
            const ValueType *    x,
            int                  incx,
            int *                maxid);
    };
  } // namespace linearAlgebra
} // namespace dftefe
#  endif
#endif // DFTEFE_WITH_DEVICE
