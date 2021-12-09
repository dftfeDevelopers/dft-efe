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

#ifndef dftefeDeviceBlasLapackExceptions_cuh
#define dftefeDeviceBlasLapackExceptions_cuh

#define CUBLAS_CHECK(cmd)                                           \
  do                                                                \
    {                                                               \
      cublasStatus_t err = cmd;                                     \
      if (err != CUBLAS_STATUS_SUCCESS)                             \
        {                                                           \
          printf("Failed: Cublas error %s:%d '%s'\n",               \
                 __FILE__,                                          \
                 __LINE__,                                          \
                 dftefe::linearAlgebra::cublasGetErrorString(err)); \
          exit(EXIT_FAILURE);                                       \
        }                                                           \
    }                                                               \
  while (0)

namespace dftefe
{
  namespace linearAlgebra
  {
    static const char *
    cublasGetErrorString(cublasStatus_t err)
    {
      switch (err)
        {
          case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

          case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

          case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

          case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

          case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

          case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

          case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

          case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        }

      return "<unknown>";
    }
  } // namespace linearAlgebra
} // namespace dftefe


#endif // dftefeDeviceBlasLapackExceptions_cuh
