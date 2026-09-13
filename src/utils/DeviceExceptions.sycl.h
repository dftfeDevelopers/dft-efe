// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
#ifndef dftefeDeviceExceptions_syclh
#define dftefeDeviceExceptions_syclh

#include <iostream>
#include <system_error>

namespace dftefe
{
  namespace utils
  {
    // deviceError_t (std::error_code) is what every device API call site in
    // this codebase passes here -- just check the code, nothing to wait on.
    inline void
    deviceApiCheck(const std::error_code &errorCode,
                   const char *           func,
                   const char *           file,
                   int                    line)
    {
      if (errorCode)
        {
          std::cerr << "SYCL error in " << func << " at " << file << ":" << line
                    << ". Error code: " << errorCode.message() << ".\n";
        }
    }
  } // namespace utils
} // namespace dftefe

#define DEVICE_API_CHECK(x) \
  dftefe::utils::deviceApiCheck(x, __func__, __FILE__, __LINE__)

#define DEVICEBLAS_API_CHECK(expr)                                 \
  do                                                               \
    {                                                              \
      try                                                          \
        {                                                          \
          (void)(expr);                                            \
        }                                                          \
      catch (sycl::exception const &__sycl_err)                    \
        {                                                          \
          std::printf("oneMKL enqueue error in %s at %s:%d: %s\n", \
                      __func__,                                    \
                      __FILE__,                                    \
                      __LINE__,                                    \
                      __sycl_err.what());                          \
        }                                                          \
    }                                                              \
  while (0)

#endif // dftefeDeviceExceptions_syclh
