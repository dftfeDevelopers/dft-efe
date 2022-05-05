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
 * @author Bikash Kanungo
 */

#ifndef dftefeMPIErrorCodeHandler_h
#define dftefeMPIErrorCodeHandler_h

#ifdef DFTEFE_WITH_MPI
#  include <mpi.h>
#endif // DFTEFE_WITH_MPI
#include <string>
namespace dftefe
{
  namespace utils
  {
    class MPIErrorCodeHandler
    {
    public:
      MPIErrorCodeHandler()  = default;
      ~MPIErrorCodeHandler() = default;
      static bool
      isSuccess(const int &errCode) const;
      static std::string
      getErrMsg(const int &errCode) const;
      static std::pair<bool, std::string>
      getIsSuccessAndMessage(const int &errCode) const;
    }; // end of mpiErrorCodes
  }    // end of namespace utils
} // end of namespace dftefe
#endif // dftefeMPIErrorCodeHandler_h
