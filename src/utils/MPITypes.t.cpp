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

#include <utils/Exceptions.h>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      template <typename T>
      MPIDatatype
      Types<T>::getMPIDatatype()
      {
        utils::throwException<utils::InvalidArgument>(
          false,
          "No equivalent primitive C++ datatype that is supported by MPI "
          "matches the template parameter T passsed to "
          "mpi::Types<T>::getMPIDatatype()");
        return MPIByte;
      }
    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
