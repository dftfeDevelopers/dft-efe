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
 * @author Bikash Kanungo, Sambit Das
 */

#ifndef dftefeMPITags_h
#define dftefeMPITags_h

#include <utils/TypeConfig.h>
#include <vector>
#include <cstdint>

#ifdef DFTEFE_WITH_MPI
#  include <mpi.h>
#endif

namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      enum class MPITags : std::uint16_t
      {
        DUMMY_MPI_TAG = 100,
        MPI_REQUESTERS_NBX_TAG,
        MPI_P2P_PATTERN_TAG,

        MPI_P2P_COMMUNICATOR_SCATTER_TAG,

        MPI_P2P_COMMUNICATOR_GATHER_TAG = MPI_P2P_COMMUNICATOR_SCATTER_TAG + 200
      };
    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
#endif // dftefeMPITags_h
