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

#ifndef dftefeMPIRequestersBase_h
#define dftefeMPIRequestersBase_h
#include <utils/TypeConfig.h>
#include <vector>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      class MPIRequestersBase
      {
        /*
         *
         * @brief A pure virtual class to evaluate the list of rank Ids that the
         * current processor needs to send data.
         *
         * In a typical case of distributed data (a vector or array), a
         * processor needs to communicate part of its part of the data to a
         * requesting processor. It is useful for the current processor to know
         * apriori which processors it has to send its part of the distributed
         * data. This base class provides an interface to indentify the Ids of
         * the processors (also known as ranks) that it has to send data to. In
         * MPI parlance, the other processors to which this processor needs to
         * send data are termed as requesting processors/ranks.
         *
         * The actual process of identifying the list of requesting processors
         * is  implemented in the derived classes. There are various different
         * algorithms with varying computational/communication complexity. Some
         * use cases are trivial, for example, (a) a serial case where
         * there are no requesting processors, (b) an all-to-all communication
         * case where all the other processors are requesting from the
         * current proccesor.
         *
         */

      public:
        virtual ~MPIRequestersBase() = default;
        virtual std::vector<size_type>
        getRequestingRankIds() = 0;
      };

    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
#endif // dftefeMPIRequestersBase_h
