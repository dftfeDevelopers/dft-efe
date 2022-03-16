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
 * @author Vishal Subramanian
 */

#ifndef dftefeQueueManager_h
#define dftefeQueueManager_h

#include <blas.hh>
#include "BlasWrappersTypedef.h"

namespace dftefe
{
  namespace linearAlgebra
  {
    class QueueManager
    {
    public:
//      blasWrapper::Queue &
//      getBlasQueue();
//
//      void
//      createBlasQueue();

    private:
      int cpuQueue;

      // FIXME Should this is be inside DFTEFE_WITH_GPU ????
      static blasWrapper::Queue blasGpuQueue;
    };

  } // namespace linearAlgebra

} // namespace dftefe

#include "QueueManager.t.cpp"
#endif // define queueManager_h
