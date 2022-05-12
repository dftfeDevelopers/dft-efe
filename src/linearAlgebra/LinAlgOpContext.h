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

#ifndef dftefeLinAlgOpContext_h
#define dftefeLinAlgOpContext_h

#include <utils/MemorySpaceType.h>
#include <utils/BlasLapackTypedef.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    template <utils::MemorySpace memorySpace>
    class LinAlgOpContext
    {
    public:
      LinAlgOpContext(BlasQueue<memorySpace> *blasQueue);
      
      void
      setBlasQueue(BlasQueue<memorySpace> *blasQueue);
      
      BlasQueue<memorySpace> &blasQueue
      getBlasQueue() const;

    private:
      BlasQueue<memorySpace> *d_blasQueue;

    }; // end of LinAlgOpContext
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/LinAlgOpContext.t.cpp>
#endif // end of dftefeLinAlgOpContext_h
