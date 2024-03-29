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
 * @author Ian C. Lin, Sambit Das
 */

namespace dftefe
{
  namespace linearAlgebra
  {
    template <utils::MemorySpace memorySpace>
    LinAlgOpContext<memorySpace>::LinAlgOpContext(
      std::shared_ptr<blasLapack::BlasQueue<memorySpace>>   blasQueue,
      std::shared_ptr<blasLapack::LapackQueue<memorySpace>> lapackQueue)
      : d_blasQueue(blasQueue)
      , d_lapackQueue(lapackQueue)
    {}

    template <utils::MemorySpace memorySpace>
    void
    LinAlgOpContext<memorySpace>::setBlasQueue(
      std::shared_ptr<blasLapack::BlasQueue<memorySpace>> blasQueue)
    {
      d_blasQueue = blasQueue;
    }

    template <utils::MemorySpace memorySpace>
    blasLapack::BlasQueue<memorySpace> &
    LinAlgOpContext<memorySpace>::getBlasQueue() const
    {
      return *d_blasQueue;
    }

    template <utils::MemorySpace memorySpace>
    void
    LinAlgOpContext<memorySpace>::setLapackQueue(
      std::shared_ptr<blasLapack::LapackQueue<memorySpace>> lapackQueue)
    {
      d_lapackQueue = lapackQueue;
    }

    template <utils::MemorySpace memorySpace>
    blasLapack::LapackQueue<memorySpace> &
    LinAlgOpContext<memorySpace>::getLapackQueue() const
    {
      return *d_lapackQueue;
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
