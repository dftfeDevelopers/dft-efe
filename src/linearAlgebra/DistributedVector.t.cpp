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

namespace dftefe
{
  namespace linearAlgebra
  {
    //
    // Constructor
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    DistributedVector<ValueType, memorySpace>::DistributedVector(
      std::shared_ptr<const MPICommunicatorP2P> mpiCommunicatorP2P,
      const ValueType                           initVal = ValueType())
      : d_mpiCommunicatorP2P(mpiCommunicatorP2P)
      , d_mpiPatternP2P(mpiCommunicatorP2P.getMPIPatternP2P())
      , d_vectorAttributes(VectorAttributes::Distribution::SERIAL)
      , d_storage(0)
      , d_localSize(0)
      , d_localOwnedSize(0)
      , d_localGhostSize(0)
      , d_globalSize(0)
    {
      d_localOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_localGhostSize = d_mpiPatternP2P->localGhostSize();
      d_localSize      = d_localOwnedSize + d_localGhostSize;
      d_storage.resize(d_localSize);
      d_globalSize = d_mpiPatternP2P->nGlobalIndices();
    }
  } // end of namespace linearAlgebra
} // namespace dftefe
