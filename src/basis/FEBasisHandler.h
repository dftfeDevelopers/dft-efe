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

#ifndef dftefeFEBasisHandler_h
#define dftefeFEBasisHandler_h

#include <basis/BasisHandler.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class to encapsulate the partitioning
     * of a finite element basis across multiple processors
     */
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    class FEBasisHandler : public BasisHandler<memorySpace>
    {
      //
      // typedefs
      //
    public:
      using SizeTypeVector = typename BasisHandler<memorySpace>::SizeTypeVector;
      using GlobalSizeTypeVector =
        typename BasisHandler<memorySpace>::GlobalSizeTypeVector;
      using LocalIndexIter       = typename SizeTypeVector::iterator;
      using const_LocalIndexIter = typename SizeTypeVector::const_iterator;
      using GlobalIndexIter      = typename GlobalSizeTypeVector::iterator;
      using const_GlobalIndexIter =
        typename GlobalSizeTypeVector::const_iterator;

    public:
      ~FEBasisHandler() = default;

      virtual std::pair<global_size_type, global_size_type>
      getLocallyOwnedRange(const std::string constraintsName) const = 0;

      virtual const GlobalSizeTypeVector &
      getGhostIndices(const std::string constraintsName) const = 0;

      virtual size_type
      nLocal(const std::string constraintsName) const = 0;

      virtual size_type
      nLocallyOwned(const std::string constraintsName) const = 0;

      virtual size_type
      nGhost(const std::string constraintsName) const = 0;

      virtual bool
      inLocallyOwnedRange(const global_size_type globalId,
                          const std::string      constraintsName) const = 0;

      virtual bool
      isGhostEntry(const global_size_type ghostId,
                   const std::string      constraintsName) const = 0;

      virtual size_type
      globalToLocalIndex(const global_size_type globalId,
                         const std::string      constraintsName) const = 0;

      virtual global_size_type
      localToGlobalIndex(const size_type   localId,
                         const std::string constraintsName) const = 0;

      //
      // FE specific functions
      //
      size_type
      numLocallyOwnedCellDofs(const size_type cellId) const = 0;

      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsBegin(
        const std::string constraintsName) const = 0;

      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsBegin(
        const size_type   cellId,
        const std::string constraintsName) const = 0;

      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsEnd(const size_type   cellId,
                                      const std::string constraintsName) const =
        0;

      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsBegin(
        const std::string constraintsName) const = 0;

      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsBegin(
        const size_type   cellId,
        const std::string constraintsName) const = 0;

      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsEnd(const size_type   cellId,
                                     const std::string constraintsName) const =
        0;
    };

  } // end of namespace basis
} // end of namespace dftefe
#endif // dftefeFEBasisHandler_h
