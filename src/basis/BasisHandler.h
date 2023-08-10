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

#ifndef dftefeBasisHandler_h
#define dftefeBasisHandler_h

#include <basis/Constraints.h>
#include <basis/BasisManager.h>
#include <utils/Point.h>
#include "DealiiConversions.h"
#include <utils/TypeConfig.h>
#include <utils/MemoryStorage.h>
#include <utils/MPIPatternP2P.h>
#include <string>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class to encapsulate the partitioning
     * of a basis across multiple processors
     *
     * @tparam ValueTypeBasisCoeff the datatype for the constraints
     * @tparam template parameter memorySpace defines the MemorySpace (i.e., HOST or
     * DEVICE) in which the data must reside.
     */
    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace>
    class BasisHandler
    {
      //
      // typedefs
      //
    public:
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;
      using GlobalSizeTypeVector =
        utils::MemoryStorage<global_size_type, memorySpace>;
      using LocalIndexIter       = typename SizeTypeVector::iterator;
      using const_LocalIndexIter = typename SizeTypeVector::const_iterator;
      using GlobalIndexIter      = typename GlobalSizeTypeVector::iterator;
      using const_GlobalIndexIter =
        typename GlobalSizeTypeVector::const_iterator;

    public:
      ~BasisHandler() = default;

      virtual bool
      isDistributed() const = 0;

      virtual const BasisManager &
      getBasisManager() const = 0;

      virtual const Constraints<ValueTypeBasisCoeff, memorySpace> &
      getConstraints(const std::string constraintsName) const = 0;

      virtual void
      setConstraints(
        std::map<
          std::string,
          std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>>
          constraintsMap) = 0;

      virtual std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
      getMPIPatternP2P(const std::string constraintsName) const = 0;

      virtual std::vector<std::pair<global_size_type, global_size_type>>
      getLocallyOwnedRanges(const std::string constraintsName) const = 0;

      virtual const GlobalSizeTypeVector &
      getGhostIndices(const std::string constraintsName) const = 0;

      virtual size_type
      nLocal(const std::string constraintsName) const = 0;

      virtual size_type
      nLocallyOwned(const std::string constraintsName) const = 0;

      virtual size_type
      nGhost(const std::string constraintsName) const = 0;

      virtual std::pair<bool, size_type>
      inLocallyOwnedRanges(const global_size_type globalId,
                           const std::string      constraintsName) const = 0;

      virtual std::pair<bool, size_type>
      isGhostEntry(const global_size_type ghostId,
                   const std::string      constraintsName) const = 0;

      virtual size_type
      globalToLocalIndex(const global_size_type globalId,
                         const std::string      constraintsName) const = 0;

      virtual global_size_type
      localToGlobalIndex(const size_type   localId,
                         const std::string constraintsName) const = 0;

      virtual void
      getBasisCenters(const size_type       localId,
                      const std::string     constraintsName,
                      dftefe::utils::Point &basisCenter) const = 0;
    };

  } // end of namespace basis
} // end of namespace dftefe
#endif // dftefeBasisHandler_h
