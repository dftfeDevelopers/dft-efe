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

#ifndef dftefeFEBasisPartitioner_h
#define dftefeFEBasisPartitioner_h

#include <basis/BasisPartitioner.h>
namespace dftefe
{
  namespace basis
  {

    /**
     * @brief An abstract class to encapsulate the partitioning 
     * of a finite element basis across multiple processors
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace, size_type dim>
      class FEBasisPartitioner : public BasisPartitioner<memorySpace>
    {

	//
	// typedefs
	//
	public:
	  using SizeTypeVector = BasisPartitioner<memorySpace>::SizeTypeVector;
	  using GlobalSizeTypeVector = BasisParitioner<memorySpace>::SizeTypeVector;

	public:
	  ~BasisPartitioner() = default;

	  std::pair<global_size_type, global_size_type>
	    getLocallyOwnedRange(const std::string constraintsName) const  = 0;

	  const GlobalSizeTypeVector & 
	    getGhostIndices(const std::string constraintsName) const = 0;

	  size_type
	    nLocalSize(const std::string constraintsName) const  = 0;

	  size_type
	    nLocallyOwnedSize(const std::string constraintsName) const  = 0;

	  size_type
	    nGhostSize(const std::string constraintsName) const  = 0;

	  bool
	    inLocallyOwnedRange(const global_size_type, const std::string constraintsName) const = 0;

	  bool
	    isGhostEntry(const global_size_type, const std::string constraintsName) const = 0;

	  size_type
	    globalToLocal(const global_size_type, const std::string constraintsName) const = 0;

	  global_size_type
	    localToGlobal(const size_type, const std::string constraintsName) const = 0;

	  //
	  // FE specific functions
	  //
	  const SizeTypeVector &
	    getLocallyOwnedCellGlobalDoFIds(const size_type cellId, const std::string constraintsName) const = 0;

	  const SizeTypeVector &
	    getLocallyOwnedCellLocalDoFIds(const size_type cellId, const std::string constraintsName) const = 0;
	  
	  const SizeTypeVector &
	    getLocallyOwnedCellGlobalDoFIds(const size_type cellId, const std::string constraintsName) const = 0;

	  const SizeTypeVector &
	    getLocallyOwnedCellLocalDoFIds(const size_type cellId, const std::string constraintsName) const = 0;
      };

  } // end of namespace basis
} // end of namespace dftefe
#endif // dftefeFEBasisPartitioner_h
