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
 * @author Bikash Kanungo, Vishal Subramanian
 */
#include<utils/Exceptions.h>
namespace dftefe 
{

  namespace basis
  {

#ifdef DFTEFE_WITH_MPI
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace, size_type dim>
	  FEBasisHandlerDealii<ValueType, memorySpace, dim>::FEBasisHandlerDealii(std::shared_ptr<const BasisManager> basisManager,
	      std::map<std::string, std::shared_ptr<const Constraints>> constraintsMap,
              const MPI_Comm &                                    mpiComm)
	  {
	    reinit(basisManager,constraintsMap,mpiComm);
	  }
    
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace, size_type dim>
	  FEBasisHandlerDealii<ValueType, memorySpace, dim>::reinit(std::shared_ptr<const BasisManager> basisManager,
	      std::map<std::string, std::shared_ptr<const Constraints>> constraintsMap,
              const MPI_Comm &                                    mpiComm)
	  {
	    d_mpiComm = mpiComm; 
	    d_feBMDealii = std::dynamic_pointer_cast<const FEBasisManagerDealii<dim>>(basisManager);
	    utils::throwException(d_feBMDealii != nullptr, 
		"Error in casting the input basis manager in FEBasisHandlerDealii to FEBasisParitionerDealii");
	    for(auto it = constraintsMap.begin(), it != constraintsMap.end(); ++it)
	    {
	      std::string constraintsName = it->first;
	     std::shared_ptr<const FEBasisConstraintsDealii<dim, ValueType>> feBasisConstraintsDealii = 
	     std::dynamic_pointer_cast<const FEBasisConstraintsDealii<dim, ValueType>>(it->second);
	     utils::throwException(feBasisConstraintsDealii != nullptr,
		 "Error in casting the input constraints to FEBasisConstraintsDealii in FEBasisHandlerDealii");
	     d_feConstraintsDealiiMap[constraintsName] = feBasisConstraintsDealii;
	    }
	    d_locallyOwnedRange	= d_feBMDealii->getLocallyOwnedRange();

	    size_type numLocallyOwnedCells = d_feBMDealii->nLocallyOwnedCells();
	    std::vector<size_type> locallyOwnedCellStartAndEndIdsTmp(2*numLocallyOwnedCells,0);
	    size_type cumulativeCellDofs = 0;
	    for(size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
	    {
	      const size_type numCellDofs = d_feBMDealii->nCellDofs(iCell);
	      cumulativeCellDofs += numCellDofs;
	      if(iCell == 0)
	      {
		locallyOwnedCellStartAndEndIdsTmp[2*iCell] = 0;
		locallyOwnedCellStartAndEndIdsTmp[2*iCell+1] = numCellDofs; 
	      }
	      else
	      {
		locallyOwnedCellStartAndEndIdsTmp[2*iCell] = locallyOwnedCellStartAndEndIdsTmp[2*iCell-1];
		locallyOwnedCellStartAndEndIdsTmp[2*iCell+1] = locallyOwnedCellStartAndEndIdsTmp[2*iCell] + numCellDofs; 
	      }
	    }

	    std::vector<global_size_type> locallyOwnedCellGlobalIndicesTmp(cumulativeCellDofs,0);
	    for(size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
	    {
	      const size_type numCellDofs = d_feBMDealii->nCellDofs(iCell);
	      locallyOwnedCellGlobalIndicesTmp
	      std::vector<global_size_type> cellGlobalIds(numCellDofs);
	      d_feBMDealii->getCellDofsGlobalIds(iCell, cellGlobalIds);
	      std::copy(cellGlobalIds.begin(), cellGlobalIds.end(),
		  locallyOwnedCellGlobalIndicesTmp.begin()+locallyOwnedCellStartAndEndIdsTmp[2*iCell]);
	    }
	  }
#else
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace, size_type dim>
	  FEBasisHandlerDealii<ValueType, memorySpace, dim>::FEBasisHandlerDealii(std::shared_ptr<const BasisManager> basisManager,
	      std::map<std::string, std::shared_ptr<const Constraints>> constraintsMap)
	  {
	    reinit(basisManager,constraintsMap);
	  }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace, size_type dim>
	  FEBasisHandlerDealii<ValueType, memorySpace, dim>::reinit(std::shared_ptr<const BasisManager> basisManager,
	      std::map<std::string, std::shared_ptr<const Constraints>> constraintsMap)
	  {
	  }
#endif // DFTEFE_WITH_MPI

  } // end of namespace basis
} // end of namespace dftefe
