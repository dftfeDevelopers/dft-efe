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
#include <utils/Exceptions.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <basis/FEBasisOperationsInternal.h>
namespace dftefe 
{
  namespace basis
  {
    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
      FEBasisOperations<ValueType, memorySpace, dim>::FEBasisOperations(
	  std::shared_ptr<const BasisDataStorgae<ValueType, memorySpace>> basisDataStorage):
      {
	d_feBasisDataStorage = std::dynamic_pointer_cast<const FEBasisDataStorage<ValueType, memorySpace, dim>>(basisDataStorage);
	utils::throwException(d_feBasisDataStorage != nullptr, 
	    "Could not cast BasisDataStorage to FEBasisDataStorage in the constructor of FEBasisOperations");
      }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
      void
      FEBasisOperations<ValueType, memorySpace, dim>::interpolate(
        const Field<ValueType, memorySpace> &       field,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        quadrarture::QuadratureValuesContainer<ValueType, memorySpace>
          &quadValuesContainer) const
      {
	const BasisHandler<memorySpace> & basisHandler = field.getBasisHandler();
	const FEBasisHandler<ValueType, memorySpace, dim> & feBasisHandler = 
	  dynamic_cast<FEBasisHandler<ValueType, memorySpace, dim> &>(basisHandler);
	utils::throwException(&feBasisHandler != nullptr,
	    "Could not cast BasisHandler to FEBasisHandler in FEBasisOperations.interpolate()");
	const BasisManager & basisManagerField = basisHandler.getBasisManager();
	const BasisManager & basisManagerDataStorage = d_feBasisDataStorage->getBasisManager();
	utils::throwException(&basisManagerField == &basisManagerDataStorage,
	    "Mismatch in BasisManager used in Field and BasisDataStorage.");
	const FEBasisManager & feBasisManager = dynamic_cast<const FEBasisManager &>(basisManagerField);
	utils::throwException(&feBasisManager != nullptr,
	    "Could not cast BasisManager to FEBasisManager in FEBasisOperations.interpolate()");

	const size_type numComponents = 1;
	const std::string constraintsName = field.getConstraintsName();
	const size_type numLocallyOwnedCells = feBasisHandler.nLocallyOwnedCells();
	const size_type numCumulativeLocallyOwnedCellDofs = feBasisHandler.nCumulativeLocallyOwnedCellDofs();
	utils::MemoryStorage<ValueType,memorySpace> fieldCellValues(numCumulativeLocallyOwnedCellDofs);
        auto itCellLocalIdsBegin = feBasisHandler.locallyOwnedCellLocalDofIdsBegin(constraintsName);
	std::vector<size_type> numCellDofs(numLocallyOwnedCells,0);
	for(size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
	  numCellDofs = feBasisHandler.nLocallyOwnedCellDofs(iCell);

	FEBasisOperationsInternal<ValueType, memorySpace>::copyFieldToCellWiseData(field.begin(),
	    numComponents,
	    itCellLocalIdsBegin,
	    numCellDofs,
	    fieldCellValues);

        const quadrature::QuadratureFamily quadratureFamily =
          quadratureRuleAttributes.getQuadratureFamily();
        bool sameQuadRuleInAllCells = false;
	if(quadratureFamily == quadrature::QuadratureFamily::GAUSS || quadratureFamily == quadrature::QuadratureFamily::GLL)
	  sameQuadRuleInAllCells = true;
	bool hpRefined = feBasisManager.isHPRefined();
	// Perform
	// Ce = Ae*Be, where Ce_ij = interpolated value of the i-th component at j-th quad point in e-th cell
	// Ae_ik = i-th field components at k-th basis function of e-th cell
	// Be_kj = k-th basis function value at j-th quad point in e-th cell
	//
	
	//
	// For better performance, we evaluate Ce for multiple cells at a time 
	//

	//
	// @note: The Be matrix is stored with the quad point as the fastest index. That is
	// Be_kj (k-th basis function value at j-th quad point in e-th cell) is stored in a 
	// row-major format. Instead of copying it to a column major format, we use the transpose
	// of Be matrix. That is, we perform Ce = Ae*(Be)^T, with Be stored in row major format
	//
	std::vector<linearAlgebra::blasLapack::Op> transA(numLocallyOwnedCells, linearAlgebra::blasLapack::Op::NoTrans); 
	std::vector<linearAlgebra::blasLapack::Op> transB(numLocallyOwnedCells, linearAlgebra::blasLapack::Op::Trans);
	linearAlgebra::blasLapack::Layout layout = linearAlgebra::blasLapack::Layout::ColMajor;
	std::vector<size_type> mSizesTmp(numLocallyOwnedCells,0);
	std::vector<size_type> nSizesTmp(numLocallyOwnedCells,0);
	std::vector<size_type> kSizesTmp(numLocallyOwnedCells,0);
	std::vector<size_type> ldaSizesTmp(numLocallyOwnedCells,0);
	std::vector<size_type> ldbSizesTmp(numLocallyOwnedCells,0);
	std::vector<size_type> ldcSizesTmp(numLocallyOwnedCells,0);
	std::vector<size_type> strideATmp(numLocallyOwnedCells,0);
	std::vector<size_type> strideBTmp(numLocallyOwnedCells,0);
	const bool zeroStrideB = sameQuadRuleInAllCells && (!hpRefined);
	for(size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
	{
	  mSizesTmp[iCell] = numComponents;
	  nSizesTmp[iCell] = quadValuesContainer.nCellQuadraturePoints(iCell);
	  kSizesTmp[iCell] = numCellDofs[iCell];
	  ldaSizesTmp[iCell] = mSizesTmp[iCell];
	  ldbSizesTmp[iCell] = nSizesTmp[iCell];
	  ldcSizesTmp[iCell] = mSizesTmp[iCell];
	  if(iCell > 0)
	    strideATmp[iCell] = strideATmp[iCell-1] + mSizesTmp[iCell-1]*kSizesTmp[iCell-1];
	  if(!zeroStrideB && iCell > 0)
	    strideBTmp[iCell] = strideBTmp[iCell-1] + nSizesTmp[iCell-1]*kSizesTmp[iCell-1];
	}
	
	utils::MemoryStorage<size_type,memorySpace> mSizes(numLocallyOwnedCells);
	utils::MemoryStorage<size_type,memorySpace> nSizes(numLocallyOwnedCells);
	utils::MemoryStorage<size_type,memorySpace> kSizes(numLocallyOwnedCells);
	utils::MemoryStorage<size_type,memorySpace> ldaSizes(numLocallyOwnedCells);
	utils::MemoryStorage<size_type,memorySpace> ldbSizes(numLocallyOwnedCells);
	utils::MemoryStorage<size_type,memorySpace> ldcSizes(numLocallyOwnedCells);
	utils::MemoryStorage<size_type,memorySpace> strideA(numLocallyOwnedCells);
	utils::MemoryStorage<size_type,memorySpace> strideB(numLocallyOwnedCells);
	utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST> memoryTransfer;
	memoryTransfer.copy(numLocallyOwnedCells, mSizes.data(), mSizesTmp.data());
	memoryTransfer.copy(numLocallyOwnedCells, nSizes.data(), nSizesTmp.data());
	memoryTransfer.copy(numLocallyOwnedCells, kSizes.data(), kSizesTmp.data());
	memoryTransfer.copy(numLocallyOwnedCells, ldaSizes.data(), ldaSizesTmp.data());
	memoryTransfer.copy(numLocallyOwnedCells, ldbSizes.data(), ldbSizesTmp.data());
	memoryTransfer.copy(numLocallyOwnedCells, ldcSizes.data(), ldcSizesTmp.data());
	memoryTransfer.copy(numLocallyOwnedCells, strideA.data(), strideATmp.data());
	memoryTransfer.copy(numLocallyOwnedCells, strideB.data(), strideBTmp.data());

	ValueType alpha = 1.0;
	ValueType beta = 0.0;
	const linearAlgebra::LinAlgContext & linAlgContext = field.getLinAlgContext();
	const ValueType * B = (d_feBasisDataStorage->getBasisInAllCells(quadratureRuleAttributes)).data();
	linearAlgebra::gemmStridedVarBatched(layout,
                            numLocallyOwnedCells,
                            transA.data(),
                            transB.data(),
                            strideA.data(),
                            strideB.data(),
                            mSizes.data(),
                            nSizes.data(),
                            kSizes.data(),
                            alpha,
                            fieldCellValues.data(),
                            ldaSizes.data(),
			    B,
                            ldbSizes.data(),
                            beta,
                            quadValuesContainer.begin(),
                            ldcSizes.data(),
                            linAlgContext.getBlasQueue());
      }

  } // end of namespace 
}//
