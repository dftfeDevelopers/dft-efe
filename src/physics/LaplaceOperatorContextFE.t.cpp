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
  namespace physics 
  {

    namespace LaplaceOperatorContextFEInternal
    {
      template <utils::MemorySpace memorySpace>  
      void storeSizes(
          utils::MemoryStorage<size_type, memorySpace> & mSizes,
          utils::MemoryStorage<size_type, memorySpace> & nSizes,
          utils::MemoryStorage<size_type, memorySpace> & kSizes,
          utils::MemoryStorage<size_type, memorySpace> & ldaSizes,
          utils::MemoryStorage<size_type, memorySpace> & ldbSizes,
          utils::MemoryStorage<size_type, memorySpace> & ldcSizes,
          utils::MemoryStorage<size_type, memorySpace> & strideA,
          utils::MemoryStorage<size_type, memorySpace> & strideB,
          utils::MemoryStorage<size_type, memorySpace> & strideC,
	  const std::vector<size_type> & numCellsInBlockDofs,
	  const size_type numVecs)
      {
	const size_type numCellsInBlock = numCellsInBlockDofs.size();
	std::vector<size_type> mSizesSTL(numCellsInBlock, 0);
	std::vector<size_type> nSizesSTL(numCellsInBlock, 0);
	std::vector<size_type> kSizesSTL(numCellsInBlock, 0);
	std::vector<size_type> ldaSizesSTL(numCellsInBlock, 0);
	std::vector<size_type> ldbSizesSTL(numCellsInBlock, 0);
	std::vector<size_type> ldcSizesSTL(numCellsInBlock, 0);
	std::vector<size_type> strideASTL(numCellsInBlock, 0);
	std::vector<size_type> strideBSTL(numCellsInBlock, 0);
	std::vector<size_type> strideCSTL(numCellsInBlock, 0);

	for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
	{
	  mSizesSTL[iCell]       = numCellsInBlockDofs[iCell];
	  nSizesSTL[iCell] = numVecs;
	  kSizesSTL[iCell]   = numCellsInBlockDofs[iCell];
	  ldaSizesSTL[iCell] = mSizesSTL[iCell];
	  ldbSizesSTL[iCell] = nSizesSTL[iCell];
	  ldcSizesSTL[iCell] = mSizesSTL[iCell];
	  strideASTL[iCell]  = mSizesSTL[iCell] * kSizesSTL[iCell];
	  strideBSTL[iCell]  = kSizesSTL[iCell] * nSizesSTL[iCell];
	  strideCSTL[iCell]  = mSizesSTL[iCell] * nSizesSTL[iCell];
	}

          mSizes.copyFrom(mSizesSTL);
          nSizes.copyFrom(nSizesSTL);
          kSizes.copyFrom(kSizesSTL);
          ldaSizes.copyFrom(ldaSizesSTL);
          ldbSizes.copyFropm(ldbSizesSTL);
          ldcSizes.copyFrom(ldcSizesSTL);
          strideA.copyFrom(strideASTL);
          strideB.copyFrom(strideBSTL);
          strideC.copyFrom(strideCSTL);
      }

    }// end of namespace LaplaceOperatorContextFEInternal


    template <typename ValueTypeOperator,
	     typename ValueTypeOperand,
	     utils::MemorySpace memorySpace,
	     size_type dim>
	       LaplaceOperatorContextFE<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>::
	       LaplaceOperatorContextFE(const basis::FEBasisHandler<ValueTypeOperator,
		   memorySpace,
		   dim > &feBasisHandler,
		   const utils::FEBasisDataStorage<ValueTypeOperator, memorySpace>
		   &                                                  feBasisDataStorage,
		   const linearAlgebra::Vector<ValueType, memorySpace> &b,
		   const std::string                                    constraintsName,
		   const QuadratureRuleAttributes &quadratureRuleAttributes,
		   const size_type maxCellTimesFieldBlock):
		 d_feBasisHandler(&feBasisHandler),
		 d_feBasisDataStorage(&feBasisDataStorage),
		 d_constraintsName(constraintsName),
		 d_quadratureRuleAttributes(quadratureRuleAttributes),
		 d_maxCellTimesNumVecs(maxCellTimesNumVecs)
    {

    }

    template <typename ValueTypeOperator,
	     typename ValueTypeOperand,
	     utils::MemorySpace memorySpace,
	     size_type dim>
	       void
	       LaplaceOperatorContextFE<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>::
	       apply(const Vector<ValueTypeOperand, memorySpace> &x,
		   Vector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
		   memorySpace> &                        y) const
	       {

		 const size_type   numLocallyOwnedCells =
		   d_feBasisHandler->nLocallyOwnedCells();
		 const size_type numVecs = 1;

		 auto gradNiGradNjInAllCells = d_feBasisDataStorage->getBasisGradNiGradNjInAllCells(d_quadratureRuleAttributes);
		 const size_type cellBlockSize = d_maxCellTimesFieldBlock / numComponents;
		 for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
		     cellStartId += cellBlockSize)
		 {
		   const size_type cellEndId =
		     std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
		   const size_type        numCellsInBlock = cellEndId - cellStartId;
		   std::vector<size_type> numCellsInBlockDofsSTL(numCellsInBlock, 0);
		   std::copy(numCellDofs.begin() + cellStartId,
		       numCellDofs.begin() + cellEndId,
		       numCellsInBlockDofsSTL.begin());

		   const size_type numCumulativeDofsCellsInBlock =
		     std::accumulate(numCellsInBlockDofsSTL.begin(),
			 numCellsInBlockDofsSTL.end(),
			 0);


		   utils::MemoryStorage<size_type, memorySpace>
		     numCellsInBlockDofs(numCellsInBlock);
		   numCellsInBlockDofs.copyFrom(numCellsInBlockDofsSTL);

		   std::vector<linearAlgebra::blasLapack::Op> transA(
		       numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
		   std::vector<linearAlgebra::blasLapack::Op> transB(
		       numCellsInBlock, linearAlgebra::blasLapack::Op::NoTrans);
		   utils::MemoryStorage<size_type, memorySpace> mSizes(numCellsInBlock);
		   utils::MemoryStorage<size_type, memorySpace> nSizes(numCellsInBlock);
		   utils::MemoryStorage<size_type, memorySpace> kSizes(numCellsInBlock);
		   utils::MemoryStorage<size_type, memorySpace> ldaSizes(
		       numCellsInBlock);
		   utils::MemoryStorage<size_type, memorySpace> ldbSizes(
		       numCellsInBlock);
		   utils::MemoryStorage<size_type, memorySpace> ldcSizes(
		       numCellsInBlock);
		   utils::MemoryStorage<size_type, memorySpace> strideA(numCellsInBlock);
		   utils::MemoryStorage<size_type, memorySpace> strideB(numCellsInBlock);
		   utils::MemoryStorage<size_type, memorySpace> strideC(numCellsInBlock);
		   
		   LaplaceOperatorContextFEInternal::getSizes(mSizes,
		     nSizes,
		     kSizes,
		     ldaSizes,
		     ldbSizes,
		     ldcSizes,
		     strideA,
		     strideB,
		     strideC,
		     numCellsInBlockDofsSTL,
		     numVecs);

		   ValueType alpha = 1.0;
		   ValueType beta = 0.0;

		 }
	       }


  }// end of namespace physics
}// end of namespace dftefe
