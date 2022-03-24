
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
#include <utils/MathFunctions.h>
#include <basis/TriangulationCellDealii.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
namespace dftefe
{
  namespace basis
  {
    namespace FEBasisDataStorageDealiiInternal {

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
	void storeValuesHRefinedSameQuadEveryCell(std::shared_ptr<const FEBasisManagerDealii<dim>> feBM,
	    std::shared_ptr<typename BasisDataStorage<ValueType, memorySpace>::Storage>  basisQuadStorage,
	    std::shared_ptr<typename BasisDataStorage<ValueType, memorySpace>::Storage>  basisGradientQuadStorage,
	    std::shared_ptr<typename BasisDataStorage<ValueType, memorySpace>::Storage>  basisHessianQuadStorage,
	    std::shared_ptr<typename BasisDataStorage<ValueType, memorySpace>::Storage>  basisOverlap,
	    const QuadratureRuleAttributes & quadratureRuleAttributes,
	    const bool storeValues,
	    const bool storeGradients,
	    const bool storeHessians)
	{
	
	  const quadrature::QuadratureFamily quadratureFamily = quadratureRuleAttributes.getQuadratureFamily();
	  const size_type num1DQuadPoints = quadratureRuleAttributes.getNum1DPoints();
	  dealii::Quadrature<dim> dealiiQuadratureRule;
	  if(quadratureFamily == quadrature::QuadratureFamily::GAUSS) 
	  {
	    dealiiQuadratureRule = dealii::QGauss<dim>(num1DQuadPoints);
	  }
	  else if(quadratureFamily == quadrature::QuadratureFamily::GLL)
	  {
	    dealiiQuadratureRule = dealii::QGaussLobatto<dim>(num1DQuadPoints);
	  }

	  else
	  {
	    utils::throwException(false, "Storing of basis values for classical finite element basis is only provided for Gauss and Gauss-Legendre-Lobatto quadrature rule"); 
	  }
	  
	  dealii::update_flags dealiiUpdateFlags = dealii::update_default;
	  
	  if(storeValues)
	    dealiiUpdateFlags |= dealii::update_values;
	  if(storeGradients)
	    dealiiUpdateFlags |= dealii::update_gradients;
	  if(storeHessians)
	    dealiiUpdateFlags |= dealii::update_hessians;
	  // NOTE: cellId 0 passed as we assume only H refined in this function
	  const size_type cellId = 0;
	  dealii::FEValues<dim> dealiiFEValues(feBM->getFEOrder(cellId),
	      dealiiQuadratureRule, dealiiUpdateFlags);
	  const size_type numLocallyOwnedCells = feBM->nLocallyOwnedCells();
	  // NOTE: cellId 0 passed as we assume only H refined in this function
	  const size_type dofsPerCell = feBM->nCellDofs(cellId);
	  bool isQuadCartesianTensorStructured = quadratureRuleAttributes.isCartesianTensorStructured();
	  utils::throwException(isQuadCartesianTensorStructured, "Storing of basis values for classical finite element basis on non-tensor structured quadrature grid is not supported"); 
	  const size_type num1DQuadPoints = quadratureRuleAttributes.getNum1DPoints();
	  const size_type numQuadPoints = utils::mathFunctions::sizeTypePow(num1DQuadPoints, dim);
	  const size_type nDimxDofsPerCellxNumQuad = dim*dofsPerCell*numQuadPoints;
	  const size_type nDimSqxDofsPerCellxNumQuad = dim*dim*dofsPerCell*numQuadPoints;
	  const size_type DofsPerCellxNumQuad = dofsPerCell*numQuadPoints;

	  std::vector<ValueType>  basisQuadStorageTmp(0);
	  std::vector<ValueType>  basisGradientQuadStorageTmp(0);
	  std::vector<ValueType>  basisHessianQuadStorageTmp(0);
	  std::vector<ValueType>  basisOverlapTmp(0);

	  if(storeValues)
	  {
	    basisQuadStorage = std::make_shared<typename BasisDataStorage<ValueType, memorySpace>::Storage>(dofsPerCell*numQuadPoints);
	    basisQuadStorageTmp.resize(dofsPerCell*numQuadPoints, ValueType(0));
	  }

	  if(storeGradients) 
	  {
	    basisGradientQuadStorage = std::make_shared<typename BasisDataStorage<ValueType, memorySpace>::Storage>(numLocallyOwnedCells*nDimxDofsPerCellxNumQuad);
	    basisGradientQuadStorageTmp.resize(numLocallyOwnedCells*nDimxDofsPerCellxNumQuad, ValueType(0));
	  }
	  if(storeHessians)
	  {
	    basisHessianQuadStorage = std::make_shared<typename BasisDataStorage<ValueType, memorySpace>::Storage>(numLocallyOwnedCells*nDimSqxDofsPerCellxNumQuad);
	    basisHessianQuadStorageTmp.resize(numLocallyOwnedCells*nDimSqxDofsPerCellxNumQuad, ValueType(0));
	  }

	  basisOverlap = std::make_shared<typename BasisDataStorage<ValueType, memorySpace>::Storage>(numLocallyOwnedCells*dofsPerCell*dofsPerCell);
	  basisOverlapTmp.resize(numLocallyOwnedCells*dofsPerCell*dofsPerCell,ValueType(0));
	  auto locallyOwnedCellIter = feBM->beginLocallyOwnedCells();
	  std::shared_ptr<FECellDealii<dim>> feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
	  utils::throwException(feCellDealii != nullptr, "Dynamic casting of FECellBase to FECellDealii not successful");

	  auto basisQuadStorageTmpIter = basisQuadStorageTmp.begin();
	  auto basisGradientQuadStorageTmpIter = basisGradientQuadStorageTmp.begin();
	  auto basisHessianQuadStorageTmpIter = basisHessianQuadStorageTmp.begin();
	  auto basisOverlapTmpIter = basisOverlapTmp.begin();
	  size_type cellIndex = 0;
	  for(; locallyOwnedCellIter != feBM->endLocallyOwnedCells(); ++locallyOwnedCellIter)
	  {
	    feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
	    dealiiFEValues.reinit(feCellDealii->getDealiiFECellIter());
	    if(storeValues && locallyOwnedCellIter == feBM->beginLocallyOwnedCells())
	    {
	      for(unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
	      {
		for( unsigned int qPoint = 0; qPoint < numQuadPoints ;qPoint++)
		{
		  *basisQuadStorageTmpIter = dealiiFEValues.shape_value(iNode,qPoint);
		  basisQuadStorageTmpIter++;
		}
	      }
	    }

	    for(unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
	    {
	      for(unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
	      {
		*basisOverlapTmpIter = 0.0;
		for( unsigned int qPoint = 0; qPoint < numQuadPoints ;qPoint++)
		{
		  *basisOverlapTmpIter += dealiiFEValues.shape_value(iNode,qPoint)*
		    dealiiFEValues.shape_value(jNode,qPoint)*
		    dealiiFEValues.JxW(qPoint);
		}
		basisOverlapTmpIter++;
	      }
	    }

	    if(storeGradients)
	    {
	      for(unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
	      {
		for( unsigned int qPoint = 0; qPoint < numQuadPoints; qPoint++)
		{
		  auto shapeGrad = dealiiFEValues.shape_grad(iNode,qPoint);
		  for( unsigned int iDim = 0; iDim < dim; iDim++)
		  {
		    auto it = basisGradientQuadStorageTmp.begin() + 
		      cellIndex*nDimxDofsPerCellxNumQuad + 
		      iDim*DofsPerCellxNumQuad + iNode*numQuadPoints + qPoint; 
		    *it = shapeGrad[dim];
		  }
		}
	      }
	    }

	    if(storeHessian)
	    {
	      for(unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
	      {
		for( unsigned int qPoint = 0; qPoint < numQuadPoints; qPoint++)
		{
		  auto shapeHessian = dealiiFEValues.shape_hessian(iNode,qPoint);
		  for( unsigned int iDim = 0; iDim < dim; iDim++)
		  {
		    for( unsigned int jDim = 0; jDim < dim; jDim++)
		    {
		      auto it = basisHessianQuadStorageTmp.begin() + 
			cellIndex*nDimsqxDofsPerCellxNumQuad + 
			iDim*nDimxDofsPerCellxNumQuad + jDim*DofsPerCellxNumQuad + iNode*numQuadPoints + qPoint;
		      *it = shapeHessian(iDim,jDim);
		    }
		  }
		}
	      }
	    }

	    cellIndex++;
	  }

	  if(storeValues)
	  {
	    utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(basisQuadStorageTmp.size(), basisQuadStorage.data(), basisQuadStorageTmp.data());
	  }

	  if(storeGradients) 
	  {
	    utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(basisGradientQuadStorageTmp.size(), basisGradientQuadStorage.data(), basisGradientQuadStorageTmp.data());
	  }
	  if(storeHessians)
	  {
	    utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(basisHessianQuadStorageTmp.size(), basisHessianQuadStorage.data(), basisHessianQuadStorageTmp.data());
	  }

	  utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(basisOverlapTmp.size(), basisOverlap.data(), basisOverlapTmp.data());
	}

    template <typename size_type dim>
      void storeValues(std::shared_ptr<const FEBasisManagerDealii<dim>> feBM,
	  std::shared_ptr<Storage>  basisQuadStorage,
	  std::shared_ptr<Storage>  basisGradientQuadStorage,
	  std::shared_ptr<Storage>  basisHessianQuadStorage,
	  std::shared_ptr<Storage>  basisOverlap,
	  const QuadratureRuleAttributes & quadratureRuleAttributes,
	  const bool storeValues,
	  const bool storeGradients,
	  const bool storeHessians)
      {
	const quadrature::QuadratureFamily quadratureFamily = quadratureRuleAttributes.getQuadratureFamily();
	if((quadratureFamily == quadrature::QuadratureFamily::GAUSS) || (quadratureFamily == quadrature::QuadratureFamily::GLL))	
	{
	  if(feBM->isHPRefined() == false)
	  {
	    storeValuesHRefinedSameQuadEveryCell(feBM,
		basisQuadStorage,
		basisGradientQuadStorage,
		basisHessianQuadStorage,
		basisOverlap,
		quadratureRuleAttributes,
		storeValues,
		storeGradients,
		storeHessians);
	  }
	}
      }
    
    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
	void storeValuesHRefinedAdaptiveQuad(std::shared_ptr<const FEBasisManagerDealii<dim>> feBM,
	    std::shared_ptr<typename BasisDataStorage<ValueType, memorySpace>::Storage>  basisQuadStorage,
	    std::shared_ptr<typename BasisDataStorage<ValueType, memorySpace>::Storage>  basisGradientQuadStorage,
	    std::shared_ptr<typename BasisDataStorage<ValueType, memorySpace>::Storage>  basisHessianQuadStorage,
	    std::shared_ptr<typename BasisDataStorage<ValueType, memorySpace>::Storage>  basisOverlap,
	    const QuadratureRuleAttributes & quadratureRuleAttributes,
	    std::shared_ptr<const quadrature::CellQuadratureContainer>
                                        quadratureContainer,
	    const bool storeValues,
	    const bool storeGradients,
	    const bool storeHessians)
	{
	  const quadrature::QuadratureFamily quadratureFamily = quadratureRuleAttributes.getQuadratureFamily();
	  if((quadratureFamily != quadrature::QuadratureFamily::GAUSS_VARIABLE) || 
	      (quadratureFamily != quadrature::QuadratureFamily::GLL_VARIABLE) ||
	      (quadratureFamily != quadrature::QuadratureFamily::ADAPTIVE))
	  {
	    utils::throwException(false, "For storing of basis values for classical finite element basis on a variable quadrature rule across cells, the underlying quadrature family has to be quadrature::QuadratureFamily::GAUSS_VARIABLE or quadrature::QuadratureFamily::GLL_VARIABLE or quadrature::QuadratureFamily::ADAPTIVE"); 
	  }

	  
	  dealii::update_flags dealiiUpdateFlags = dealii::update_default;
	  
	  if(storeValues)
	    dealiiUpdateFlags |= dealii::update_values;
	  if(storeGradients)
	    dealiiUpdateFlags |= dealii::update_gradients;
	  if(storeHessians)
	    dealiiUpdateFlags |= dealii::update_hessians;
	  // NOTE: cellId 0 passed as we assume only H refined in this function
	  const size_type cellId = 0;
	  dealii::FEValues<dim> dealiiFEValues(feBM->getFEOrder(cellId),
	      dealiiQuadratureRule, dealiiUpdateFlags);
	  const size_type numLocallyOwnedCells = feBM->nLocallyOwnedCells();

	  // NOTE: cellId 0 passed as we assume only H refined in this function
	  const size_type cellId = 0;
	  const size_type numLocallyOwnedCells = feBM->nLocallyOwnedCells();
	  // NOTE: cellId 0 passed as we assume only H refined in this function
	  const size_type dofsPerCell = feBM->nCellDofs(cellId);

	  std::vector<ValueType>  basisQuadStorageTmp(0);
	  std::vector<ValueType>  basisGradientQuadStorageTmp(0);
	  std::vector<ValueType>  basisHessianQuadStorageTmp(0);
	  std::vector<ValueType>  basisOverlapTmp(0);

	  const size_type nTotalQuadPoints = quadratureContainer->nQuadraturePoints();
	  if(storeValues)
	  {
	    basisQuadStorage = std::make_shared<typename BasisDataStorage<ValueType, memorySpace>::Storage>(dofsPerCell*nTotalQuadPoints);
	    basisQuadStorageTmp.resize(dofsPerCell*nTotalQuadPoints, ValueType(0));
	  }

	  if(storeGradients) 
	  {
	    basisGradientQuadStorage = std::make_shared<typename BasisDataStorage<ValueType, memorySpace>::Storage>(dofsPerCell*dim*nTotalQuadPoints);
	    basisGradientQuadStorageTmp.resize(dofsPerCell*dim*nTotalQuadPoints, ValueType(0));
	  }
	  if(storeHessians)
	  {
	    basisHessianQuadStorage = std::make_shared<typename BasisDataStorage<ValueType, memorySpace>::Storage>(dofsPerCell*dim*dim*nTotalQuadPoints);
	    basisHessianQuadStorageTmp.resize(dofsPerCell*dim*dim*nTotalQuadPoints, ValueType(0));
	  }

	  basisOverlap = std::make_shared<typename BasisDataStorage<ValueType, memorySpace>::Storage>(numLocallyOwnedCells*dofsPerCell*dofsPerCell);
	  basisOverlapTmp.resize(numLocallyOwnedCells*dofsPerCell*dofsPerCell,ValueType(0));
	  auto locallyOwnedCellIter = feBM->beginLocallyOwnedCells();
	  std::shared_ptr<FECellDealii<dim>> feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
	  utils::throwException(feCellDealii != nullptr, "Dynamic casting of FECellBase to FECellDealii not successful");

	  auto basisQuadStorageTmpIter = basisQuadStorageTmp.begin();
	  auto basisGradientQuadStorageTmpIter = basisGradientQuadStorageTmp.begin();
	  auto basisHessianQuadStorageTmpIter = basisHessianQuadStorageTmp.begin();
	  auto basisOverlapTmpIter = basisOverlapTmp.begin();
	  size_type cellIndex = 0;

	  // get the dealii FiniteElement object
	  std::shared_ptr<const dealii::DoFHandler<dim>> dealiiDofHandler = feBM->getDoFHandler();
	  dealii::FiniteElement<dim> & dealiiFEObj = dealiiDofHandler->get_fe();
	  for(; locallyOwnedCellIter != feBM->endLocallyOwnedCells(); ++locallyOwnedCellIter)
	  {
	    feCellDealii = std::dynamic_pointer_cast<FECellDealii<dim>>(*locallyOwnedCellIter);
	    size_type nQuadPointInCell =  quadratureContainer->nCellQuadraturePoints(cellIndex);
	    const std::vector<dftefe::utils::Point> & cellParametricQuadPoints = 
	      quadratureContainer->getCellParametricPoints(cellIndex);
            std::vector<double> cellJxWValues = quadratureContainer->getCellJxW(cellIndex);
            std::vector<dealii::Point<dim, double>> dealiiParametricQuadPoints(0);
	    convertToDealiiPoint<dim>(cellParametricQuadPoints,
                         dealiiParametricQuadPoints);
	    dealii::Quadrature<dim> dealiiQuadratureRule(dealiiParametricQuadPoints);
	    if(storeValues)
	    {
	      for(unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
	      {
		for( unsigned int qPoint = 0; qPoint < nQuadPointInCell ;qPoint++)
		{
		  *basisQuadStorageTmpIter = dealiiFEObj.shape_value(iNode,dealiiParametricQuadPoints[qPoint]);
		  basisQuadStorageTmpIter++;
		}
	      }
	    }

	    for(unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
	    {
	      for(unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
	      {
		*basisOverlapTmpIter = 0.0;
		for( unsigned int qPoint = 0; qPoint < numQuadPoints ;qPoint++)
		{
		  *basisOverlapTmpIter += dealiiFEObj.shape_value(iNode,dealiiParametricQuadPoints[qPoint])*
		    dealiiFEValues.shape_value(jNode,dealiiParametricQuadPoints[qPoint])*
		    cellJxWValues[qPoint];
		}
		basisOverlapTmpIter++;
	      }
	    }

	    if(storeGradients)
	    {
	      for(unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
	      {
		for( unsigned int qPoint = 0; qPoint < numQuadPoints; qPoint++)
		{
		  auto shapeGrad = dealiiFEValues.shape_grad(iNode,qPoint);
		  for( unsigned int iDim = 0; iDim < dim; iDim++)
		  {
		    auto it = basisGradientQuadStorageTmp.begin() + 
		      cellIndex*nDimxDofsPerCellxNumQuad + 
		      iDim*DofsPerCellxNumQuad + iNode*numQuadPoints + qPoint; 
		    *it = shapeGrad[dim];
		  }
		}
	      }
	    }

	    if(storeHessian)
	    {
	      for(unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
	      {
		for( unsigned int qPoint = 0; qPoint < numQuadPoints; qPoint++)
		{
		  auto shapeHessian = dealiiFEValues.shape_hessian(iNode,qPoint);
		  for( unsigned int iDim = 0; iDim < dim; iDim++)
		  {
		    for( unsigned int jDim = 0; jDim < dim; jDim++)
		    {
		      auto it = basisHessianQuadStorageTmp.begin() + 
			cellIndex*nDimsqxDofsPerCellxNumQuad + 
			iDim*nDimxDofsPerCellxNumQuad + jDim*DofsPerCellxNumQuad + iNode*numQuadPoints + qPoint;
		      *it = shapeHessian(iDim,jDim);
		    }
		  }
		}
	      }
	    }

	    cellIndex++;
	  }

	  if(storeValues)
	  {
	    utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(basisQuadStorageTmp.size(), basisQuadStorage.data(), basisQuadStorageTmp.data());
	  }

	  if(storeGradients) 
	  {
	    utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(basisGradientQuadStorageTmp.size(), basisGradientQuadStorage.data(), basisGradientQuadStorageTmp.data());
	  }
	  if(storeHessians)
	  {
	    utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(basisHessianQuadStorageTmp.size(), basisHessianQuadStorage.data(), basisHessianQuadStorageTmp.data());
	  }

	  utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(basisOverlapTmp.size(), basisOverlap.data(), basisOverlapTmp.data());
	}

    }

    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
      FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      FEBasisDataStorageDealii(
	  std::shared_ptr<const FEBasisManagerDealii<dim>>           feBM,
	  std::vector<std::shared_ptr<const ConstraintsDealii>> constraintsVec,
	  const std::vector<QuadratureRuleAttributes> &quadratureRuleAttribuesVec,
	  const bool storeValues,
	  const bool storeGradients,
	  const bool storeHessians,
	  const bool storeJxW,
	  const bool storeQuadRealPoints):
	d_feBM(feBM)
    {
      const size_type numConstraints  = constraintsVec.size();
      const size_type numQuadRuleType = quadratureRuleAttribuesVec.size();
      std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
	feBM->getDoFHandler();
      std::vector<const dealii::DoFHandler<dim> *> dofHandlerVec(
	  numConstraints, dofHandler.get());
      std::vector<const dealii::AffineConstraints<ValueType> *>
	dealiiAffineConstraintsVec(numConstraints, nullptr);
      for (size_type i = 0; i < numConstraints; ++i)
      {
	dealiiAffineConstraintsVec[i] =
	  (constraintsVec[i]->getAffineConstraints()).get();
      }

      std::vector<dealii::QuadratureType> dealiiQuadratureTypeVec(0);
      for (size_type i = 0; i < numQuadRuleType; ++i)
      {
	size_type num1DQuadPoints =
	  quadratureRuleAttribuesVec[i].getNum1DPoints();
	quadrature::QuadratureFamily quadFamily =
	  quadratureRuleAttribuesVec[i].getQuadratureFamily();
	if (quadFamily == quadrature::QuadratureFamily::GAUSS)
	  dealiiQuadratureTypeVec.push_back(
	      dealii::QGauss<1>(num1DQuadPoints));
	else if (quadFamily == quadrature::QuadratureFamily::GLL)
	  dealiiQuadratureTypeVec.push_back(
	      dealii::QGaussLobatto<1>(num1DQuadPoints));
	else
	  utils::throwException<utils::InvalidArgument>(
	      false,
	      "Quadrature family is undefined. Currently, only Gauss and GLL quadrature families are supported.")
      }

      typename dealii::MatrixFree<dim>::AdditionalData dealiiAdditionalData;
      dealiiAdditionalData.tasks_parallel_scheme = dealii::MatrixFree<dim>::AdditionalData::partition_partition;
      dealii::update_flags dealiiUpdateFlags = dealii::update_default;

      if(storeValues)
	dealiiUpdateFlags |= dealii::update_values;
      if(storeGradients)
	dealiiUpdateFlags |= dealii::update_gradients;
      if(storeHessians)
	dealiiUpdateFlags |= dealii::update_hessians;
      if(storeJxW)
	dealiiUpdateFlags |= deali::update_JxW_values;
      if(storeQuadRealPoints)
	dealiiUpdateFlags |= dealii::update_quadrature_points;

      additionalData.mapping_update_flags = dealiiUpdateFlags; 

      d_dealiiMatrixFree =
	std::make_shared<dealii::MatrixFree<dim, ValueType>>();
      d_dealiiMatrixFree->clear();
      d_dealiiMatrixFree->reinit(dofHandlerVec,
	  dealiiAffineConstraintsVec,
	  dealiiQuadratureTypeVec, 
	  additionalData);
    }

  } // end of namespace basis
} // end of namespace dftefe
