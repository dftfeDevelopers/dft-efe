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
 * @author Avirup Sircar
 */
#include <basis/Defaults.h>
#include <utils/PointImpl.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/LinearSolverFunction.h>
#include <basis/L2ProjectionLinearSolverFunction.h>
#include <linearAlgebra/CGLinearSolver.h>
#include <algorithm>
#include <string>
#include <basis/FEOverlapInverseOperatorContext.h>

namespace dftefe
{
  namespace basis
  {
    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::EnrichmentClassicalInterfaceSpherical(
      std::shared_ptr<const FEOverlapOperatorContext<ValueTypeBasisData, ValueTypeBasisData, 
        memorySpace, dim>> cfeBasisOverlapOperator,  
      std::shared_ptr<const FEBasisDataStorage<ValueTypeBasisData, memorySpace>> cfeBasisDataStorageRhs,     
      std::shared_ptr<const FEBasisManager> cfeBasisManager,
      std::shared_ptr<const FEBasisHandler<ValueTypeBasisData, memorySpace, dim>> cfeBasisHandler,
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const double                     atomPartitionTolerance,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      std::string                       basisInterfaceCoeffConstraint,
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContext,
      const utils::mpi::MPIComm &      comm)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_enrichmentIdsPartition(nullptr)
      , d_cfeBasisHandler(cfeBasisHandler)
      , d_basisInterfaceCoeffConstraint(basisInterfaceCoeffConstraint)
      , d_atomIdsPartition(nullptr)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(atomCoordinatesVec)
      , d_fieldName(fieldName)
      , d_cfeBasisManager(cfeBasisManager)
      , d_overlappingEnrichmentIdsInCells(0)
    {
      d_isOrthogonalized = true;

      if(dim != 3)
        utils::throwException(
        false,
        "Dimension should be 3 for Spherical Enrichment Dofs.");

      utils::throwException(
      (&(cfeBasisDataStorageRhs->getBasisManager()) == &(cfeBasisHandler->getBasisManager()))
      && (&(cfeBasisOverlapOperator->getFEBasisDataStorage().getBasisManager()) == &(cfeBasisHandler->getBasisManager())),
      "The BasisManager of the dataStorage and basisHandler should be same in EnrichmentClassicalInterfaceSpherical() ");

      const FEBasisManager &feBM =
        dynamic_cast<const FEBasisManager &>(cfeBasisHandler->getBasisManager());
      utils::throwException(
        &feBM != nullptr,
        "Could not cast BasisManager to FEBasisManager "
        "in EnrichmentClassicalInterfaceSpherical");

      utils::throwException(
      (&(feBM) ==  cfeBasisManager.get()),
      "The Input basismanager does not match with the basismanager used for creating the basishandler in EnrichmentClassicalInterfaceSpherical() ");

      d_triangulation = feBM.getTriangulation();

      // Partition the enriched dofs based on the BCs and Orthogonalized EFE

      std::vector<utils::Point> cellVertices(0, utils::Point(dim, 0.0));
      std::vector<std::vector<utils::Point>> cellVerticesVector(0);
      auto cell = d_triangulation->beginLocal();
      auto endc = d_triangulation->endLocal();

      for (; cell != endc; cell++)
      {
        (*cell)->getVertices(cellVertices);
        cellVerticesVector.push_back(cellVertices);
      }

      std::vector<double> minbound;
      std::vector<double> maxbound;
      maxbound.resize(dim, 0);
      minbound.resize(dim, 0);

      for (unsigned int k = 0; k < dim; k++)
        {
          double maxtmp = -DBL_MAX, mintmp = DBL_MAX;
          auto   cellIter = cellVerticesVector.begin();
          for (; cellIter != cellVerticesVector.end(); ++cellIter)
            {
              auto cellVertices = cellIter->begin();
              for (; cellVertices != cellIter->end(); ++cellVertices)
                {
                  if (maxtmp <= *(cellVertices->begin() + k))
                    maxtmp = *(cellVertices->begin() + k);
                  if (mintmp >= *(cellVertices->begin() + k))
                    mintmp = *(cellVertices->begin() + k);
                }
            }
          maxbound[k] = maxtmp;
          minbound[k] = mintmp;
        }

      // Create atomIdsPartition Object.
      d_atomIdsPartition =
        std::make_shared<const AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                      minbound,
                                                      maxbound,
                                                      cellVerticesVector,
                                                      atomPartitionTolerance,
                                                      comm);

      // Create enrichmentIdsPartition Object.
      d_enrichmentIdsPartition =
        std::make_shared<const EnrichmentIdsPartition<dim>>(
          d_atomSphericalDataContainer,
          d_atomIdsPartition,
          atomSymbolVec,
          atomCoordinatesVec,
          fieldName,
          minbound,
          maxbound,
          d_triangulation->maxCellDiameter(),
          cellVerticesVector,
          comm);

      d_overlappingEnrichmentIdsInCells = d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();

      // For Non-Periodic BC, a sparse vector d_i with hanging with homogenous BC will be formed which 
      // will be solved by Md = integrateWithBasisValues( homogeneous BC).
      // Form quadRuleContainer for Pristine enrichment.
      // Form OperatorContext object for OverlapMatrix.
      // Form L2ProjectionLinearSolverContext.
      // Get the multiVector for d_basisInterfaceCoeff.

      // Form the quadValuesContainer for pristine enrichment N_A quadValuesEnrichmentFunction

      // Find the total number of local and ghost enrichment ids = num Componebts of the quadValuesContainer

      size_type nTotalEnrichmentIds = d_enrichmentIdsPartition->nTotalEnrichmentIds();

      d_basisInterfaceCoeff = linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>
      (cfeBasisHandler->getMPIPatternP2P(basisInterfaceCoeffConstraint),
        linAlgOpContext, nTotalEnrichmentIds, ValueTypeBasisData());

      quadrature::QuadratureValuesContainer<ValueTypeBasisData, memorySpace> 
        quadValuesEnrichmentFunction(cfeBasisDataStorageRhs->getQuadratureRuleContainer(), nTotalEnrichmentIds) ;

      const size_type numLocallyOwnedCells = cfeBasisManager->nLocallyOwnedCells();
      std::vector<size_type> nQuadPointsInCell(0);
      nQuadPointsInCell.resize(numLocallyOwnedCells, 0);
      size_type cellIndex = 0;
      auto locallyOwnedCellIter = cfeBasisManager->beginLocallyOwnedCells();
      for (; locallyOwnedCellIter != cfeBasisManager->endLocallyOwnedCells();
            ++locallyOwnedCellIter)
      {
        size_type nQuadPointInCell = cfeBasisDataStorageRhs->getQuadratureRuleContainer()
          ->nCellQuadraturePoints(cellIndex);
        std::vector<utils::Point> quadRealPointsVec =
              cfeBasisDataStorageRhs->getQuadratureRuleContainer()->getCellRealPoints(cellIndex);

        for (unsigned int qPoint = 0; qPoint < nQuadPointInCell; qPoint++)
        {
          std::vector<ValueTypeBasisData> enrichmentQuadValue(0);
          enrichmentQuadValue.resize(nTotalEnrichmentIds,(ValueTypeBasisData)0);
          // if the cell has enrichment ids then get the values of them at the quadpoints
            for( auto enrichmentId : d_overlappingEnrichmentIdsInCells[cellIndex])
            {
                size_type atomId =
                  d_enrichmentIdsPartition->getAtomId(enrichmentId);
                size_type qNumberId =
                  (d_enrichmentIdsPartition->getEnrichmentIdAttribute(
                    enrichmentId))
                    .localIdInAtom;
                std::string  atomSymbol = d_atomSymbolVec[atomId];
                utils::Point origin(d_atomCoordinatesVec[atomId]);
                std::vector<std::vector<int>> qNumbers(0);
                qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                    d_fieldName);
                auto sphericalData =
                  d_atomSphericalDataContainer->getSphericalData(
                    atomSymbol, d_fieldName, qNumbers[qNumberId]);
                enrichmentQuadValue[enrichmentId] = 
                  sphericalData->getValue(quadRealPointsVec[qPoint], origin);
            }
            quadValuesEnrichmentFunction.template setCellQuadValues
              <utils::MemorySpace::HOST> (cellIndex, qPoint, enrichmentQuadValue.data());
        }
        cellIndex= cellIndex+1;
      }

      std::shared_ptr<linearAlgebra::LinearSolverFunction<ValueTypeBasisData,
                                                      ValueTypeBasisData,
                                                      memorySpace>> linearSolverFunction =
        std::make_shared<L2ProjectionLinearSolverFunction<ValueTypeBasisData,
                                                      ValueTypeBasisData,
                                                      memorySpace,
                                                      dim>>
                                                      ( cfeBasisHandler,
                                                        cfeBasisOverlapOperator,
                                                        cfeBasisDataStorageRhs,
                                                        quadValuesEnrichmentFunction,
                                                        basisInterfaceCoeffConstraint,
                                                        L2ProjectionDefaults::PC_TYPE,
                                                        linAlgOpContext,
                                                        L2ProjectionDefaults::MAX_CELL_TIMES_NUMVECS);

      linearAlgebra::LinearAlgebraProfiler profiler;

      std::shared_ptr<linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                      ValueTypeBasisData,
                                                      memorySpace>> CGSolve =
        std::make_shared<linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                      ValueTypeBasisData,
                                                      memorySpace>>
                                                      (L2ProjectionDefaults::MAX_ITER,
                                                      L2ProjectionDefaults::ABSOLUTE_TOL,
                                                      L2ProjectionDefaults::RELATIVE_TOL,
                                                      L2ProjectionDefaults::DIVERGENCE_TOL,
                                                      profiler);

      CGSolve->solve(*linearSolverFunction);
      linearSolverFunction->getSolution(d_basisInterfaceCoeff);

      // // Can also do via the M^(-1) route. 

      // linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace> d(
      //       cfeBasisHandler->getMPIPatternP2P(basisInterfaceCoeffConstraint),
      //       linAlgOpContext,
      //       nTotalEnrichmentIds);
      // d.setValue(0.0);

      // FEBasisOperations<ValueTypeBasisData, ValueTypeBasisData, memorySpace, dim> cfeBasisOperations(cfeBasisDataStorageRhs, L2ProjectionDefaults::MAX_CELL_TIMES_NUMVECS);

      // // Integrate this with different quarature rule. (i.e. adaptive for the enrichment functions) , inp will be in adaptive grid
      // cfeBasisOperations.integrateWithBasisValues(quadValuesEnrichmentFunction,
      //                                            *cfeBasisHandler,
      //                                            basisInterfaceCoeffConstraint,
      //                                            d);

      // std::shared_ptr<dftefe::linearAlgebra::OperatorContext<ValueTypeBasisData,
      //                                              ValueTypeBasisData,
      //                                              memorySpace>> MInvContext =
      // std::make_shared<dftefe::basis::FEOverlapInverseOperatorContext<ValueTypeBasisData,
      //                                              ValueTypeBasisData,
      //                                              memorySpace,
      //                                              dim>>
      //                                              (*cfeBasisHandler,
      //                                               *cfeBasisOverlapOperator,
      //                                               basisInterfaceCoeffConstraint,
      //                                               linAlgOpContext);

      // MInvContext->apply(d,d_basisInterfaceCoeff);
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::EnrichmentClassicalInterfaceSpherical(
      std::shared_ptr<const TriangulationBase> triangulation,
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const double                     atomPartitionTolerance,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      const utils::mpi::MPIComm &      comm)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_enrichmentIdsPartition(nullptr)
      , d_atomIdsPartition(nullptr)
      , d_atomSymbolVec(atomSymbolVec)
      , d_atomCoordinatesVec(atomCoordinatesVec)
      , d_fieldName(fieldName)
      , d_triangulation(triangulation)
      , d_overlappingEnrichmentIdsInCells(0)
    {
      d_isOrthogonalized = false;

      if(dim != 3)
        utils::throwException(
        false,
        "Dimension should be 3 for Spherical Enrichment Dofs.");

      // Partition the enriched dofs with pristine enrichment

      std::vector<utils::Point> cellVertices(0, utils::Point(dim, 0.0));
      std::vector<std::vector<utils::Point>> cellVerticesVector(0);
      auto cell = triangulation->beginLocal();
      auto endc = triangulation->endLocal();

      for (; cell != endc; cell++)
      {
        (*cell)->getVertices(cellVertices);
        cellVerticesVector.push_back(cellVertices);
      }

      std::vector<double> minbound;
      std::vector<double> maxbound;
      maxbound.resize(dim, 0);
      minbound.resize(dim, 0);

      for (unsigned int k = 0; k < dim; k++)
        {
          double maxtmp = -DBL_MAX, mintmp = DBL_MAX;
          auto   cellIter = cellVerticesVector.begin();
          for (; cellIter != cellVerticesVector.end(); ++cellIter)
            {
              auto cellVertices = cellIter->begin();
              for (; cellVertices != cellIter->end(); ++cellVertices)
                {
                  if (maxtmp <= *(cellVertices->begin() + k))
                    maxtmp = *(cellVertices->begin() + k);
                  if (mintmp >= *(cellVertices->begin() + k))
                    mintmp = *(cellVertices->begin() + k);
                }
            }
          maxbound[k] = maxtmp;
          minbound[k] = mintmp;
        }

      // Create atomIdsPartition Object.
      d_atomIdsPartition =
        std::make_shared<const AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                      minbound,
                                                      maxbound,
                                                      cellVerticesVector,
                                                      atomPartitionTolerance,
                                                      comm);

      // Create enrichmentIdsPartition Object.
      d_enrichmentIdsPartition =
        std::make_shared<const EnrichmentIdsPartition<dim>>(
          d_atomSphericalDataContainer,
          d_atomIdsPartition,
          atomSymbolVec,
          atomCoordinatesVec,
          fieldName,
          minbound,
          maxbound,
          0,
          cellVerticesVector,
          comm);

      d_overlappingEnrichmentIdsInCells = d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const atoms::AtomSphericalDataContainer>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getAtomSphericalDataContainer() const
    {
      return d_atomSphericalDataContainer;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const EnrichmentIdsPartition<dim>>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getEnrichmentIdsPartition() const
    {
      return d_enrichmentIdsPartition;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const AtomIdsPartition<dim>>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getAtomIdsPartition() const
    {
      return d_atomIdsPartition;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const FEBasisHandler<ValueTypeBasisData, memorySpace, dim>>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getCFEBasisHandler() const
    {
      if( !d_isOrthogonalized )
        utils::throwException(
        false,
        "Cannot call getCFEBasisHandler() for no orthogonalization of EFE mesh.");

      return d_cfeBasisHandler;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const FEBasisManager>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getCFEBasisManager() const
    {
      if( !d_isOrthogonalized )
        utils::throwException(
        false,
        "Cannot call getCFEBasisManager() for no orthogonalization of EFE mesh.");

      return d_cfeBasisManager;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::string
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getBasisInterfaceCoeffConstraint() const
    {
      if( !d_isOrthogonalized )
        utils::throwException(
        false,
        "Cannot call getBasisInterfaceCoeffConstraint() for no orthogonalization of EFE mesh.");

      return d_basisInterfaceCoeffConstraint;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    const linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace> &
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getBasisInterfaceCoeff() const
    {
      if( !d_isOrthogonalized )
        utils::throwException(
        false,
        "Cannot call getBasisInterfaceCoeff() for no orthogonalization of EFE mesh.");

      return d_basisInterfaceCoeff;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    bool
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::isOrthgonalized() const
    {
      return d_isOrthogonalized;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::vector<std::string>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getAtomSymbolVec() const
    {
      return d_atomSymbolVec;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::vector<utils::Point>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getAtomCoordinatesVec() const
    {
      return d_atomCoordinatesVec;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::string
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getFieldName() const
    {
      return d_fieldName;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::shared_ptr<const TriangulationBase>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getTriangulation() const
    {
      return d_triangulation;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    global_size_type
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getEnrichmentId(
      size_type cellId,
      size_type enrichmentCellLocalId
    ) const
    {
      global_size_type enrichmentId = UINT_MAX;
      if (!d_overlappingEnrichmentIdsInCells[cellId].empty())
        {
          if (d_overlappingEnrichmentIdsInCells[cellId].size() >
              enrichmentCellLocalId)
            {
              enrichmentId =
                d_overlappingEnrichmentIdsInCells[cellId]
                                                 [enrichmentCellLocalId];
            }
          else
            {
              utils::throwException(
                false,
                "The requested cell local enrichment id does not exist.");
            }
        }
      else
        {
          utils::throwException(
            false,
            "The requested cell does not have any enrichment ids overlapping with it.");
        }
      return enrichmentId;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    size_type
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getEnrichmentLocalId(
      size_type cellId,
      size_type enrichmentCellLocalId
    ) const
    {
      global_size_type enrichmentId = UINT_MAX;
      size_type enrichmentLocalId = UINT_MAX;
      if (!d_overlappingEnrichmentIdsInCells[cellId].empty())
        {
          if (d_overlappingEnrichmentIdsInCells[cellId].size() >
              enrichmentCellLocalId)
            {
              enrichmentId =
                d_overlappingEnrichmentIdsInCells[cellId]
                                                 [enrichmentCellLocalId];
              enrichmentLocalId = getEnrichmentLocalId(enrichmentId);
            }
          else
            {
              utils::throwException(
                false,
                "The requested cell local enrichment id does not exist.");
            }
        }
      else
        {
          utils::throwException(
            false,
            "The requested cell does not have any enrichment ids overlapping with it.");
        }
      return enrichmentLocalId;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    size_type
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getEnrichmentLocalId(
      global_size_type enrichmentId) const
    {
      size_type enrichmentLocalId = UINT_MAX;
      global_size_type locallyOwnedIdsBegin = d_enrichmentIdsPartition->locallyOwnedEnrichmentIds().first;
      global_size_type locallyOwnedIdsEnd = d_enrichmentIdsPartition->locallyOwnedEnrichmentIds().second;

      if(enrichmentId < locallyOwnedIdsEnd && 
          enrichmentId >= locallyOwnedIdsBegin)
        enrichmentLocalId = enrichmentId - locallyOwnedIdsBegin;

      else
      {
        int c = 0;
        for(auto it : d_enrichmentIdsPartition->ghostEnrichmentIds())
        {
          if(it == enrichmentId)
            {return c; break;}
          else c+=1;
        }
        // auto it = std::find(d_enrichmentIdsPartition->ghostEnrichmentIds().begin(), 
        //   d_enrichmentIdsPartition->ghostEnrichmentIds().end(), enrichmentId);
        // if(it != d_enrichmentIdsPartition->ghostEnrichmentIds().end())
        //   enrichmentLocalId = locallyOwnedIdsEnd + it - d_enrichmentIdsPartition->ghostEnrichmentIds().begin();
        if(c == d_enrichmentIdsPartition->ghostEnrichmentIds().size())
        {
          utils::throwException(
            false,
            "The requested enrichmentId is not found in the locally owned or ghost set." + std::to_string(enrichmentId) + " " + 
             std::to_string(locallyOwnedIdsBegin) + " " + std::to_string(locallyOwnedIdsEnd) + " " + 
             std::to_string(d_enrichmentIdsPartition->ghostEnrichmentIds()[0]) );
        }
      }
      return enrichmentLocalId;
    }

  } // namespace basis
} // namespace dftefe