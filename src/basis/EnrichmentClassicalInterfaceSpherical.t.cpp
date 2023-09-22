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
#include <utils/Point.h>
#include <basis/Defaults.h>
#include <utils/PointImpl.h>
#include <basis/EnrichementClassicalInterfaceSpherical.h>

namespace dftefe
{
  namespace basis
  {
    namespace EnrichmentClassicalInterfaceSphericalInternal
    {
      template <typename ValueTypeBasisData>
      ValueTypeBasisData
      getEnrichmentValue(std::shared_ptr<const EnrichmentIdsPartition<dim>>
                                  enrichmentIdsPartition,
                                  std::shared_ptr<const AtomSphericalDataContainer> atomSphericalDataContainer,
                                  const std::vector<std::string> & atomSymbolVec,
                                  const std::vector<utils::Point> &atomCoordinatesVec,
                                  const std::string  fieldName,
                                  const utils::Point &point)
      {
        ValueTypeBasisData retValue = (ValueTypeBasisData)0;
        std::pair<global_size_type, global_size_type> locallyOwnedEnrichemntIds =
          enrichmentIdsPartition->locallyOwnedEnrichmentIds();
        std::vector<global_size_type> ghostEnrichmentIds =
          enrichmentIdsPartition->ghostEnrichmentIds();
            for (global_size_type i = locallyOwnedEnrichemntIds.first;
                i < locallyOwnedEnrichemntIds.second;
                i++)
              {
                size_type atomId = enrichmentIdsPartition->getAtomId(i);
                size_type qNumberId =
                  (enrichmentIdsPartition->getEnrichmentIdAttribute(i))
                    .localIdInAtom;
                std::string  atomSymbol = atomSymbolVec[atomId];
                utils::Point origin(atomCoordinatesVec[atomId]);
                std::vector<std::vector<int>> qNumbers(0);
                qNumbers = atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                    fieldName);
                auto sphericalData =
                  atomSphericalDataContainer->getSphericalData(
                    atomSymbol, fieldName, qNumbers[qNumberId]);
                retValue += sphericalData->getValue(point, origin);
              }
            for (auto i : ghostEnrichmentIds)
              {
                size_type atomId = enrichmentIdsPartition->getAtomId(i);
                size_type qNumberId =
                  (enrichmentIdsPartition->getEnrichmentIdAttribute(i))
                    .localIdInAtom;
                std::string  atomSymbol = atomSymbolVec[atomId];
                utils::Point origin(atomCoordinatesVec[atomId]);
                std::vector<std::vector<int>> qNumbers(0);
                qNumbers = atomSphericalDataContainer->getQNumbers(atomSymbol,
                                                                    fieldName);
                auto sphericalData =
                  atomSphericalDataContainer->getSphericalData(
                    atomSymbol, fieldName, qNumbers[qNumberId]);
                retValue += sphericalData->getValue(point, origin);
              }
        return retValue;
      }
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::EnrichmentClassicalInterfaceSpherical(
      std::shared_ptr<const FEBasisDataStorage> cfeBasisDataStorage,
      std::shared_ptr<const FEBasisHandler> cfeBasisHandler,
      std::shared_ptr<const quadrature::QuadratureRuleAttributes> l2ProjQuadAttr,
      std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const double                     atomPartitionTolerance,
      const std::vector<std::string> & atomSymbolVec,
      const std::vector<utils::Point> &atomCoordinatesVec,
      const std::string                fieldName,
      std::string                       basisInterfaceCoeffConstraint,
      std::shared_ptr< const linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContext,
      const utils::mpi::MPIComm &      comm)
      : d_atomSphericalDataContainer(atomSphericalDataContainer)
      , d_enrichmentIdsPartition(nullptr)
      , d_cfeBasisHandler(cfeBasisHandler)
      , d_basisInterfaceCoeffConstraint(basisInterfaceCoeffConstraint)
      , d_atomIdsPartition(nullptr)
    {
      d_isOrthogonalized = true;
      const size_type nEnrichmentRanges = 1;
      d_basisInterfaceCoeff(cfeBasisHandler->getMPIPatternP2P(basisInterfaceCoeffConstraint), 
        linAlgOpContext, nEnrichmentRanges, ValueTypeBasisData());

      if(dim != 3)
        utils::throwException(
        false,
        "Dimension should be 3 for Spherical Enrichment Dofs.");

      // Partition the enriched dofs based on the BCs and Orthogonalized EFE

      std::vector<utils::Point> cellVertices(0, utils::Point(dim, 0.0));
      std::vector<std::vector<utils::Point>> cellVerticesVector(0);
      auto cell = (cfeBasisDataStorage->getBasisManager).beginLocallyOwnedCells();
      auto endc = (cfeBasisDataStorage->getBasisManager).endLocallyOwnedCells();

      for (; cell != endc; cell++)
      {
        (cell)->getVertices(cellVertices);
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
        std::make_shared<const AtomIdsPartition<dim>>(d_atomCoordinatesVec,
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
          cfeBasisDataStorage->getTriangulation()->maxCellDiameter(),
          cellVerticesVector,
          comm);

      // For Non-Periodic BC, a sparse vector d_i with hanging with homogenous BC will be formed which 
      // will be solved by Md = integrateWithBasisValues( homogeneous BC).
      // Form quadRuleContainer for Pristine enrichment.
      // Form OperatorContext object for OverlapMatrix.
      // Form L2ProjectionLinearSolverContext.
      // Get the multiVector for d_basisInterfaceCoeff.

      // Form the quadValuesContainer for pristine enrichment N_A quadValuesEnrichmentFunction

      quadrature::QuadratureValuesContainer<ValueTypeBasisData, memorySpace> 
        quadValuesEnrichmentFunction(cfeBasisDataStorage->getQuadratureRuleContainer(l2ProjQuadAttr), 
        nEnrichmentRanges);

      for(size_type i = 0 ; i < quadValuesEnrichmentFunction.nCells() ; i++)
      {
        size_type quadId = 0;
        for (auto quadPoint : cfeBasisDataStorage->getQuadratureRuleContainer().getCellRealPoints(i))
        {
          std::vector<ValueTypeBasisData> enrichmentQuadValue(nEnrichmentRanges, 0);
          for ( size_type i = 0 ; i < nEnrichmentRanges ; i++)
          {
            enrichmentQuadValue[i] = EnrichmentClassicalInterfaceSphericalInternal::
                                    getEnrichmentValue<ValueTypeBasisData>(
                                                                d_enrichmentIdsPartition,
                                                                d_atomSphericalDataContainer,
                                                                d_atomSymbolVec,
                                                                d_atomCoordinatesVec,
                                                                d_fieldName,
                                                                quadPoint)
          }
          ValueTypeBasisData* quadValuePtr = enrichmentQuadValue.data();
          quadValuesEnrichmentFunction.setCellQuadValues<utils::MemorySpace::HOST> (i, quadId, quadValuePtr);
          quadId = quadId + 1;
        }
      }

      // Set up basis Operations
      FEBasisOperations<ValueTypeBasisData, ValueTypeBasisData, memorySpace, dim> cfeBasisOp(cfeBasisDataStorage, L2ProjectionDefaults::MAX_CELL_TIMES_NUMVECS);

      std::shared_ptr<linearAlgebra::LinearSolverFunction<ValueTypeBasisData,
                                                      ValueTypeBasisData,
                                                      memorySpace>> linearSolverFunction =
        std::make_shared<physics::L2ProjectionLinearSolverFunctionFE<ValueTypeBasisData,
                                                      ValueTypeBasisData,
                                                      memorySpace,
                                                      dim>>
                                                      ( cfeBasisHandler,
                                                        cfeBasisDataStorage,
                                                        cfeBasisOp,
                                                        quadValuesEnrichmentFunction,
                                                        l2ProjectectionQuadAttr,
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
    {
      d_isOrthogonalized = false;

      if(dim != 3)
        utils::throwException(
        false,
        "Dimension should be 3 for Spherical Enrichment Dofs.");

      // Partition the enriched dofs based on the BCs and Orthogonalized EFE

      std::vector<utils::Point> cellVertices(0, utils::Point(dim, 0.0));
      std::vector<std::vector<utils::Point>> cellVerticesVector(0);
      auto cell = triangulation->beginLocal();
      auto endc = triangulation->endLocal();

      for (; cell != endc; cell++)
      {
        (cell)->getVertices(cellVertices);
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
        std::make_shared<const AtomIdsPartition<dim>>(d_atomCoordinatesVec,
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
    std::shared_ptr<const FEBasisHandler>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getCFEBasisHandler() const
    {
      if( !d_isOrthgonalized )
        utils::throwException(
        false,
        "Cannot call getCFEBasisHandler() for no orthogonalization of EFE.");
      else
      return d_CFEBasisHandler;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    std::string
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getBasisInterfaceCoeffConstraint() const
    {
      if( !d_isOrthgonalized )
        utils::throwException(
        false,
        "Cannot call getBasisInterfaceCoeffConstraint() for no orthogonalization of EFE.");
      else
      return d_basisInterfaceCoeffConstraint;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::getBasisInterfaceCoeff() const
    {
      if( !d_isOrthgonalized )
        utils::throwException(
        false,
        "Cannot call getBasisInterfaceCoeff() for no orthogonalization of EFE.");
      else
      return d_basisInterfaceCoeff;
    }

    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace, size_type dim>
    bool
    EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData, memorySpace,  dim>::isOrthgonalized() const
    {
      return d_isOrthgonalized;
    }

  } // namespace basis
} // namespace dftefe
