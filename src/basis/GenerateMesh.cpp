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
 * The GenerateMesh for adaptive mesh generation around atom is borrowed from
 * dftfe.
 */

#include <basis/Defaults.h>
#include <basis/GenerateMesh.h>
#include <cmath>

namespace dftefe
{
  namespace basis
  {
    namespace GenerateMeshInternal
    {
      void
      checkTriangulationEqualityAcrossProcessorPools(
        const TriangulationBase &triangulation,
        const size_type          numLocallyOwnedCells,
        const MPI_Comm &         interpoolComm)
      {
        size_type numberGlobalCellsParallelMinPools =
          triangulation.nGlobalCells();
        utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          &numberGlobalCellsParallelMinPools,
          1,
          utils::mpi::Types<size_type>::getMPIDatatype(),
          utils::mpi::MPIMin,
          interpoolComm);

        size_type numberGlobalCellsParallelMaxPools =
          triangulation.nGlobalCells();
        utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          &numberGlobalCellsParallelMaxPools,
          1,
          utils::mpi::Types<size_type>::getMPIDatatype(),
          utils::mpi::MPIMax,
          interpoolComm);

        DFTEFE_AssertWithMsg(
          numberGlobalCellsParallelMinPools ==
            numberGlobalCellsParallelMaxPools,
          "Number of global cells are different across pools.");

        size_type numberLocalCellsMinPools = numLocallyOwnedCells;
        utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          &numberLocalCellsMinPools,
          1,
          utils::mpi::Types<size_type>::getMPIDatatype(),
          utils::mpi::MPIMin,
          interpoolComm);

        size_type numberLocalCellsMaxPools = numLocallyOwnedCells;
        utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          &numberLocalCellsMaxPools,
          1,
          utils::mpi::Types<size_type>::getMPIDatatype(),
          utils::mpi::MPIMax,
          interpoolComm);

        DFTEFE_AssertWithMsg(
          numberLocalCellsMinPools == numberLocalCellsMaxPools,
          "Number of local cells are different across pools or "
          "in other words the physical partitions don't have the "
          "same ordering across pools.");
      }
    } // namespace GenerateMeshInternal

    GenerateMesh::GenerateMesh(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<utils::Point> &domainBoundingVectors,
      double                           radiusAroundAtom,
      double                           meshSizeAroundAtom,
      const std::vector<bool> &        isPeriodicFlags,
      const basis::CellMappingBase &   cellMapping,
      const MPI_Comm &                 mpiDomainCommunicator,
      const MPI_Comm &                 mpiInterpoolCommunicator)
      : d_dim(atomCoordinates[0].size())
      , d_atomCoordinates(atomCoordinates)
      , d_domainBoundingVectors(domainBoundingVectors)
      , d_radiusAtAtom(0)
      , d_radiusAroundAtom(radiusAroundAtom)
      , d_meshSizeAroundAtom(meshSizeAroundAtom)
      , d_meshSizeAtAtom(1e10)
      , d_mpiDomainCommunicator(mpiDomainCommunicator)
      , d_mpiInterpoolCommunicator(mpiInterpoolCommunicator)
      , d_isPeriodicFlags(isPeriodicFlags)
      , d_cellMapping(cellMapping)
      , d_rootCout(std::cout)
      , d_maxRefinementSteps(GenerateMeshDefaults::MAX_REFINEMENT_STEPS)
    {
      d_adaptiveWithFineMesh = false;
      int rank;
      utils::mpi::MPICommRank(mpiDomainCommunicator, &rank);
      d_rootCout.setCondition(rank == 0);
    }

    GenerateMesh::GenerateMesh(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<utils::Point> &domainBoundingVectors,
      double                           radiusAtAtom,
      double                           meshSizeAtAtom,
      double                           radiusAroundAtom,
      double                           meshSizeAroundAtom,
      const std::vector<bool> &        isPeriodicFlags,
      const basis::CellMappingBase &   cellMapping,
      const MPI_Comm &                 mpiDomainCommunicator,
      const MPI_Comm &                 mpiInterpoolCommunicator)
      : d_dim(atomCoordinates[0].size())
      , d_atomCoordinates(atomCoordinates)
      , d_domainBoundingVectors(domainBoundingVectors)
      , d_radiusAtAtom(radiusAtAtom)
      , d_radiusAroundAtom(radiusAroundAtom)
      , d_meshSizeAroundAtom(meshSizeAroundAtom)
      , d_meshSizeAtAtom(meshSizeAtAtom)
      , d_mpiDomainCommunicator(mpiDomainCommunicator)
      , d_mpiInterpoolCommunicator(mpiInterpoolCommunicator)
      , d_isPeriodicFlags(isPeriodicFlags)
      , d_cellMapping(cellMapping)
      , d_rootCout(std::cout)
      , d_maxRefinementSteps(GenerateMeshDefaults::MAX_REFINEMENT_STEPS)
    {
      d_adaptiveWithFineMesh = true;
      int rank;
      utils::mpi::MPICommRank(mpiDomainCommunicator, &rank);
      d_rootCout.setCondition(rank == 0);
    }

    void
    GenerateMesh::generateCoarseMesh(TriangulationBase &      triangulation,
                                     const std::vector<bool> &isPeriodicFlags)
    {
      //
      // compute magnitudes of domainBounding Vectors
      //
      // CHANGE - dim
      std::vector<double> domainBoundingVectorMag(d_dim, 0);

      for (size_type i = 0; i < d_dim; i++)
        {
          double sum = 0;
          for (size_type j = 0; j < d_dim; j++)
            {
              sum +=
                d_domainBoundingVectors[i][j] * d_domainBoundingVectors[i][j];
            }
          domainBoundingVectorMag[i] = std::sqrt(sum);
        }

      std::vector<size_type> subdivisions(d_dim, 1.0);

      std::vector<double> numberIntervalsEachDirection(0);

      double largestMeshSizeAroundAtom = d_meshSizeAroundAtom;

      std::vector<double> baseMeshSize(d_dim, 0.0);
      if (std::any_of(isPeriodicFlags.begin(),
                      isPeriodicFlags.end(),
                      [](bool v) { return v; }))
        {
          const double targetBaseMeshSize =
            (*(std::min_element(domainBoundingVectorMag.begin(),
                                domainBoundingVectorMag.end())) > 50.0) ?
              (4.0) :
              std::max(2.0, largestMeshSizeAroundAtom);

          for (auto &i : baseMeshSize)
            {
              i = std::pow(2,
                           round(log2(targetBaseMeshSize /
                                      largestMeshSizeAroundAtom))) *
                  largestMeshSizeAroundAtom;
            }
        }
      else
        {
          for (auto &i : baseMeshSize)
            {
              i =
                std::pow(2,
                         round(log2(
                           (std::max(8.0, largestMeshSizeAroundAtom) /*4.0*/) /
                           largestMeshSizeAroundAtom))) *
                largestMeshSizeAroundAtom;
            }
        }

      for (size_type i = 0; i < d_dim; i++)
        numberIntervalsEachDirection.push_back(domainBoundingVectorMag[i] /
                                               baseMeshSize[i]);

      for (size_type i = 0; i < d_dim; i++)
        {
          const double temp = numberIntervalsEachDirection[i] -
                              std::floor(numberIntervalsEachDirection[i]);
          if (temp >= 0.5)
            subdivisions[i] = std::ceil(numberIntervalsEachDirection[i]);
          else
            subdivisions[i] = std::floor(numberIntervalsEachDirection[i]);
        }

      triangulation.createUniformParallelepiped(subdivisions,
                                                d_domainBoundingVectors,
                                                isPeriodicFlags);

      //
      // Translate the main grid so that midpoint is at center
      //
      utils::Point translation(d_dim, 0);
      for (auto &i : d_domainBoundingVectors)
        translation += 0.5 * i;
      triangulation.shiftTriangulation(-1.0 * translation);
      triangulation.finalizeTriangulationConstruction();

      /*
      // collect periodic faces of the first level mesh to set up periodic
      // boundary conditions later
      //
      meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,
                                                   d_domainBoundingVectors,
                                                   d_mpiCommParent,
                                                   d_dftParams);
      */

      d_rootCout << std::endl
                 << "Coarse triangulation number of elements: "
                 << triangulation.nGlobalCells() << std::endl;
    }

    bool
    GenerateMesh::refinementAlgorithm(
      TriangulationBase &             triangulation,
      std::vector<size_type> &        locallyOwnedCellsRefineFlags,
      std::map<size_type, size_type> &cellIdToCellRefineFlagMapLocal,
      const basis::CellMappingBase &  cellMapping)
    {
      locallyOwnedCellsRefineFlags.clear();
      cellIdToCellRefineFlagMapLocal.clear();
      auto cell = triangulation.beginLocal();
      auto endc = triangulation.endLocal();

      std::map<size_type, size_type> cellIdToLocallyOwnedId;
      size_type                      locallyOwnedCount = 0;

      bool   isAnyCellRefined           = false;
      double smallestMeshSizeAroundAtom = d_meshSizeAroundAtom;

      std::vector<double>    atomPointsLocal;
      std::vector<size_type> atomIdsLocal;
      std::vector<double>    meshSizeAroundAtomLocalAtoms;
      std::vector<double>    radiusAroundAtomLocalAtoms;
      for (size_type iAtom = 0; iAtom < (d_atomCoordinates.size()); iAtom++)
        {
          for (size_type i = 0; i < d_dim; i++)
            atomPointsLocal.push_back(d_atomCoordinates[iAtom][i]);
          atomIdsLocal.push_back(iAtom);

          meshSizeAroundAtomLocalAtoms.push_back(d_meshSizeAroundAtom);
          radiusAroundAtomLocalAtoms.push_back(d_radiusAroundAtom);
        }

      size_type cellIndex = 0;
      for (; cell != endc; ++cell)
        {
          cellIdToLocallyOwnedId[cellIndex] = locallyOwnedCount;
          locallyOwnedCount++;

          utils::Point center(d_dim, 0.0);
          (*cell)->center(center);

          double currentMeshSize = (*cell)->minimumVertexDistance();

          bool cellRefineFlag = false;

          // loop over all atoms
          double       distanceToClosestAtom = 1e8;
          utils::Point closestAtom(d_dim, 0.0);
          size_type    closestId = 0;
          for (size_type n = 0; n < d_atomCoordinates.size(); n++)
            {
              double dist = 0;
              for (size_type j = 0; j < d_dim; j++)
                {
                  dist += std::pow((center[j] - d_atomCoordinates[n][j]), 2);
                }
              dist = std::sqrt(dist);

              if (dist < distanceToClosestAtom)
                {
                  distanceToClosestAtom = dist;
                  closestAtom           = d_atomCoordinates[n];
                  closestId             = n;
                }
            }

          bool inOuterAtomBall = false;

          if (distanceToClosestAtom <= radiusAroundAtomLocalAtoms[closestId])
            inOuterAtomBall = true;

          if (inOuterAtomBall &&
              (currentMeshSize > 1.2 * meshSizeAroundAtomLocalAtoms[closestId]))
            cellRefineFlag = true;

          if (d_adaptiveWithFineMesh)
            {
              bool inInnerAtomBall = false;

              if (distanceToClosestAtom <= d_radiusAtAtom)
                inInnerAtomBall = true;

              if (inInnerAtomBall && currentMeshSize > 1.2 * d_meshSizeAtAtom)
                cellRefineFlag = true;
            }


          bool         isPointInside = true;
          utils::Point closestAtomParametricPoint(d_dim, 0);
          (*cell)->getParametricPoint(closestAtom,
                                      cellMapping,
                                      closestAtomParametricPoint);

          double dist = (*cell)->distanceToUnitCell(closestAtomParametricPoint);

          if (dist < 1e-08 &&
              ((d_adaptiveWithFineMesh && currentMeshSize > d_meshSizeAtAtom) ||
               (currentMeshSize > meshSizeAroundAtomLocalAtoms[closestId])))
            cellRefineFlag = true;


          size_type cellRefineFlagSizeType = (size_type)cellRefineFlag;

          utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &cellRefineFlagSizeType,
            1,
            utils::mpi::Types<size_type>::getMPIDatatype(),
            utils::mpi::MPIMax,
            d_mpiInterpoolCommunicator);

          cellRefineFlag = cellRefineFlagSizeType;

          //
          // set refine flags
          if (cellRefineFlag)
            {
              locallyOwnedCellsRefineFlags.push_back(1);
              cellIdToCellRefineFlagMapLocal[cellIndex] = 1;
              (*cell)->setRefineFlag();
              isAnyCellRefined = true;
            }
          else
            {
              cellIdToCellRefineFlagMapLocal[cellIndex] = 0;
              locallyOwnedCellsRefineFlags.push_back(0);
            }
          cellIndex += 1;
        }
      return isAnyCellRefined;
    }

    bool
    GenerateMesh::refineInsideSystemNonPeriodicAlgorithm(
      TriangulationBase &  triangulation,
      std::vector<double> &maxAtomCoordinates,
      std::vector<double> &minAtomCoordinates)
    {
      auto cell = triangulation.beginLocal();
      auto endc = triangulation.endLocal();

      double    currentMeshSize  = (*cell)->minimumVertexDistance();
      bool      isAnyCellRefined = false;
      size_type cellIndex        = 0;
      for (; cell != endc; ++cell)
        {
          utils::Point center(d_dim, 0.0);
          (*cell)->center(center);

          bool cellRefineFlag = false;

          double boundingBoxDom = 0;
          for (int i = 0; i < d_dim; i++)
            {
              double axesLen = std::max(std::abs(maxAtomCoordinates[i]),
                                        std::abs(minAtomCoordinates[i])) +
                               std::max(15.0, d_radiusAroundAtom);
              if (boundingBoxDom < axesLen)
                boundingBoxDom = axesLen;
            }
          bool val = true;
          for (int i = 0; i < d_dim; i++)
            {
              if (std::abs(center[i]) > boundingBoxDom)
                {
                  val = false;
                  break;
                }
              // val += std::pow((center[i]/ axesLen),2);
            }
          if (val && currentMeshSize > 4)
            {
              cellRefineFlag = true;
            }

          size_type cellRefineFlagSizeType = (size_type)cellRefineFlag;

          utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &cellRefineFlagSizeType,
            1,
            utils::mpi::Types<size_type>::getMPIDatatype(),
            utils::mpi::MPIMax,
            d_mpiInterpoolCommunicator);

          cellRefineFlag = cellRefineFlagSizeType;

          //
          // set coarsen flags
          if (cellRefineFlag)
            {
              (*cell)->setRefineFlag();
              isAnyCellRefined = true;
            }
          cellIndex += 1;
        }
      return isAnyCellRefined;
    }

    void
    GenerateMesh::createMesh(TriangulationBase &triangulation)
    {
      triangulation.initializeTriangulationConstruction();
      generateCoarseMesh(triangulation, d_isPeriodicFlags);

      bool refineFlag = true;
      // all directions non periodic, note, can be extended to some dicrecs.
      // non-periodic
      if (!(std::any_of(d_isPeriodicFlags.begin(),
                        d_isPeriodicFlags.end(),
                        [](bool v) { return v; })))
        {
          std::vector<double> maxAtomCoordinates(d_dim, 0);
          std::vector<double> minAtomCoordinates(d_dim, 0);

          for (int j = 0; j < d_dim; j++)
            {
              for (int i = 0; i < d_atomCoordinates.size(); i++)
                {
                  if (maxAtomCoordinates[j] < d_atomCoordinates[i][j])
                    maxAtomCoordinates[j] = d_atomCoordinates[i][j];
                  if (minAtomCoordinates[j] > d_atomCoordinates[i][j])
                    minAtomCoordinates[j] = d_atomCoordinates[i][j];
                }
            }
          refineFlag =
            refineInsideSystemNonPeriodicAlgorithm(triangulation,
                                                   maxAtomCoordinates,
                                                   minAtomCoordinates);

          // This sets the global refinement sweep flag
          size_type refineFlagSizeType = (size_type)refineFlag;

          utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &refineFlagSizeType,
            1,
            utils::mpi::Types<size_type>::getMPIDatatype(),
            utils::mpi::MPIMax,
            d_mpiDomainCommunicator);

          refineFlag = refineFlagSizeType;
        }
      triangulation.executeCoarseningAndRefinement();
      triangulation.finalizeTriangulationConstruction();

      d_triaCurrentRefinement.clear();

      //
      // Call only refinementAlgorithm. Multilevel refinement is
      // performed until refinementAlgorithm does not set refinement flags on
      // any cell.
      //
      size_type numLevels = 0;
      refineFlag          = true;
      while (refineFlag)
        {
          refineFlag = false;
          std::vector<size_type>         locallyOwnedCellsRefineFlags;
          std::map<size_type, size_type> cellIdToCellRefineFlagMapLocal;

          refineFlag = refinementAlgorithm(triangulation,
                                           locallyOwnedCellsRefineFlags,
                                           cellIdToCellRefineFlagMapLocal,
                                           d_cellMapping);

          // This sets the global refinement sweep flag
          size_type refineFlagSizeType = (size_type)refineFlag;

          utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &refineFlagSizeType,
            1,
            utils::mpi::Types<size_type>::getMPIDatatype(),
            utils::mpi::MPIMax,
            d_mpiDomainCommunicator);

          refineFlag = refineFlagSizeType;

          // Refine
          if (refineFlag)
            {
              if (numLevels < d_maxRefinementSteps)
                {
                  d_rootCout << "refinement in progress, level: " << numLevels
                             << std::endl;

                  d_triaCurrentRefinement.push_back(std::vector<bool>());
                  triangulation.saveRefineFlags(
                    d_triaCurrentRefinement[numLevels]);

                  triangulation.executeCoarseningAndRefinement();
                  numLevels++;
                }
              else
                {
                  refineFlag = false;
                }
            }
          triangulation.finalizeTriangulationConstruction();
        }

      //
      // compute some adaptive mesh metrics
      //
      double    minElemLength        = d_meshSizeAroundAtom;
      double    maxElemLength        = 0.0;
      auto      cell                 = triangulation.beginLocal();
      auto      endc                 = triangulation.endLocal();
      size_type numLocallyOwnedCells = 0;
      for (; cell != endc; ++cell)
        {
          numLocallyOwnedCells++;
          if ((*cell)->minimumVertexDistance() < minElemLength)
            minElemLength = (*cell)->minimumVertexDistance();

          if ((*cell)->minimumVertexDistance() > maxElemLength)
            maxElemLength = (*cell)->minimumVertexDistance();
        }

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &minElemLength,
        1,
        utils::mpi::Types<double>::getMPIDatatype(),
        utils::mpi::MPIMin,
        d_mpiDomainCommunicator);

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &maxElemLength,
        1,
        utils::mpi::Types<double>::getMPIDatatype(),
        utils::mpi::MPIMax,
        d_mpiDomainCommunicator);

      //
      // print out adaptive mesh metrics and check mesh generation
      // synchronization across pools
      //

      d_rootCout << "Triangulation generation summary: " << std::endl
                 << " num elements: " << triangulation.nGlobalCells()
                 << ", num refinement levels: " << numLevels
                 << ", min element length: " << minElemLength
                 << ", max element length: " << maxElemLength << std::endl;

      GenerateMeshInternal::checkTriangulationEqualityAcrossProcessorPools(
        triangulation, numLocallyOwnedCells, d_mpiInterpoolCommunicator);

      const size_type numberGlobalCellsParallel = triangulation.nGlobalCells();

      d_rootCout << " numParallelCells: " << numberGlobalCellsParallel
                 << std::endl;
    }

  } // end of namespace basis
} // end of namespace dftefe
