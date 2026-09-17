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

/*
 * Parallel test of EnrichmentIdsPartition, with and without periodicity.
 *
 * The periodic run is the point of the test. It is checked against an oracle
 * built inside the test rather than against stored reference output, so that
 * the check stays valid when the mesh, the atom count or the number of
 * processors changes:
 *
 *   Oracle A (partition logic). For every locally owned cell the test
 *     recomputes, from the atom coordinates and the generator's truncated
 *     image list, which enrichments overlap that cell and which origins --
 *     master atom and periodic images -- reach it, applying the same
 *     per orbital cutoff the class applies. The recomputed sequence must
 *     match overlappingEnrichmentIdsInCells() and getAtomIdsForCellEnrich()
 *     entry by entry. This pins the extended atom id encoding, the offsets
 *     into the per cell origin list, the per orbital filtering and, in
 *     parallel, the case a serial run cannot reach: an atom that is only
 *     visible on this processor through one of its images.
 *
 *   Oracle B (envelope coverage). Independently of the generator, the test
 *     enumerates lattice translations of every atom and requires that any
 *     translation reaching a locally owned cell is present in the truncated
 *     image list. A truncated envelope that is too small would silently drop
 *     enrichments rather than fail, so it is checked directly.
 *
 * On top of those, the ownership of the enrichment ids across processors is
 * checked to be a contiguous disjoint tiling, and the ghost set of each
 * processor is pinned exactly: it must be the ids its own cells overlap and it
 * does not own, no fewer and no more. Too few would leave a processor unable
 * to evaluate an enrichment its cells need; too many would inflate the halo
 * payload without ever failing anything, which is the cost periodicity is most
 * likely to introduce quietly. Each ghost is also required to be owned by
 * exactly one processor.
 *
 * The periodic geometry is sized from the enrichment cutoff read out of the
 * atom data file so that images are guaranteed to matter: the box edge is 2.5
 * times the largest enrichment reach, which leaves cells that no master atom
 * reaches but some image does. The test refuses to pass if no such cell turned
 * up, so that it cannot succeed by exercising nothing.
 *
 * Both meshes are then refined twice in shrinking shells around the atoms, so
 * the cells come in three sizes and carry hanging nodes. A uniform mesh gives
 * every cell the same relation to the cutoff balls and tests one case many
 * times over; refined cells vary in size by a factor of four, and since the
 * atoms sit next to the faces the finest of them straddle the periodic
 * boundary, which is where an image reaches a cell its master cannot.
 *
 * The non periodic run is a separate, larger box with the atoms pulled into
 * the middle. Without periodicity an enrichment ball is not allowed to spill
 * out of the domain, so that geometry keeps every ball clear of the boundary
 * by more than one reach. It exists to confirm the periodic additions left the
 * non periodic behaviour untouched: one origin per enrichment, the master
 * itself.
 */

// For the Base class
#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/TriangulationCellBase.h>
#include <basis/AtomIdsPartition.h>
#include <basis/EnrichmentIdsPartition.h>
#include <basis/PeriodicImageAtomGenerator.h>
#include <atoms/AtomSphericalDataContainer.h>

// Header for the utils class
#include <utils/PointImpl.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <cfloat>

namespace
{
  const unsigned int dim = 3;

  using dftefe::global_size_type;
  using dftefe::size_type;
  using dftefe::utils::Point;

  // The overlap test of EnrichmentIdsPartition, reproduced so the oracle
  // admits exactly what the class admits: an axis aligned box of half width
  // cutoff about origin, against the cell bounding box.
  bool
  ballOverlapsBox(const Point &              origin,
                  const double               cutoff,
                  const std::vector<double> &boxMin,
                  const std::vector<double> &boxMax)
  {
    for (size_type k = 0; k < dim; k++)
      {
        const double a = boxMin[k];
        const double b = boxMax[k];
        const double c = origin[k] - cutoff;
        const double d = origin[k] + cutoff;
        if ((c < a && d < a) || (c > b && d > b))
          return false;
      }
    return true;
  }

  void
  boundingBox(const std::vector<Point> &vertices,
              std::vector<double> &     boxMin,
              std::vector<double> &     boxMax)
  {
    boxMin.resize(dim);
    boxMax.resize(dim);
    for (size_type k = 0; k < dim; k++)
      {
        double minTmp = DBL_MAX, maxTmp = -DBL_MAX;
        for (auto &vertex : vertices)
          {
            minTmp = std::min(minTmp, vertex[k]);
            maxTmp = std::max(maxTmp, vertex[k]);
          }
        boxMin[k] = minTmp;
        boxMax[k] = maxTmp;
      }
  }

  // The per orbital reach of an enrichment, as EnrichmentIdsPartition computes
  // it. Any divergence here would make the oracle agree with a wrong
  // implementation, so it is written once and used everywhere in this file.
  double
  enrichmentReach(const std::shared_ptr<const dftefe::atoms::SphericalData> &sd,
                  const double additionalCutoff)
  {
    return sd->getCutoff() + sd->getCutoff() / sd->getSmoothness() +
           additionalCutoff;
  }

  // Distance to an atom, wrapped across whichever directions are periodic, so
  // that a refinement region around an atom sitting next to a face closes up
  // on the other side the way a real periodic mesh does.
  double
  minimumImageDistance(const Point &             a,
                       const Point &             b,
                       const std::vector<Point> &domainVectors,
                       const std::vector<bool> & isPeriodicFlags)
  {
    double distanceSquared = 0.0;
    for (size_type k = 0; k < dim; k++)
      {
        double delta = a[k] - b[k];
        if (isPeriodicFlags[k])
          {
            const double edge = domainVectors[k][k];
            delta -= edge * std::round(delta / edge);
          }
        distanceSquared += delta * delta;
      }
    return std::sqrt(distanceSquared);
  }

  // A parallelepiped refined in shrinking shells around the atoms, its locally
  // owned cell vertices and the processor bounding box those cells span, which
  // is what both partition classes take.
  //
  // The refinement is what makes the overlap checks worth running: a uniform
  // mesh gives every cell the same size and the same relation to the cutoff
  // balls, whereas here the cells nearest the atoms are a quarter of the
  // coarse size and hang off their neighbours, and because the atoms sit next
  // to the faces the fine cells straddle the periodic boundary, which is
  // exactly where an image reaches a cell its master cannot.
  //
  // refineRadii holds one radius per refinement sweep, applied in order; a
  // cell is flagged when its centre is within the radius of any atom.
  std::shared_ptr<dftefe::basis::TriangulationBase>
  buildMesh(const dftefe::utils::mpi::MPIComm &comm,
            const std::vector<size_type> &     subdivisions,
            const std::vector<Point> &         domainVectors,
            const std::vector<bool> &          isPeriodicFlags,
            const std::vector<Point> &         atomCoordinates,
            const std::vector<double> &        refineRadii,
            std::vector<std::vector<Point>> &  cellVerticesVector,
            std::vector<double> &              minbound,
            std::vector<double> &              maxbound)
  {
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulation =
      std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);

    triangulation->initializeTriangulationConstruction();
    triangulation->createUniformParallelepiped(subdivisions,
                                               domainVectors,
                                               isPeriodicFlags);
    // A no-op when nothing is flagged periodic, and it has to happen on the
    // coarsest level, before any refinement.
    triangulation->markPeriodicFaces(isPeriodicFlags, domainVectors);
    triangulation->finalizeTriangulationConstruction();

    // The flagging criterion is purely geometric, so every processor makes the
    // same decision about the cells it owns and the refined mesh does not
    // depend on how the cells were distributed.
    for (auto radius : refineRadii)
      {
        for (auto cellIter = triangulation->beginLocal();
             cellIter != triangulation->endLocal();
             ++cellIter)
          {
            Point center(dim, 0.0);
            (*cellIter)->center(center);
            for (auto &atom : atomCoordinates)
              if (minimumImageDistance(center,
                                       atom,
                                       domainVectors,
                                       isPeriodicFlags) < radius)
                {
                  (*cellIter)->setRefineFlag();
                  break;
                }
          }
        triangulation->executeCoarseningAndRefinement();
        triangulation->finalizeTriangulationConstruction();
      }

    cellVerticesVector.resize(0);
    std::vector<Point> cellVertices;
    for (auto cellIter = triangulation->beginLocal();
         cellIter != triangulation->endLocal();
         ++cellIter)
      {
        (*cellIter)->getVertices(cellVertices);
        cellVerticesVector.push_back(cellVertices);
      }

    minbound.assign(dim, 0.0);
    maxbound.assign(dim, 0.0);
    bool first = true;
    for (auto &vertices : cellVerticesVector)
      {
        std::vector<double> cellMin, cellMax;
        boundingBox(vertices, cellMin, cellMax);
        for (size_type k = 0; k < dim; k++)
          {
            minbound[k] = first ? cellMin[k] : std::min(minbound[k], cellMin[k]);
            maxbound[k] = first ? cellMax[k] : std::max(maxbound[k], cellMax[k]);
          }
        first = false;
      }

    return triangulation;
  }

  std::string
  pointToString(const Point &p)
  {
    std::stringstream ss;
    ss << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")";
    return ss.str();
  }
} // namespace

int
main()
{
#ifdef DFTEFE_WITH_MPI

  dftefe::utils::mpi::MPIComm comm = dftefe::utils::mpi::MPICommWorld;

  dftefe::utils::mpi::MPIInit(NULL, NULL);

  int numProcs, rank;
  dftefe::utils::mpi::MPICommSize(comm, &numProcs);
  dftefe::utils::mpi::MPICommRank(comm, &rank);

  //
  // Atom data. The enrichment cutoffs come out of the file and both domains
  // are then sized from them, so the test does not silently stop exercising
  // the image path if the data file is regenerated with different cutoffs.
  //
  const std::string        atomDataFile = "pseudodojo.C.json";
  const std::string        fieldName    = "orbital";
  std::vector<std::string> fieldNames{fieldName};
  std::vector<std::string> metadataNames{"symbol", "Z", "charge", "NR"};

  std::shared_ptr<const dftefe::atoms::AtomSphericalDataContainer>
    atomSphericalDataContainer =
      std::make_shared<const dftefe::atoms::AtomSphericalDataContainer>(
        dftefe::atoms::AtomSphericalDataType::ENRICHMENT,
        std::map<std::string, std::string>{{"C", atomDataFile}},
        fieldNames,
        metadataNames,
        std::map<std::string, std::string>{{"PSP/AE", "PSP"}});

  const double additionalCutoff = 0.0;

  double maxEnrichmentReach = 0.0;
  for (auto &sd : atomSphericalDataContainer->getSphericalData("C", fieldName))
    maxEnrichmentReach =
      std::max(maxEnrichmentReach, enrichmentReach(sd, additionalCutoff));

  // Where the atoms sit, as a fraction of the box edge. Near the faces and
  // corners, so that images wrap in every direction, while staying strictly
  // inside the box, which AtomIdsPartition needs in order to give each atom an
  // owner.
  const double atomFractions[4][dim] = {{0.20, 0.20, 0.20},
                                        {0.90, 0.50, 0.50},
                                        {0.50, 0.05, 0.80},
                                        {0.97, 0.97, 0.03}};
  const size_type          nAtoms = 4;
  std::vector<std::string> atomSymbolVec(nAtoms, "C");
  std::vector<double>      atomChargesVec(nAtoms, 4.0);

  // A coarse mesh plus two refinement sweeps around the atoms, so the cells
  // that matter come in three sizes and carry hanging nodes.
  std::vector<size_type> subdivisions(dim, 4);
  const std::vector<double> refineRadiusFractions{0.35, 0.18};
  const double              tolerance = 1e-6;

  if (rank == 0)
    {
      std::cout << "Enrichment cutoffs for C, field " << fieldName << ":\n";
      for (auto &sd :
           atomSphericalDataContainer->getSphericalData("C", fieldName))
        std::cout << "  cutoff " << sd->getCutoff() << ", smoothness "
                  << sd->getSmoothness() << ", reach "
                  << enrichmentReach(sd, additionalCutoff) << "\n";
      std::cout << "Largest reach: " << maxEnrichmentReach
                << ", processors: " << numProcs << "\n"
                << std::flush;
    }

  //
  // ---------------------------------------------------------------------
  // Periodic case
  // ---------------------------------------------------------------------
  //
  // An edge of 2.5 reaches keeps the far side of the box out of every master
  // atom's ball while still inside the ball of that atom's nearest image,
  // which is the configuration the periodic path exists for.
  //
  const double       domainLength = 2.5 * maxEnrichmentReach;
  std::vector<bool>  isPeriodicFlags(dim, true);
  std::vector<Point> domainVectors(dim, Point(dim, 0.0));
  for (size_type k = 0; k < dim; k++)
    domainVectors[k][k] = domainLength;

  std::vector<Point> atomCoordinatesVec(nAtoms, Point(dim, 0.0));
  for (size_type i = 0; i < nAtoms; i++)
    for (size_type k = 0; k < dim; k++)
      atomCoordinatesVec[i][k] = atomFractions[i][k] * domainLength;

  std::vector<double> refineRadii;
  refineRadii.resize(refineRadiusFractions.size(), 0.0);
  for (size_type i = 0; i < refineRadiusFractions.size(); i++)
    refineRadii[i] = refineRadiusFractions[i] * domainLength;

  std::vector<std::vector<Point>> cellVerticesVector;
  std::vector<double>             minbound, maxbound;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulation =
    buildMesh(comm,
              subdivisions,
              domainVectors,
              isPeriodicFlags,
              atomCoordinatesVec,
              refineRadii,
              cellVerticesVector,
              minbound,
              maxbound);

  const size_type nLocalCells = cellVerticesVector.size();

  // Both element length queries reduce over the domain communicator, so every
  // processor has to reach them even though only one prints.
  const size_type nGlobalCells    = triangulation->nGlobalCells();
  const double    minElementLength = triangulation->minElementLength();
  const double    maxElementLength = triangulation->maxElementLength();
  if (rank == 0)
    std::cout << "Periodic mesh: " << nGlobalCells
              << " cells, element length from " << minElementLength << " to "
              << maxElementLength << "\n"
              << std::flush;

  std::shared_ptr<const dftefe::basis::AtomIdsPartition<dim>> atomIdsPartition =
    std::make_shared<dftefe::basis::AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                           minbound,
                                                           maxbound,
                                                           cellVerticesVector,
                                                           tolerance,
                                                           comm);

  // The truncated envelope has to cover the widest enrichment, which is the
  // sizing rule the callers of this class are expected to apply.
  const double cutOffTrunc =
    std::max(dftefe::basis::PeriodicImageAtomGeneratorDefaults::CUTOFF_TRUNC,
             maxEnrichmentReach);

  std::shared_ptr<const dftefe::basis::PeriodicImageAtomGenerator>
    imageAtomGenerator =
      std::make_shared<const dftefe::basis::PeriodicImageAtomGenerator>(
        atomCoordinatesVec,
        atomChargesVec,
        domainVectors,
        isPeriodicFlags,
        dftefe::basis::PeriodicImageAtomGeneratorDefaults::CUTOFF,
        cutOffTrunc);

  std::shared_ptr<const dftefe::basis::EnrichmentIdsPartition<dim>>
    enrichmentIdsPartition =
      std::make_shared<dftefe::basis::EnrichmentIdsPartition<dim>>(
        atomSphericalDataContainer,
        atomIdsPartition,
        atomSymbolVec,
        atomCoordinatesVec,
        fieldName,
        minbound,
        maxbound,
        additionalCutoff,
        domainVectors,
        isPeriodicFlags,
        cellVerticesVector,
        comm,
        imageAtomGenerator);

  //
  // Each failure appends to failureMsg and bumps nFailures, which is summed
  // over the processors at the end, so that one processor's failure fails the
  // run and the diagnostics of every processor survive.
  //
  size_type         nFailures = 0;
  std::stringstream failureMsg;

  const std::vector<size_type> newAtomIds = atomIdsPartition->newAtomIds();
  const std::vector<global_size_type> offsets =
    enrichmentIdsPartition->newAtomIdToEnrichmentIdOffset();
  const std::vector<std::vector<global_size_type>> overlapIds =
    enrichmentIdsPartition->overlappingEnrichmentIdsInCells();

  const std::vector<Point> &imagePositionsTrunc =
    imageAtomGenerator->getImagePositionsTrunc();
  // imageIdsTrunc[iImage] is the master atom that image belongs to.
  const std::vector<size_type> &imageMasterIdsTrunc =
    imageAtomGenerator->getImageIdsTrunc();

  if (overlapIds.size() != nLocalCells)
    {
      nFailures++;
      failureMsg << "rank " << rank << ": overlappingEnrichmentIdsInCells has "
                 << overlapIds.size() << " cells, expected " << nLocalCells
                 << "\n";
    }

  // How much of the periodic machinery actually got exercised. Summed over
  // processors at the end and required to be non zero.
  size_type nCellEnrichWithImage = 0, nCellEnrichImageOnly = 0,
            nCellEnrichPeriodic = 0;

  // Oracle B, first half: enumerate the lattice translations of every atom
  // that come close enough to this processor's cells to be worth testing, so
  // the per cell work below is a short list rather than a triple loop. The
  // shell count is at least as wide as the one the generator uses.
  const int nShells = (int)std::ceil(2.0 * cutOffTrunc / domainLength) + 1;
  std::vector<std::vector<Point>> candidateTranslates(nAtoms);
  for (size_type iAtom = 0; iAtom < nAtoms; iAtom++)
    for (int ix = -nShells; ix <= nShells; ix++)
      for (int iy = -nShells; iy <= nShells; iy++)
        for (int iz = -nShells; iz <= nShells; iz++)
          {
            if (ix == 0 && iy == 0 && iz == 0)
              continue;
            const int shift[dim] = {ix, iy, iz};
            Point     translate  = atomCoordinatesVec[iAtom];
            for (size_type k = 0; k < dim; k++)
              translate[k] += shift[k] * domainLength;
            if (ballOverlapsBox(translate,
                                maxEnrichmentReach,
                                minbound,
                                maxbound))
              candidateTranslates[iAtom].push_back(translate);
          }

  for (size_type iCell = 0; iCell < nLocalCells; iCell++)
    {
      std::vector<double> cellMin, cellMax;
      boundingBox(cellVerticesVector[iCell], cellMin, cellMax);

      //
      // Oracle A: recompute which enrichments overlap this cell and, for
      // each of them, which origins reach it.
      //
      std::vector<global_size_type>       expectedIds;
      std::vector<std::vector<size_type>> expectedOrigins;

      for (size_type iAtom = 0; iAtom < nAtoms; iAtom++)
        {
          const std::vector<std::vector<int>> qNumbers =
            atomSphericalDataContainer->getQNumbers(atomSymbolVec[iAtom],
                                                    fieldName);
          for (size_type iOrbital = 0; iOrbital < qNumbers.size(); iOrbital++)
            {
              const double cutoff = enrichmentReach(
                atomSphericalDataContainer->getSphericalData(
                  atomSymbolVec[iAtom], fieldName, qNumbers[iOrbital]),
                additionalCutoff);

              std::vector<size_type> origins;
              if (ballOverlapsBox(atomCoordinatesVec[iAtom],
                                  cutoff,
                                  cellMin,
                                  cellMax))
                origins.push_back(iAtom);

              for (auto iImage :
                   imageAtomGenerator->getImageIdsForMasterTrunc(iAtom))
                if (ballOverlapsBox(imagePositionsTrunc[iImage],
                                    cutoff,
                                    cellMin,
                                    cellMax))
                  origins.push_back(nAtoms + iImage);

              if (origins.empty())
                continue;

              const global_size_type enrichmentId =
                (newAtomIds[iAtom] != 0) ?
                  offsets[newAtomIds[iAtom] - 1] + iOrbital :
                  iOrbital;
              expectedIds.push_back(enrichmentId);
              expectedOrigins.push_back(origins);
            }
        }

      if (iCell < overlapIds.size() && overlapIds[iCell] != expectedIds)
        {
          nFailures++;
          failureMsg << "rank " << rank << ", cell " << iCell
                     << ": overlapping enrichment ids differ from the "
                        "recomputed ones. got "
                     << overlapIds[iCell].size() << " ids, expected "
                     << expectedIds.size() << "\n";
          continue;
        }

      // Each enrichment appears at most once per cell, periodic or not: the
      // images add origins, never ids.
      std::set<global_size_type> uniqueIds(expectedIds.begin(),
                                           expectedIds.end());
      if (uniqueIds.size() != expectedIds.size())
        {
          nFailures++;
          failureMsg << "rank " << rank << ", cell " << iCell
                     << ": an enrichment id is listed more than once\n";
        }

      nCellEnrichPeriodic += expectedIds.size();

      for (size_type enrichIdInCell = 0; enrichIdInCell < expectedIds.size();
           enrichIdInCell++)
        {
          const std::vector<size_type> origins =
            enrichmentIdsPartition->getAtomIdsForCellEnrich(iCell,
                                                            enrichIdInCell);

          if (origins != expectedOrigins[enrichIdInCell])
            {
              nFailures++;
              failureMsg << "rank " << rank << ", cell " << iCell
                         << ", enrichment " << enrichIdInCell << " of the cell"
                         << " (id " << expectedIds[enrichIdInCell]
                         << "): origins differ. got " << origins.size()
                         << ", expected "
                         << expectedOrigins[enrichIdInCell].size() << "\n";
              continue;
            }

          const size_type masterOfEnrich =
            enrichmentIdsPartition->getAtomId(expectedIds[enrichIdInCell]);

          bool hasMaster = false, hasImage = false;
          for (auto extId : origins)
            {
              // The encoding must decode back to the position the oracle used,
              // and every origin of one enrichment must belong to the master
              // owns the enrichment.
              const bool  isImage = (extId >= nAtoms);
              const Point expectedPosition =
                isImage ? imagePositionsTrunc[extId - nAtoms] :
                          atomCoordinatesVec[extId];
              const Point gotPosition =
                enrichmentIdsPartition->getPositionOfAtomId(extId);
              for (size_type k = 0; k < dim; k++)
                if (std::abs(gotPosition[k] - expectedPosition[k]) > 1e-12)
                  {
                    nFailures++;
                    failureMsg << "rank " << rank << ", cell " << iCell
                               << ", enrichment " << enrichIdInCell
                               << " of the cell: extended atom id " << extId
                               << " decodes to "
                               << pointToString(gotPosition) << ", expected "
                               << pointToString(expectedPosition) << "\n";
                    break;
                  }

              const size_type masterOfOrigin =
                isImage ? imageMasterIdsTrunc[extId - nAtoms] : extId;
              if (isImage)
                hasImage = true;
              else
                hasMaster = true;

              if (masterOfOrigin != masterOfEnrich)
                {
                  nFailures++;
                  failureMsg << "rank " << rank << ", cell " << iCell
                             << ", enrichment " << enrichIdInCell
                             << " of the cell: extended atom id " << extId
                             << " belongs to atom " << masterOfOrigin
                             << " but the enrichment belongs to atom "
                             << masterOfEnrich << "\n";
                }
            }

          if (origins.empty())
            {
              nFailures++;
              failureMsg << "rank " << rank << ", cell " << iCell
                         << ", enrichment " << enrichIdInCell
                         << " of the cell: it overlaps the cell with no "
                            "origin reaching it\n";
            }

          if (hasImage)
            nCellEnrichWithImage++;
          if (hasImage && !hasMaster)
            nCellEnrichImageOnly++;
        }

      //
      // Oracle B, second half: any lattice translation that actually reaches
      // this cell has to be in the truncated envelope, or enrichments are lost
      // quietly. Testing against the cell rather than the processor box keeps
      // this from failing on a translation the envelope is entitled to drop.
      //
      for (size_type iAtom = 0; iAtom < nAtoms; iAtom++)
        {
          const std::vector<size_type> &truncIds =
            imageAtomGenerator->getImageIdsForMasterTrunc(iAtom);
          for (auto &translate : candidateTranslates[iAtom])
            {
              if (!ballOverlapsBox(translate,
                                   maxEnrichmentReach,
                                   cellMin,
                                   cellMax))
                continue;

              bool found = false;
              for (auto iImage : truncIds)
                {
                  bool same = true;
                  for (size_type k = 0; k < dim; k++)
                    if (std::abs(imagePositionsTrunc[iImage][k] -
                                 translate[k]) > 1e-8)
                      {
                        same = false;
                        break;
                      }
                  if (same)
                    {
                      found = true;
                      break;
                    }
                }
              if (!found)
                {
                  nFailures++;
                  failureMsg << "rank " << rank << ", cell " << iCell
                             << ": image " << pointToString(translate)
                             << " of atom " << iAtom
                             << " reaches this cell but is missing from the "
                                "truncated image list\n";
                }
            }
        }
    }

  //
  // Ownership of the enrichment ids across the processors. Periodicity must
  // not move an id's owner: images extend an atom's footprint, they do not own
  // anything.
  //
  const std::pair<global_size_type, global_size_type> ownedRange =
    enrichmentIdsPartition->locallyOwnedEnrichmentIds();
  const std::vector<global_size_type> ghostIds =
    enrichmentIdsPartition->ghostEnrichmentIds();
  const global_size_type nTotalEnrichmentIds =
    enrichmentIdsPartition->nTotalEnrichmentIds();

  std::vector<global_size_type> sendRange{ownedRange.first, ownedRange.second};
  std::vector<global_size_type> allRanges(2 * numProcs, 0);
  dftefe::utils::mpi::MPIAllgather<dftefe::utils::MemorySpace::HOST>(
    sendRange.data(),
    2,
    dftefe::utils::mpi::MPIUnsignedLong,
    allRanges.data(),
    2,
    dftefe::utils::mpi::MPIUnsignedLong,
    comm);

  if (rank == 0)
    {
      global_size_type expectedStart = 0;
      for (int iProc = 0; iProc < numProcs; iProc++)
        {
          const global_size_type first = allRanges[2 * iProc];
          const global_size_type last  = allRanges[2 * iProc + 1];

          // A processor that owns no atom owns no enrichment either, and the
          // class leaves its pair at the default [0, 0) rather than at an
          // empty range positioned after its predecessor. Such a processor
          // contributes nothing to the tiling, so step over it.
          if (first == last)
            continue;

          if (first != expectedStart || last < first)
            {
              nFailures++;
              failureMsg << "the locally owned enrichment id ranges do not "
                            "tile contiguously: processor "
                         << iProc << " owns [" << first << ", " << last
                         << ") after " << expectedStart << "\n";
            }
          expectedStart = last;
        }
      if (expectedStart != nTotalEnrichmentIds)
        {
          nFailures++;
          failureMsg << "the owned ranges cover " << expectedStart
                     << " enrichment ids but there are " << nTotalEnrichmentIds
                     << " in total\n";
        }
    }

  // The ghost list has to be sorted, free of repeats, disjoint from the owned
  // range, and inside the global id space.
  for (size_type i = 0; i < ghostIds.size(); i++)
    {
      if (i > 0 && ghostIds[i] <= ghostIds[i - 1])
        {
          nFailures++;
          failureMsg << "rank " << rank
                     << ": the ghost enrichment ids are not strictly "
                        "increasing at position "
                     << i << "\n";
        }
      if (ghostIds[i] >= nTotalEnrichmentIds)
        {
          nFailures++;
          failureMsg << "rank " << rank << ": ghost enrichment id "
                     << ghostIds[i] << " is outside the global id space\n";
        }
      if (ghostIds[i] >= ownedRange.first && ghostIds[i] < ownedRange.second)
        {
          nFailures++;
          failureMsg << "rank " << rank << ": enrichment id " << ghostIds[i]
                     << " is both owned and ghost\n";
        }
    }

  // The ghost set is pinned exactly, in both directions, because the two
  // directions fail differently and both matter under periodicity.
  //
  // Too few ghosts is a correctness failure: an image reaching a processor
  // whose master atom lives elsewhere, and the id never getting enrolled, so
  // the halo exchange never brings its coefficients over.
  //
  // Too many is a cost failure that stays silent: images widen an atom's
  // footprint, so a master that was ghost on one processor before can become
  // ghost on many, and the halo payload grows with it. An id enrolled as ghost
  // that no local cell actually needs would never be noticed by a subset test,
  // so the set is required to be exactly the overlapping ids that this
  // processor does not own.
  {
    std::set<global_size_type> ghostSet(ghostIds.begin(), ghostIds.end());

    std::set<global_size_type> expectedGhostSet;
    for (size_type iCell = 0; iCell < overlapIds.size(); iCell++)
      for (auto id : overlapIds[iCell])
        if (!(id >= ownedRange.first && id < ownedRange.second))
          expectedGhostSet.insert(id);

    for (auto id : expectedGhostSet)
      if (ghostSet.count(id) == 0)
        {
          nFailures++;
          failureMsg << "rank " << rank << ": enrichment id " << id
                     << " overlaps a local cell and is not owned here, but is "
                        "missing from the ghost list\n";
        }

    for (auto id : ghostSet)
      if (expectedGhostSet.count(id) == 0)
        {
          nFailures++;
          failureMsg << "rank " << rank << ": enrichment id " << id
                     << " is listed as ghost but no local cell needs it\n";
        }

    // A ghost has to be owned by exactly one other processor, or the halo
    // exchange has no one to fetch it from. allRanges came from the gather
    // above, which every processor took part in.
    for (auto id : ghostSet)
      {
        int nOwners = 0;
        for (int iProc = 0; iProc < numProcs; iProc++)
          if (id >= allRanges[2 * iProc] && id < allRanges[2 * iProc + 1])
            nOwners++;
        if (nOwners != 1)
          {
            nFailures++;
            failureMsg << "rank " << rank << ": ghost enrichment id " << id
                       << " is owned by " << nOwners
                       << " processors, expected exactly one\n";
          }
      }
  }

  //
  // ---------------------------------------------------------------------
  // Non periodic case, on its own larger box
  // ---------------------------------------------------------------------
  //
  // Without periodicity an enrichment ball may not spill out of the domain, so
  // this box is four reaches wide and the atoms are pulled into the middle
  // fifth to fourth of it. Every ball then clears the boundary by more than a
  // full reach. The run confirms the periodic additions left this path alone:
  // one origin per enrichment, the master itself.
  //
  const double       domainLengthNonPeriodic = 4.0 * maxEnrichmentReach;
  std::vector<bool>  noPeriodicFlags(dim, false);
  std::vector<Point> domainVectorsNonPeriodic(dim, Point(dim, 0.0));
  for (size_type k = 0; k < dim; k++)
    domainVectorsNonPeriodic[k][k] = domainLengthNonPeriodic;

  // The same relative arrangement, squeezed into [0.3, 0.7] of the edge.
  std::vector<Point> atomCoordinatesNonPeriodic(nAtoms, Point(dim, 0.0));
  for (size_type i = 0; i < nAtoms; i++)
    for (size_type k = 0; k < dim; k++)
      atomCoordinatesNonPeriodic[i][k] =
        (0.3 + 0.4 * atomFractions[i][k]) * domainLengthNonPeriodic;

  std::vector<double> refineRadiiNonPeriodic;
  refineRadiiNonPeriodic.resize(refineRadiusFractions.size(), 0.0);
  for (size_type i = 0; i < refineRadiusFractions.size(); i++)
    refineRadiiNonPeriodic[i] =
      refineRadiusFractions[i] * domainLengthNonPeriodic;

  std::vector<std::vector<Point>> cellVerticesVectorNonPeriodic;
  std::vector<double>             minboundNonPeriodic, maxboundNonPeriodic;
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationNonPeriodic =
    buildMesh(comm,
              subdivisions,
              domainVectorsNonPeriodic,
              noPeriodicFlags,
              atomCoordinatesNonPeriodic,
              refineRadiiNonPeriodic,
              cellVerticesVectorNonPeriodic,
              minboundNonPeriodic,
              maxboundNonPeriodic);

  const size_type nGlobalCellsNonPeriodic =
    triangulationNonPeriodic->nGlobalCells();
  const double minElementLengthNonPeriodic =
    triangulationNonPeriodic->minElementLength();
  const double maxElementLengthNonPeriodic =
    triangulationNonPeriodic->maxElementLength();
  if (rank == 0)
    std::cout << "Non periodic mesh: " << nGlobalCellsNonPeriodic
              << " cells, element length from "
              << minElementLengthNonPeriodic << " to "
              << maxElementLengthNonPeriodic << "\n"
              << std::flush;

  std::shared_ptr<const dftefe::basis::AtomIdsPartition<dim>>
    atomIdsPartitionNonPeriodic =
      std::make_shared<dftefe::basis::AtomIdsPartition<dim>>(
        atomCoordinatesNonPeriodic,
        minboundNonPeriodic,
        maxboundNonPeriodic,
        cellVerticesVectorNonPeriodic,
        tolerance,
        comm);

  std::shared_ptr<const dftefe::basis::EnrichmentIdsPartition<dim>>
    enrichmentIdsPartitionNonPeriodic =
      std::make_shared<dftefe::basis::EnrichmentIdsPartition<dim>>(
        atomSphericalDataContainer,
        atomIdsPartitionNonPeriodic,
        atomSymbolVec,
        atomCoordinatesNonPeriodic,
        fieldName,
        minboundNonPeriodic,
        maxboundNonPeriodic,
        additionalCutoff,
        domainVectorsNonPeriodic,
        noPeriodicFlags,
        cellVerticesVectorNonPeriodic,
        comm);

  const std::vector<std::vector<global_size_type>> overlapIdsNonPeriodic =
    enrichmentIdsPartitionNonPeriodic->overlappingEnrichmentIdsInCells();

  size_type nCellEnrichNonPeriodic = 0;
  for (size_type iCell = 0; iCell < overlapIdsNonPeriodic.size(); iCell++)
    {
      nCellEnrichNonPeriodic += overlapIdsNonPeriodic[iCell].size();
      for (size_type enrichIdInCell = 0;
           enrichIdInCell < overlapIdsNonPeriodic[iCell].size();
           enrichIdInCell++)
        {
          const std::vector<size_type> origins =
            enrichmentIdsPartitionNonPeriodic->getAtomIdsForCellEnrich(
              iCell, enrichIdInCell);
          const size_type master =
            enrichmentIdsPartitionNonPeriodic->getAtomId(
              overlapIdsNonPeriodic[iCell][enrichIdInCell]);
          if (origins.size() != 1 || origins[0] != master)
            {
              nFailures++;
              failureMsg << "rank " << rank << ", non periodic cell " << iCell
                         << ", enrichment " << enrichIdInCell
                         << " of the cell: without periodicity the origin "
                            "list must be the master alone, got "
                         << origins.size() << " entries\n";
            }
        }
    }

  //
  // Reductions: the run fails if any processor failed, and it also fails if
  // the periodic path turned out to be unexercised.
  //
  std::vector<size_type> localCounts{nFailures,
                                     nCellEnrichWithImage,
                                     nCellEnrichImageOnly,
                                     nCellEnrichPeriodic,
                                     nCellEnrichNonPeriodic,
                                     (size_type)ghostIds.size()};
  std::vector<size_type> globalCounts(localCounts.size(), 0);
  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
    localCounts.data(),
    globalCounts.data(),
    (int)localCounts.size(),
    dftefe::utils::mpi::MPIUnsignedLong,
    dftefe::utils::mpi::MPISum,
    comm);

  // Flushed explicitly: the checks below end the run by throwing, and an
  // uncaught exception terminates without flushing the streams, which would
  // lose exactly the diagnostics that say what went wrong.
  if (nFailures > 0)
    std::cout << failureMsg.str() << std::flush;

  if (rank == 0)
    {
      std::cout << "Periodic box edge: " << domainLength
                << ", non periodic box edge: " << domainLengthNonPeriodic
                << "\n";
      std::cout << "Total enrichment ids: " << nTotalEnrichmentIds << "\n";
      std::cout << "Truncated images generated: " << imagePositionsTrunc.size()
                << "\n";
      std::cout << "Cell-enrichment pairs over all cells, periodic: "
                << globalCounts[3] << ", non periodic: " << globalCounts[4]
                << "\n";
      std::cout << "Cell-enrichment pairs reached by at least one image: "
                << globalCounts[1]
                << ", of which reached only by images: " << globalCounts[2]
                << "\n";
      // Worth watching rather than asserting on: periodicity widens each
      // atom's footprint, so this is the halo payload the MPI pattern will
      // carry.
      std::cout << "Ghost enrichment ids summed over processors: "
                << globalCounts[5] << "\n"
                << std::flush;
    }

  // A pass with no image contribution anywhere would mean the geometry stopped
  // exercising the periodic path, so treat it as a failure rather than let the
  // test go quietly green. Every processor has the reduced counts, so every
  // processor decides the same way.
  dftefe::utils::throwException(
    globalCounts[1] > 0,
    "No enrichment was reached in any cell by a periodic image, so the "
    "periodic path "
    "of EnrichmentIdsPartition was never exercised. The test geometry no "
    "longer matches the enrichment cutoffs.");

  dftefe::utils::throwException(
    globalCounts[2] > 0,
    "No enrichment was reached in any cell only by images, so the case a "
    "master atom "
    "cannot reach on its own was never exercised. The test geometry no longer "
    "matches the enrichment cutoffs.");

  dftefe::utils::throwException(globalCounts[0] == 0,
                                "EnrichmentIdsPartition parallel periodic "
                                "test found " +
                                  std::to_string(globalCounts[0]) +
                                  " mismatch(es); see the messages above.");

  if (rank == 0)
    std::cout << "TestEnrichmentIdsPartitionParallel passed.\n";

  dftefe::utils::mpi::MPIFinalize();
#endif
}
