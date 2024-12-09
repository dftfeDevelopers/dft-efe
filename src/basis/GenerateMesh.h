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

#ifndef dftefeGenerateMesh_h
#define dftefeGenerateMesh_h

#include <utils/MPICommunicatorP2P.h>
#include <utils/ConditionalOStream.h>
#include <basis/TriangulationBase.h>
#include <basis/CellMappingBase.h>
#include <utils/Point.h>
#include <vector>
#include <string>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * An abstract class to handle GenerateMesh, for mesh
     * generation around atom.
     */
    class GenerateMesh
    {
    public:
      /**
       * @brief This class generates and stores adaptive finite element meshes for the real-space dft problem.
       *
       * The class uses an adpative mesh generation strategy to generate finite
       * element mesh for given domain based on: atomBallRadius,
       * meshSizeAroundAtom. Mainly to be used in PseudoPotential Caluclations.
       * Additionaly, this class also applies periodicity to mesh.
       *
       *  @author dftefe
       */
      GenerateMesh(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<utils::Point> &domainBoundingVectors,
        double                           radiusAroundAtom,
        double                           meshSizeAroundAtom,
        const std::vector<bool> &        isPeriodicFlags,
        const basis::CellMappingBase &   cellMapping,
        const MPI_Comm &                 mpiDomainCommunicator,
        const MPI_Comm &mpiInterpoolCommunicator = utils::mpi::MPICommSelf);

      /**
       * @brief This class generates and stores adaptive finite element meshes for the real-space dft problem.
       *
       * The class uses an adpative mesh generation strategy to generate finite
       * element mesh for given domain based on: atomBallRadius,
       * meshSizeAroundAtom, meshSizeAtAtom .Mainly used in AllElectron
       * Calculations. Additionaly, this class also applies periodicity to mesh.
       *
       *  @author dftefe
       */
      GenerateMesh(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<utils::Point> &domainBoundingVectors,
        double                           radiusAtAtom,
        double                           meshSizeAtAtom,
        double                           radiusAroundAtom,
        double                           meshSizeAroundAtom,
        const std::vector<bool> &        isPeriodicFlags,
        const basis::CellMappingBase &   cellMapping,
        const MPI_Comm &                 mpiDomainCommunicator,
        const MPI_Comm &mpiInterpoolCommunicator = utils::mpi::MPICommSelf);

      ~GenerateMesh() = default;

      void
      createMesh(TriangulationBase &triangulation);

    private:
      void
      generateCoarseMesh(TriangulationBase &      triangulation,
                         const std::vector<bool> &isPeriodicFlags);

      bool
      refinementAlgorithm(
        TriangulationBase &             triangulation,
        std::vector<size_type> &        locallyOwnedCellsRefineFlags,
        std::map<size_type, size_type> &cellIdToCellRefineFlagMapLocal,
        const basis::CellMappingBase &  cellMapping);

      bool
      refineInsideSystemNonPeriodicAlgorithm(
        TriangulationBase &  triangulation,
        std::vector<double> &maxAtomCoordinates,
        std::vector<double> &minAtomCoordinates);

      bool                             d_adaptiveWithFineMesh;
      size_type                        d_dim;
      double                           d_radiusAtAtom;
      double                           d_radiusAroundAtom;
      double                           d_meshSizeAtAtom;
      double                           d_meshSizeAroundAtom;
      size_type                        d_maxRefinementSteps;
      const std::vector<utils::Point> &d_atomCoordinates;
      const std::vector<utils::Point> &d_domainBoundingVectors;
      const std::vector<bool> &        d_isPeriodicFlags;
      const MPI_Comm &                 d_mpiDomainCommunicator;
      const MPI_Comm &                 d_mpiInterpoolCommunicator;
      const basis::CellMappingBase &   d_cellMapping;
      utils::ConditionalOStream        d_rootCout;
      std::vector<std::vector<bool>>   d_triaCurrentRefinement;

    }; // end of GenerateMesh
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeGenerateMesh_h
