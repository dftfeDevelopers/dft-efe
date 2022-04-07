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
 * @author Vishal Subramanian
 */

#ifndef dftefeTriangulationDealiiParallel_h
#define dftefeTriangulationDealiiParallel_h

#include "TriangulationBase.h"
#include <utils/TypeConfig.h>
#include "TriangulationCellDealii.h"
#include <deal.II/distributed/tria.h>
namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    class TriangulationDealiiParallel : public TriangulationBase
    {
    public:
      TriangulationDealiiParallel(const MPI_Comm &mpi_communicator);
      ~TriangulationDealiiParallel();

      void
      initializeTriangulationConstruction() override;
      void
      finalizeTriangulationConstruction() override;
      void
      createUniformParallelepiped(
        const std::vector<unsigned int> &subdivisions,
        const std::vector<utils::Point> &domainVectors,
        const std::vector<bool> &        isPeriodicFlags) override;
      void
      createSingleCellTriangulation(
        const std::vector<utils::Point> &vertices) override;
      void
      shiftTriangulation(const utils::Point &origin) override;
      void
      refineGlobal(const unsigned int times = 1) override;
      void
      coarsenGlobal(const unsigned int times = 1) override;
      void
      clearUserFlags() override;
      void
      executeCoarseningAndRefinement() override;
      size_type
      nLocalCells() const override;
      size_type
      nGlobalCells() const override;
      /**
       * \todo
       * TODO:
       * Implement it to get the user specified boundary Ids on different
       * faces of the triangulation
       */
      std::vector<size_type>
      getBoundaryIds() const override;
      TriangulationBase::TriangulationCellIterator
      beginLocal() override;
      TriangulationBase::TriangulationCellIterator
      endLocal() override;
      TriangulationBase::const_TriangulationCellIterator
      beginLocal() const override;
      TriangulationBase::const_TriangulationCellIterator
      endLocal() const override;
      unsigned int
      getDim() const override;

      // Class specific member function


      dealii::parallel::Distributed::Triangulation<dim> &
        returnDealiiTria();

    private:
      /**
       * \todo
       * TODO:
       * 1. Implement for periodic case
       * 2. Check if the domainvectors argument is redundant (i.e., if they can
       *  be fetched from the d_triangulationDealii)
       */
      void
      markPeriodicFaces(const std::vector<bool> &        isPeriodicFlags,
                        const std::vector<utils::Point> &domainVectors);

    private:
      bool                                                isInitialized;
      bool                                                isFinalized;
      dealii::parallel::Distributed::Triangulation<dim>   d_triangulationDealii;
      std::vector<std::shared_ptr<TriangulationCellBase>> d_triaVectorCell;

    }; // end of class TriangulationDealiiParallel

  } // end of namespace basis

} // end of namespace dftefe
#include "TriangulationDealiiParallel.t.cpp"
#endif // # ifndef dftefeTriangulationDealiiParallel_h
