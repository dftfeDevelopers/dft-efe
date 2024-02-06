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
 * @author Bikash Kanungo, Vishal Subramanian, Avirup Sircar
 */

#ifndef dftefeCFEBasisDofHandlerDealii_h
#define dftefeCFEBasisDofHandlerDealii_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <basis/FEBasisDofHandler.h>
#include <memory>
#include <deal.II/fe/fe_q.h>

#include <basis/ConstraintsLocal.h>
#include <basis/CFEConstraintsLocalDealii.h>
#include <memory>
#include <map>

/// dealii includes
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace dftefe
{
  namespace basis
  {
    /**
     * A derived class of FEBasisDofHandler to handle the FE basis evaluations
     * through dealii
     */
    template <typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type                  dim>
    class CFEBasisDofHandlerDealii
      : public FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>
    {
    public:
      using FECellIterator       = typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::FECellIterator;
      using const_FECellIterator = typename FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>::const_FECellIterator;

      CFEBasisDofHandlerDealii(
        std::shared_ptr<const TriangulationBase> triangulation,
        const size_type                          feOrder,
        const utils::mpi::MPIComm &              mpiComm);

      CFEBasisDofHandlerDealii(
        std::shared_ptr<const TriangulationBase> triangulation,
        const size_type                          feOrder);

      double
      getBasisFunctionValue(const size_type     basisId,
                            const utils::Point &point) const override;
      std::vector<double>
      getBasisFunctionDerivative(
        const size_type     basisId,
        const utils::Point &point,
        const size_type     derivativeOrder = 1) const override;

      ////// FE specific  member functions /////
      void
      reinit(std::shared_ptr<const TriangulationBase> triangulation,
             const size_type                          feOrder,
             const utils::mpi::MPIComm &              mpiComm);

      void
      reinit(std::shared_ptr<const TriangulationBase> triangulation,
             const size_type                          feOrder);

      std::shared_ptr<const TriangulationBase>
      getTriangulation() const override;

      size_type
      nLocalCells() const override;
      size_type
      nLocallyOwnedCells() const override;

      size_type
      nGlobalCells() const override;
      size_type
      getFEOrder(size_type cellId) const override;

      size_type
      nCellDofs(size_type cellId) const override;

      bool
      isVariableDofsPerCell() const override;

      size_type
      nLocalNodes() const override;

      std::vector<std::pair<global_size_type, global_size_type>>
      getLocallyOwnedRanges() const override;

      std::vector<std::pair<global_size_type, global_size_type>>
      getGlobalRanges() const override;

      std::map<BasisIdAttribute, size_type>
      getBasisAttributeToRangeIdMap() const override;

      global_size_type
      nGlobalNodes() const override;

      std::vector<size_type>
      getLocalNodeIds(size_type cellId) const override;

      std::vector<size_type>
      getGlobalNodeIds() const override;

      void
      getCellDofsGlobalIds(
        size_type                      cellId,
        std::vector<global_size_type> &vecGlobalNodeId) const override;

      const std::vector<global_size_type> &
      getBoundaryIds() const override;

      FECellIterator
      beginLocallyOwnedCells() override;

      FECellIterator
      endLocallyOwnedCells() override;

      const_FECellIterator
      beginLocallyOwnedCells() const override;

      const_FECellIterator
      endLocallyOwnedCells() const override;

      FECellIterator
      beginLocalCells() override;
      FECellIterator
      endLocalCells() override;
      const_FECellIterator
      beginLocalCells() const override;
      const_FECellIterator
      endLocalCells() const override;
      unsigned int
      getDim() const override;

      size_type
      nCumulativeLocallyOwnedCellDofs() const override;

      size_type
      nCumulativeLocalCellDofs() const override;

      size_type
      totalRanges() const override;

      // This assumes a linear cell mapping
      void
      getBasisCenters(
        std::map<global_size_type, utils::Point> &dofCoords) const override;

      // Additional functions for getting geometric constriants matrix
      // Additional functions for getting the communication pattern object
      // for MPI case

      std::shared_ptr<const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
        getIntrinsicConstraints() const override;

      // use this to add extra constraints on top of geometric constraints
      std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
       createConstraintsStart() const override;

      // call this after calling start
      void
      createConstraintsEnd(
         std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, 
         memorySpace>> constraintsLocal) const override;

      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
      getMPIPatternP2P() const override;

      bool isDistributed() const override;


      //
      // dealii specific functions
      //
      std::shared_ptr<const dealii::DoFHandler<dim>>
      getDoFHandler() const;

      const dealii::FiniteElement<dim> &
      getReferenceFE(const size_type cellId) const;

    private:
      std::shared_ptr<const TriangulationBase> d_triangulation;
      std::shared_ptr<dealii::DoFHandler<dim>> d_dofHandler;
      bool                                     d_isVariableDofsPerCell;
      std::vector<std::shared_ptr<FECellBase>> d_localCells;
      std::vector<std::shared_ptr<FECellBase>> d_locallyOwnedCells;
      size_type d_numCumulativeLocallyOwnedCellDofs;
      size_type d_numCumulativeLocalCellDofs;
      size_type d_totalRanges;

      std::vector<global_size_type> d_boundaryIds;
      bool d_isDistributed;
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>> d_mpiPatternP2P;
      std::shared_ptr<const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>> d_constraintsLocal;

    }; // end of CFEBasisDofHandlerDealii
  }    // end of namespace basis
} // end of namespace dftefe
#include "CFEBasisDofHandlerDealii.t.cpp"
#endif // dftefeCFEBasisDofHandlerDealii_h
