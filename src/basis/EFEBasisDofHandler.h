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

#ifndef dftefeEFEBasisDofHandler_h
#define dftefeEFEBasisDofHandler_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <basis/FEBasisDofHandler.h>
#include <basis/TriangulationBase.h>
#include <basis/FECellBase.h>
#include <map>
#include <basis/EnrichmentClassicalInterfaceSpherical.h>
#include <basis/EnrichmentIdsPartition.h>
#include <basis/ConstraintsLocal.h>
#include <utils/MPIPatternP2P.h>

namespace dftefe
{
  namespace basis
  {
    /**
     * An abstract class to handle enriched finite element (FE) basis related
     * operations, such as evaluating the value and gradients of any basis
     * function at a point, getting cell and nodal information, etc.
     *
     */
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    class EFEBasisDofHandler
      : public FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>

    {
    public:
      virtual double
      getBasisFunctionValue(const size_type     basisId,
                            const utils::Point &point) const = 0;
      virtual std::vector<double>
      getBasisFunctionDerivative(const size_type     basisId,
                                 const utils::Point &point,
                                 const size_type derivativeOrder = 1) const = 0;

      ////// FE specific virtual member functions /////
      // virtual void
      // reinit(std::shared_ptr<const TriangulationBase> triangulation,
      //        const size_type                          feOrder) = 0;

      virtual std::shared_ptr<const TriangulationBase>
      getTriangulation() const = 0;

      virtual size_type
      nLocalCells() const = 0;
      virtual size_type
      nLocallyOwnedCells() const = 0;
      virtual size_type
      nGlobalCells() const = 0;
      virtual size_type
      getFEOrder(size_type cellId) const = 0;
      virtual size_type
      nCellDofs(size_type cellId) const = 0;
      virtual bool
      isVariableDofsPerCell() const = 0;

      virtual std::vector<std::pair<global_size_type, global_size_type>>
      getLocallyOwnedRanges() const = 0;

      virtual std::vector<std::pair<global_size_type, global_size_type>>
      getGlobalRanges() const = 0;

      virtual std::map<BasisIdAttribute, size_type>
      getBasisAttributeToRangeIdMap() const = 0;

      virtual size_type
      nLocalNodes() const = 0;
      virtual global_size_type
      nGlobalNodes() const = 0;
      virtual std::vector<size_type>
      getLocalNodeIds(size_type cellId) const = 0;
      virtual std::vector<size_type>
      getGlobalNodeIds() const = 0;
      virtual void
      getCellDofsGlobalIds(
        size_type                      cellId,
        std::vector<global_size_type> &vecGlobalNodeId) const = 0;
      virtual const std::vector<global_size_type> &
      getBoundaryIds() const = 0;
      virtual typename FEBasisDofHandler<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::FECellIterator
      beginLocallyOwnedCells() = 0;
      virtual typename FEBasisDofHandler<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::FECellIterator
      endLocallyOwnedCells() = 0;
      virtual typename FEBasisDofHandler<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::const_FECellIterator
      beginLocallyOwnedCells() const = 0;
      virtual typename FEBasisDofHandler<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::const_FECellIterator
      endLocallyOwnedCells() const = 0;
      virtual typename FEBasisDofHandler<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::FECellIterator
      beginLocalCells() = 0;
      virtual typename FEBasisDofHandler<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::FECellIterator
      endLocalCells() = 0;
      virtual typename FEBasisDofHandler<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::const_FECellIterator
      beginLocalCells() const = 0;
      virtual typename FEBasisDofHandler<ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::const_FECellIterator
      endLocalCells() const = 0;

      virtual size_type
      nCumulativeLocallyOwnedCellDofs() const = 0;

      virtual size_type
      nCumulativeLocalCellDofs() const = 0;

      // This assumes a linear cell mapping
      virtual void
      getBasisCenters(
        std::map<global_size_type, utils::Point> &dofCoords) const = 0;

      virtual unsigned int
      getDim() const = 0;

      // Enrichment specific functions.
      virtual double
      getEnrichmentValue(const size_type             cellId,
                         const size_type             cellLocalEnrichmentId,
                         const dftefe::utils::Point &point) const = 0;

      virtual std::vector<double>
      getEnrichmentDerivative(const size_type             cellId,
                              const size_type             cellLocalEnrichmentId,
                              const dftefe::utils::Point &point) const = 0;

      virtual std::vector<double>
      getEnrichmentHessian(const size_type             cellId,
                           const size_type             cellLocalEnrichmentId,
                           const dftefe::utils::Point &point) const = 0;

      // Enrichment specific functions.
      virtual std::vector<double>
      getEnrichmentValue(
        const size_type                          cellId,
        const size_type                          cellLocalEnrichmentId,
        const std::vector<dftefe::utils::Point> &points) const = 0;

      virtual std::vector<double>
      getEnrichmentDerivative(
        const size_type                          cellId,
        const size_type                          cellLocalEnrichmentId,
        const std::vector<dftefe::utils::Point> &points) const = 0;

      virtual std::vector<double>
      getEnrichmentHessian(
        const size_type                          cellId,
        const size_type                          cellLocalEnrichmentId,
        const std::vector<dftefe::utils::Point> &points) const = 0;

      virtual std::vector<global_size_type>
      getGhostEnrichmentGlobalIds() const = 0;

      virtual global_size_type
      nGlobalEnrichmentNodes() const = 0;

      virtual std::shared_ptr<const EnrichmentIdsPartition<dim>>
      getEnrichmentIdsPartition() const = 0;

      virtual std::shared_ptr<
        const EnrichmentClassicalInterfaceSpherical<ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>>
      getEnrichmentClassicalInterface() const = 0;

      virtual bool
      isOrthogonalized() const = 0;

      virtual size_type
      totalRanges() const = 0;

      // Additional functions for getting geometric constriants matrix
      // Additional functions for getting the communication pattern object
      // for MPI case

      virtual std::shared_ptr<
        const ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
      getIntrinsicConstraints() const = 0;

      // use this to add extra constraints on top of geometric constraints
      virtual std::shared_ptr<
        ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
      createConstraintsStart() const = 0;

      // call this after calling start
      virtual void
      createConstraintsEnd(
        std::shared_ptr<ConstraintsLocal<ValueTypeBasisCoeff, memorySpace>>
          constraintsLocal) const = 0;

      virtual std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
      getMPIPatternP2P() const = 0;

      virtual bool
      isDistributed() const = 0;

    }; // end of EFEBasisDofHandler
  }    // end of namespace basis
} // end of namespace dftefe
#endif // dftefeEFEBasisDofHandler_h
