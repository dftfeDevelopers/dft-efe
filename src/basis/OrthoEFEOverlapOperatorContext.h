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

#ifndef dftefeEFEOverlapOperatorContext_h
#define dftefeEFEOverlapOperatorContext_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <basis/FEBasisManager.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/FEBasisDofHandler.h>
#include <basis/EFEBasisDofHandler.h>
#include <basis/EFEBasisDataStorage.h>
#include <quadrature/QuadratureAttributes.h>
#include <memory>

namespace dftefe
{
  namespace basis
  {
    /**
     *@brief A derived class of linearAlgebra::OperatorContext to encapsulate
     * the action of a discrete operator on vectors, matrices, etc. for enriched
     *basis
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator e.g.
     * the \f$\integral_{\omega} N_i^E(x) N_j^E(x) dx\f$ in this case where E is
     *the enrichment functions.
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act.
     * @tparam memorySpace The memory sapce (HOST, DEVICE, HOST_PINNED, etc.) in which the data of the operator
     * and its operands reside.
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class OrthoEFEOverlapOperatorContext
      : public linearAlgebra::
          OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
    {
    public:
      /**
       * @brief define ValueType as the superior (bigger set) of the
       * ValueTypeOperator and ValueTypeOperand
       * (e.g., between double and complex<double>, complex<double>
       * is the bigger set)
       */
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

      using Storage =
        dftefe::utils::MemoryStorage<ValueTypeOperator, memorySpace>;

    public:
      /**
       * @brief Constructor where the classical dofs have a different quadrature rule than that of the enrichment dofs.
       * This can happen when the classical dofs have a different quadrature
       * than that of the enrichment dofs. For example one can have Adaptive
       * quadrature for the enrichment functions and GLL for the classical dofs.
       * @tparam feBasisManager FEBasisManager object for getting the processor local to cell mapping of the distributed vector
       * @tparam cfeBasisDataStorage Classical FEBasisDataStorage object for getting the basisvalues of the classical dofs
       * @tparam efeBasisDataStorage Enrichment FEBasisDataStorage object for getting the basisvalues of the enrichment dofs
       * @tparam constraintsX Constraints for X
       * @tparam constraintsY Constraints for Y
       * @tparam maxCellTimesNumVecs cell times number of vectors
       */
      OrthoEFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             enrichmentBlockClassicalBasisDataStorage,
        const size_type maxCellTimesNumVecs,
        const bool      calculateWings = true);

      /**
       * @brief Constructor where the classical dofs have a different quadrature rule than that of the enrichment dofs.
       * This can happen when the classical dofs have a different quadrature
       * than that of the enrichment dofs. For example one can have Adaptive
       * quadrature for the enrichment functions and GLL for the classical dofs.
       * @tparam feBasisManager FEBasisManager object for getting the processor local to cell mapping of the distributed vector
       * @tparam cfeBasisDataStorage Classical FEBasisDataStorage object for getting the basisvalues of the classical dofs
       * @tparam efeBasisDataStorage Enrichment FEBasisDataStorage object for getting the basisvalues of the enrichment dofs
       * @tparam constraintsX Constraints for X
       * @tparam constraintsY Constraints for Y
       * @tparam maxCellTimesNumVecs cell times number of vectors
       */
      OrthoEFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             enrichmentBlockBasisDataStorage,
        const size_type maxCellTimesNumVecs,
        const bool      calculateWings = true);

      /**
       * @brief Constructor where the classical dofs have a different quadrature rule than that of the enrichment dofs.
       * This can happen when the classical dofs have a different quadrature
       * than that of the enrichment dofs. For example one can have Adaptive
       * quadrature for the enrichment functions and GLL for the classical dofs.
       * @tparam feBasisManager FEBasisManager object for getting the processor local to cell mapping of the distributed vector
       * @tparam cfeBasisDataStorage Classical FEBasisDataStorage object for getting the basisvalues of the classical dofs
       * @tparam efeBasisDataStorage Enrichment FEBasisDataStorage object for getting the basisvalues of the enrichment dofs
       * @tparam constraintsX Constraints for X
       * @tparam constraintsY Constraints for Y
       * @tparam maxCellTimesNumVecs cell times number of vectors
       */
      OrthoEFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);

      /**
       * @brief Constructor where the classical dofs have a different quadrature rule than that of the enrichment dofs.
       * This can happen when the classical dofs have a different quadrature
       * than that of the enrichment dofs. For example one can have Adaptive
       * quadrature for the enrichment functions and GLL for the classical dofs.
       * @tparam feBasisManager FEBasisManager object for getting the processor local to cell mapping of the distributed vector
       * @tparam cfeBasisDataStorage Classical FEBasisDataStorage object for getting the basisvalues of the classical dofs
       * @tparam efeBasisDataStorage Enrichment FEBasisDataStorage object for getting the basisvalues of the enrichment dofs
       * @tparam constraintsX Constraints for X
       * @tparam constraintsY Constraints for Y
       * @tparam maxCellTimesNumVecs cell times number of vectors
       */
      OrthoEFEOverlapOperatorContext(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);

      /**
       * @brief Apply AX = B where A is the discretized matrix, X is the operand and B is the result.
       */
      void
      apply(
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
        linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const override;

      // get overlap of two basis functions in a cell
      Storage
      getBasisOverlap(const size_type cellId,
                      const size_type basisId1,
                      const size_type basisId2) const;

      // get overlap of all the basis functions in a cell
      Storage
      getBasisOverlapInCell(const size_type cellId) const;

      // get overlap of all the basis functions in all cells
      const Storage &
      getBasisOverlapInAllCells() const;


    private:
      const FEBasisManager<ValueTypeOperand,
                           ValueTypeOperator,
                           memorySpace,
                           dim> *d_feBasisManager;
      std::shared_ptr<Storage>   d_basisOverlap;
      std::vector<size_type>     d_cellStartIdsBasisOverlap;
      std::vector<size_type>     d_dofsInCell;
      const size_type            d_maxCellTimesNumVecs;

      bool d_isMassLumping;
      std::shared_ptr<linearAlgebra::Vector<ValueTypeOperator, memorySpace>>
        d_diagonal;
      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
                                     d_basisOverlapEnrichmentBlock;
      const EFEBasisDofHandler<ValueTypeOperand,
                               ValueTypeOperator,
                               memorySpace,
                               dim> *d_efebasisDofHandler;
      size_type                      d_nglobalEnrichmentIds;

    }; // end of class OrthoEFEOverlapOperatorContext
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/OrthoEFEOverlapOperatorContext.t.cpp>
#endif // dftefeEFEOverlapOperatorContext_h
