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

#ifndef dftefeFEBasisOperations_h
#define dftefeFEBasisOperations_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/BasisOperations.h>
#include <basis/BasisDataStorage.h>
#include <basis/FEBasisManager.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    enum class FEBasisOpScratchSpaceAttr
      {
        fieldCellValues,
        basisDataInCellRange,
        basisGradientDataInCellRange,
        JxWxNBlock,
        JxWxGradNBlock
      };
    /**
     * An abstract class to handle interactions between a basis and a
     * field (e.g., integration of field with basis).
     */
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class FEBasisOperations : public BasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace>
    {
      //
      // typedefs
      //
    public:
      //
      // Get the union of the ValueTypeBasisCoeff and ValueTypeBasisData
      // (.e.g, the union of double and complex<double> is complex<double>)
      //
      using ValueTypeUnion =
        typename BasisOperations<ValueTypeBasisCoeff,
                                 ValueTypeBasisData,
                                 memorySpace>::ValueTypeUnion;

      using StorageUnion = typename BasisOperations<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace>::StorageUnion;

      using StorageBasis = typename BasisOperations<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace>::StorageBasis;

      FEBasisOperations(
        std::shared_ptr<const BasisDataStorage<ValueTypeBasisData, memorySpace>>
                        basisDataStorage,
        const size_type maxCellBlockSize,
        const size_type maxFieldBlockSize = 0);

      void
      reinit(const size_type maxCellBlockSize,
             const size_type maxFieldBlockSize = 0);

      ~FEBasisOperations() = default;

      void
      interpolate(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                                   vectorData,
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                 ValueTypeBasisData>,
          memorySpace> &quadValuesContainer) const override;

      void
      interpolateWithBasisGradient(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                                   vectorData,
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                 ValueTypeBasisData>,
          memorySpace> &quadValuesContainer) const override;

      void
      interpolateWithBasisGradient(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                                   vectorData,
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        const std::pair<size_type, size_type>                 cellRange,
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                               ValueTypeBasisData>
          *quadValuesInCellRangePtr) const;

      void
      integrateWithBasisValues(
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                                 ValueTypeBasisData>,
          memorySpace> &                                      inp,
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &vectorData) const override;

      void
      interpolate(
        const linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                                   vectorData,
        const BasisManager<ValueTypeBasisCoeff, memorySpace> &basisManager,
        const std::pair<size_type, size_type>                 cellRange,
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisCoeff,
                                               ValueTypeBasisData>
          *quadValuesInCellRangePtr) const;

      /* FE functions for local kernel computations*/
      /* \integral (L1 op1 N1) f (L2 op2 N2) dx */
      void
      computeFEMatrices(
        realspace::LinearLocalOp L1,
        realspace::VectorMathOp  Op1,
        const quadrature::QuadratureValuesContainer<ValueTypeUnion, memorySpace>
          &                                          f,
        realspace::VectorMathOp  L2,
        realspace::LinearLocalOp Op2,
        StorageUnion &                               cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext) const;

      /* FE functions for local kernel computations*/
      /* \integral f L12 op12 (N1.N2) dx */
      void
      computeFEMatrices(
        const quadrature::QuadratureValuesContainer<ValueTypeUnion, memorySpace>
          &                                          f,
        realspace::VectorMathOp  Op12,
        realspace::LinearLocalOp L12,
        StorageUnion &                               cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext) const;

     /* FE functions for local kernel computations*/
      /* \integral (L1 op1 N1) (L2 op2 N2) dx */
      void
      computeFEMatrices(
        realspace::LinearLocalOp                     L1,
        realspace::VectorMathOp                      Op1,
        realspace::LinearLocalOp                     L2,
        StorageBasis &                               cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext) const;

    private:
      std::shared_ptr<const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        d_feBasisDataStorage;
      size_type d_maxCellBlock;
      size_type d_maxFieldBlock;

      /**---temporary scratch spaces----- */
      mutable StorageBasis d_basisDataInCellRange, d_basisGradientDataInCellRange, d_JxWxNBlock, d_JxWxGradNBlock;
      mutable StorageUnion d_fieldCellValues, d_fxJxWxNBlock;
      mutable std::pair<size_type, size_type> d_cellRangeForBasisDataCache;
      mutable bool                            d_isBasisDataCellRangeCached;
      mutable std::pair<size_type, size_type> d_cellRangeForBasisGradientDataCache;
      mutable bool                            d_isBasisGradientDataCellRangeCached;
      /**---temporary scratch spaces----- */

      std::vector<size_type> d_numCellDofs;
      std::vector<size_type> d_numCellQuad;
      const FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim> *d_feBasisDofHandler;
      size_type d_maxDofInCell;
      size_type d_maxQuadInCell;
      size_type d_numLocallyOwnedCells;
      bool d_variableDofsPerCell, d_sameQuadRuleInAllCells;
      std::shared_ptr<const quadrature::QuadratureRuleContainer> d_quadratureRuleContainer;

      void deleteScratch() const;

      void
      BasisWeakFormKernelWithField(
        realspace::LinearLocalOp L1,
        realspace::VectorMathOp  Op1,
        const quadrature::QuadratureValuesContainer<ValueTypeUnion,
          memorySpace> &                                     f,
        realspace::VectorMathOp                              Op2,
        realspace::LinearLocalOp                             L2,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                                             feBasisDataStorage,
        const size_type                                      cellBlockSize,
        StorageUnion &cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &        linAlgOpContext) const;

      void
      BasisWeakFormKernelWithField(
        const quadrature::QuadratureValuesContainer<ValueTypeUnion,
          memorySpace> &                                     f,
        realspace::VectorMathOp                              Op12,
        realspace::LinearLocalOp                             L12,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                                             feBasisDataStorage,
        const size_type                                      cellBlockSize,
        StorageUnion &cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &        linAlgOpContext) const;

      void
      BasisWeakFormKernel(
        realspace::LinearLocalOp L1,
        realspace::VectorMathOp  Op1,
        realspace::LinearLocalOp L2,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                                             feBasisDataStorage,
        const size_type                                      cellBlockSize,
        StorageBasis &cellWiseFEData,
        linearAlgebra::LinAlgOpContext<memorySpace> &        linAlgOpContext) const;

    }; // end of FEBasisOperations
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/FEBasisOperations.t.cpp>
#endif // dftefeBasisOperations_h
