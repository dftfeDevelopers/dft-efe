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

#ifndef dftefeEFEOverlapInverseOperatorContext_h
#define dftefeEFEOverlapInverseOperatorContext_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/OperatorContext.h>
#include <basis/OrthoEFEOverlapOperatorContext.h>
#include <basis/EFEBasisDataStorage.h>
#include <basis/FEBasisManager.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/FEBasisDofHandler.h>
#include <basis/EFEBasisDofHandler.h>
#include <quadrature/QuadratureAttributes.h>
#include <basis/FECellWiseDataOperations.h>
#include <vector>
#include <memory>

namespace dftefe
{
  namespace basis
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class OrthoEFEOverlapInverseOpContextGLL
      : public linearAlgebra::
          OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
    {
    public:
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

    public:
      OrthoEFEOverlapInverseOpContextGLL(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
            &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);


      // --------------DEBUG ONLY (direct inversion)---------
      OrthoEFEOverlapInverseOpContextGLL(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
            &                                      feBasisManager,
        const OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                             ValueTypeOperand,
                                             memorySpace,
                                             dim> &MContext,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
             linAlgOpContext,
        bool isCGSolved = true);

      ~OrthoEFEOverlapInverseOpContextGLL() = default;


      void
      apply(
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
        linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const override;

    private:
      const FEBasisManager<ValueTypeOperand,
                           ValueTypeOperator,
                           memorySpace,
                           dim> *    d_feBasisManager;
      const EFEBasisDofHandler<ValueTypeOperand,
                               ValueTypeOperator,
                               memorySpace,
                               dim> *d_efebasisDofHandler;
      linearAlgebra::Vector<ValueTypeOperator, memorySpace> d_diagonalInv;
      std::shared_ptr<utils::MemoryStorage<ValueTypeOperator, memorySpace>>
                d_basisOverlapEnrichmentBlock;
      size_type d_nglobalEnrichmentIds;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;


      global_size_type d_nglobalIds;
      bool             d_isCGSolved;
      std::shared_ptr<linearAlgebra::LinearSolverFunction<ValueTypeOperator,
                                                          ValueTypeOperand,
                                                          memorySpace>>
        d_overlapInvPoisson;
      std::shared_ptr<linearAlgebra::LinearSolverImpl<ValueTypeOperator,
                                                      ValueTypeOperand,
                                                      memorySpace>>
        d_CGSolve;

    }; // end of class BasisOverlapOperatorContext
  }    // namespace basis
} // end of namespace dftefe
#include <basis/OrthoEFEOverlapInverseOpContextGLL.t.cpp>
#endif // dftefeEFEOverlapInverseOperatorContext_h
