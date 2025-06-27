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

#ifndef dftefeOrthonormalizationFunctions_h
#define dftefeOrthonormalizationFunctions_h

#include <utils/TypeConfig.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <string>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/OperatorContext.h>
#include <linearAlgebra/IdentityOperatorContext.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    class OrthonormalizationFunctions
    {
    public:
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
      using RealType = blasLapack::real_type<ValueType>;
      using OpContext =
        OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>;

      OrthonormalizationFunctions(
        const size_type       eigenVectorBatchSize,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>> mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext);

      /**
       *@brief Default Destructor
       *
       */
      ~OrthonormalizationFunctions() = default;

      OrthonormalizationError
      CholeskyGramSchmidt(
        MultiVector<ValueTypeOperand, memorySpace> &X,
        MultiVector<ValueType, memorySpace> &       orthogonalizedX,
        const OpContext &B = IdentityOperatorContext<ValueTypeOperator,
                                                     ValueTypeOperand,
                                                     memorySpace>());

      OrthonormalizationError
      MultipassLowdin(
        MultiVector<ValueTypeOperand, memorySpace> &X,
        size_type                                   maxPass,
        RealType                                    shiftTolerance,
        RealType                                    identityTolerance,
        MultiVector<ValueType, memorySpace> &       orthogonalizedX,
        const OpContext &B = IdentityOperatorContext<ValueTypeOperator,
                                                     ValueTypeOperand,
                                                     memorySpace>());

      static OrthonormalizationError
      ModifiedGramSchmidt(
        MultiVector<ValueTypeOperand, memorySpace> &X,
        MultiVector<ValueType, memorySpace> &       orthogonalizedX,
        const OpContext &B = IdentityOperatorContext<ValueTypeOperator,
                                                     ValueTypeOperand,
                                                     memorySpace>());

      private:

        void computeXTransBX(MultiVector<ValueTypeOperand, memorySpace> &X, 
            utils::MemoryStorage<ValueType, memorySpace> &S,
            const OpContext &B);

        std::shared_ptr<MultiVector<ValueType, memorySpace>> d_XinBatchSmall,
          d_XinBatch, d_XoutBatchSmall, d_XoutBatch;

        size_type d_eigenVecBatchSize , d_batchSizeSmall;

    }; // end of class OrthonormalizationFunctions
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/OrthonormalizationFunctions.t.cpp>
#endif
