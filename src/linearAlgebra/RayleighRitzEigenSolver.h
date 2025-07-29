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

#ifndef dftefeRayleighRitzEigenSolver_h
#define dftefeRayleighRitzEigenSolver_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/OperatorContext.h>
#include <memory>
#include <linearAlgebra/ElpaScalapackManager.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *@brief A derived class of OperatorContext to encapsulate
     * the action of a discrete operator on vectors, matrices, etc.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     * on which the operator will act
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNED, etc.) in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    class RayleighRitzEigenSolver
    {
    public:
      /**
       * @brief define ValueType as the superior (bigger set) of the
       * ValueTypeOperator and ValueTypeOperand
       * (e.g., between double and complex<double>, complex<double>
       * is the bigger set)
       */
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
      using RealType = blasLapack::real_type<ValueType>;
      using OpContext =
        OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>;

    public:
      /**
       * @brief Constructor
       */
      RayleighRitzEigenSolver(
        const size_type             eigenVectorBatchSize,
        const ElpaScalapackManager &elpaScala,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                      mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext);

      /**
       *@brief Default Destructor
       *
       */
      ~RayleighRitzEigenSolver() = default;

      // In this case we can solve the KohnSham GHEP if the X's are B
      // orthogonalized. X_MO^T H X_MO Q = \lambda X_MO^T M X_MO Q , or, H X =
      // \lambda M X if X = X_MO Q
      EigenSolverError
      solve(const OpContext &                           A,
            MultiVector<ValueTypeOperand, memorySpace> &X,
            std::vector<RealType> &                     eigenValues,
            MultiVector<ValueType, memorySpace> &       eigenVectors,
            bool computeEigenVectors = false);

      // In this case we solve the Kohn Sham GHEP if X's are othogonalized.
      // X_O^T H X_O Q = \lambda X_O^T M X_O Q , or,
      // H X = \lambda M X if X = X_O Q
      EigenSolverError
      solve(const OpContext &                           A,
            const OpContext &                           B,
            MultiVector<ValueTypeOperand, memorySpace> &X,
            std::vector<RealType> &                     eigenValues,
            MultiVector<ValueType, memorySpace> &       eigenVectors,
            bool computeEigenVectors = false);

      // /**  In this case we solve the Kohn Sham GHEP for any general X
      // * Performs cholesky factorization for orthogonalization internally.
      // * SConj = X^T M X
      // * SConj=LConj*L^{T}
      // * Lconj^{-1} compute
      // * compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C  (C denotes
      // *     conjugate transpose LAPACK notation)
      // * compute standard eigendecomposition HSConjProj: {QConjPrime,D}
      // * HSConjProj=QConjPrime*D*QConjPrime^{C} QConj={Lc^{-1}}^{C}*QConjPrime
      // *     rotate the basis in the subspace
      // * X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, stored in the column major
      // *     format In the above we use Q^{T}={QConjPrime}^{C}*LConj^{-1}
      // * In other words, X_O^T H X_O Q = \lambda X_O^T M X_O Q , or,
      // * H X = \lambda M X if X = X_O Q where X_O = X((LInv)^C) (C is conj
      // trans)
      // **/
      // EigenSolverError
      // solveGEPNoOrtho(const OpContext &                 A,
      //       const OpContext &                           B,
      //       MultiVector<ValueTypeOperand, memorySpace> &X,
      //       std::vector<RealType> &                     eigenValues,
      //       MultiVector<ValueType, memorySpace> &       eigenVectors,
      //       bool computeEigenVectors = false);

    private:
      void
      computeXTransOpX(MultiVector<ValueTypeOperand, memorySpace> &  X,
                       utils::MemoryStorage<ValueType, memorySpace> &S,
                       const OpContext &                             Op);

      std::shared_ptr<MultiVector<ValueType, memorySpace>> d_XinBatchSmall,
        d_XinBatch, d_XoutBatchSmall, d_XoutBatch;

      size_type d_eigenVecBatchSize, d_batchSizeSmall;

      const ElpaScalapackManager *d_elpaScala;
      const bool            d_useELPA;

    }; // end of class RayleighRitzEigenSolver
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/RayleighRitzEigenSolver.t.cpp>
#endif // dftefeRayleighRitzEigenSolver_h
