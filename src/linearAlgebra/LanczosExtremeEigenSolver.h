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

#ifndef dftefeLanczosExtremeEigenSolver_h
#define dftefeLanczosExtremeEigenSolver_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <linearAlgebra/IdentityOperatorContext.h>
#include <linearAlgebra/HermitianIterativeEigenSolver.h>
#include <memory>

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
    class LanczosExtremeEigenSolver
      : public HermitianIterativeEigenSolver<ValueTypeOperator,
                                             ValueTypeOperand,
                                             memorySpace>
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
        typename HermitianIterativeEigenSolver<ValueTypeOperator,
                                               ValueTypeOperand,
                                               memorySpace>::OpContext;

      /**
       * @brief Constructor
       */
      LanczosExtremeEigenSolver(
        const size_type                              maxKrylovSubspaceSize,
        const size_type                              numLowerExtermeEigenValues,
        const size_type                              numUpperExtermeEigenValues,
        std::vector<double> &                        tolerance,
        double                                       lanczosBetaTolerance,
        const Vector<ValueTypeOperand, memorySpace> &initialGuess,
        bool                                         isAdaptiveSolve = true);

      LanczosExtremeEigenSolver(
        const size_type      maxKrylovSubspaceSize,
        const size_type      numLowerExtermeEigenValues,
        const size_type      numUpperExtermeEigenValues,
        std::vector<double> &tolerance,
        double               lanczosBetaTolerance,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                      mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext,
        bool                                          isAdaptiveSolve = true);

      /**
       *@brief Default Destructor
       *
       */
      ~LanczosExtremeEigenSolver() = default;

      void
      reinit(const size_type      maxKrylovSubspaceSize,
             const size_type      numLowerExtermeEigenValues,
             const size_type      numUpperExtermeEigenValues,
             std::vector<double> &tolerance,
             double               lanczosBetaTolerance,
             const Vector<ValueTypeOperand, memorySpace> &initialGuess);

      void
      reinit(const size_type      maxKrylovSubspaceSize,
             const size_type      numLowerExtermeEigenValues,
             const size_type      numUpperExtermeEigenValues,
             std::vector<double> &tolerance,
             double               lanczosBetaTolerance,
             std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                           mpiPatternP2P,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext);

      EigenSolverError
      solve(const OpContext &                    A,
            std::vector<RealType> &              eigenValues,
            MultiVector<ValueType, memorySpace> &eigenVectors,
            bool                                 computeEigenVectors = false,
            const OpContext &B = IdentityOperatorContext<ValueTypeOperator,
                                                         ValueTypeOperand,
                                                         memorySpace>(),
            const OpContext &BInv =
              IdentityOperatorContext<ValueTypeOperator,
                                      ValueTypeOperand,
                                      memorySpace>()) override;

      void
      getTridiagonalMatrix(std::vector<RealType> &diagonal,
                           std::vector<RealType> &subDiagonal) const;

    private:
      Vector<ValueTypeOperand, memorySpace> d_initialGuess;
      size_type                             d_maxKrylovSubspaceSize;
      size_type                             d_numLowerExtermeEigenValues;
      size_type                             d_numUpperExtermeEigenValues;
      std::vector<double>                   d_tolerance;
      double                                d_lanczosBetaTolerance;
      std::vector<RealType>                 d_diagonal;
      std::vector<RealType>                 d_subDiagonal;
      bool                                  d_isSolved;
      const bool                            d_isAdaptiveSolve;


    }; // end of class LanczosExtremeEigenSolver
  }    // namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/LanczosExtremeEigenSolver.t.cpp>
#endif // dftefeLanczosExtremeEigenSolver_h
