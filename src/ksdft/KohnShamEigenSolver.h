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

#ifndef dftefeKohnShamEigenSolver_h
#define dftefeKohnShamEigenSolver_h

#include <utils/ScalarFunction.h>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <linearAlgebra/HermitianIterativeEigenSolver.h>
#include <linearAlgebra/RayleighRitzEigenSolver.h>
#include <linearAlgebra/ChebyshevFilter.h>
#include <memory>

namespace dftefe
{
  namespace ksdft
  {
    /**
     *@brief
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
    class KohnShamEigenSolver
      : public linearAlgebra::HermitianIterativeEigenSolver<ValueTypeOperator,
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
      using ValueType = typename linearAlgebra::HermitianIterativeEigenSolver<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace>::ValueType;
      using RealType = typename linearAlgebra::HermitianIterativeEigenSolver<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace>::RealType;
      using OpContext = typename linearAlgebra::HermitianIterativeEigenSolver<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace>::OpContext;

    public:
      /**
       * @brief Constructor
       * The occupation Tolerance is such that one considers the
       * Kohn Sham Orbitals occupied if occupationTolerance < f_i(from
       * diftribution) < 1-1e-12
       */
      KohnShamEigenSolver(
        const utils::ScalarFunctionReal occupationDistribution,
        const double                    occupationTolerance,
        const double                    residualTolerance,
        const size_type                 chebyshevPolynomialDegree,
        const size_type                 maxChebyshevFilterPass,
        const MultiVector<ValueTypeOperand, memorySpace>
          &             waveFunctionSubspaceGuess,
        const size_type waveFunctionBlockSize = 0);

      /**
       *@brief Default Destructor
       *
       */
      ~KohnShamEigenSolver() = default;

      void
      reinit(const utils::ScalarFunctionReal &occupationDistribution,
             const double                     occupationTolerance,
             const double                     residualTolerance,
             const size_type                  maxChebyshevFilterPass,
             const MultiVector<ValueTypeOperand, memorySpace>
               &             waveFunctionSubspaceGuess,
             const size_type waveFunctionBlockSize = 0);

      EigenSolverError
      solve(const OpContext &                    kohnShamOperator,
            std::vector<RealType> &              kohnShamEnergies,
            MultiVector<ValueType, memorySpace> &kohnShamWaveFunctions,
            bool                                 computeWaveFunctions = false,
            const OpContext &M = IdentityOperatorContext<ValueTypeOperator,
                                                         ValueTypeOperand,
                                                         memorySpace>(),
            const OpContext &MInv =
              IdentityOperatorContext<ValueTypeOperator,
                                      ValueTypeOperand,
                                      memorySpace>()) override;

    private:
      double                                     d_occupationTolerance;
      double                                     d_residualTolerance;
      size_type                                  d_maxFilterPass;
      size_type                                  d_polynomialDegree;
      MultiVector<ValueTypeOperand, memorySpace> d_waveFunctionSubspaceGuess;

    }; // end of class KohnShamEigenSolver
  }    // namespace ksdft
} // end of namespace dftefe
#include <linearAlgebra/KohnShamEigenSolver.t.cpp>
#endif // dftefeKohnShamEigenSolver_h
