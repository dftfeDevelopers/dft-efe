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

#ifndef dftefeChebyshevFilteredEigenSolver_h
#define dftefeChebyshevFilteredEigenSolver_h

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
#include <utils/Profiler.h>

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
    class ChebyshevFilteredEigenSolver
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

    public:
      /**
       * @brief Constructor
       */
      ChebyshevFilteredEigenSolver(
        const double                                wantedSpectrumLowerBound,
        const double                                wantedSpectrumUpperBound,
        const double                                unWantedSpectrumUpperBound,
        const double                                polynomialDegree,
        const double                                illConditionTolerance,
        MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess,
        bool            isResidualChebyshevFilter = true,
        const size_type eigenVectorBlockSize      = 0);

      /**
       *@brief Destructor
       *
       */
      ~ChebyshevFilteredEigenSolver() = default;

      void
      reinit(const double wantedSpectrumLowerBound,
             const double wantedSpectrumUpperBound,
             const double unWantedSpectrumUpperBound,
             const double polynomialDegree,
             const double illConditionTolerance,
             MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess,
             const size_type                             eigenVectorBlockSize);

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

      MultiVector<ValueType, memorySpace> &
      getFilteredSubspace();

      MultiVector<ValueType, memorySpace> &
      getOrthogonalizedFilteredSubspace();

    private:
      double                                      d_wantedSpectrumLowerBound;
      double                                      d_wantedSpectrumUpperBound;
      double                                      d_unWantedSpectrumUpperBound;
      double                                      d_polynomialDegree;
      double                                      d_illConditionTolerance;
      MultiVector<ValueTypeOperand, memorySpace> *d_eigenSubspaceGuess;
      size_type                                   d_eigenVectorBlockSize;

      std::shared_ptr<MultiVector<ValueType, memorySpace>>
        d_filteredSubspaceOrtho;
      std::shared_ptr<MultiVector<ValueType, memorySpace>> d_filteredSubspace;

      std::shared_ptr<
        RayleighRitzEigenSolver<ValueTypeOperator, ValueType, memorySpace>>
                      d_rr;
      utils::Profiler d_p;
      const bool      d_isResidualChebyFilter;

    }; // end of class ChebyshevFilteredEigenSolver
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/ChebyshevFilteredEigenSolver.t.cpp>
#endif // dftefeChebyshevFilteredEigenSolver_h
