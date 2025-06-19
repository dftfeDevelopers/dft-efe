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

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/OperatorContext.h>
#include <linearAlgebra/ChebyshevFilteredEigenSolver.h>
#include <linearAlgebra/HermitianIterativeEigenSolver.h>
#include <memory>
#include <utils/ConditionalOStream.h>
#include <utils/Profiler.h>

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
        const size_type numElectrons,
        const double    smearingTemperature,
        const double    fermiEnergyTolerance,
        const double    fracOccupancyTolerance,
        const double    eigenSolveResidualTolerance,
        const size_type maxChebyshevFilterPass,
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace>
          &waveFunctionSubspaceGuess,
        linearAlgebra::Vector<ValueTypeOperand, memorySpace> &lanczosGuess,
        bool             isResidualChebyshevFilter = true,
        const size_type  waveFunctionBlockSize     = 0,
        const OpContext &MLanczos =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MInvLanczos =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        bool  storeIntermediateSubspaces = false);

      /**
       *@brief Default Destructor
       *
       */
      ~KohnShamEigenSolver() = default;

      void
      reinitBasis(
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace>
          &waveFunctionSubspaceGuess,
        linearAlgebra::Vector<ValueTypeOperand, memorySpace> &lanczosGuess,
        const OpContext &                                     MLanczos =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MInvLanczos =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>());

      void
      reinitBounds(double wantedSpectrumLowerBound,
                   double wantedSpectrumUpperBound);

      RealType
      getFermiEnergy();

      std::vector<RealType>
      getFractionalOccupancy();

      std::vector<RealType>
      getEigenSolveResidualNorm();

      void
      setChebyPolyScalingFactor(double scalingFactor);

      void
      setChebyshevPolynomialDegree(size_type chebyPolyDeg);

      void
      setResidualChebyshevFilterFlag(bool flag);

      linearAlgebra::EigenSolverError
      solve(const OpContext &      kohnShamOperator,
            std::vector<RealType> &kohnShamEnergies,
            linearAlgebra::MultiVector<ValueType, memorySpace>
              &              kohnShamWaveFunctions,
            bool             computeWaveFunctions = false,
            const OpContext &M =
              linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                     ValueTypeOperand,
                                                     memorySpace>(),
            const OpContext &MInv =
              linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                     ValueTypeOperand,
                                                     memorySpace>()) override;

      linearAlgebra::MultiVector<ValueType, memorySpace> &
      getFilteredSubspace();

      linearAlgebra::MultiVector<ValueType, memorySpace> &
      getOrthogonalizedFilteredSubspace();

    private:
      double    d_smearingTemperature;
      double    d_fermiEnergyTolerance;
      double    d_fracOccupancyTolerance;
      double    d_eigenSolveResidualTolerance;
      size_type d_maxChebyshevFilterPass;
      size_type d_chebyshevPolynomialDegree;
      size_type d_numWantedEigenvalues;
      size_type d_waveFunctionBlockSize;
      linearAlgebra::MultiVector<ValueTypeOperand, memorySpace>
        *d_waveFunctionSubspaceGuess;
      linearAlgebra::Vector<ValueTypeOperand, memorySpace> *d_lanczosGuess;
      const OpContext *                                     d_MLanczos;
      const OpContext *                                     d_MInvLanczos;
      std::vector<RealType>                                 d_fracOccupancy;
      std::vector<RealType>                                 d_eigSolveResNorm;
      RealType                                              d_fermiEnergy;
      bool                                                  d_isSolved;
      const size_type                                       d_numElectrons;
      utils::ConditionalOStream                             d_rootCout;
      double d_wantedSpectrumLowerBound;
      double d_wantedSpectrumUpperBound;
      bool   d_isBoundKnown;
      double d_chebyPolyScalingFactor;
      bool   d_setChebyPolDegExternally;

      std::shared_ptr<
        linearAlgebra::ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                                    ValueTypeOperand,
                                                    memorySpace>>
        d_chfsi;

      linearAlgebra::MultiVector<ValueType, memorySpace>
        *d_filteredSubspaceOrtho;
      linearAlgebra::MultiVector<ValueType, memorySpace> *d_filteredSubspace;
      utils::Profiler                                     d_p;
      bool d_isResidualChebyFilter;
      const bool  d_storeIntermediateSubspaces;

    }; // end of class KohnShamEigenSolver
  }    // namespace ksdft
} // end of namespace dftefe
#include <ksdft/KohnShamEigenSolver.t.cpp>
#endif // dftefeKohnShamEigenSolver_h
