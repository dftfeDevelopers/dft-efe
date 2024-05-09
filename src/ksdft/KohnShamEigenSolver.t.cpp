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

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      KohnShamEigenSolver(
        const double                                eigenSolveResidualTolerance,
        const size_type                             chebyshevPolynomialDegree,
        const size_type                             maxChebyshevFilterPass,
        MultiVector<ValueTypeOperand, memorySpace> &waveFunctionSubspaceGuess,
        const size_type                             waveFunctionBlockSize)
      : d_numWantedEigenvalues(waveFunctionSubspaceGuess.getNumberComponents())
      , d_eigenSolveResidualTolerance(eigenSolveResidualTolerance)
      , d_chebyshevPolynomialDegree(chebyshevPolynomialDegree)
      , d_maxChebyshevFilterPass(maxChebyshevFilterPass)
      , d_waveFunctionBlockSize(waveFunctionBlockSize)
    {
      reinit(waveFunctionSubspaceGuess);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      reinit(
        MultiVector<ValueTypeOperand, memorySpace> &waveFunctionSubspaceGuess)
    {
      d_waveFunctionSubspaceGuess = &waveFunctionSubspaceGuess;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      solve(const OpContext &                    kohnShamOperator,
            std::vector<RealType> &              kohnShamEnergies,
            MultiVector<ValueType, memorySpace> &kohnShamWaveFunctions,
            bool                                 computeWaveFunctions,
            const OpContext &                    M,
            const OpContext &                    MInv)
    {
      // get bounds from lanczos
      std::vector<double> tol{
        ksdft::LinearEigenSolverDefaults::LANCZOS_EXTREME_EIGENVAL_TOL,
        ksdft::LinearEigenSolverDefaults::LANCZOS_EXTREME_EIGENVAL_TOL};
      linearAlgebra::LanczosExtremeEigenSolver<ValueTypeOperator,
                                               ValueTypeOperand,
                                               memorySpace>
        lanczos(maxKrylovSubspaceSize,
                1,
                1,
                tol,
                lanczosBetaTolerance,
                mpiPatternP2P,
                *kohnShamWaveFunctions.getLinAlgOpContext());

      linearAlgebra::MultiVector<ValueType, memorySpace> eigenVectorsLanczos(0,
                                                                             0);
      std::vector<RealType>                              eigenValuesLanczos(2);
      lanczos.solve(kohnShamOperator,
                    eigenValuesLanczos,
                    eigenVectorsLanczos,
                    false,
                    M,
                    MInv);

      std::vector<RealType> diagonal(0), subDiagonal(0);
      lanczos.getTridiagonalMatrix(diagonal, subDiagonal);
      RealType residual = subDiagonal[subDiagonal.size() - 1];

      double wantedSpectrumUpperBound =
        (eigenValuesLanczos[1] - eigenValuesLanczos[0]) *
          ((double)(d_numWantedEigenvalues +
                    (int)(0.05 * numWantedEigenValues)) /
           globalSize) +
        eigenValuesLanczos[0];

      MultiVector<ValueType, memorySpace> HX(kohnShamWaveFunctions),
        MX(kohnShamWaveFunctions), residualEigenSolver(kohnShamWaveFunctions);

      for (size_type iPass = 0; iPass < d_maxChebyshevFilterPass; iPass++)
        {
          // do chebyshev filetered eigensolve

          std::shared_ptr<
            linearAlgebra::HermitianIterativeEigenSolver<ValueTypeOperator,
                                                         ValueTypeOperand,
                                                         memorySpace>>
            chfsi = std::make_shared<
              linearAlgebra::ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                                          ValueTypeOperand,
                                                          memorySpace>>(
              eigenValuesLanczos[0],
              wantedSpectrumUpperBound,
              eigenValuesLanczos[1] + residual,
              d_chebyshevPolynomialDegree,
              ksdft::LinearEigenSolverDefaults::ILL_COND_TOL,
              *d_waveFunctionSubspaceGuess,
              d_waveFunctionBlockSize);

          linearAlgebra::EigenSolverError err =
            chfsi->solve(kohnShamOperator,
                         kohnShamEnergies,
                         kohnShamWaveFunctions,
                         computeWaveFunctions,
                         M,
                         MInv);

          // Calculate the chemical potential using newton raphson

          // Calculate the occupation vector

          // calculate residualEigenSolver
          kohnShamOperator.apply(kohnShamWaveFunctions, HX);
          M.apply(kohnShamWaveFunctions, MX);

          linearAlgebra::blasLapack::
            axpbyBlocked<ValueType, ValueType, memorySpace>(
              MX.locallyOwnedSize(),
              MX.getNumberComponents(),
              ones.data(),
              HX.data(),
              kohnShamEnergiesMemSpace.data(),
              MX.data(),
              residualEigenSolver.data(),
              *MX.getLinAlgOpContext());

          double eigenSolveResidual = /*Calculate this*/

            if (eigenSolveResidual < d_eigenSolveResidualTolerance) break;
          else *d_waveFunctionSubspaceGuess = kohnShamWaveFunctions;
        }
    }

  } // namespace ksdft
} // end of namespace dftefe
