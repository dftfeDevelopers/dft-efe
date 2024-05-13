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

#include <ksdft/Defaults.h>
#include <linearAlgebra/LanczosExtremeEigenSolver.h>
#include <linearAlgebra/ChebyshevFilteredEigenSolver.h>
#include <ksdft/FractionalOccupancyFunction.h>
namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      KohnShamEigenSolver(
        const double    smearingTemperature,
        const double    fermiEnergyTolerance,
        const double    fracOccupancyTolerance,
        const double    eigenSolveResidualTolerance,
        const size_type chebyshevPolynomialDegree,
        const size_type maxChebyshevFilterPass,
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace>
          &             waveFunctionSubspaceGuess,
        const size_type waveFunctionBlockSize)
      : d_numWantedEigenvalues(waveFunctionSubspaceGuess.getNumberComponents())
      , d_eigenSolveResidualTolerance(eigenSolveResidualTolerance)
      , d_chebyshevPolynomialDegree(chebyshevPolynomialDegree)
      , d_maxChebyshevFilterPass(maxChebyshevFilterPass)
      , d_waveFunctionBlockSize(waveFunctionBlockSize)
      , d_fermiEnergyTolerance(fermiEnergyTolerance)
      , d_fracOccupancyTolerance(fracOccupancyTolerance)
      , d_smearingTemperature(smearingTemperature)
      , d_fracOccupancy(d_numWantedEigenvalues)
    {
      reinit(waveFunctionSubspaceGuess);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      reinit(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace>
               &waveFunctionSubspaceGuess)
    {
      d_isSolved                  = false;
      d_waveFunctionSubspaceGuess = &waveFunctionSubspaceGuess;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    linearAlgebra::EigenSolverError
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      solve(const OpContext &      kohnShamOperator,
            std::vector<RealType> &kohnShamEnergies,
            linearAlgebra::MultiVector<ValueType, memorySpace>
              &              kohnShamWaveFunctions,
            bool             computeWaveFunctions,
            const OpContext &M,
            const OpContext &MInv)
    {
      d_isSolved                          = true;
      global_size_type globalSize         = kohnShamWaveFunctions.globalSize();
      double           eigenSolveResidual = 0;

      linearAlgebra::EigenSolverError     returnValue;
      linearAlgebra::EigenSolverError     lanczosErr;
      linearAlgebra::EigenSolverError     chfsiErr;
      linearAlgebra::EigenSolverErrorCode err;

      // get bounds from lanczos
      std::vector<double> tol{
        ksdft::LinearEigenSolverDefaults::LANCZOS_EXTREME_EIGENVAL_TOL,
        ksdft::LinearEigenSolverDefaults::LANCZOS_EXTREME_EIGENVAL_TOL};
      linearAlgebra::LanczosExtremeEigenSolver<ValueTypeOperator,
                                               ValueTypeOperand,
                                               memorySpace>
        lanczos(globalSize,
                1,
                1,
                tol,
                ksdft::LinearEigenSolverDefaults::LANCZOS_BETA_TOL,
                kohnShamWaveFunctions.getMPIPatternP2P(),
                *kohnShamWaveFunctions.getLinAlgOpContext());

      linearAlgebra::MultiVector<ValueType, memorySpace> eigenVectorsLanczos(0,
                                                                             0);
      std::vector<RealType>                              eigenValuesLanczos(2);
      lanczosErr = lanczos.solve(kohnShamOperator,
                                 eigenValuesLanczos,
                                 eigenVectorsLanczos,
                                 false,
                                 M,
                                 MInv);

      size_type iPass = 0;
      if (lanczosErr.isSuccess)
        {
          std::vector<RealType> diagonal(0), subDiagonal(0);
          lanczos.getTridiagonalMatrix(diagonal, subDiagonal);
          RealType residual = subDiagonal[subDiagonal.size() - 1];

          double wantedSpectrumUpperBound =
            (eigenValuesLanczos[1] - eigenValuesLanczos[0]) *
              ((double)(d_numWantedEigenvalues) / globalSize) +
            eigenValuesLanczos[0];

          linearAlgebra::MultiVector<ValueType, memorySpace> HX(
            kohnShamWaveFunctions.getMPIPatternP2P(),
            kohnShamWaveFunctions.getLinAlgOpContext(),
            1),
            MX(HX), residualEigenSolver(HX), waveFnResidualCheck(HX);

          utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

          for (; iPass < d_maxChebyshevFilterPass; iPass++)
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

              chfsiErr = chfsi->solve(kohnShamOperator,
                                      kohnShamEnergies,
                                      kohnShamWaveFunctions,
                                      computeWaveFunctions,
                                      M,
                                      MInv);

              // Calculate the chemical potential using newton raphson

              std::shared_ptr<ksdft::FractionalOccupancyFunction> fOcc =
                std::make_shared<ksdft::FractionalOccupancyFunction>(
                  kohnShamEnergies,
                  d_numWantedEigenvalues,
                  Constants::BOLTZMANN_CONST_HARTREE,
                  d_smearingTemperature);

              linearAlgebra::NewtonRaphsonSolver<RealType> nrs(
                NewtonRaphsonSolverDefaults::MAX_ITER,
                d_fermiEnergyTolerance,
                NewtonRaphsonSolverDefaults::FORCE_TOL);

              linearAlgebra::NewtonRaphsonError err = nrs.solve(*fOcc);

              fOcc->getSolution(d_fermiEnergy);

              // Calculate the frac occupancy vector
              for (size_type i = 0; i < d_fracOccupancy.size(); i++)
                {
                  d_fracOccupancy[i] =
                    fermiDirac(kohnShamEnergies[i],
                               d_fermiEnergy,
                               Constants::BOLTZMANN_CONST_HARTREE,
                               d_smearingTemperature);
                }

              // kohnshamenergies are in ascending order from lapack assumed
              size_type residualCheckIndex;
              for (size_type i = d_fracOccupancy.size(); i > 0; i--)
                {
                  if (d_fracOccupancy[i] > d_fracOccupancyTolerance)
                    {
                      residualCheckIndex = i;
                      break;
                    }
                }

              for (size_type iSize = 0;
                   iSize < kohnShamWaveFunctions.locallyOwnedSize();
                   iSize++)
                memoryTransfer.copy(1,
                                    waveFnResidualCheck.data(),
                                    kohnShamWaveFunctions.data() +
                                      iSize * d_numWantedEigenvalues +
                                      residualCheckIndex);

              // calculate residualEigenSolver
              kohnShamOperator.apply(waveFnResidualCheck, HX);
              M.apply(waveFnResidualCheck, MX);
              RealType nenergyAtResidualCheck =
                -kohnShamEnergies[waveFnResidualCheck];
              linearAlgebra::blasLapack::
                axpbyBlocked<ValueType, ValueType, memorySpace>(
                  MX.locallyOwnedSize(),
                  (ValueType)1.0,
                  HX.data(),
                  (ValueType)(nenergyAtResidualCheck),
                  MX.data(),
                  residualEigenSolver.data(),
                  *MX.getLinAlgOpContext());

              eigenSolveResidual = (residualEigenSolver.l2Norms())[0];

              if (eigenSolveResidual < d_eigenSolveResidualTolerance ||
                  !chfsiErr.isSuccess)
                break;
              else
                *d_waveFunctionSubspaceGuess = kohnShamWaveFunctions;
            }
        }
      else if (!lanczosErr.isSuccess)
        {
          err = linearAlgebra::EigenSolverErrorCode::KS_LANCZOS_ERROR;
          returnValue =
            linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
          returnValue.msg += lanczosErr.msg;
        }
      else if (!chfsiErr.isSuccess)
        {
          err = linearAlgebra::EigenSolverErrorCode::KS_CHFSI_ERROR;
          returnValue =
            linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
          returnValue.msg += chfsiErr.msg;
        }
      else if (iPass > d_maxChebyshevFilterPass)
        {
          err = linearAlgebra::EigenSolverErrorCode::KS_MAX_PASS_ERROR;
          returnValue =
            linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
        }
      else
        {
          err = linearAlgebra::EigenSolverErrorCode::SUCCESS;
          returnValue =
            linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
          returnValue.msg += "Number of CHFSI passes required are " +
                             std::to_string(iPass) + ".";
        }

      return returnValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    typename linearAlgebra::HermitianIterativeEigenSolver<ValueTypeOperator,
                                                          ValueTypeOperand,
                                                          memorySpace>::RealType
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      getFermiEnergy()
    {
      utils::throwException(
        d_isSolved,
        "Cannot call getFermiEnergy() before solving the eigenproblem.");
      return d_fermiEnergy;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    std::vector<typename linearAlgebra::HermitianIterativeEigenSolver<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace>::RealType>
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      getFractionalOccupancy()
    {
      utils::throwException(
        d_isSolved,
        "Cannot call getFractionalOccupancy() before solving the eigenproblem.");
      return d_fracOccupancy;
    }

  } // namespace ksdft
} // end of namespace dftefe
