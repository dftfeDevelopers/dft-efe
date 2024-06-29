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
        const size_type numElectrons,
        const double    smearingTemperature,
        const double    fermiEnergyTolerance,
        const double    fracOccupancyTolerance,
        const double    eigenSolveResidualTolerance,
        const size_type chebyshevPolynomialDegree,
        const size_type maxChebyshevFilterPass,
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace>
          &waveFunctionSubspaceGuess,
        linearAlgebra::Vector<ValueTypeOperand, memorySpace> &lanczosGuess,
        const size_type  waveFunctionBlockSize,
        const OpContext &MLanczos,
        const OpContext &MInvLanczos)
      : d_numWantedEigenvalues(waveFunctionSubspaceGuess.getNumberComponents())
      , d_eigenSolveResidualTolerance(eigenSolveResidualTolerance)
      , d_chebyshevPolynomialDegree(chebyshevPolynomialDegree)
      , d_maxChebyshevFilterPass(maxChebyshevFilterPass)
      , d_waveFunctionBlockSize(waveFunctionBlockSize)
      , d_fermiEnergyTolerance(fermiEnergyTolerance)
      , d_fracOccupancyTolerance(fracOccupancyTolerance)
      , d_smearingTemperature(smearingTemperature)
      , d_fracOccupancy(d_numWantedEigenvalues)
      , d_eigSolveResNorm(d_numWantedEigenvalues)
      , d_numElectrons(numElectrons)
      , d_rootCout(std::cout)
    {
      reinitBasis(waveFunctionSubspaceGuess,
                  lanczosGuess,
                  MLanczos,
                  MInvLanczos);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      reinitBasis(
        linearAlgebra::MultiVector<ValueTypeOperand, memorySpace>
          &waveFunctionSubspaceGuess,
        linearAlgebra::Vector<ValueTypeOperand, memorySpace> &lanczosGuess,
        const OpContext &                                     MLanczos,
        const OpContext &                                     MInvLanczos)
    {
      d_isSolved                  = false;
      d_isBoundKnown              = false;
      d_waveFunctionSubspaceGuess = &waveFunctionSubspaceGuess;
      d_lanczosGuess              = &lanczosGuess;
      d_MLanczos                  = &MLanczos;
      d_MInvLanczos               = &MInvLanczos;
      int rank;
      utils::mpi::MPICommRank(
        lanczosGuess.getMPIPatternP2P()->mpiCommunicator(), &rank);
      d_rootCout.setCondition(rank == 0);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      reinitBounds(double wantedSpectrumLowerBound,
                   double wantedSpectrumUpperBound)
    {
      d_isSolved                 = false;
      d_isBoundKnown             = true;
      d_wantedSpectrumLowerBound = wantedSpectrumLowerBound;
      d_wantedSpectrumUpperBound = wantedSpectrumUpperBound;
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
      d_isSolved                  = true;
      global_size_type globalSize = kohnShamWaveFunctions.globalSize();
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        linAlgOpContext = kohnShamWaveFunctions.getLinAlgOpContext();
      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
                                          memoryTransfer;
      linearAlgebra::EigenSolverError     returnValue;
      linearAlgebra::EigenSolverError     lanczosErr;
      linearAlgebra::EigenSolverError     chfsiErr;
      linearAlgebra::EigenSolverErrorCode err =
        linearAlgebra::EigenSolverErrorCode::OTHER_ERROR;
      linearAlgebra::NewtonRaphsonError nrErr;

      // get bounds from lanczos
      std::vector<double> tol{
        ksdft::LinearEigenSolverDefaults::LANCZOS_EXTREME_EIGENVAL_TOL,
        ksdft::LinearEigenSolverDefaults::LANCZOS_EXTREME_EIGENVAL_TOL};
      linearAlgebra::LanczosExtremeEigenSolver<ValueTypeOperator,
                                               ValueTypeOperand,
                                               memorySpace>
        lanczos(ksdft::LinearEigenSolverDefaults::LANCZOS_MAX_KRYLOV_SUBSPACE,
                1,
                1,
                tol,
                ksdft::LinearEigenSolverDefaults::LANCZOS_BETA_TOL,
                *d_lanczosGuess);

      linearAlgebra::MultiVector<ValueType, memorySpace> eigenVectorsLanczos;

      std::vector<RealType> eigenValuesLanczos(2);
      lanczosErr = lanczos.solve(kohnShamOperator,
                                 eigenValuesLanczos,
                                 eigenVectorsLanczos,
                                 false,
                                 *d_MLanczos,
                                 *d_MInvLanczos);

      std::vector<RealType> diagonal(0), subDiagonal(0);
      lanczos.getTridiagonalMatrix(diagonal, subDiagonal);
      RealType residual = subDiagonal[subDiagonal.size() - 1];

      size_type iPass = 0;
      if (lanczosErr.isSuccess ||
          lanczosErr.err ==
            linearAlgebra::EigenSolverErrorCode::LANCZOS_SUBSPACE_INSUFFICIENT)
        {
          //--------------------CHANGE THIS ------------------------------
          if (!d_isBoundKnown)
            {
              d_wantedSpectrumLowerBound = eigenValuesLanczos[0];
              d_wantedSpectrumUpperBound =
                (eigenValuesLanczos[1] - eigenValuesLanczos[0]) *
                  ((double)(d_numWantedEigenvalues * 200.0) / globalSize) +
                eigenValuesLanczos[0];
            }

          d_rootCout << "wantedSpectrumLowerBound: "
                     << d_wantedSpectrumLowerBound << std::endl;
          d_rootCout << "wantedSpectrumUpperBound: "
                     << d_wantedSpectrumUpperBound << "\n";
          d_rootCout << "unWantedSpectrumUpperBound: "
                     << eigenValuesLanczos[1] + residual << "\n";

          linearAlgebra::ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                                      ValueTypeOperand,
                                                      memorySpace>
            chfsi(d_wantedSpectrumLowerBound,
                  d_wantedSpectrumUpperBound,
                  eigenValuesLanczos[1] + residual,
                  d_chebyshevPolynomialDegree,
                  ksdft::LinearEigenSolverDefaults::ILL_COND_TOL,
                  *d_waveFunctionSubspaceGuess,
                  d_waveFunctionBlockSize);

          linearAlgebra::MultiVector<ValueType, memorySpace> HX(
            kohnShamWaveFunctions, (ValueType)0.0),
            MX(kohnShamWaveFunctions, (ValueType)0.0),
            residualEigenSolver(kohnShamWaveFunctions, (ValueType)0.0);

          linearAlgebra::Vector<ValueType, memorySpace>
            kohnShamEnergiesMemspace(d_numWantedEigenvalues,
                                     linAlgOpContext,
                                     (ValueType)0),
            nOnes(kohnShamEnergiesMemspace, (ValueType)-1.0);

          for (; iPass < d_maxChebyshevFilterPass; iPass++)
            {
              // do chebyshev filetered eigensolve

              chfsiErr = chfsi.solve(kohnShamOperator,
                                     kohnShamEnergies,
                                     kohnShamWaveFunctions,
                                     computeWaveFunctions,
                                     M,
                                     MInv);

              d_rootCout << "Chebyshev Filter Pass: [" << iPass << "] "
                         << chfsiErr.msg << std::endl;

              // for(auto &i : kohnShamEnergies)
              //   std::cout <<  i <<", ";
              // std::cout << "\n";

              // Calculate the chemical potential using newton raphson

              std::shared_ptr<ksdft::FractionalOccupancyFunction> fOcc =
                std::make_shared<ksdft::FractionalOccupancyFunction>(
                  kohnShamEnergies,
                  d_numElectrons,
                  Constants::BOLTZMANN_CONST_HARTREE,
                  d_smearingTemperature,
                  kohnShamEnergies
                    [std::ceil(static_cast<double>(d_numElectrons) / 2.0) - 1]);

              linearAlgebra::NewtonRaphsonSolver<double> nrs(
                NewtonRaphsonSolverDefaults::MAX_ITER,
                d_fermiEnergyTolerance,
                NewtonRaphsonSolverDefaults::FORCE_TOL);

              nrErr = nrs.solve(*fOcc);

              fOcc->getSolution(d_fermiEnergy);

              size_type numLevelsBelowFermiEnergy = 0;
              // Calculate the frac occupancy vector
              for (size_type i = 0; i < d_fracOccupancy.size(); i++)
                {
                  d_fracOccupancy[i] =
                    fermiDirac(kohnShamEnergies[i],
                               d_fermiEnergy,
                               Constants::BOLTZMANN_CONST_HARTREE,
                               d_smearingTemperature);
                  if (d_fracOccupancy[i] > d_fracOccupancyTolerance)
                    numLevelsBelowFermiEnergy += 1;
                }

              // TODO : Implement blocked approach for wavefns
              // calculate residualEigenSolver
              kohnShamOperator.apply(kohnShamWaveFunctions, HX);
              M.apply(kohnShamWaveFunctions, MX);

              memoryTransfer.copy(d_numWantedEigenvalues,
                                  kohnShamEnergiesMemspace.data(),
                                  kohnShamEnergies.data());

              linearAlgebra::blasLapack::
                axpbyBlocked<ValueType, ValueType, memorySpace>(
                  MX.locallyOwnedSize(),
                  d_numWantedEigenvalues,
                  nOnes.data(),
                  HX.data(),
                  kohnShamEnergiesMemspace.data(),
                  MX.data(),
                  residualEigenSolver.data(),
                  *linAlgOpContext);

              d_eigSolveResNorm = residualEigenSolver.l2Norms();
              size_type numLevelsBelowFermiEnergyResidualConverged = 0;
              for (size_type i = 0; i < d_numWantedEigenvalues; i++)
                {
                  if (d_fracOccupancy[i] > d_fracOccupancyTolerance &&
                      d_eigSolveResNorm[i] <= d_eigenSolveResidualTolerance)
                    numLevelsBelowFermiEnergyResidualConverged += 1;
                }

              d_rootCout << "*****************The CHFSI results are: "
                            "******************\n";
              d_rootCout << "Fermi Energy is : " << d_fermiEnergy << "\n";
              d_rootCout << "Fermi Energy residual is : " << nrs.getResidual()
                         << "\n";
              d_rootCout
                << "Kohn Sham Energy\t\tFractional Occupancy\t\tEigen Solve Residual Norm\n";
              for (size_type i = 0; i < d_numWantedEigenvalues; i++)
                d_rootCout << kohnShamEnergies[i] << "\t\t"
                           << d_fracOccupancy[i] << "\t\t"
                           << d_eigSolveResNorm[i] << "\n";
              d_rootCout << "\n";

              if (numLevelsBelowFermiEnergy ==
                    numLevelsBelowFermiEnergyResidualConverged ||
                  !chfsiErr.isSuccess || !nrErr.isSuccess)
                break;
              else
                {
                  d_waveFunctionSubspaceGuess = &kohnShamWaveFunctions;
                  d_wantedSpectrumLowerBound  = kohnShamEnergies[0];
                  d_wantedSpectrumUpperBound =
                    kohnShamEnergies[d_numWantedEigenvalues - 1];
                  chfsi.reinit(d_wantedSpectrumLowerBound,
                               d_wantedSpectrumUpperBound,
                               eigenValuesLanczos[1] + residual,
                               d_chebyshevPolynomialDegree,
                               ksdft::LinearEigenSolverDefaults::ILL_COND_TOL,
                               *d_waveFunctionSubspaceGuess,
                               d_waveFunctionBlockSize);
                }
            }
          if (!chfsiErr.isSuccess)
            {
              err = linearAlgebra::EigenSolverErrorCode::KS_CHFSI_ERROR;
              returnValue =
                linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
              returnValue.msg += chfsiErr.msg;
            }
          else if (!nrErr.isSuccess)
            {
              err =
                linearAlgebra::EigenSolverErrorCode::KS_NEWTON_RAPHSON_ERROR;
              returnValue =
                linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
              returnValue.msg += nrErr.msg;
            }
          else if (iPass >= d_maxChebyshevFilterPass && chfsiErr.isSuccess &&
                   nrErr.isSuccess)
            {
              err = linearAlgebra::EigenSolverErrorCode::KS_MAX_PASS_ERROR;
              returnValue =
                linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
            }
          else if (iPass < d_maxChebyshevFilterPass && chfsiErr.isSuccess &&
                   nrErr.isSuccess)
            {
              err = linearAlgebra::EigenSolverErrorCode::SUCCESS;
              returnValue =
                linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
              returnValue.msg += "Number of CHFSI passes required are " +
                                 std::to_string(iPass + 1) + ".";
            }
          else
            {
              returnValue =
                linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
            }
        }
      else
        {
          err = linearAlgebra::EigenSolverErrorCode::KS_LANCZOS_ERROR;
          returnValue =
            linearAlgebra::EigenSolverErrorMsg::isSuccessAndMsg(err);
          returnValue.msg += lanczosErr.msg;
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

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    std::vector<typename linearAlgebra::HermitianIterativeEigenSolver<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace>::RealType>
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      getEigenSolveResidualNorm()
    {
      utils::throwException(
        d_isSolved,
        "Cannot call getEigenSolveResidualNorm() before solving the eigenproblem.");
      return d_eigSolveResNorm;
    }

  } // namespace ksdft
} // end of namespace dftefe
