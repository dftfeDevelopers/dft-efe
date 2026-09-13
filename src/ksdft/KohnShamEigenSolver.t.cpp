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
#include <linearAlgebra/MultiVectorOps.h>
#include <ksdft/FractionalOccupancyFunction.h>
#include <linearAlgebra/BisectionSolver.h>
#include <iomanip>
#include <cmath>
namespace dftefe
{
  namespace ksdft
  {
    namespace
    {
      size_type
      getChebyPolynomialDegree(size_type unWantedSpectrumUpperBound)
      {
        auto lower = LinearEigenSolverDefaults::CHEBY_ORDER_LOOKUP.lower_bound(
          unWantedSpectrumUpperBound);
        size_type val =
          lower != LinearEigenSolverDefaults::CHEBY_ORDER_LOOKUP.end() ?
            lower->second :
            1250;
        return val;
      }

      /**
       * Largest Chebyshev degree an all-electron spectrum can tolerate.
       *
       * The filter maps [a, b] = [wantedUpper, unWantedUpper] onto [-1, 1]
       * through x(lambda) = (lambda - c)/e, e = (b - a)/2, c = (b + a)/2
       * (see ChebyshevFilter.t.cpp), so the lowest wanted eigenvalue a0
       * lands at
       *
       *   |x_low| = (b + a - 2*a0)/(b - a) = 1 + 2*rho,
       *   rho     = (a - a0)/(b - a).
       *
       * Outside [-1, 1] the Chebyshev polynomial grows as
       * T_m(x) = cosh(m*acosh|x|), i.e. log10|T_m| ~ m*acosh|x|/ln(10).
       * The filtered subspace therefore spans that many decades between its
       * lowest and highest wanted state. Once that exceeds the ~16 decades
       * of a double, the upper states fall below round-off relative to the
       * lowest one, the subspace collapses in rank and Rayleigh-Ritz returns
       * garbage for everything above the deepest few states. So invert the
       * relation for m at CHEBY_FILTER_TARGET_DECADES.
       *
       * Only used for all-electron. A pseudopotential spectrum has a shallow
       * a0 (rho ~ 1e-4), which puts this bound in the thousands where it
       * never binds - hence CHEBY_ORDER_LOOKUP alone has always sufficed
       * there, and that path is left untouched.
       */
      size_type
      getChebyPolynomialDegreeAllElectron(double wantedSpectrumLowerBound,
                                          double wantedSpectrumUpperBound,
                                          double unWantedSpectrumUpperBound)
      {
        //
        // Number of decades of dynamic range the Chebyshev filter is allowed
        // to span across the wanted spectrum in an all-electron calculation.
        // Kept below the ~16 decades of a double so that the Gram-Schmidt
        // orthogonalization still has headroom to recover the upper states.
        //
        const double CHEBY_FILTER_TARGET_DECADES = 11.0;
        //
        // Floor for the all-electron Chebyshev degree, so a pathological
        // (or inverted) set of spectrum bounds still yields a usable filter.
        //
        const size_type CHEBY_ORDER_ALLELECTRON_MIN = 10;

        const double unWantedWidth =
          unWantedSpectrumUpperBound - wantedSpectrumUpperBound;
        const double wantedWidth =
          wantedSpectrumUpperBound - wantedSpectrumLowerBound;

        // Degenerate or inverted bounds: e <= 0 in the filter, so there is
        // no meaningful dynamic range to solve for. Fall back to the floor
        // rather than feeding acosh an argument below 1.
        if (unWantedWidth <= 0.0 || wantedWidth <= 0.0)
          return CHEBY_ORDER_ALLELECTRON_MIN;

        const double xLow = 1.0 + 2.0 * wantedWidth / unWantedWidth;
        const double degree =
          CHEBY_FILTER_TARGET_DECADES * std::log(10.0) / std::acosh(xLow);

        return std::max(CHEBY_ORDER_ALLELECTRON_MIN, (size_type)degree);
      }
    } // namespace

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
        const size_type maxChebyshevFilterPass,
        const size_type numWantedEigenvalues,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
          mpiPatternP2P,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                                   linAlgOpContext,
        const linearAlgebra::ElpaScalapackManager &elpaScala,
        bool                                       isResidualChebyshevFilter,
        const size_type                            waveFunctionBatchSize,
        const OpContext &                          MLanczos,
        const OpContext &                          MInvLanczos,
        const bool                                 isGHEP,
        linearAlgebra::OrthogonalizationType       orthoType,
        bool                                       storeIntermediateSubspaces,
        bool                                       useSameScratchInEigenSolver,
        SpinMode                                   spinMode,
        CalculationType                            calculationType)
      : d_spinMode(spinMode)
      , d_S((spinMode == SpinMode::Unpolarized) ? 1 : 2)
      , d_numWantedEigenvalues(numWantedEigenvalues)
      , d_eigenSolveResidualTolerance(eigenSolveResidualTolerance)
      , d_maxChebyshevFilterPass(maxChebyshevFilterPass)
      , d_waveFunctionBatchSize(spinMode == SpinMode::Collinear ?
                                  waveFunctionBatchSize / 2 :
                                  waveFunctionBatchSize)
      , d_fermiEnergyTolerance(fermiEnergyTolerance)
      , d_fracOccupancyTolerance(fracOccupancyTolerance)
      , d_smearingTemperature(smearingTemperature)
      , d_fracOccupancy(d_S * d_numWantedEigenvalues)
      , d_eigSolveResNorm(d_S * d_numWantedEigenvalues)
      , d_numElectrons(numElectrons)
      , d_rootCout(std::cout)
      , d_batchSizeSmall(0)
      , d_p(mpiPatternP2P->mpiCommunicator(), "Kohn Sham EigenSolver")
      , d_chebyPolyScalingFactor(1.0)
      , d_isResidualChebyFilter(isResidualChebyshevFilter)
      , d_setChebyPolDegExternally(false)
      , d_isChebyPolDegComputed(false)
      , d_calculationType(calculationType)
      , d_storeIntermediateSubspaces(storeIntermediateSubspaces)
      , d_filteredSubspace(nullptr)
      , d_filteredSubspaceOrtho((nullptr))
      , d_orthoType(orthoType)
      , d_elpaScala(&elpaScala)
      , d_isGHEP(isGHEP)
      , d_useSameScratch(useSameScratchInEigenSolver)
      , d_scratch(nullptr)
      , d_pTotal(mpiPatternP2P->mpiCommunicator(),
                 "Kohn Sham EigenSolver Solve Time")
    {
      reinitBasis(mpiPatternP2P, linAlgOpContext, MLanczos, MInvLanczos);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      reinitBasis(std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                    mpiPatternP2P,
                  std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
                  const OpContext &MLanczos,
                  const OpContext &MInvLanczos)
    {
      d_isSolved     = false;
      d_isBoundKnown = false;
      // A new basis means a new spectrum, so any all-electron degree latched
      // against the old bounds is stale and has to be derived again.
      d_isChebyPolDegComputed = false;
      d_mpiPatternP2P         = mpiPatternP2P;
      d_linAlgOpContext       = linAlgOpContext;
      d_MLanczos              = &MLanczos;
      d_MInvLanczos           = &MInvLanczos;
      int rank;
      utils::mpi::MPICommRank(mpiPatternP2P->mpiCommunicator(), &rank);
      d_rootCout.setCondition(rank == 0);

      const size_type eigenVecBatchSize = d_S * d_waveFunctionBatchSize;

      d_waveFnBatch =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          mpiPatternP2P, linAlgOpContext, eigenVecBatchSize, ValueType());
      d_HXBatch =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          mpiPatternP2P, linAlgOpContext, eigenVecBatchSize, ValueType());

      if (d_useSameScratch)
        d_scratch = std::make_shared<
          linearAlgebra::MultivectorScratch<ValueType, memorySpace>>(
          d_waveFnBatch, d_HXBatch);
      else
        d_scratch = nullptr;

      d_MXBatch =
        std::make_shared<linearAlgebra::MultiVector<ValueType, memorySpace>>(
          mpiPatternP2P, linAlgOpContext, eigenVecBatchSize, ValueType());

      d_kohnShamEnergiesMemspace = utils::MemoryStorage<ValueType, memorySpace>(
        d_S * d_numWantedEigenvalues, (ValueType)0),
      d_nOnes = utils::MemoryStorage<ValueType, memorySpace>(
        d_S * d_numWantedEigenvalues, (ValueType)-1.0);

      d_chfsi = std::make_shared<
        linearAlgebra::ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                                    ValueTypeOperand,
                                                    memorySpace>>(
        0,
        0,
        0,
        0,
        ksdft::LinearEigenSolverDefaults::ILL_COND_TOL,
        mpiPatternP2P,
        linAlgOpContext,
        *d_elpaScala,
        d_isResidualChebyFilter,
        eigenVecBatchSize,
        d_isGHEP,
        d_orthoType,
        d_storeIntermediateSubspaces,
        d_scratch);
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
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      setChebyPolyScalingFactor(double scalingFactor)
    {
      d_chebyPolyScalingFactor = scalingFactor;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      setChebyshevPolynomialDegree(size_type chebyPolyDeg)
    {
      d_setChebyPolDegExternally  = true;
      d_chebyshevPolynomialDegree = chebyPolyDeg;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      setResidualChebyshevFilterFlag(bool flag)
    {
      d_isResidualChebyFilter = flag;
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
      // d_fracOccupancy memory layout: spin-major, size S*N (= d_S *
      // d_numWantedEigenvalues).
      //   d_fracOccupancy[ s*N + n ]  — fractional occupancy for spin s,
      //   orbital n. Mirrors kohnShamEnergies layout exactly (filled via
      //   fermiDirac(kohnShamEnergies[i])).
      //
      // d_eigSolveResNorm memory layout: spin-major, size S*N (= d_S *
      // d_numWantedEigenvalues).
      //   d_eigSolveResNorm[ s*N + n ]  — ||H*psi_{s,n} - E_{s,n}*M*psi_{s,n}||
      //   / ||psi_{s,n}||. Scattered by getLinearEigenSolveResidual:
      //   residualVec[s*numVecPerSpace + n].
      d_p.reset();
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
                d_mpiPatternP2P,
                d_linAlgOpContext,
                false,
                d_S);

      linearAlgebra::MultiVector<ValueType, memorySpace> eigenVectorsLanczos;

      std::vector<RealType> eigenValuesLanczos(2);
      d_p.registerStart("Lanczos Solve");
      d_pTotal.registerStart("Lanczos Solve");
      lanczosErr = lanczos.solve(kohnShamOperator,
                                 eigenValuesLanczos,
                                 eigenVectorsLanczos,
                                 false,
                                 *d_MLanczos,
                                 *d_MInvLanczos);
      d_p.registerEnd("Lanczos Solve");
      d_pTotal.registerEnd("Lanczos Solve");

      std::vector<RealType> diagonal(0), subDiagonal(0);
      lanczos.getTridiagonalMatrix(diagonal, subDiagonal);
      RealType residual = subDiagonal[subDiagonal.size() - 1];
      residual =
        residual / 10; // Done in dftfe because the subspace size is 20. TODO.

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
                (eigenValuesLanczos[1] + residual - eigenValuesLanczos[0]) *
                  ((double)(d_numWantedEigenvalues * 200.0) / globalSize) +
                eigenValuesLanczos[0];
              if (d_wantedSpectrumUpperBound >=
                  eigenValuesLanczos[1] + residual)
                {
                  d_wantedSpectrumUpperBound =
                    (eigenValuesLanczos[1] + residual + eigenValuesLanczos[0]) *
                    0.5;
                }
            }

          d_rootCout << "wantedSpectrumLowerBound: "
                     << d_wantedSpectrumLowerBound << std::endl;
          d_rootCout << "wantedSpectrumUpperBound: "
                     << d_wantedSpectrumUpperBound << "\n";
          d_rootCout << "unWantedSpectrumUpperBound: "
                     << eigenValuesLanczos[1] + residual << "\n";

          // All-electron derives the degree from the spectrum bounds, which
          // are set by the deep core states and so barely move once the run
          // is under way. It is therefore computed once and kept for every
          // later SCF. Pseudopotential keeps recomputing from the lookup
          // each SCF, as before.
          if (!d_setChebyPolDegExternally &&
              !(d_calculationType == CalculationType::AE &&
                d_isChebyPolDegComputed))
            {
              d_chebyshevPolynomialDegree =
                d_calculationType == CalculationType::AE ?
                  getChebyPolynomialDegreeAllElectron(
                    d_wantedSpectrumLowerBound,
                    d_wantedSpectrumUpperBound,
                    eigenValuesLanczos[1] + residual) :
                  getChebyPolynomialDegree(eigenValuesLanczos[1] + residual);

              d_chebyshevPolynomialDegree =
                d_chebyshevPolynomialDegree * d_chebyPolyScalingFactor;

              // Latch only once the bounds are the ones the run actually
              // uses. On the very first SCF they still come from the initial
              // guess above, which is a far narrower window than the
              // Ritz-based one reinitBounds() supplies from the next SCF on,
              // so a degree latched there would be calibrated for a window
              // that is never seen again.
              d_isChebyPolDegComputed = d_isBoundKnown;
            }

          d_rootCout << "Chebyshev Polynomial Degree : "
                     << d_chebyshevPolynomialDegree << "\n";

          d_p.registerStart("Reinit CHFSI");
          d_pTotal.registerStart("Reinit CHFSI");
          d_chfsi->reinit(d_wantedSpectrumLowerBound,
                          d_wantedSpectrumUpperBound,
                          eigenValuesLanczos[1] + residual,
                          d_chebyshevPolynomialDegree,
                          ksdft::LinearEigenSolverDefaults::ILL_COND_TOL,
                          kohnShamWaveFunctions.getMPIPatternP2P(),
                          kohnShamWaveFunctions.getLinAlgOpContext());
          d_pTotal.registerStart("Reinit CHFSI");
          d_p.registerEnd("Reinit CHFSI");

          for (; iPass < d_maxChebyshevFilterPass; iPass++)
            {
              // do chebyshev filetered eigensolve

              d_p.registerStart("Solve CHFSI");
              d_pTotal.registerStart("Solve CHFSI");
              chfsiErr = d_chfsi->solve(kohnShamOperator,
                                        kohnShamEnergies,
                                        kohnShamWaveFunctions,
                                        computeWaveFunctions,
                                        M,
                                        MInv);

              kohnShamWaveFunctions.updateGhostValues();
              d_pTotal.registerEnd("Solve CHFSI");
              d_p.registerEnd("Solve CHFSI");

              /*
              // Compute projected hamiltonian = Y^H M Y

              size_type                           numVec  =
          kohnShamWaveFunctions.getNumberComponents(); size_type vecSize =
          kohnShamWaveFunctions.locallyOwnedSize();
              linearAlgebra::MultiVector<ValueType, memorySpace>
          temp(kohnShamWaveFunctions, (ValueType)0);

              utils::MemoryStorage<ValueType, memorySpace> temp1(
                numVec * numVec, utils::Types<ValueType>::zero);

              M.apply(kohnShamWaveFunctions, temp, true, true);

              linearAlgebra::blasLapack::gemm<ValueType, ValueType,
          memorySpace>('N',
                'C',
                numVec,
                numVec,
                vecSize,
                (ValueType)1,
                temp.data(),
                numVec,
                kohnShamWaveFunctions.data(),
                numVec,
                (ValueType)0,
                temp1.data(),
                numVec,
                *kohnShamWaveFunctions.getLinAlgOpContext());

              int mpierr = utils::mpi::MPIAllreduce<memorySpace>(
                utils::mpi::MPIInPlace,
                temp1.data(),
                temp1.size(),
                utils::mpi::Types<ValueType>::getMPIDatatype(),
                utils::mpi::MPISum,
                kohnShamWaveFunctions.getMPIPatternP2P()->mpiCommunicator());

          d_rootCout<< "Y^TMY:" <<std::endl;
            for(size_type i= 0 ; i < numVec ; i++)
            {
              d_rootCout << "[";
              for(size_type j= 0 ; j < numVec ; j++)
                d_rootCout << *(temp1.data() + numVec * i + j) << ",";
              d_rootCout<< "]" << std::endl;
            }
            */

              if (d_storeIntermediateSubspaces)
                {
                  d_filteredSubspace = &d_chfsi->getFilteredSubspace();
                  d_filteredSubspaceOrtho =
                    &d_chfsi->getOrthogonalizedFilteredSubspace();
                }

              d_rootCout << "Chebyshev Filter Pass: [" << iPass << "] "
                         << chfsiErr.msg << std::endl;

              d_p.registerStart("Compute chemical potential");
              d_pTotal.registerStart("Compute chemical potential");
              // Calculate the chemical potential: bisect within the
              // bracket of all filtered eigenvalues first (see
              // BisectionSolver - immune to sharply-peaked/vanishing
              // derivatives since it only ever evaluates getValue()), then
              // hand off to Newton

              const size_type initialGuessIdx =
                (d_spinMode == SpinMode::Collinear) ?
                  std::ceil(static_cast<double>(d_numElectrons) / 2.0) - 1 :
                  std::ceil(static_cast<double>(d_numElectrons * d_S) / 2.0) -
                    1;

              std::shared_ptr<ksdft::FractionalOccupancyFunction> fOcc =
                std::make_shared<ksdft::FractionalOccupancyFunction>(
                  kohnShamEnergies,
                  d_numElectrons * d_S,
                  Constants::BOLTZMANN_CONST_HARTREE,
                  d_smearingTemperature,
                  kohnShamEnergies[initialGuessIdx]);

              linearAlgebra::BisectionSolver<double> bisectionSolver(
                BisectionSolverDefaults::MAX_ITER,
                BisectionSolverDefaults::TOL);

              linearAlgebra::BisectionError bisectionErr =
                bisectionSolver.solve(*fOcc);

              utils::throwException(
                bisectionErr.isSuccess,
                "KohnShamEigenSolver: bisection pre-pass for the Fermi "
                "energy failed - " +
                  bisectionErr.msg +
                  " This should not happen for a monotonic occupancy "
                  "function; check for NaN/Inf eigenvalues.");

              linearAlgebra::NewtonRaphsonSolver<double> nrs(
                NewtonRaphsonSolverDefaults::MAX_ITER,
                d_fermiEnergyTolerance,
                NewtonRaphsonSolverDefaults::FORCE_TOL);

              nrErr = nrs.solve(*fOcc);

              if (!nrErr.isSuccess &&
                  nrErr.err ==
                    linearAlgebra::NewtonRaphsonErrorCode::FORCE_TOLERANCE_ERR)
                {
                  d_rootCout
                    << "KohnShamEigenSolver: Newton-Raphson found the "
                       "derivative already vanished at the bisected Fermi "
                       "energy estimate (likely a HOMO-LUMO gap); keeping "
                       "the bisected value. "
                    << nrErr.msg << std::endl;

                  // The bisected estimate is being deliberately accepted
                  // here, so this is not a genuine failure - reset nrErr
                  // to SUCCESS so it does not spuriously trip the CHFSI
                  // break condition or get reported as a
                  // KS_NEWTON_RAPHSON_ERROR below.
                  nrErr = linearAlgebra::NewtonRaphsonErrorMsg::isSuccessAndMsg(
                    linearAlgebra::NewtonRaphsonErrorCode::SUCCESS);
                }
              else
                {
                  utils::throwException(
                    nrErr.isSuccess,
                    "KohnShamEigenSolver: Newton-Raphson polish for the "
                    "Fermi energy failed - " +
                      nrErr.msg +
                      " This should not happen starting from a "
                      "bisection-refined estimate; check for NaN/Inf "
                      "eigenvalues.");
                }

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

              // // TODO : Implement blocked approach for wavefns
              // // calculate residualEigenSolver
              const size_type       numVecPerSpace   = d_numWantedEigenvalues;
              const size_type       eigVecBatchPerSp = d_waveFunctionBatchSize;
              std::vector<RealType> energiesBatchMajor(d_S *
                                                       d_numWantedEigenvalues);
              size_type             dstOffset = 0;
              for (size_type psiStart = 0; psiStart < numVecPerSpace;
                   psiStart += eigVecBatchPerSp)
                {
                  const size_type batchN =
                    std::min(eigVecBatchPerSp, numVecPerSpace - psiStart);
                  for (size_type s = 0; s < d_S; ++s)
                    {
                      std::copy(kohnShamEnergies.begin() + s * numVecPerSpace +
                                  psiStart,
                                kohnShamEnergies.begin() + s * numVecPerSpace +
                                  psiStart + batchN,
                                energiesBatchMajor.begin() + dstOffset);
                      dstOffset += batchN;
                    }
                }
              memoryTransfer.copy(d_S * d_numWantedEigenvalues,
                                  d_kohnShamEnergiesMemspace.data(),
                                  energiesBatchMajor.data());

              d_p.registerEnd("Compute chemical potential");
              d_pTotal.registerEnd("Compute chemical potential");
              d_p.registerStart("Compute Residuals");
              d_pTotal.registerStart("Compute Residuals");

              size_type numLevelsBelowFermiEnergyResidualConverged = 0;
              if (computeWaveFunctions)
                {
                  d_eigSolveResNorm =
                    getLinearEigenSolveResidual(kohnShamOperator,
                                                kohnShamWaveFunctions,
                                                M);

                  for (size_type i = 0; i < d_S * d_numWantedEigenvalues; i++)
                    {
                      if (d_fracOccupancy[i] > d_fracOccupancyTolerance &&
                          d_eigSolveResNorm[i] <= d_eigenSolveResidualTolerance)
                        numLevelsBelowFermiEnergyResidualConverged += 1;
                    }

                  d_rootCout << "*****************The CHFSI results are: "
                                "******************\n";
                  d_rootCout << "Fermi Energy is : " << d_fermiEnergy << "\n";
                  d_rootCout
                    << "Fermi Energy residual is : " << nrs.getResidual()
                    << "\n";
                  {
                    std::ostream &          os       = d_rootCout.getOStream();
                    std::ios_base::fmtflags oldFlag  = os.flags();
                    std::streamsize         oldPrec  = os.precision();
                    std::streamsize         oldWidth = os.width();
                    os << std::scientific << std::right;
                    if (d_spinMode == SpinMode::Collinear)
                      {
                        d_rootCout << std::setw(6) << "No." << std::setw(24)
                                   << "[Spin 0] KS Energy" << std::setw(24)
                                   << "[Spin 1] KS Energy" << std::setw(22)
                                   << "[Spin 0] Frac. Occ." << std::setw(22)
                                   << "[Spin 1] Frac. Occ." << std::setw(22)
                                   << "[Spin 0] Residual" << std::setw(22)
                                   << "[Spin 1] Residual"
                                   << "\n";
                        for (size_type i = 0; i < d_numWantedEigenvalues; i++)
                          d_rootCout
                            << std::setw(6) << i << std::setw(24)
                            << std::setprecision(10) << kohnShamEnergies[i]
                            << std::setw(24) << std::setprecision(10)
                            << kohnShamEnergies[d_numWantedEigenvalues + i]
                            << std::setw(22) << std::setprecision(8)
                            << d_fracOccupancy[i] << std::setw(22)
                            << std::setprecision(8)
                            << d_fracOccupancy[d_numWantedEigenvalues + i]
                            << std::setw(22) << std::setprecision(8)
                            << d_eigSolveResNorm[i] << std::setw(22)
                            << std::setprecision(8)
                            << d_eigSolveResNorm[d_numWantedEigenvalues + i]
                            << "\n";
                      }
                    else
                      {
                        d_rootCout << std::setw(6) << "No." << std::setw(24)
                                   << "Kohn Sham Energy" << std::setw(22)
                                   << "Frac. Occupancy" << std::setw(22)
                                   << "Residual Norm"
                                   << "\n";
                        for (size_type i = 0; i < d_S * d_numWantedEigenvalues;
                             i++)
                          d_rootCout
                            << std::setw(6) << i + 1 << std::setw(24)
                            << std::setprecision(10) << kohnShamEnergies[i]
                            << std::setw(22) << std::setprecision(8)
                            << d_fracOccupancy[i] << std::setw(22)
                            << std::setprecision(8) << d_eigSolveResNorm[i]
                            << "\n";
                      }
                    os.flags(oldFlag);
                    os.precision(oldPrec);
                    os.width(oldWidth);
                  }
                  d_rootCout << "\n";
                }
              else
                {
                  d_rootCout
                    << "Not Computing EigenVectors. Linear Eigensolve break condition only satisfied by Max Cheby Filter Pass.";
                }
              d_p.registerEnd("Compute Residuals");
              d_pTotal.registerEnd("Compute Residuals");

              // *d_waveFunctionSubspaceGuess = kohnShamWaveFunctions;

              if (numLevelsBelowFermiEnergy ==
                    numLevelsBelowFermiEnergyResidualConverged ||
                  !chfsiErr.isSuccess || !nrErr.isSuccess)
                break;
              else
                {
                  d_wantedSpectrumLowerBound = kohnShamEnergies[0];
                  d_wantedSpectrumUpperBound =
                    kohnShamEnergies[d_S * d_numWantedEigenvalues - 1];
                  d_pTotal.registerStart("Reinit CHFSI");
                  d_p.registerStart("Reinit CHFSI");
                  d_chfsi->reinit(
                    d_wantedSpectrumLowerBound,
                    d_wantedSpectrumUpperBound,
                    eigenValuesLanczos[1] + residual,
                    d_chebyshevPolynomialDegree,
                    ksdft::LinearEigenSolverDefaults::ILL_COND_TOL,
                    kohnShamWaveFunctions.getMPIPatternP2P(),
                    kohnShamWaveFunctions.getLinAlgOpContext());
                  d_p.registerEnd("Reinit CHFSI");
                  d_pTotal.registerEnd("Reinit CHFSI");
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

      d_chebyPolyScalingFactor = 1.0;

      d_p.print();
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
    std::vector<double>
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      getLinearEigenSolveResidual(
        const OpContext &kohnShamOperator,
        const linearAlgebra::MultiVector<ValueType, memorySpace>
          &              kohnShamWaveFunctions,
        const OpContext &M)
    {
      std::shared_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>>
        HXBatch = nullptr,
        MXBatch = nullptr, XBatch = nullptr;

      size_type numEigenVectors = kohnShamWaveFunctions.getNumberComponents();
      std::vector<double> residualVec(numEigenVectors, 0);
      size_type           eigenVecLocalSize = kohnShamWaveFunctions.localSize();
      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

      const size_type eigenVecBatchSize = d_S * d_waveFunctionBatchSize;

      auto *Xps = static_cast<
        const linearAlgebra::MultiVectorProductSpace<ValueType, memorySpace> *>(
        &kohnShamWaveFunctions);
      const size_type numSpaces        = Xps->numSpaces();
      const size_type numVecPerSpace   = Xps->numVectorsPerSpace();
      const size_type eigVecBatchPerSp = eigenVecBatchSize / numSpaces;

      if (d_scratch)
        d_scratch->acquire();

      for (size_type waveFnStartId = 0; waveFnStartId < numVecPerSpace;
           waveFnStartId += eigVecBatchPerSp)
        {
          const size_type numEigVecInBatch =
            std::min(waveFnStartId + eigVecBatchPerSp, numVecPerSpace) -
            waveFnStartId;
          const size_type numEigVecInBatchTotal = numSpaces * numEigVecInBatch;

          if (numEigVecInBatch == eigVecBatchPerSp)
            {
              linearAlgebra::MultiVectorOps::copyToBatch(
                *Xps,
                waveFnStartId,
                numEigVecInBatch,
                *d_waveFnBatch,
                *kohnShamWaveFunctions.getLinAlgOpContext());

              XBatch  = d_waveFnBatch;
              HXBatch = d_HXBatch;
              MXBatch = d_MXBatch;
            }
          else if (numEigVecInBatch == d_batchSizeSmall)
            {
              linearAlgebra::MultiVectorOps::copyToBatch(
                *Xps,
                waveFnStartId,
                numEigVecInBatch,
                *d_waveFnBatchSmall,
                *kohnShamWaveFunctions.getLinAlgOpContext());

              XBatch  = d_waveFnBatchSmall;
              HXBatch = d_HXBatchSmall;
              MXBatch = d_MXBatchSmall;
            }
          else
            {
              d_batchSizeSmall = numEigVecInBatch;

              const bool useSmallScratch =
                d_scratch != nullptr && d_scratch->hasXinBatchSmall() &&
                d_scratch->hasXoutBatchSmall() &&
                d_scratch->getXinBatchSmallSize() == numEigVecInBatchTotal;

              if (useSmallScratch)
                {
                  d_waveFnBatchSmall = d_scratch->getXinBatchSmall();
                  d_HXBatchSmall     = d_scratch->getXoutBatchSmall();
                }
              else
                {
                  d_waveFnBatchSmall = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    kohnShamWaveFunctions.getMPIPatternP2P(),
                    kohnShamWaveFunctions.getLinAlgOpContext(),
                    numEigVecInBatchTotal,
                    ValueType());

                  d_HXBatchSmall = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    kohnShamWaveFunctions.getMPIPatternP2P(),
                    kohnShamWaveFunctions.getLinAlgOpContext(),
                    numEigVecInBatchTotal,
                    ValueType());
                  if (d_scratch != nullptr)
                    {
                      d_scratch->setXinBatchSmall(d_waveFnBatchSmall);
                      d_scratch->setXoutBatchSmall(d_HXBatchSmall);
                    }
                }

              d_MXBatchSmall = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                kohnShamWaveFunctions.getMPIPatternP2P(),
                kohnShamWaveFunctions.getLinAlgOpContext(),
                numEigVecInBatchTotal,
                ValueType());

              linearAlgebra::MultiVectorOps::copyToBatch(
                *Xps,
                waveFnStartId,
                numEigVecInBatch,
                *d_waveFnBatchSmall,
                *kohnShamWaveFunctions.getLinAlgOpContext());

              XBatch  = d_waveFnBatchSmall;
              HXBatch = d_HXBatchSmall;
              MXBatch = d_MXBatchSmall;
            }

          kohnShamOperator.apply(*XBatch, *HXBatch, true, true);

          M.apply(*XBatch, *MXBatch, true, true);

          linearAlgebra::blasLapack::
            axpbyBlocked<ValueType, ValueType, memorySpace>(
              eigenVecLocalSize,
              numEigVecInBatchTotal,
              1,
              d_nOnes.data(),
              HXBatch->data(),
              1,
              d_kohnShamEnergiesMemspace.data() + waveFnStartId * numSpaces,
              MXBatch->data(),
              XBatch->data(),
              *kohnShamWaveFunctions.getLinAlgOpContext());

          std::vector<double> normVec = XBatch->l2Norms();

          for (size_type s = 0; s < numSpaces; ++s)
            std::copy(normVec.begin() + s * numEigVecInBatch,
                      normVec.begin() + (s + 1) * numEigVecInBatch,
                      residualVec.begin() + s * numVecPerSpace + waveFnStartId);
        }

      if (d_scratch)
        d_scratch->release();

      return residualVec;
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

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    linearAlgebra::MultiVector<
      typename linearAlgebra::HermitianIterativeEigenSolver<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace>::ValueType,
      memorySpace> &
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      getFilteredSubspace()
    {
      utils::throwException(
        d_isSolved,
        "Cannot call getEigenSolveResidualNorm() before solving the eigenproblem.");
      return *d_filteredSubspace;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    linearAlgebra::MultiVector<
      typename linearAlgebra::HermitianIterativeEigenSolver<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace>::ValueType,
      memorySpace> &
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      getOrthogonalizedFilteredSubspace()
    {
      utils::throwException(
        d_isSolved,
        "Cannot call getEigenSolveResidualNorm() before solving the eigenproblem.");
      return *d_filteredSubspaceOrtho;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      printTotalInScopeTimings()
    {
      d_chfsi->printTotalInScopeTimings();
      d_pTotal.print();
    }

  } // namespace ksdft
} // end of namespace dftefe
