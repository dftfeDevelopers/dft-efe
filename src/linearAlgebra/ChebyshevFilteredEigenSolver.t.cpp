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

#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/DataTypeOverloads.h>
#include <linearAlgebra/Defaults.h>
#include <utils/ConditionalOStream.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::
      ChebyshevFilteredEigenSolver(
        const double                                wantedSpectrumLowerBound,
        const double                                wantedSpectrumUpperBound,
        const double                                unWantedSpectrumUpperBound,
        const double                                polynomialDegree,
        const double                                illConditionTolerance,
        MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess,
        const size_type                             eigenVectorBlockSize)
      : d_p(eigenSubspaceGuess.getMPIPatternP2P()->mpiCommunicator(), "CHFSI")
    {
      d_filteredSubspaceOrtho =
        std::make_shared<MultiVector<ValueType, memorySpace>>(
          eigenSubspaceGuess, (ValueType)0);
      d_filteredSubspace =
        std::make_shared<MultiVector<ValueType, memorySpace>>(
          eigenSubspaceGuess, (ValueType)0);

      reinit(wantedSpectrumLowerBound,
             wantedSpectrumUpperBound,
             unWantedSpectrumUpperBound,
             polynomialDegree,
             illConditionTolerance,
             eigenSubspaceGuess,
             eigenVectorBlockSize);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::
      reinit(const double wantedSpectrumLowerBound,
             const double wantedSpectrumUpperBound,
             const double unWantedSpectrumUpperBound,
             const double polynomialDegree,
             const double illConditionTolerance,
             MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess,
             const size_type                             eigenVectorBlockSize)
    {
      d_eigenSubspaceGuess         = &eigenSubspaceGuess;
      d_polynomialDegree           = polynomialDegree;
      d_wantedSpectrumLowerBound   = wantedSpectrumLowerBound;
      d_wantedSpectrumUpperBound   = wantedSpectrumUpperBound;
      d_unWantedSpectrumUpperBound = unWantedSpectrumUpperBound;

      d_rr = std::make_shared<
        RayleighRitzEigenSolver<ValueTypeOperator, ValueType, memorySpace>>();

      /*create new filtered subspace vec after calling isCompatible*/

      if (!d_filteredSubspace->isCompatible(eigenSubspaceGuess))
        {
          d_filteredSubspaceOrtho =
            std::make_shared<MultiVector<ValueType, memorySpace>>(
              eigenSubspaceGuess, (ValueType)0);
          d_filteredSubspace =
            std::make_shared<MultiVector<ValueType, memorySpace>>(
              eigenSubspaceGuess, (ValueType)0);
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::
      solve(const OpContext &                    A,
            std::vector<RealType> &              eigenValues,
            MultiVector<ValueType, memorySpace> &eigenVectors, /*in/out*/
            bool                                 computeEigenVectors,
            const OpContext &                    B,
            const OpContext &                    BInv)
    {
      d_p.reset();
      EigenSolverError        retunValue;
      EigenSolverErrorCode    err;
      OrthonormalizationError orthoerr;
      EigenSolverError        rrerr;

      // MultiVector<ValueType, memorySpace> filteredSubspace(
      //   *d_eigenSubspaceGuess, (ValueType)0);

      // [CF] Chebyshev filtering of \psi

      int rank;
      utils::mpi::MPICommRank(
        d_eigenSubspaceGuess->getMPIPatternP2P()->mpiCommunicator(), &rank);
      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      rootCout << "d_eigenSubspaceGuess l2norms CHFSI: ";
      for (auto &i : d_eigenSubspaceGuess->l2Norms())
        rootCout << i << "\t";
      rootCout << "\n";

      d_p.registerStart("Chebyshev Filter");
      ChebyshevFilter<ValueTypeOperator, ValueTypeOperand, memorySpace>(
        A,
        BInv,
        *d_eigenSubspaceGuess, /*scratch1*/
        d_polynomialDegree,
        d_wantedSpectrumLowerBound,
        d_wantedSpectrumUpperBound,
        d_unWantedSpectrumUpperBound,
        *d_filteredSubspace); /*scratch2*/
      d_p.registerEnd("Chebyshev Filter");

      rootCout << "d_filteredSubspace l2norms CHFSI: ";
      for (auto &i : d_filteredSubspace->l2Norms())
        rootCout << i << "\t";
      rootCout << "\n";

      // MultiVector<ValueType, memorySpace> filteredSubspaceOrtho(
      //   *d_filteredSubspace, (ValueType)0);

      // B orthogonalization required of X -> X_O

      // orthoerr = OrthonormalizationFunctions<
      //   ValueTypeOperator,
      //   ValueTypeOperand,
      //   memorySpace>::CholeskyGramSchmidt(*d_filteredSubspace,
      //                                     *d_filteredSubspaceOrtho,
      //                                     B);
      /*scratch2->eigenvector*/

      d_p.registerStart("OrthoNormalization");
      orthoerr = linearAlgebra::
        OrthonormalizationFunctions<ValueType, ValueType, memorySpace>::
          MultipassLowdin(*d_filteredSubspace, /*in/out, eigenvector*/
                          linearAlgebra::MultiPassLowdinDefaults::MAX_PASS,
                          linearAlgebra::MultiPassLowdinDefaults::SHIFT_TOL,
                          linearAlgebra::MultiPassLowdinDefaults::IDENTITY_TOL,
                          *d_filteredSubspaceOrtho, // go away
                          B);
      d_p.registerEnd("OrthoNormalization");

      // orthoerr = linearAlgebra::OrthonormalizationFunctions<
      //   ValueType,
      //   ValueType,
      //   memorySpace>::ModifiedGramSchmidt(*d_filteredSubspace,
      //                                     *d_filteredSubspaceOrtho,
      //                                     B);

      rootCout << "d_filteredSubspaceOrtho l2norms CHFSI: ";
      for (auto &i : d_filteredSubspaceOrtho->l2Norms())
        rootCout << i << "\t";
      rootCout << "\n";

      // [RR] Perform the Rayleigh–Ritz procedure for *d_filteredSubspace

      d_p.registerStart("RR Step");
      rrerr = d_rr->solve(A,
                          *d_filteredSubspaceOrtho, // go away
                          eigenValues,
                          eigenVectors, /*in/out*/
                          computeEigenVectors);
      d_p.registerEnd("RR Step");
      d_p.print();

      rootCout << "eigenVectors l2norms CHFSI: ";
      for (auto &i : eigenVectors.l2Norms())
        rootCout << i << "\t";
      rootCout << "\n";

      /* Using GHEP with B = Identity in orthogonalization
       * does not work. Prob due to no distribute C2P
       * in IdenstiyOperator. */
      // rrerr = d_rr->solve(A,
      //                     B,
      //                     *d_filteredSubspaceOrtho,//go away
      //                     eigenValues,
      //                     eigenVectors,/*in/out*/
      //                     computeEigenVectors);

      if (!orthoerr.isSuccess)
        {
          err        = EigenSolverErrorCode::CHFSI_ORTHONORMALIZATION_ERROR;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += orthoerr.msg;
        }
      else if (!rrerr.isSuccess)
        {
          err        = EigenSolverErrorCode::CHFSI_RAYLEIGH_RITZ_ERROR;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += rrerr.msg;
        }
      else
        {
          err        = EigenSolverErrorCode::SUCCESS;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += orthoerr.msg;
        }

      return retunValue;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    MultiVector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                memorySpace> &
    ChebyshevFilteredEigenSolver<ValueTypeOperator,
                                 ValueTypeOperand,
                                 memorySpace>::getFilteredSubspace()
    {
      return *d_filteredSubspace;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    MultiVector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                memorySpace> &
    ChebyshevFilteredEigenSolver<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace>::getOrthogonalizedFilteredSubspace()
    {
      return *d_filteredSubspaceOrtho;
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
