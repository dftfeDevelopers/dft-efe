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
    {
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
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    ChebyshevFilteredEigenSolver<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace>::solve(const OpContext &                    A,
                          std::vector<RealType> &              eigenValues,
                          MultiVector<ValueType, memorySpace> &eigenVectors,
                          bool             computeEigenVectors,
                          const OpContext &B,
                          const OpContext &BInv)
    {
      EigenSolverError        retunValue;
      EigenSolverErrorCode    err;
      OrthonormalizationError orthoerr;
      EigenSolverError        rrerr;

      MultiVector<ValueType, memorySpace> filteredSubspace(
        *d_eigenSubspaceGuess, (ValueType)0);

      // [CF] Chebyshev filtering of \psi

      ChebyshevFilter<ValueTypeOperator, ValueTypeOperand, memorySpace>(
        A,
        BInv,
        *d_eigenSubspaceGuess,
        d_polynomialDegree,
        d_wantedSpectrumLowerBound,
        d_wantedSpectrumUpperBound,
        d_unWantedSpectrumUpperBound,
        filteredSubspace);


      MultiVector<ValueType, memorySpace> filteredSubspaceOrtho(
        filteredSubspace, (ValueType)0);

      // B orthogonalization required of X -> X_O

      // orthoerr = OrthonormalizationFunctions<
      //   ValueTypeOperator,
      //   ValueTypeOperand,
      //   memorySpace>::CholeskyGramSchmidt(filteredSubspace,
      //                                     filteredSubspaceOrtho,
      //                                     B);

      orthoerr = linearAlgebra::OrthonormalizationFunctions<
        ValueType,
        ValueType,
        memorySpace>::MultipassLowdin(filteredSubspace,
                                      10,
                                      1e-12,
                                      1e-12,
                                      filteredSubspaceOrtho,
                                      B);

      // [RR] Perform the Rayleigh–Ritz procedure for filteredSubspace

      rrerr = d_rr->solve(A,
                          filteredSubspaceOrtho,
                          eigenValues,
                          eigenVectors,
                          computeEigenVectors);

      if (!orthoerr.isSuccess)
        {
          err        = EigenSolverErrorCode::CHFSI_ORTHONORMALIZATION_ERROR;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += orthoerr.msg;
        }
      if (!rrerr.isSuccess)
        {
          err        = EigenSolverErrorCode::CHFSI_RAYLEIGH_RITZ_ERROR;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
          retunValue.msg += rrerr.msg;
        }
      else
        {
          err        = EigenSolverErrorCode::SUCCESS;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
        }

      return retunValue;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
