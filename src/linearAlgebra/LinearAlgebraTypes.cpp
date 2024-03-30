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
 * @author Bikash Kanungo, Avirup Sircar
 */

#include <linearAlgebra/LinearAlgebraTypes.h>
#include <utils/Exceptions.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    const std::map<LapackErrorCode, std::string> LapackErrorMsg::d_errToMsgMap = {
      {LapackErrorCode::SUCCESS, "Success"},
      {LapackErrorCode::FAILED_DENSE_MATRIX_INVERSE,
       "Dense matrix inversion failed with either lapack::getrf or lapack::getri with error codes "},
      {LapackErrorCode::FAILED_TRIA_MATRIX_INVERSE,
       "Triangular matrix inversion failed with error code "},
      {LapackErrorCode::FAILED_CHOLESKY_FACTORIZATION,
       "Cholesky factorization failed with error code "},
      {LapackErrorCode::FAILED_REAL_TRIDIAGONAL_EIGENPROBLEM,
       "Real Tridiagonal Standard eigenproblem decomposition failed with error code "},
      {LapackErrorCode::FAILED_STANDARD_EIGENPROBLEM,
       "Standard eigenproblem decomposition failed with error code "},
      {LapackErrorCode::FAILED_GENERALIZED_EIGENPROBLEM,
       "Generalized eigenproblem decomposition failed with error code "}};

    const std::map<LinearSolverErrorCode, std::string>
      LinearSolverErrorMsg::d_errToMsgMap = {
        {LinearSolverErrorCode::SUCCESS, "Success"},
        {LinearSolverErrorCode::FAILED_TO_CONVERGE, "Failed to converge"},
        {LinearSolverErrorCode::RESIDUAL_DIVERGENCE, "Residual diverged"},
        {LinearSolverErrorCode::DIVISON_BY_ZERO,
         "Division by zero encountered"},
        {LinearSolverErrorCode::OTHER_ERROR, "Other error encountered"}};

    const std::map<EigenSolverErrorCode, std::string>
      EigenSolverErrorMsg::d_errToMsgMap = {
        {EigenSolverErrorCode::SUCCESS, "Success"},
        {EigenSolverErrorCode::LAPACK_ERROR, "LAPACK function failed. "},
        {EigenSolverErrorCode::LANCZOS_BETA_ZERO,
         "Could not create more B-orthonormal krylov subspace vectors in Lanczos."},
        {EigenSolverErrorCode::LANCZOS_SUBSPACE_INSUFFICIENT,
         "Maximum Krylov Subspace Size given is insufficient for Lanczos convergence."},
        {EigenSolverErrorCode::OTHER_ERROR, "Other error encountered"}};

    const std::map<OrthonormalizationErrorCode, std::string>
      OrthonormalizationErrorMsg::d_errToMsgMap = {
        {OrthonormalizationErrorCode::SUCCESS, "Success"},
        {OrthonormalizationErrorCode::NON_ORTHONORMALIZABLE_MULTIVECTOR,
         "Failed to converge"}};

    LinearSolverError
    LinearSolverErrorMsg::isSuccessAndMsg(const LinearSolverErrorCode &error)
    {
      LinearSolverError ret;
      auto              it = d_errToMsgMap.find(error);
      if (it != d_errToMsgMap.end())
        {
          if (error == LinearSolverErrorCode::SUCCESS)
            ret.isSuccess = true;
          else
            ret.isSuccess = false;
          ret.err = error;
          ret.msg = it->second;
          // returnValue = std::make_pair(error == Error::SUCCESS, it->second);
        }
      else
        {
          utils::throwException<utils::InvalidArgument>(
            false, "Invalid linearAlgebra::LinearSolverErrorCode passed.");
        }
      return ret;
    }

    LapackError
    LapackErrorMsg::isSuccessAndMsg(const LapackErrorCode &error)
    {
      LapackError ret;
      auto        it = d_errToMsgMap.find(error);
      if (it != d_errToMsgMap.end())
        {
          if (error == LapackErrorCode::SUCCESS)
            ret.isSuccess = true;
          else
            ret.isSuccess = false;
          ret.err = error;
          ret.msg = it->second;
        }
      else
        {
          utils::throwException<utils::InvalidArgument>(
            false, "Invalid linearAlgebra::LapackErrorCode passed.");
        }
      return ret;
    }

    EigenSolverError
    EigenSolverErrorMsg::isSuccessAndMsg(const EigenSolverErrorCode &error)
    {
      EigenSolverError ret;
      auto             it = d_errToMsgMap.find(error);
      if (it != d_errToMsgMap.end())
        {
          if (error == EigenSolverErrorCode::SUCCESS)
            ret.isSuccess = true;
          else
            ret.isSuccess = false;
          ret.err = error;
          ret.msg = it->second;
        }
      else
        {
          utils::throwException<utils::InvalidArgument>(
            false, "Invalid linearAlgebra::EigenSolverErrorCode passed.");
        }
      return ret;
    }

    OrthonormalizationError
    OrthonormalizationErrorMsg::isSuccessAndMsg(
      const OrthonormalizationErrorCode &error)
    {
      OrthonormalizationError ret;
      auto                    it = d_errToMsgMap.find(error);
      if (it != d_errToMsgMap.end())
        {
          if (error == OrthonormalizationErrorCode::SUCCESS)
            ret.isSuccess = true;
          else
            ret.isSuccess = false;
          ret.err = error;
          ret.msg = it->second;
        }
      else
        {
          utils::throwException<utils::InvalidArgument>(
            false,
            "Invalid linearAlgebra::OrthonormalizationErrorCode passed.");
        }
      return ret;
    }

  } // namespace linearAlgebra
} // namespace dftefe
