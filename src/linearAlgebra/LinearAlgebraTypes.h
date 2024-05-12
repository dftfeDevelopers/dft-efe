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

#ifndef dftefeLinearAlgebraTypes_h
#define dftefeLinearAlgebraTypes_h

#include <map>
#include <string>
namespace dftefe
{
  namespace linearAlgebra
  {
    enum class LinearSolverType
    {
      CG
    };

    enum class PreconditionerType
    {
      NONE,
      JACOBI
    };

    enum class NonLinearSolverType
    {
      CG,
      LBFGS
    };

    enum class ParallelPrintType
    {
      NONE,
      ROOT_ONLY,
      ALL
    };

    enum class LapackErrorCode
    {
      SUCCESS,
      FAILED_DENSE_MATRIX_INVERSE,
      FAILED_TRIA_MATRIX_INVERSE,
      FAILED_CHOLESKY_FACTORIZATION,
      FAILED_REAL_TRIDIAGONAL_EIGENPROBLEM,
      FAILED_STANDARD_EIGENPROBLEM,
      FAILED_GENERALIZED_EIGENPROBLEM,
    };

    enum class LinearSolverErrorCode
    {
      SUCCESS,
      FAILED_TO_CONVERGE,
      RESIDUAL_DIVERGENCE,
      DIVISON_BY_ZERO,
      OTHER_ERROR
    };

    enum class EigenSolverErrorCode
    {
      SUCCESS,
      LAPACK_ERROR,
      LANCZOS_BETA_ZERO,
      LANCZOS_SUBSPACE_INSUFFICIENT,
      CHFSI_ORTHONORMALIZATION_ERROR,
      CHFSI_RAYLEIGH_RITZ_ERROR,
      KS_MAX_PASS_ERROR,
      KS_CHFSI_ERROR,
      KS_LANCZOS_ERROR,
      OTHER_ERROR
    };

    enum class OrthonormalizationErrorCode
    {
      SUCCESS,
      LAPACK_ERROR,
      NON_ORTHONORMALIZABLE_MULTIVECTOR,
      MAX_PASS_EXCEEDED
    };

    enum class NewtonRaphsonErrorCode
    {
      SUCCESS,
      FORCE_TOLERANCE_ERR,
      FAILED_TO_CONVERGE,
      OTHER_ERROR
    };

    struct LapackError
    {
      bool            isSuccess;
      LapackErrorCode err;
      std::string     msg;
    };

    struct LinearSolverError
    {
      bool                  isSuccess;
      LinearSolverErrorCode err;
      std::string           msg;
    };

    struct EigenSolverError
    {
      bool                 isSuccess;
      EigenSolverErrorCode err;
      std::string          msg;
    };

    struct OrthonormalizationError
    {
      bool                        isSuccess;
      OrthonormalizationErrorCode err;
      std::string                 msg;
    };

    struct NewtonRaphsonError
    {
      bool                   isSuccess;
      NewtonRaphsonErrorCode err;
      std::string            msg;
    };

    /**
     * @brief A class to map Error to a message.
     * @note: This class only has static const data members.
     */
    class LapackErrorMsg
    {
    public:
      static LapackError
      isSuccessAndMsg(const LapackErrorCode &errorCode);

    private:
      static const std::map<LapackErrorCode, std::string> d_errToMsgMap;
    }; // end of class LapackErrorMsg

    /**
     * @brief A class to map Error to a message.
     * @note: This class only has static const data members.
     */
    class LinearSolverErrorMsg
    {
    public:
      static LinearSolverError
      isSuccessAndMsg(const LinearSolverErrorCode &errorCode);

    private:
      static const std::map<LinearSolverErrorCode, std::string> d_errToMsgMap;
    }; // end of class LinearSolverErrorMsg

    class EigenSolverErrorMsg
    {
    public:
      static EigenSolverError
      isSuccessAndMsg(const EigenSolverErrorCode &errorCode);

    private:
      static const std::map<EigenSolverErrorCode, std::string> d_errToMsgMap;
    }; // end of class EigenSolverErrorMsg

    class OrthonormalizationErrorMsg
    {
    public:
      static OrthonormalizationError
      isSuccessAndMsg(const OrthonormalizationErrorCode &errorCode);

    private:
      static const std::map<OrthonormalizationErrorCode, std::string>
        d_errToMsgMap;
    }; // end of class OrthonormalizationErrorMsg


    class NewtonRaphsonErrorMsg
    {
    public:
      static NewtonRaphsonError
      isSuccessAndMsg(const NewtonRaphsonErrorCode &errorCode);

    private:
      static const std::map<NewtonRaphsonErrorCode, std::string> d_errToMsgMap;
    }; // end of class NewtonRaphsonErrorMsg

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearAlgebraTypes_h
