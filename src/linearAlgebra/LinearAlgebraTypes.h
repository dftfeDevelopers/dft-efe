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
      LAPACK_STEQR_ERROR,
      LANCZOS_BETA_ZERO,
      OTHER_ERROR
    };

    enum class OrthonormalizationErrorCode
    {
      SUCCESS,
      NON_ORTHONORMALIZABLE_MULTIVECTOR
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

  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif // dftefeLinearAlgebraTypes_h
