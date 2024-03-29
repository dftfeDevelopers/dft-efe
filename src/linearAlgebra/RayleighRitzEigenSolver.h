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

#ifndef dftefeRayleighRitzEigenSolver_h
#define dftefeRayleighRitzEigenSolver_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/OperatorContext.h>
#include <memory>

namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *@brief A derived class of OperatorContext to encapsulate
     * the action of a discrete operator on vectors, matrices, etc.
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
    class RayleighRitzEigenSolver
    {
    public:
      /**
       * @brief define ValueType as the superior (bigger set) of the
       * ValueTypeOperator and ValueTypeOperand
       * (e.g., between double and complex<double>, complex<double>
       * is the bigger set)
       */
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
      using RealType = blasLapack::real_type<ValueType>;
      using OpContext =
        OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>;

    public:
      /**
       * @brief Constructor
       */
      RayleighRitzEigenSolver();

      /**
       *@brief Default Destructor
       *
       */
      ~RayleighRitzEigenSolver() = default;

      EigenSolverError
      solve(const OpContext &                                 A,
            const MultiVector<ValueTypeOperand, memorySpace> &X,
            std::vector<RealType> &                           eigenValues,
            MultiVector<ValueType, memorySpace> &             eigenVectors,
            bool computeEigenVectors = false) override;

      EigenSolverError
      solve(const OpContext &                                 A,
            const OpContext &                                 B,
            const MultiVector<ValueTypeOperand, memorySpace> &X,
            std::vector<RealType> &                           eigenValues,
            MultiVector<ValueType, memorySpace> &             eigenVectors,
            bool computeEigenVectors = false) override;

    private:
    }; // end of class RayleighRitzEigenSolver
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/RayleighRitzEigenSolver.t.cpp>
#endif // dftefeRayleighRitzEigenSolver_h
