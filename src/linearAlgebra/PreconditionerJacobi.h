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
 * @author Bikash Kanungo
 */

#ifndef dftefePreconditionerJacobi_h
#define dftefePreconditionerJacobi_h

#include <utils/TypeConfig.h>
#include <linearAlgebra/Preconditioner.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/BlasLapackTypedef.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *@brief Class to encapsulate the Jacobi preconditioner
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying preconditioner
     * @tparam ValueTypeOperand The datatype (float, double, complex<double>, etc.) of the vector, matrices, etc.
     *  on which the preconditioner will act.
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNES, etc.) in which the data of the preconditioner
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    class PreconditionerJacobi
      : public Preconditioner<ValueTypeOperator, ValueTypeOperand, memorySpace>
    {
    public:
      /**
       * @brief Constructor
       *
       * @param[in] diagonal Vector object containing the diagonal vector of a
       * matrix. The Vector can be serial or distributed.
       */
      PreconditionerJacobi(
        const Vector<ValueTypeOperator, memoryStorageSrc> &diagonal);

      /**
       *@brief Default Destructor
       *
       */
      ~PreconditionerJacobi() = default;

      void
      apply(const Vector<ValueTypeOperand, memorySpace> &x,
            Vector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                   memorySpace> &                        y) const override;

      void
      apply(const MultiVector<ValueTypeOperand, memorySpace> &X,
            MultiVector<scalar_type<ValueTypeOperator, ValueTypeOperand>,
                        memorySpace> &                        Y) const override;

      SolverTypes::PreconditionerType
      getPreconditionerType() const = override;
      //
      // TODO: Uncomment the following and implement in all the derived classes
      //

      // virtual
      //  apply(const AbstractMatrix<ValueTypeOperand, memorySpace> & X,
      //    AbstractMatrix<scalar_type<ValueTypeOperator, ValueTypeOperand>,
      //    memorySpace> & Y) const = 0;

    private:
      Vector<ValueTypeOperator, memoryStorage> d_invDiagonal;
    };


  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/PreconditionerJacobi.t.cpp>
#endif // dftefePreconditionerJacobi_h
