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

#ifndef dftefePreconditionerNone_h
#define dftefePreconditionerNone_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/Preconditioner.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <linearAlgebra/BlasLapackTypedef.h>
namespace dftefe
{
  namespace linearAlgebra
  {
    /**
     *@brief Class to encapsulate the NONE preconditioner. Just takes in a vector and returns it.
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
    class PreconditionerNone
      : public Preconditioner<ValueTypeOperator, ValueTypeOperand, memorySpace>
    {
      //
      // typedefs
      //
    public:
      //
      // alias to define the union of ValueTypeOperator and ValueTypeOperand
      // (e.g., the union of double and complex<double> is complex<double>)
      //
      using ValueTypeUnion =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

    public:
      /**
       * @brief Constructor
       *
       */
      PreconditionerNone();

      /**
       *@brief Default Destructor
       *
       */
      ~PreconditionerNone() = default;

      /*
       * @brief Function to apply the Jacobi preconditioner on an input Vector \p X
       * and store the output in \p Y. That is, it stores \f$Y_i=1/X_i$\f
       *
       * @param[in] X Input Vector
       * @param[out] Y Output Vector
       *
       * @note The input Vector \p X can be modified inside the function for
       * performance reasons. If the user needs \p X to be constant
       * (un-modified), we suggest the user to make a copy of \p X
       * prior to calling this function
       *
       */
      void
      apply(MultiVector<ValueTypeOperand, memorySpace> &X,
            MultiVector<ValueTypeUnion, memorySpace> &  Y) const override;

      PreconditionerType
      getPreconditionerType() const override;

    private:
      PreconditionerType d_pcType;
    };


  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/PreconditionerNone.t.cpp>
#endif // dftefePreconditionerNone_h
