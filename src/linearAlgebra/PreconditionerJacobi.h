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
    class PreconditionerJacobi : public Preconditioner<ValueTypeOperator,
                                                       ValueTypeOperand,
                                                       ValueTypeOperand>
    {
    public:
      /**
       * @brief Constructor based on input MemoryStorage
       *
       * @param[in] diagonal MemoryStorage object containing the local part of
       * diagonal of the matrix. The local part denotes the union of the locally
       * owned and the ghost part of the matrix (i.e, the locally relevant part)
       *
       * @tparam memorySpaceSrc The memory space in which the input parameter diagonal resides. This allows
       * us to seamlessly transfer data from any memory space (i.e.,
       * memorySpaceSrc) to the memory space of the PreconditionerJacobi object
       * (i.e., memorySpace)
       *
       */
      template <utils::MemorySpace memorySpaceSrc>
      PreconditionerJacobi(
        const utils::MemoryStorage<ValueTypeOperator, memoryStorageSrc>
          &diagonal);

      /**
       * @brief Constructor based on input pointer
       *
       * @param[in] diagData Pointer to data containing the local part of
       * diagonal of the matrix. The local part denotes the union of the locally
       * owned and the ghost part of the matrix (i.e, the locally relevant part)
       *
       * @param[in] N size of the local part of the diagonal of the matrix (..e,
       * the size of the locally owned plus the ghost indices)
       *
       * @note: This constructor assumes that the data pointed by diagData
       * resides in the memory space of the PreconditionerJacobi (i.e.,
       * memorySpace).
       *
       * @note: The diagData must be appropriately allocated
       *
       */
      PreconditionerJacobi(const ValueTypeOperator *diagonal, size_type N);

      /**
       *@brief Default Destructor
       *
       */
      ~PreconditionerJacobi() = default;

      void
      apply(const Vector<ValueTypeOperand, memorySpace> &x,
            Vector<scalar_type<ValueTypeOperator, ValueTypeOperand>,
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
      utils::MemoryStorage<ValueTypeOperator, memoryStorageSrc> d_invDiagonal;
    };


  } // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/PreconditionerJacobi.t.cpp>
#endif // dftefePreconditionerJacobi_h
