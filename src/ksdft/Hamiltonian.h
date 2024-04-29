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

#ifndef dftefeHamiltonian_h
#define dftefeHamiltonian_h

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeOperator, utils::MemorySpace memorySpace>
    class Hamiltonian
    {
    public:
      using Storage =
        dftefe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>;

    public:
      virtual Storage
      getLocal() const = 0;

      // virtual void
      // applyNonLocal(const linearAlgebra::MultiVector<ValueTypeBasisCoeff,
      // memorySpace> &X,const linearAlgebra::MultiVector<ValueTypeBasisCoeff,
      // memorySpace> & Y) const = 0;

    }; // end of Hamiltonian
  }    // end of namespace ksdft
} // end of namespace dftefe
#endif // dftefeHamiltonian_h
