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

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticAllElectronFE<ValueTypeOperator,
                             ValueTypeOperand,
                             memorySpace,
                             dim>::
      ElectrostaticAllElectronFE(
        const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &             feBasisDataStorage)
      : d_feBasisDataStorage(&feBasisDataStorage)
    {}

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    Storage
    ElectrostaticAllElectronFE<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::getHamiltonian() const
    {
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticAllElectronFE<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::evalEnergy() const
    {
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    RealType
    ElectrostaticAllElectronFE<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::getEnergy() const
    {
      return d_energy;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
