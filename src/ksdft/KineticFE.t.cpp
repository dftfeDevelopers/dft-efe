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
    KineticFE<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>::KineticFE(
      const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
        &feBasisDataStorage)
      : d_feBasisDataStorage(&feBasisDataStorage)
    {}

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    Storage
    KineticFE<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>::
      getHamiltonian() const
    {
      return d_feBasisDataStorage->getGradNiNjInAllCells();
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KineticFE<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>::
      evalEnergy(const std::vector<RealType> &orbitalOccupancy,
                 const std::vector<RealType> &eigenEnergy,
                 const Storage &              density,
                 const Storage &              kohnShamPotential) const
    {
      RealType bandEnergy;
      for (size_Type i = 0; i < orbitalOccupancy.size(); i++)
        {
          bandEnergy += orbitalOccupancy[i] * eigenEnergy[i];
        }
      d_energy = 2 * bandEnergy - kohnShamEnergy;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KineticFE<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>::
      evalEnergy(const std::vector<RealType> &              orbitalOccupancy,
                 const MultiVector<ValueType, memorySpace> &waveFunction) const
    {
      d_energy = /*\sum f_i c_i^2 \integral \grad N_i \grad N_i*/
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    RealType
    KineticFE<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>::
      getEnergy() const
    {
      return d_energy;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
