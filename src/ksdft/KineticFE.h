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

#ifndef dftefeKineticFE_h
#define dftefeKineticFE_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <ksdft/Hamiltonian.h>
#include <ksdft/Energy.h>
#include <basis/FEBasisDataStorage.h>

namespace dftefe
{
  namespace ksdft
  {
    /**
     *@brief A derived class of linearAlgebra::OperatorContext to encapsulate
     * the action of a discrete operator on vectors, matrices, etc.
     *
     * @tparam ValueTypeOperator The datatype (float, double, complex<double>, etc.) for the underlying operator
     * on which the operator will act
     * @tparam memorySpace The meory sapce (HOST, DEVICE, HOST_PINNED, etc.) in which the data of the operator
     * and its operands reside
     *
     */
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class KineticFE : public Hamiltonian<ValueTypeOperator, memorySpace>,
                      public Energy<ValueTypeOperator>
    {
    public:
      using Storage = Hamiltonian<ValueTypeOperator, memorySpace>::Storage;

    public:
      /**
       * @brief Constructor
       */
      KineticFE(const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
                  &feBasisDataStorage);

      Storage
      getLocal() const override;
      void
      evalEnergy(const std::vector<RealType> &orbitalOccupancy,
                 const std::vector<RealType> &eigenEnergy,
                 const Storage &              density,
                 const Storage &              kohnShamPotential) const;
      void
      evalEnergy(const std::vector<RealType> &              orbitalOccupancy,
                 const MultiVector<ValueType, memorySpace> &waveFunction) const;
      RealType
      getEnergy() const override;

    private:
      const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
        *d_feBasisDataStorage;
    }; // end of class KineticFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/KineticFE.t.cpp>
#endif // dftefeKineticFE_h
