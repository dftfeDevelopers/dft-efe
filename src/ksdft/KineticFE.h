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
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class KineticFE : public Hamiltonian<ValueTypeBasisData, memorySpace>,
                      public Energy<linearAlgebra::blasLapack::scalar_type
                        <ValueTypeBasisData, ValueTypeBasisCoeff>>
    {
    public:
      using Storage = Hamiltonian<ValueTypeBasisData, memorySpace>::Storage;

      using ValueType = <linearAlgebra::blasLapack::scalar_type
                        <ValueTypeBasisData, ValueTypeBasisCoeff>;

    public:
      /**
       * @brief Constructor
       */
      KineticFE(const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>
                  &feBasisDataStorage);

      void
      getLocal(Storage cellWiseStorage) const override;
      void
      evalEnergy(const std::vector<RealType>               &orbitalOccupancy,
                 const MultiVector<ValueType, memorySpace> &waveFunction) const;
      RealType<ValueType>
      getEnergy() const override;

    private:
      const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>
        *d_feBasisDataStorage;
    }; // end of class KineticFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/KineticFE.t.cpp>
#endif // dftefeKineticFE_h
