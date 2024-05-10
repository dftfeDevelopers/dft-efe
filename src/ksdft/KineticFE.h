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

#include <linearAlgebra/MultiVector.h>
#include <ksdft/Hamiltonian.h>
#include <ksdft/Energy.h>
#include <basis/FEBasisDataStorage.h>
#include <basis/FEBasisOperations.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class KineticFE
      : public Hamiltonian<ValueTypeBasisData, memorySpace>,
        public Energy<linearAlgebra::blasLapack::real_type<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff>>>
    {
    public:
      using Storage =
        typename Hamiltonian<ValueTypeBasisData, memorySpace>::Storage;

      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                               ValueTypeBasisCoeff>;

      using RealType = linearAlgebra::blasLapack::real_type<ValueType>;

    public:
      /**
       * @brief Constructor
       */
      KineticFE(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                        feBasisDataStorage,
        const size_type cellBlockSize,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext);

      void
      reinit(std::shared_ptr<
             const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
               feBasisDataStorage);

      void
      getLocal(Storage cellWiseStorage) const override;
      void
      evalEnergy(const std::vector<RealType> &                  occupation,
                 const basis::FEBasisManager<ValueTypeBasisCoeff,
                                             ValueTypeBasisData,
                                             memorySpace,
                                             dim>               feBMPsi,
                 const linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                  memorySpace> &waveFunc,
                 const size_type waveFuncBatchSize) const;
      void
      getEnergy(RealType energy) const override;

    private:
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        d_feBasisDataStorage;
      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>>
                      d_feBasisOp;
      RealType        d_energy;
      const size_type d_cellBlockSize;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;

    }; // end of class KineticFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/KineticFE.t.cpp>
#endif // dftefeKineticFE_h
