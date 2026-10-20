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
#include <electrostatics/LaplaceOperatorContextFE.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEBasisDofHandler.h>
#include <basis/FEBasisManager.h>
#include <basis/ReciprocalLattice.h>
#include <map>
#include <ksdft/KSAttributes.h>
#include <ksdft/RDM1Spectral.h>
#include <ksdft/HamiltonianSpinBlockCopyKernels.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class KineticFE
      : public Hamiltonian<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace>,
        public Energy<linearAlgebra::blasLapack::real_type<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff>>>
    {
    public:
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                               ValueTypeBasisCoeff>;

      using RealType = linearAlgebra::blasLapack::real_type<ValueType>;

      // the assembled cell matrix is handed to the operator context, which
      // multiplies it against the coefficients, so it carries the union type
      using Storage = utils::MemoryStorage<ValueType, memorySpace>;

    public:
      /**
       * @brief Constructor
       */
      KineticFE(
        const std::map<
          basis::kpoint::KPointId,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>>
          &feBasisDataStorage,
        const std::map<
          basis::kpoint::KPointId,
          std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                      ValueTypeBasisData,
                                                      memorySpace,
                                                      dim>>> &feBasisManager,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                                            linAlgOpContext,
        const size_type                                     maxCellBlock,
        const size_type                                     waveFuncBatchSize,
        SpinMode                                            spinMode,
        const std::map<basis::kpoint::KPointId, basis::kpoint::KPointData> &kPoints);

      ~KineticFE() = default;

      void
      reinit(
        const std::map<
          basis::kpoint::KPointId,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>>
          &feBasisDataStorage);

      /**
       * @brief Select the k that getLocal and evalEnergy then use.
       */
      void
      activateKpt(const basis::kpoint::KPointId kPointId) override;

      void
      getLocal(Storage &cellWiseStorage) const override;
      /**
       * @brief sum_k w_k sum_n f_nk <psi|T(k)|psi>, with the orbitals and
       * occupancies read from the spectral RDM1.
       */
      void
      evalEnergy(RDM1<ValueType, memorySpace> &rdm1);
      RealType
      getEnergy() const override;

      void
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const override;

      bool
      hasLocalComponent() const override;

      bool
      hasNonLocalComponent() const override;

    private:
      // per-k containers, nKLocal long, indexed by the LOCAL index
      std::vector<std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>>
        d_feBasisDataStorage;
      std::vector<
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>>>
        d_feBasisManager;
      std::vector<std::shared_ptr<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace,
                                                           dim>>>
        d_feBasisOp;
      // global ids of the local slots: keys into SpectralRep, never an index
      std::vector<basis::kpoint::KPointId> d_kPointIds;
      std::vector<utils::Point>    d_kPointCoordinates;
      std::vector<double>          d_kPointWeights;
      size_type                    d_currentKPointIndex;
      RealType        d_energy;
      const size_type d_maxCellBlock, d_waveFuncBatchSize;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;
      // quadrature::QuadratureValuesContainer<ValueType, memorySpace>
      // *d_gradPsi;
      std::vector<std::shared_ptr<Storage>> d_cellWiseStorageKineticEnergy;

      SpinMode               d_spinMode;
      size_type              d_S;
      SpinStorageLayout      d_layout;
      std::vector<size_type> d_numCellDofs;
      size_type              d_basisOverlapSize;

      std::shared_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>>
        d_psiBatchSmall, d_psiBatch, d_YBatch, d_YBatchSmall;
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
        d_mpiPatternP2P;

      std::vector<std::shared_ptr<
        electrostatics::LaplaceOperatorContextFE<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff,
                                                 memorySpace,
                                                 dim>>>
        d_laplaceOp;

    }; // end of class KineticFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/KineticFE.t.cpp>
#endif // dftefeKineticFE_h
