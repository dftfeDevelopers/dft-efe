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

#ifndef dftefe_RDM1FE_h
#define dftefe_RDM1FE_h

// C++ standard libraries
#include <vector>
#include <set>
#include <utility>
#include <unordered_map>
#include <memory>
#include <ksdft/RDM1Spectral.h>
#include <ksdft/DensityCalculator.h>
#include <basis/FEBasisDataStorage.h>
#include <utils/Point.h>
#include <basis/FEBasisManager.h>
#include <basis/ReciprocalLattice.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <map>

namespace dftefe
{
  namespace ksdft
  {
    /**
     * @brief Class to store the one-particle reduced density matrix
     * using its spectral (eigen) representation in the finite-element
     * basis
     */
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    class RDM1FE
      : public RDM1Spectral<
          linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff>,
          memorySpace>
    {
    public:
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeBasisData,
                                               ValueTypeBasisCoeff>;
      using AttrStorage = typename RDM1<ValueType, memorySpace>::AttrStorage;

    public:
      virtual ~RDM1FE() = default;

      RDM1FE(
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
                                                      dim>>> &feBMPsi,
        const std::map<basis::kpoint::KPointId, basis::kpoint::KPointData> &  kPoints,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &mpiCommDomain,
        const size_type            cellBlockSize,
        const size_type            waveFuncBatchSize,
        const SpinMode             spinMode = SpinMode::Unpolarized,
        const bool                 isSOC    = false);

      void
      reinit(
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
                                                      dim>>> &feBMPsi);

      void
      setEvalDescrFlag(const bool evalFlag) override;

      void
      getDescriptors(
        const std::set<DensityDescrAttr> &                 densityAttrs,
        const std::set<WfcDescrAttr> &                     wfcAttrs,
        std::unordered_map<DensityDescrAttr, AttrStorage> &densityAttrVals,
        std::unordered_map<WfcDescrAttr, AttrStorage> &    wfcAttrVals);

      void
      setDescriptors(
        const std::unordered_map<DensityDescrAttr, AttrStorage>
          &                                                  densityAttrVals,
        const std::unordered_map<WfcDescrAttr, AttrStorage> &wfcAttrVals);

      void
      getDensityObs(
        const std::set<DensityObsAttr> &densityObsAttrs,
        std::unordered_map<DensityObsAttr, std::vector<std::vector<double>>>
          &densityObsAttrVals);

      SpinMode
      spinMode() const;

      bool
      isSOC() const;

      size_type
      getnKSOrbs() const;

      std::vector<utils::Point>
      getkPointCoords() const;

      std::vector<double>
      getkPointWeights() const;

      std::unique_ptr<RDM1<ValueType, memorySpace>>
      clone() const;

      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
      getFEBasisDataStorage() const;

      const basis::FEBasisManager<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  memorySpace,
                                  dim> &
      getFEBasisManager() const;

    private:
      // the per-k inputs, held as given: this class runs no k loop, it only
      // forwards them to the DensityCalculator and to clone()
      std::map<
        basis::kpoint::KPointId,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>>
        d_feBasisDataStorage;
      std::map<basis::kpoint::KPointId,
               std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                           ValueTypeBasisData,
                                                           memorySpace,
                                                           dim>>>
        d_feBMPsi;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;
      std::shared_ptr<DensityCalculator<ValueTypeBasisData,
                                        ValueTypeBasisCoeff,
                                        memorySpace,
                                        dim>>
        d_densCalc;

      std::map<basis::kpoint::KPointId, basis::kpoint::KPointData>      d_kPoints;
      SpinMode                                          d_spinMode;
      bool                                              d_isSOC;
      std::reference_wrapper<const utils::mpi::MPIComm> d_mpiCommDomain;
      size_type                                         d_cellBlockSize;
      size_type                                         d_waveFuncBatchSize;
      std::unordered_map<DensityDescrAttr, AttrStorage> d_densityAttrVals;
      std::unordered_map<WfcDescrAttr, AttrStorage>     d_wfcAttrVals;
    }; // end of RDM1FE class

  } // namespace ksdft
} // namespace dftefe
#include "RDM1FE.t.cpp"
#endif // dftefe_RDM1FE_h
