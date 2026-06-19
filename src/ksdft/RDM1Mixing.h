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

#ifndef dftefe_RDM1Mixing_h
#define dftefe_RDM1Mixing_h

#include <vector>
#include <set>
#include <unordered_map>
#include <memory>
#include <functional>

#include <ksdft/RDM1.h>
#include <ksdft/MixingScheme.h>
#include <utils/MemoryStorage.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <linearAlgebra/BlasLapack.h>

namespace dftefe
{
  namespace ksdft
  {
    /**
     * @brief RDM1 implementation that performs Anderson mixing of the electron
     * density. Derives directly from RDM1 — no spectral decomposition.
     *
     * setRDM1(rdm1)  — stores the RDM1 pointer.
     * mix()          — explicit call from KohnShamDFT after each eigensolve.
     *                  Fetches densOut from d_rdm1Ptr->getDescriptors(),
     *                  computes residual = densOut - in, and performs Anderson
     *                  mixing. No normalization is done here; KohnShamDFT
     *                  normalizes densIn and densOut externally.
     * getDescriptors — returns d_densityInAttrVals (the last mixed in-density).
     *
     * The caller (KohnShamDFT) owns and configures the MixingScheme object
     * (including addMixingVariable) before the first setRDM1() call.
     * d_rdm1Ptr is null until the first setRDM1() call.
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    class RDM1Mixing : public RDM1<ValueType, memorySpace>
    {
    public:
      using RealType    = linearAlgebra::blasLapack::real_type<ValueType>;
      using AttrStorage = typename RDM1<ValueType, memorySpace>::AttrStorage;

    public:
      /**
       * @brief Constructor.
       *
       * @param mixingScheme  Anderson mixing scheme owned by the caller.
       *   RDM1Mixing calls addVariableToInHist, addVariableToResidualHist,
       *   popOldHistory, computeAndersonMixingCoeff, and mixVariable on it
       *   inside mix(). The caller is responsible for calling
       *   addMixingVariable() on the scheme before the first setRDM1() call.
       * @param mixingHistory     Anderson mixing history length.
       * @param numElectrons  Number of electrons (unused internally; kept for
       *   interface compatibility).
       * @param linAlgOpContextHost  LinAlg context for HOST-side operations.
       * @param mpiCommDomain MPI communicator.
       */
      RDM1Mixing(
        MixingScheme<RealType, RealType> &                                   mixingScheme,
        const size_type                                                      mixingHistory,
        const std::vector<RealType> &                                        jxwDataHost,
        const double                                                         mixingParameter,
        const bool                                                           isAdaptiveMixingParameter,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>>
          linAlgOpContextHost,
        const MPI_Comm &mpiCommDomain);

      virtual ~RDM1Mixing() = default;

      /**
       * @brief Store rdm1 as a weak_ptr (non-owning).
       */
      void
      setRDM1(std::shared_ptr<RDM1<ValueType, memorySpace>> rdm1);

      const RDM1<ValueType, memorySpace> &
      getRDM1() const;

      /**
       * @brief Execute the Anderson mixing pipeline using the stored d_rdm1Ptr.
       *        On the first call d_densityInAttrVals is adopted from densOut.
       *        Subsequent calls compute residual, mix, and update
       *        d_densityInAttrVals. No normalization is performed here.
       */
      void
      mix();

      // ---- RDM1 interface ------------------------------------------------

      /**
       * @brief Returns d_densityInAttrVals (the last mixed in-density).
       */
      void
      getDescriptors(
        const std::set<DensityDescrAttr> &densityAttrs,
        const std::set<WfcDescrAttr> &    wfcAttrs,
        std::unordered_map<DensityDescrAttr, AttrStorage> &densityAttrVals,
        std::unordered_map<WfcDescrAttr, AttrStorage> &   wfcAttrVals)
        override;

      void
      setDescriptors(
        const std::unordered_map<DensityDescrAttr, AttrStorage> &densityAttrVals,
        const std::unordered_map<WfcDescrAttr, AttrStorage> &    wfcAttrVals)
        override;

      void
      getDensityObs(
        const std::set<DensityObsAttr> &densityObsAttrs,
        std::unordered_map<DensityObsAttr, std::vector<std::vector<double>>>
          &densityObsAttrVals) override;

      void      setEvalDescrFlag(const bool evalFlag) override;
      bool      isSpinPolarized() const override;
      bool      isNonCollinear() const override;
      bool      isSOC() const override;
      size_type getnKSOrbs() const override;
      std::vector<double> getkPointCoords() const override;
      std::vector<double> getkPointWeights() const override;
      std::unique_ptr<RDM1<ValueType, memorySpace>> clone() const override;

    private:
      std::weak_ptr<RDM1<ValueType, memorySpace>> d_rdm1Ptr;

      MixingScheme<RealType, RealType> &     d_mixingScheme;
      size_type                              d_mixingHistory;
      size_type                              d_numElectrons;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>>
                                             d_linAlgOpContextHost;
      std::reference_wrapper<const MPI_Comm> d_mpiCommDomain;

      AttrStorage d_densityInAttrVals;
      AttrStorage d_gradDensityInAttrVals;
    };

  } // namespace ksdft
} // namespace dftefe
#include "RDM1Mixing.t.cpp"
#endif // dftefe_RDM1Mixing_h
