
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
#include <utils/Exceptions.h>
#include <ksdft/DensityCalculator.h>
#include <ksdft/RDM1Spectral.h>
#include <memory>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::RDM1FE(
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                        feBasisDataStorage,
      const basis::FEBasisManager<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  memorySpace,
                                  dim> &feBMPsi,
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                 linAlgOpContext,
      const utils::mpi::MPIComm &mpiCommDomain,
      const size_type            cellBlockSize,
      const size_type            waveFuncBatchSize,
      const SpinMode             spinMode /*= SpinMode::Unpolarized*/,
      const bool                 isSOC /*= false*/,
      const std::vector<double> &kPointCoords /*= std::vector<double>{}*/,
      const std::vector<double> &kPointWeights /*= std::vector<double>{}*/)
      : d_feBasisDataStorage(feBasisDataStorage)
      , d_feBMPsiPtr(&feBMPsi)
      , d_linAlgOpContext(linAlgOpContext)
      , d_kPointCoords(kPointCoords)
      , d_kPointWeights(kPointWeights)
      , d_spinMode(spinMode)
      , d_isSOC(isSOC)
      , d_mpiCommDomain(mpiCommDomain)
      , d_cellBlockSize(cellBlockSize)
      , d_waveFuncBatchSize(waveFuncBatchSize)
    {
      dftefe::utils::throwException<dftefe::utils::InvalidArgument>(
        !isSOC, "RDM1FE does not yet support SOC density computation.");

      d_densCalc = std::make_shared<DensityCalculator<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpace,
                                                      dim>>(feBasisDataStorage,
                                                            feBMPsi,
                                                            linAlgOpContext,
                                                            cellBlockSize,
                                                            waveFuncBatchSize);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::reinit(
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
                                        feBasisDataStorage,
      const basis::FEBasisManager<ValueTypeBasisCoeff,
                                  ValueTypeBasisData,
                                  memorySpace,
                                  dim> &feBMPsi)
    {
      d_feBasisDataStorage = feBasisDataStorage;
      d_feBMPsiPtr         = &feBMPsi;
      d_densCalc->reinit(feBasisDataStorage, feBMPsi);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    SpinMode
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      spinMode() const
    {
      return d_spinMode;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    bool
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::isSOC()
      const
    {
      return d_isSOC;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    size_type
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getnKSOrbs() const
    {
      return this->getSpectral().nKSOrbs;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<double>
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getkPointCoords() const
    {
      return d_kPointCoords;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::vector<double>
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getkPointWeights() const
    {
      return d_kPointWeights;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      setEvalDescrFlag(const bool evalFlag)
    {
      this->d_evalFlag = evalFlag;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getDescriptors(
        const std::set<DensityDescrAttr> &                 densityAttrs,
        const std::set<WfcDescrAttr> &                     wfcAttrs,
        std::unordered_map<DensityDescrAttr, AttrStorage> &densityAttrVals,
        std::unordered_map<WfcDescrAttr, AttrStorage> &    wfcAttrVals)
    {
      // Determine whether re-evaluation is needed.
      bool needEval = this->d_evalFlag;
      if (!needEval)
        {
          for (const auto &attr : densityAttrs)
            if (d_densityAttrVals.find(attr) == d_densityAttrVals.end())
              {
                needEval = true;
                break;
              }
        }
      if (!needEval)
        {
          for (const auto &attr : wfcAttrs)
            if (d_wfcAttrVals.find(attr) == d_wfcAttrVals.end())
              {
                needEval = true;
                break;
              }
        }

      if (needEval)
        {
          dftefe::utils::throwException<dftefe::utils::LogicError>(
            this->isSpectralSet(),
            "Re-evaluation of descriptors in RDM1FE::getDescriptors() requires "
            "the KS orbitals to be set via RDM1Spectral::setSpectral().");

          DFTEFE_AssertWithMsg(
            wfcAttrs.find(WfcDescrAttr::Tau) == wfcAttrs.end(),
            "Tau computation not yet implemented in dftefe "
            "RDM1FE::getDescriptors - required for mGGA functionals.");

          size_type ncomp = 1;
          if (d_spinMode == SpinMode::Collinear)
            ncomp = 2;
          else if (d_spinMode == SpinMode::NonCollinear)
            ncomp = 4;

          const std::vector<double> &occ = this->d_spectral->occupancies;

          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainer =
              d_feBasisDataStorage->getQuadratureRuleContainer();

          auto &densVal = d_densityAttrVals[DensityDescrAttr::Val];
          if (densVal.size() != ncomp ||
              densVal[0].getQuadratureRuleContainer() != quadRuleContainer)
            {
              densVal = AttrStorage(
                ncomp,
                quadrature::QuadratureValuesContainer<double,
                                                      utils::MemorySpace::HOST>(
                  quadRuleContainer, 1, 0.0));
            }

          auto &     gradDensVal = d_densityAttrVals[DensityDescrAttr::Grad];
          const bool needGrad    = densityAttrs.count(DensityDescrAttr::Grad);
          if (needGrad)
            {
              if (gradDensVal.size() != ncomp ||
                  gradDensVal[0].getQuadratureRuleContainer() !=
                    quadRuleContainer)
                gradDensVal = AttrStorage(
                  ncomp,
                  quadrature::QuadratureValuesContainer<
                    double,
                    utils::MemorySpace::HOST>(quadRuleContainer, dim, 0.0));
            }
          else
            {
              if (gradDensVal.empty())
                gradDensVal.resize(1);
            }

          std::vector<
            quadrature::QuadratureValuesContainer<double,
                                                  utils::MemorySpace::HOST> *>
            rhoVec(ncomp);
          for (size_type ic = 0; ic < ncomp; ++ic)
            rhoVec[ic] = &densVal[ic];
          const size_type gradNcomp = needGrad ? ncomp : 1;
          std::vector<
            quadrature::QuadratureValuesContainer<double,
                                                  utils::MemorySpace::HOST> *>
            gradRhoVec(gradNcomp);
          for (size_type ic = 0; ic < gradNcomp; ++ic)
            gradRhoVec[ic] = &gradDensVal[ic];
          d_densCalc->computeRho(occ,
                                 *this->d_spectral->ksOrbs,
                                 rhoVec,
                                 gradRhoVec,
                                 needGrad,
                                 d_spinMode);

          this->d_evalFlag = false;
        }

      // Light check: cached Quads must still be on the same quad rule as
      // d_feBasisDataStorage (catches reinit() without setEvalDescrFlag(true)).
      if (!d_densityAttrVals.empty())
        {
          auto it = d_densityAttrVals.find(DensityDescrAttr::Val);
          if (it != d_densityAttrVals.end() && !it->second.empty())
            dftefe::utils::throwException(
              it->second[0].getQuadratureRuleContainer() ==
                d_feBasisDataStorage->getQuadratureRuleContainer(),
              "RDM1FE::getDescriptors: cached density Quad has a different "
              "QuadratureRuleContainer than the current d_feBasisDataStorage. "
              "Call setEvalDescrFlag(true) after reinit().");
        }

      for (const auto &attr : densityAttrs)
        densityAttrVals[attr] = d_densityAttrVals.at(attr);
      for (const auto &attr : wfcAttrs)
        wfcAttrVals[attr] = d_wfcAttrVals.at(attr);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      setDescriptors(
        const std::unordered_map<DensityDescrAttr, AttrStorage>
          &                                                  densityAttrVals,
        const std::unordered_map<WfcDescrAttr, AttrStorage> &wfcAttrVals)
    {
      for (const auto &kv : densityAttrVals)
        d_densityAttrVals[kv.first] = kv.second;
      for (const auto &kv : wfcAttrVals)
        d_wfcAttrVals[kv.first] = kv.second;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::unique_ptr<RDM1<linearAlgebra::blasLapack::
                           scalar_type<ValueTypeBasisData, ValueTypeBasisCoeff>,
                         memorySpace>>
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::clone()
      const
    {
      return std::make_unique<
        RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>>(
        d_feBasisDataStorage,
        *d_feBMPsiPtr,
        d_linAlgOpContext,
        d_mpiCommDomain.get(),
        d_cellBlockSize,
        d_waveFuncBatchSize,
        d_spinMode,
        d_isSOC,
        d_kPointCoords,
        d_kPointWeights);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    void
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getDensityObs(
        const std::set<DensityObsAttr> &densityObsAttrs,
        std::unordered_map<DensityObsAttr, std::vector<std::vector<double>>>
          &densityObsAttrVals)
    {
      size_type ncomp = 1;
      if (d_spinMode == SpinMode::Collinear)
        ncomp = 2;
      else if (d_spinMode == SpinMode::NonCollinear)
        ncomp = 4;

      // Ensure density has been computed.
      const bool needDensityEval =
        this->d_evalFlag || (d_densityAttrVals.find(DensityDescrAttr::Val) ==
                             d_densityAttrVals.end());
      if (needDensityEval)
        {
          dftefe::utils::throwException<dftefe::utils::LogicError>(
            this->isSpectralSet(),
            "Re-evaluation of density observables in RDM1FE::getDensityObs() "
            "requires KS orbitals set via RDM1Spectral::setSpectral().");

          const std::vector<double> &occ = this->d_spectral->occupancies;

          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainer =
              d_feBasisDataStorage->getQuadratureRuleContainer();

          auto &densVal = d_densityAttrVals[DensityDescrAttr::Val];
          if (densVal.size() != ncomp ||
              densVal[0].getQuadratureRuleContainer() != quadRuleContainer)
            densVal = AttrStorage(
              ncomp,
              quadrature::QuadratureValuesContainer<double,
                                                    utils::MemorySpace::HOST>(
                quadRuleContainer, 1, 0.0));

          auto &gradDensVal = d_densityAttrVals[DensityDescrAttr::Grad];
          if (gradDensVal.empty())
            gradDensVal.resize(1);

          std::vector<
            quadrature::QuadratureValuesContainer<double,
                                                  utils::MemorySpace::HOST> *>
            rhoVec2(ncomp), gradRhoVec2(1);
          for (size_type ic = 0; ic < ncomp; ++ic)
            rhoVec2[ic] = &densVal[ic];
          gradRhoVec2[0] = &gradDensVal[0];
          d_densCalc->computeRho(occ,
                                 *this->d_spectral->ksOrbs,
                                 rhoVec2,
                                 gradRhoVec2,
                                 false,
                                 d_spinMode);
        }

      // Determine the maximum moment order requested.
      size_type maxMoment = 0;
      for (const auto &attr : densityObsAttrs)
        {
          if (attr == DensityObsAttr::Dipole && maxMoment < 1)
            maxMoment = 1;
          if (attr == DensityObsAttr::Quadrupole && maxMoment < 2)
            maxMoment = 2;
        }

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBasisDataStorage->getQuadratureRuleContainer();

      const std::vector<double> &      jxw = quadRuleContainer->getJxW();
      const std::vector<utils::Point> &pts = quadRuleContainer->getRealPoints();
      const size_type nq = quadRuleContainer->nQuadraturePoints();

      for (const auto &attr : densityObsAttrs)
        densityObsAttrVals[attr].resize(ncomp);

      for (size_type iComp = 0; iComp < ncomp; ++iComp)
        {
          const double *densityPtr =
            d_densityAttrVals[DensityDescrAttr::Val][iComp].data();

          // Pre-compute rho * JxW.
          const size_type cappedMaxMoment = std::min(maxMoment, size_type(2));
          size_type       nMoments        = 1;
          if (cappedMaxMoment >= 1)
            nMoments += 3;
          if (cappedMaxMoment >= 2)
            nMoments += 9;

          std::vector<double> moments(nMoments, 0.0);

          // Monopole
          double mono = 0.0;
          for (size_type q = 0; q < nq; ++q)
            mono += densityPtr[q] * jxw[q];
          moments[0] = mono;

          if (cappedMaxMoment >= 1)
            {
              double dx = 0.0, dy = 0.0, dz = 0.0;
              for (size_type q = 0; q < nq; ++q)
                {
                  const double rw = densityPtr[q] * jxw[q];
                  dx += rw * pts[q][0];
                  dy += rw * pts[q][1];
                  dz += rw * pts[q][2];
                }
              moments[1] = dx;
              moments[2] = dy;
              moments[3] = dz;
            }

          if (cappedMaxMoment >= 2)
            {
              double Qxx = 0, Qyy = 0, Qzz = 0, Qxy = 0, Qxz = 0, Qyz = 0;
              for (size_type q = 0; q < nq; ++q)
                {
                  const double rw = densityPtr[q] * jxw[q];
                  const double x = pts[q][0], y = pts[q][1], z = pts[q][2];
                  const double r2 = x * x + y * y + z * z;
                  Qxx += rw * (3 * x * x - r2);
                  Qyy += rw * (3 * y * y - r2);
                  Qzz += rw * (3 * z * z - r2);
                  Qxy += rw * 3 * x * y;
                  Qxz += rw * 3 * x * z;
                  Qyz += rw * 3 * y * z;
                }
              moments[4]  = Qxx;
              moments[5]  = Qxy;
              moments[6]  = Qxz;
              moments[7]  = Qxy;
              moments[8]  = Qyy;
              moments[9]  = Qyz;
              moments[10] = Qxz;
              moments[11] = Qyz;
              moments[12] = Qzz;
            }

          MPI_Allreduce(MPI_IN_PLACE,
                        moments.data(),
                        static_cast<int>(nMoments),
                        MPI_DOUBLE,
                        MPI_SUM,
                        d_mpiCommDomain.get());

          for (const auto &attr : densityObsAttrs)
            {
              if (attr == DensityObsAttr::Monopole)
                densityObsAttrVals[attr][iComp] = {moments[0]};
              else if (attr == DensityObsAttr::Dipole)
                densityObsAttrVals[attr][iComp] =
                  std::vector<double>(moments.begin() + 1, moments.begin() + 4);
              else if (attr == DensityObsAttr::Quadrupole)
                densityObsAttrVals[attr][iComp] =
                  std::vector<double>(moments.begin() + 4, moments.end());
            }
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    std::shared_ptr<
      const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
    RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
      getFEBasisDataStorage() const
    {
      return d_feBasisDataStorage;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    const basis::
      FEBasisManager<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace, dim>
        &
        RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>::
          getFEBasisManager() const
    {
      return *d_feBMPsiPtr;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
