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

#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <cmath>
#include <string>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    RDM1Mixing<ValueType, memorySpace>::RDM1Mixing(
      MixingScheme<RealType, RealType> &mixingScheme,
      const size_type                   mixingHistory,
      const std::vector<RealType> &     jxwDataHost,
      const double                      mixingParameter,
      const bool                        isAdaptiveMixingParameter,
      const double                      spinMixingEnhancementFactor,
      const SpinMode                    spinMode,
      const std::string &               xcType,
      std::shared_ptr<linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>>
                      linAlgOpContextHost,
      const MPI_Comm &mpiCommDomain)
      : d_mixingScheme(mixingScheme)
      , d_mixingHistory(mixingHistory)
      , d_linAlgOpContextHost(linAlgOpContextHost)
      , d_mpiCommDomain(mpiCommDomain)
    {
      utils::MemoryStorage<RealType, utils::MemorySpace::HOST> jxwStorage(
        jxwDataHost.size());
      jxwStorage.copyFrom(jxwDataHost);
      const double spinMixingParameter =
        mixingParameter * spinMixingEnhancementFactor;
      d_mixingScheme.addMixingVariable(mixingVariable::rho,
                                       jxwStorage,
                                       true,
                                       mixingParameter,
                                       isAdaptiveMixingParameter);
      if (spinMode != SpinMode::Unpolarized)
        {
          d_mixingScheme.addMixingVariable(mixingVariable::magZ,
                                           jxwStorage,
                                           true,
                                           spinMixingParameter,
                                           isAdaptiveMixingParameter);
          if (spinMode == SpinMode::NonCollinear)
            {
              d_mixingScheme.addMixingVariable(mixingVariable::magY,
                                               jxwStorage,
                                               true,
                                               spinMixingParameter,
                                               isAdaptiveMixingParameter);
              d_mixingScheme.addMixingVariable(mixingVariable::magX,
                                               jxwStorage,
                                               true,
                                               spinMixingParameter,
                                               isAdaptiveMixingParameter);
            }
        }
      if (xcType.rfind("GGA", 0) == 0)
        {
          d_mixingScheme.addMixingVariable(
            mixingVariable::gradRho,
            utils::MemoryStorage<RealType, utils::MemorySpace::HOST>(),
            false,
            mixingParameter,
            isAdaptiveMixingParameter);
          if (spinMode != SpinMode::Unpolarized)
            {
              d_mixingScheme.addMixingVariable(
                mixingVariable::gradMagZ,
                utils::MemoryStorage<RealType, utils::MemorySpace::HOST>(),
                false,
                spinMixingParameter,
                isAdaptiveMixingParameter);
              if (spinMode == SpinMode::NonCollinear)
                {
                  d_mixingScheme.addMixingVariable(
                    mixingVariable::gradMagY,
                    utils::MemoryStorage<RealType, utils::MemorySpace::HOST>(),
                    false,
                    spinMixingParameter,
                    isAdaptiveMixingParameter);
                  d_mixingScheme.addMixingVariable(
                    mixingVariable::gradMagX,
                    utils::MemoryStorage<RealType, utils::MemorySpace::HOST>(),
                    false,
                    spinMixingParameter,
                    isAdaptiveMixingParameter);
                }
            }
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Mixing<ValueType, memorySpace>::setRDM1(
      std::shared_ptr<RDM1<ValueType, memorySpace>> rdm1)
    {
      d_rdm1Ptr = rdm1;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    const RDM1<ValueType, memorySpace> &
    RDM1Mixing<ValueType, memorySpace>::getRDM1() const
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::getRDM1() called before setRDM1() or stored RDM1 has expired.");
      return *rdm1;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Mixing<ValueType, memorySpace>::mix()
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::mix() called but stored RDM1 has expired or setRDM1() was not called.");
      dftefe::utils::throwException<dftefe::utils::InvalidArgument>(
        !rdm1->isSOC(), "RDM1Mixing::mix() does not support SOC.");

      const SpinMode       sm          = rdm1->spinMode();
      const size_type      ncomp       = (sm == SpinMode::Unpolarized) ? 1 :
                                         (sm == SpinMode::Collinear)   ? 2 :
                                                                         4;
      const mixingVariable valVars[4]  = {mixingVariable::rho,
                                         mixingVariable::magZ,
                                         mixingVariable::magY,
                                         mixingVariable::magX};
      const mixingVariable gradVars[4] = {mixingVariable::gradRho,
                                          mixingVariable::gradMagZ,
                                          mixingVariable::gradMagY,
                                          mixingVariable::gradMagX};

      // Get output density (and gradient for GGA) from the stored rdm1.
      std::unordered_map<DensityDescrAttr, AttrStorage> densOutAttr;
      std::unordered_map<WfcDescrAttr, AttrStorage>     wfcAttr;
      std::set<DensityDescrAttr> descrSet = {DensityDescrAttr::Val};
      if (d_mixingScheme.hasVariable(mixingVariable::gradRho))
        descrSet.insert(DensityDescrAttr::Grad);
      rdm1->getDescriptors(descrSet, {}, densOutAttr, wfcAttr);

      AttrStorage &   densOut = densOutAttr.at(DensityDescrAttr::Val);
      const size_type nq      = densOut[0].nQuadraturePoints();

      // density components: residual = densOut - in; update history.
      for (size_type ic = 0; ic < ncomp; ++ic)
        {
          utils::MemoryStorage<double, utils::MemorySpace::HOST> residual(nq,
                                                                          0.0);
          {
            const double *outPtr = densOut[ic].data();
            const double *inPtr  = d_densityInAttrVals[ic].data();
            double *      resPtr = residual.data();
            for (size_type i = 0; i < nq; ++i)
              resPtr[i] = outPtr[i] - inPtr[i];
          }
          d_mixingScheme.template addVariableToInHist<utils::MemorySpace::HOST>(
            valVars[ic], d_densityInAttrVals[ic].data(), nq);
          d_mixingScheme
            .template addVariableToResidualHist<utils::MemorySpace::HOST>(
              valVars[ic], residual.data(), nq);
        }

      // grad density components: dependent variables (empty weights).
      if (d_mixingScheme.hasVariable(mixingVariable::gradRho))
        {
          AttrStorage &gradDensOut = densOutAttr.at(DensityDescrAttr::Grad);
          for (size_type ic = 0; ic < ncomp; ++ic)
            {
              const size_type nGrad = gradDensOut[ic].nEntries();
              utils::MemoryStorage<double, utils::MemorySpace::HOST>
                gradResidual(nGrad, 0.0);
              {
                const double *outPtr = gradDensOut[ic].data();
                const double *inPtr  = d_gradDensityInAttrVals[ic].data();
                double *      resPtr = gradResidual.data();
                for (size_type i = 0; i < nGrad; ++i)
                  resPtr[i] = outPtr[i] - inPtr[i];
              }
              d_mixingScheme
                .template addVariableToInHist<utils::MemorySpace::HOST>(
                  gradVars[ic], d_gradDensityInAttrVals[ic].data(), nGrad);
              d_mixingScheme
                .template addVariableToResidualHist<utils::MemorySpace::HOST>(
                  gradVars[ic], gradResidual.data(), nGrad);
            }
        }

      d_mixingScheme.popOldHistory(d_mixingHistory);

      // Anderson coefficients from all primary density variables.
      std::vector<mixingVariable> andersonVars(valVars, valVars + ncomp);
      d_mixingScheme.computeAndersonMixingCoeff(andersonVars,
                                                *d_linAlgOpContextHost);

      for (size_type ic = 0; ic < ncomp; ++ic)
        d_mixingScheme.template mixVariable<utils::MemorySpace::HOST>(
          valVars[ic], d_densityInAttrVals[ic].data(), nq);

      if (d_mixingScheme.hasVariable(mixingVariable::gradRho))
        {
          AttrStorage &gradDensOut = densOutAttr.at(DensityDescrAttr::Grad);
          for (size_type ic = 0; ic < ncomp; ++ic)
            {
              const size_type nGrad = gradDensOut[ic].nEntries();
              d_mixingScheme.template mixVariable<utils::MemorySpace::HOST>(
                gradVars[ic], d_gradDensityInAttrVals[ic].data(), nGrad);
            }
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Mixing<ValueType, memorySpace>::getDescriptors(
      const std::set<DensityDescrAttr> &densityAttrs,
      const std::set<WfcDescrAttr> & /*wfcAttrs*/,
      std::unordered_map<DensityDescrAttr, AttrStorage> &densityAttrVals,
      std::unordered_map<WfcDescrAttr, AttrStorage> & /*wfcAttrVals*/)
    {
      for (const auto &attr : densityAttrs)
        {
          if (attr == DensityDescrAttr::Val)
            densityAttrVals[attr] = d_densityInAttrVals;
          else if (attr == DensityDescrAttr::Grad &&
                   d_mixingScheme.hasVariable(mixingVariable::gradRho))
            densityAttrVals[attr] = d_gradDensityInAttrVals;
        }
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Mixing<ValueType, memorySpace>::setDescriptors(
      const std::unordered_map<DensityDescrAttr, AttrStorage> &densityAttrVals,
      const std::unordered_map<WfcDescrAttr, AttrStorage> & /*wfcAttrVals*/)
    {
      auto it = densityAttrVals.find(DensityDescrAttr::Val);
      if (it != densityAttrVals.end())
        d_densityInAttrVals = it->second;
      auto itG = densityAttrVals.find(DensityDescrAttr::Grad);
      if (itG != densityAttrVals.end())
        d_gradDensityInAttrVals = itG->second;
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Mixing<ValueType, memorySpace>::getDensityObs(
      const std::set<DensityObsAttr> &densityObsAttrs,
      std::unordered_map<DensityObsAttr, std::vector<std::vector<double>>>
        &densityObsAttrVals)
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::getDensityObs(): stored RDM1 has expired or setRDM1() was not called.");
      rdm1->getDensityObs(densityObsAttrs, densityObsAttrVals);
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    RDM1Mixing<ValueType, memorySpace>::setEvalDescrFlag(
      const bool /*evalFlag*/)
    {}

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    SpinMode
    RDM1Mixing<ValueType, memorySpace>::spinMode() const
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::spinMode(): stored RDM1 has expired or setRDM1() was not called.");
      return rdm1->spinMode();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    bool
    RDM1Mixing<ValueType, memorySpace>::isSOC() const
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::isSOC(): stored RDM1 has expired or setRDM1() was not called.");
      return rdm1->isSOC();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    size_type
    RDM1Mixing<ValueType, memorySpace>::getnKSOrbs() const
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::getnKSOrbs(): stored RDM1 has expired or setRDM1() was not called.");
      return rdm1->getnKSOrbs();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    RDM1Mixing<ValueType, memorySpace>::getkPointCoords() const
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::getkPointCoords(): stored RDM1 has expired or setRDM1() was not called.");
      return rdm1->getkPointCoords();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::vector<double>
    RDM1Mixing<ValueType, memorySpace>::getkPointWeights() const
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::getkPointWeights(): stored RDM1 has expired or setRDM1() was not called.");
      return rdm1->getkPointWeights();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    std::unique_ptr<RDM1<ValueType, memorySpace>>
    RDM1Mixing<ValueType, memorySpace>::clone() const
    {
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        false, "RDM1Mixing::clone() is not supported.");
      return nullptr;
    }

  } // namespace ksdft
} // namespace dftefe
