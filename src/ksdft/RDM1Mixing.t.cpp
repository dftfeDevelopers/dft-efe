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
      d_mixingScheme.addMixingVariable(mixingVariable::rho,
                                       jxwStorage,
                                       true,
                                       mixingParameter,
                                       isAdaptiveMixingParameter);
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
        !rdm1->isSpinPolarized() && !rdm1->isNonCollinear() && !rdm1->isSOC(),
        "RDM1Mixing::mix() only supports SpinMode::Unpolarized. "
        "Spin-polarized, non-collinear, and SOC mixing are not yet implemented.");

      // Get output density (and gradient for GGA) from the stored rdm1.
      std::unordered_map<DensityDescrAttr, AttrStorage> densOutAttr;
      std::unordered_map<WfcDescrAttr, AttrStorage>     wfcAttr;
      std::set<DensityDescrAttr> descrSet = {DensityDescrAttr::Val};
      if (d_mixingScheme.hasVariable(mixingVariable::gradRho))
        descrSet.insert(DensityDescrAttr::Grad);
      rdm1->getDescriptors(descrSet, {}, densOutAttr, wfcAttr);

      AttrStorage &   densOut = densOutAttr.at(DensityDescrAttr::Val);
      const size_type nq      = densOut[0].nQuadraturePoints();

      if (d_densityInAttrVals.empty())
        {
          // First call: adopt densOut (and gradDensOut) directly.
          d_densityInAttrVals = densOut;
          if (d_mixingScheme.hasVariable(mixingVariable::gradRho))
            d_gradDensityInAttrVals = densOutAttr.at(DensityDescrAttr::Grad);
        }
      else
        {
          // rho: residual = densOut - in; update history; mix.
          utils::MemoryStorage<double, utils::MemorySpace::HOST> rhoResidual(
            nq, 0.0);
          {
            const double *outPtr = densOut[0].data();
            const double *inPtr  = d_densityInAttrVals[0].data();
            double *      resPtr = rhoResidual.data();
            for (size_type i = 0; i < nq; ++i)
              resPtr[i] = outPtr[i] - inPtr[i];
          }
          d_mixingScheme.template addVariableToInHist<utils::MemorySpace::HOST>(
            mixingVariable::rho, d_densityInAttrVals[0].data(), nq);
          d_mixingScheme
            .template addVariableToResidualHist<utils::MemorySpace::HOST>(
              mixingVariable::rho, rhoResidual.data(), nq);

          // gradRho: same pattern, dependent variable (same coefficients as
          // rho).
          if (d_mixingScheme.hasVariable(mixingVariable::gradRho))
            {
              AttrStorage &gradDensOut = densOutAttr.at(DensityDescrAttr::Grad);
              const size_type nGrad    = gradDensOut[0].nEntries();
              utils::MemoryStorage<double, utils::MemorySpace::HOST>
                gradResidual(nGrad, 0.0);
              {
                const double *outPtr = gradDensOut[0].data();
                const double *inPtr  = d_gradDensityInAttrVals[0].data();
                double *      resPtr = gradResidual.data();
                for (size_type i = 0; i < nGrad; ++i)
                  resPtr[i] = outPtr[i] - inPtr[i];
              }
              d_mixingScheme
                .template addVariableToInHist<utils::MemorySpace::HOST>(
                  mixingVariable::gradRho,
                  d_gradDensityInAttrVals[0].data(),
                  nGrad);
              d_mixingScheme
                .template addVariableToResidualHist<utils::MemorySpace::HOST>(
                  mixingVariable::gradRho, gradResidual.data(), nGrad);
            }

          d_mixingScheme.popOldHistory(d_mixingHistory);

          // Coefficients determined by rho residual only.
          d_mixingScheme.computeAndersonMixingCoeff({mixingVariable::rho},
                                                    *d_linAlgOpContextHost);

          d_mixingScheme.template mixVariable<utils::MemorySpace::HOST>(
            mixingVariable::rho, d_densityInAttrVals[0].data(), nq);
          if (d_mixingScheme.hasVariable(mixingVariable::gradRho))
            {
              AttrStorage &gradDensOut = densOutAttr.at(DensityDescrAttr::Grad);
              const size_type nGrad    = gradDensOut[0].nEntries();
              d_mixingScheme.template mixVariable<utils::MemorySpace::HOST>(
                mixingVariable::gradRho,
                d_gradDensityInAttrVals[0].data(),
                nGrad);
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
    bool
    RDM1Mixing<ValueType, memorySpace>::isSpinPolarized() const
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::isSpinPolarized(): stored RDM1 has expired or setRDM1() was not called.");
      return rdm1->isSpinPolarized();
    }

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    bool
    RDM1Mixing<ValueType, memorySpace>::isNonCollinear() const
    {
      auto rdm1 = d_rdm1Ptr.lock();
      dftefe::utils::throwException<dftefe::utils::LogicError>(
        rdm1 != nullptr,
        "RDM1Mixing::isNonCollinear(): stored RDM1 has expired or setRDM1() was not called.");
      return rdm1->isNonCollinear();
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
