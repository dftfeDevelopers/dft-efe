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

#include <utils/DataTypeOverloads.h>
#include <ksdft/Defaults.h>
#include <ksdft/RDM1FE.h>
#include <ksdft/RDM1Mixing.h>
#include <quadrature/QuadratureValuesContainer.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::
      ExchangeCorrelationFE(
        RDM1<ValueType, memorySpace>                                     &rdm1,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize)
      : d_cellBlockSize(cellBlockSize)
      , d_linAlgOpContext(linAlgOpContext)
    {
      d_funcX = new xc_func_type;
      d_funcC = new xc_func_type;
      int         err;
      std::string msg;
      err = xc_func_init(d_funcX, 1, XC_UNPOLARIZED);
      msg = "LDA Exchange Functional not found\n";
      utils::throwException(err == 0, msg);
      err = xc_func_init(d_funcC, 12, XC_UNPOLARIZED);
      msg = "LDA Correlation Functional not found\n";
      utils::throwException(err == 0, msg);
      xc_func_set_dens_threshold(d_funcX, LibxcDefaults::DENSITY_ZERO_TOL);
      xc_func_set_dens_threshold(d_funcC, LibxcDefaults::DENSITY_ZERO_TOL);

      reinitBasis(rdm1);
      reinitField(rdm1);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::
      ExchangeCorrelationFE(
        RDM1<ValueType, memorySpace>                                     &rdm1,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                        atomSphericalDataContainerPSP,
        const std::vector<std::string> &atomSymbolVec,
        const std::vector<utils::Point> &atomCoordinates)
      : d_cellBlockSize(cellBlockSize)
      , d_linAlgOpContext(linAlgOpContext)
    {
      d_funcX = new xc_func_type;
      d_funcC = new xc_func_type;
      int         err;
      std::string msg;
      err = xc_func_init(d_funcX, 1, XC_UNPOLARIZED);
      msg = "LDA Exchange Functional not found\n";
      utils::throwException(err == 0, msg);
      err = xc_func_init(d_funcC, 12, XC_UNPOLARIZED);
      msg = "LDA Correlation Functional not found\n";
      utils::throwException(err == 0, msg);
      xc_func_set_dens_threshold(d_funcX, LibxcDefaults::DENSITY_ZERO_TOL);
      xc_func_set_dens_threshold(d_funcC, LibxcDefaults::DENSITY_ZERO_TOL);

      reinitBasis(rdm1);

      auto quadRuleContainer =
        d_feBasisDataStorage->getQuadratureRuleContainer();

      d_coreCorrDensUPF = std::make_shared<
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>(
        quadRuleContainer, 1, (RealType)0.0);

      const atoms::AtomSevereFunction<memorySpace>
        rhoCoreCorrection(atomSphericalDataContainerPSP,
                          atomSymbolVec,
                          atomCoordinates,
                          "nlcc",
                          0,
                          1,
                          1,
                          linAlgOpContext.get());

      utils::MemoryStorage<RealType, memorySpace>
        coreCorrDensUPFMemspace(quadRuleContainer->nQuadraturePoints());

      rhoCoreCorrection.template eval<memorySpace>(
        quadRuleContainer->nQuadraturePoints(),
        quadRuleContainer->template getRealPointsPtr<memorySpace>(),
        coreCorrDensUPFMemspace.data());

      utils::MemoryTransfer<memorySpaceHost, memorySpace> memTrans;
      memTrans.copy(coreCorrDensUPFMemspace.size(),
                    d_coreCorrDensUPF->begin(),
                    coreCorrDensUPFMemspace.data());

      reinitField(rdm1);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::~ExchangeCorrelationFE()
    {
      if (d_funcX != nullptr)
        {
          xc_func_end(d_funcX);
          delete d_funcX;
          d_funcX = nullptr;
        }
      if (d_funcC != nullptr)
        {
          xc_func_end(d_funcC);
          delete d_funcC;
          d_funcC = nullptr;
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::reinitBasis(RDM1<ValueType, memorySpace> &rdm1)
    {
      using RDM1FEType = RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>;

      const RDM1FEType *rdm1FEPtr = dynamic_cast<const RDM1FEType *>(&rdm1);

      if (rdm1FEPtr == nullptr)
        {
          const auto *mixPtr =
            dynamic_cast<const RDM1Mixing<ValueType, memorySpace> *>(&rdm1);
          if (mixPtr != nullptr)
            rdm1FEPtr =
              dynamic_cast<const RDM1FEType *>(&mixPtr->getRDM1());
        }

      utils::throwException(
        rdm1FEPtr != nullptr,
        "ExchangeCorrelationFE::reinitBasis: could not resolve RDM1 to RDM1FE. "
        "Pass an RDM1FE or an RDM1Mixing wrapping one.");

      auto feBasisDataStorage = rdm1FEPtr->getFEBasisDataStorage();
      d_feBasisDataStorage    = feBasisDataStorage;
      d_xcPotentialQuad    = std::make_shared<
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>(
        feBasisDataStorage->getQuadratureRuleContainer(), 1);
      d_xcPotentialQuadMemspace = std::make_shared<
        quadrature::QuadratureValuesContainer<RealType, memorySpace>>(
        feBasisDataStorage->getQuadratureRuleContainer(), 1);
      d_feBasisOp =
        std::make_shared<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>>(
          feBasisDataStorage,
          d_cellBlockSize * d_xcPotentialQuad->getNumberComponents());

      std::shared_ptr<const basis::BasisDofHandler> basisDofHandlerData =
        feBasisDataStorage->getBasisDofHandler();
      d_feBasisDofHandler = std::dynamic_pointer_cast<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
        basisDofHandlerData);
      utils::throwException(
        d_feBasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input Field to FEBasisDofHandler "
        "in ExchangeCorrelationFE::reinitBasis()");
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::reinitField(RDM1<ValueType, memorySpace> &rdm1)
    {
      using AttrStorage = typename RDM1<ValueType, memorySpace>::AttrStorage;
      using RDM1FEType = RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>;
      const RDM1FEType *p = dynamic_cast<const RDM1FEType *>(&rdm1);
      if (p == nullptr)
        {
          const auto *mix =
            dynamic_cast<const RDM1Mixing<ValueType, memorySpace> *>(&rdm1);
          if (mix != nullptr)
            p = dynamic_cast<const RDM1FEType *>(&mix->getRDM1());
        }
      utils::throwException(
        p != nullptr &&
          p->getFEBasisDataStorage() == d_feBasisDataStorage,
        "ExchangeCorrelationFE::reinitField: RDM1 uses a different "
        "FEBasisDataStorage than the one set in reinitBasis(). "
        "Call reinitBasis(rdm1) whenever the basis is reinitialised.");

      std::unordered_map<DensityDescrAttr, AttrStorage> densAttr;
      std::unordered_map<WfcDescrAttr, AttrStorage>     wfcAttr;
      rdm1.getDescriptors({DensityDescrAttr::Val}, {}, densAttr, wfcAttr);
      const auto &rhoQuad = densAttr.at(DensityDescrAttr::Val)[0];
      utils::throwException(
        rhoQuad.getQuadratureRuleContainer() ==
          d_feBasisDataStorage->getQuadratureRuleContainer(),
        "ExchangeCorrelationFE::reinitField: density Quad from "
        "RDM1::getDescriptors has a different QuadratureRuleContainer "
        "than d_feBasisDataStorage.");

      const size_type lenRho =
        d_xcPotentialQuad->getQuadratureRuleContainer()->nQuadraturePoints();

      utils::MemoryStorage<RealType, utils::MemorySpace::HOST> rho(lenRho),
        vcRho(lenRho), vxRho(lenRho);
      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpaceHost>
        memoryTransfer;
      memoryTransfer.copy(lenRho, rho.data(), rhoQuad.begin());
      if (d_coreCorrDensUPF != nullptr)
        {
          const RealType *corrPtr = d_coreCorrDensUPF->begin();
          for (size_type i = 0; i < lenRho; ++i)
            rho.data()[i] += corrPtr[i];
        }
      xc_lda_vxc(d_funcX, lenRho, rho.data(), vxRho.data());
      xc_lda_vxc(d_funcC, lenRho, rho.data(), vcRho.data());

      size_type count = 0;
      for (size_type iCell = 0; iCell < d_xcPotentialQuad->nCells(); iCell++)
        {
          std::vector<RealType> a(
            d_xcPotentialQuad->getQuadratureRuleContainer()
              ->nCellQuadraturePoints(iCell));
          for (size_type quadId = 0; quadId < d_xcPotentialQuad->getQuadratureRuleContainer()
                ->nCellQuadraturePoints(iCell); quadId++)
            {
              a[quadId] = *(vxRho.data() + count) + *(vcRho.data() + count);
              count += 1;
            }
          d_xcPotentialQuad->template setCellValues<utils::MemorySpace::HOST>(
            iCell, a.data());
        }

      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
        memoryTransferH2M;
      memoryTransferH2M.copy(d_xcPotentialQuad->nEntries(),
                             d_xcPotentialQuadMemspace->data(),
                             d_xcPotentialQuad->data());
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getLocal(Storage &cellWiseStorage) const
    {
      d_feBasisOp->computeFEMatrices(basis::realspace::LinearLocalOp::IDENTITY,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::LinearLocalOp::IDENTITY,
                                     *d_xcPotentialQuadMemspace,
                                     cellWiseStorage,
                                     *d_linAlgOpContext);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::evalEnergy(RDM1<ValueType, memorySpace> &rdm1,
                                           const utils::mpi::MPIComm    &comm)
    {
      using AttrStorage = typename RDM1<ValueType, memorySpace>::AttrStorage;
      using RDM1FEType =
        RDM1FE<ValueTypeBasisData, ValueTypeBasisCoeff, memorySpace, dim>;
      const RDM1FEType *p = dynamic_cast<const RDM1FEType *>(&rdm1);
      if (p == nullptr)
        {
          const auto *mix =
            dynamic_cast<const RDM1Mixing<ValueType, memorySpace> *>(&rdm1);
          if (mix != nullptr)
            p = dynamic_cast<const RDM1FEType *>(&mix->getRDM1());
        }
      utils::throwException(
        p != nullptr &&
          p->getFEBasisDataStorage() == d_feBasisDataStorage,
        "ExchangeCorrelationFE::evalEnergy: RDM1 uses a different "
        "FEBasisDataStorage than the one set in reinitBasis(). "
        "Call reinitBasis(rdm1) whenever the basis is reinitialised.");

      std::unordered_map<DensityDescrAttr, AttrStorage> densAttr;
      std::unordered_map<WfcDescrAttr, AttrStorage>     wfcAttr;
      rdm1.getDescriptors({DensityDescrAttr::Val}, {}, densAttr, wfcAttr);
      const auto &rhoQuad = densAttr.at(DensityDescrAttr::Val)[0];
      utils::throwException(
        rhoQuad.getQuadratureRuleContainer() ==
          d_feBasisDataStorage->getQuadratureRuleContainer(),
        "ExchangeCorrelationFE::evalEnergy: density Quad from "
        "RDM1::getDescriptors has a different QuadratureRuleContainer "
        "than d_feBasisDataStorage.");

      const size_type lenRho =
        d_xcPotentialQuad->getQuadratureRuleContainer()->nQuadraturePoints();

      utils::MemoryStorage<RealType, utils::MemorySpace::HOST> rho(lenRho),
        ecRho(lenRho), exRho(lenRho);
      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpaceHost>
        memoryTransfer;
      memoryTransfer.copy(lenRho, rho.data(), rhoQuad.begin());
      if (d_coreCorrDensUPF != nullptr)
        {
          const RealType *corrPtr = d_coreCorrDensUPF->begin();
          for (size_type i = 0; i < lenRho; ++i)
            rho.data()[i] += corrPtr[i];
        }
      xc_lda_exc(d_funcX, lenRho, rho.data(), exRho.data());
      xc_lda_exc(d_funcC, lenRho, rho.data(), ecRho.data());

      // ∫ (εx + εc) * (ρ_val + ρ_core) dV
      const auto &    jxw    = d_feBasisDataStorage->getQuadratureRuleContainer()->getJxW();
      const RealType *rhoPtr = rho.data();
      RealType        totalEnergy = (RealType)0;
      for (size_type i = 0; i < lenRho; ++i)
        totalEnergy +=
          (*(exRho.data() + i) + *(ecRho.data() + i)) * rhoPtr[i] * jxw[i];

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &totalEnergy,
        1,
        utils::mpi::Types<RealType>::getMPIDatatype(),
        utils::mpi::MPISum,
        comm);

      d_energy = totalEnergy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename ExchangeCorrelationFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   memorySpace,
                                   dim>::RealType
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getEnergy() const
    {
      return d_energy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const quadrature::QuadratureValuesContainer<
      typename ExchangeCorrelationFE<ValueTypeBasisData,
                                     ValueTypeBasisCoeff,
                                     memorySpace,
                                     dim>::ValueType,
      memorySpace> &
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getFunctionalDerivative() const
    {
      return *d_xcPotentialQuadMemspace;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const
    {
      utils::throwException(
        false,
        "Non-Local component not present to call in ExchangeCorrelationFE.h");
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::hasLocalComponent() const
    {
      return true;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::hasNonLocalComponent() const
    {
      return false;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                   ValueTypeBasisData,
                                                   memorySpace,
                                                   dim>>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getHamiltonianFEBasisOperations() const
    {
      return d_feBasisOp;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::getLinAlgOpContext() const
    {
      return d_linAlgOpContext;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
