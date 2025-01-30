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

namespace dftefe
{
  namespace ksdft
  {
    namespace ExchangeCorrelationFEInternal
    {
      /* Assumption : field and rho have numComponents = 1 */
      template <typename ValueTypeBasisData,
                typename ValueTypeBasisCoeff,
                utils::MemorySpace memorySpace,
                size_type          dim>
      typename ExchangeCorrelationFE<ValueTypeBasisData,
                                     ValueTypeBasisCoeff,
                                     memorySpace,
                                     dim>::RealType
      getIntegralFieldTimesRho(
        const quadrature::QuadratureValuesContainer<
          typename ExchangeCorrelationFE<ValueTypeBasisData,
                                         ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::RealType,
          memorySpace> &field,
        const quadrature::QuadratureValuesContainer<
          typename ExchangeCorrelationFE<ValueTypeBasisData,
                                         ValueTypeBasisCoeff,
                                         memorySpace,
                                         dim>::RealType,
          memorySpace> &                                             rho,
        const utils::MemoryStorage<ValueTypeBasisData, memorySpace> &jxwStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm)
      {
        using RealType = typename ExchangeCorrelationFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpace,
                                                        dim>::RealType;

        RealType        value                = 0;
        const RealType *fieldIter            = field.begin();
        const RealType *rhoIter              = rho.begin();
        const RealType *jxwStorageIter       = jxwStorage.data();
        size_type       cumulativeQuadInCell = 0;

        for (size_type iCell = 0; iCell < field.nCells(); iCell++)
          {
            size_type numQuadInCell = field.nCellQuadraturePoints(iCell);
            for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
              {
                const RealType jxwVal =
                  jxwStorageIter[cumulativeQuadInCell + iQuad];
                const RealType fieldVal =
                  fieldIter[cumulativeQuadInCell + iQuad];
                const RealType rhoVal = rhoIter[cumulativeQuadInCell + iQuad];
                value += rhoVal * fieldVal * jxwVal;
              }
            cumulativeQuadInCell += numQuadInCell;
          }

        // quadrature::QuadratureValuesContainer<RealType, memorySpace>
        // fieldxrho(
        //   field);

        // linearAlgebra::blasLapack::
        //   hadamardProduct<RealType, RealType, memorySpace>(
        //     field.nEntries(),
        //     field.begin(),
        //     rho.begin(),
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     fieldxrho.begin(),
        //     *linAlgOpContext);

        // linearAlgebra::blasLapack::
        //   hadamardProduct<RealType, RealType, memorySpace>(
        //     fieldxrho.nEntries(),
        //     fieldxrho.begin(),
        //     jxwStorage.data(),
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     fieldxrho.begin(),
        //     *linAlgOpContext);

        // for (size_type iCell = 0; iCell < fieldxrho.nCells(); iCell++)
        //   {
        //     std::vector<RealType> a(
        //       fieldxrho.getQuadratureRuleContainer()->nCellQuadraturePoints(
        //         iCell));
        //     fieldxrho.template getCellValues<utils::MemorySpace::HOST>(
        //       iCell, a.data());
        //     value += std::accumulate(a.begin(), a.end(), (RealType)0);
        //   }

        int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          &value,
          1,
          utils::mpi::Types<RealType>::getMPIDatatype(),
          utils::mpi::MPISum,
          comm);

        return value;
      }
    } // namespace ExchangeCorrelationFEInternal

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ExchangeCorrelationFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          memorySpace,
                          dim>::
      ExchangeCorrelationFE(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize)
      : d_cellBlockSize(cellBlockSize)
      , d_linAlgOpContext(linAlgOpContext)
    {
      d_funcX = new xc_func_type;
      d_funcC = new xc_func_type;
      int err;
      err             = xc_func_init(d_funcX,
                         1,
                         XC_UNPOLARIZED); // LDA_X (id=1): Slater exchange
      std::string msg = "LDA Exchange Functional not found\n";
      utils::throwException(err == 0, msg);
      err = xc_func_init(d_funcC,
                         12,
                         XC_UNPOLARIZED); // LDA_C_PW (id=12): Perdew & Wang
      msg = "LDA Correlation Functional not found\n";
      utils::throwException(err == 0, msg);
      xc_func_set_dens_threshold(
        d_funcX, LibxcDefaults::DENSITY_ZERO_TOL); // makes e and v zero if \rho
                                                   // < rho_zero_tol
      xc_func_set_dens_threshold(
        d_funcC, LibxcDefaults::DENSITY_ZERO_TOL); // makes e and v zero if \rho
                                                   // < rho_zero_tol

      reinitBasis(feBasisDataStorage);
      reinitField(electronChargeDensity);
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
      if (d_rho != nullptr)
        {
          delete d_rho;
          d_rho = nullptr;
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
                          dim>::
      reinitBasis(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBasisDataStorage)
    {
      d_feBasisDataStorage = feBasisDataStorage;
      d_xcPotentialQuad    = std::make_shared<
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
        "in FEBasisOperations ExchangeCorrelationFE()");

      d_rho = new utils::MemoryStorage<RealType, utils::MemorySpace::HOST>(
        d_xcPotentialQuad->getQuadratureRuleContainer()->nQuadraturePoints());
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
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity) /*Assumes rho has 1 component*/
    {
      d_electronChargeDensity = &electronChargeDensity;
      size_type lenRho = d_electronChargeDensity->getQuadratureRuleContainer()
                           ->nQuadraturePoints();

      utils::throwException(
        d_electronChargeDensity->getQuadratureRuleContainer() ==
          d_xcPotentialQuad->getQuadratureRuleContainer(),
        "The electron density should have same quadRuleContainer as input BasisDataStorage.");

      /*
       * Compute exc (energy density) and vxc (dexc/d\rho) from libxc
       * note for spin up and down the parameters change as
       * xc_lda_vxc(d_funcX,nPoints, nPoints, rhoUp.data(),
       * rhoDown.data(),vxRho.data()); vx or vc has length 2*nPoints , where  as
       * exc will be still nPoints length.
       */

      utils::MemoryStorage<RealType, utils::MemorySpace::HOST> vcRho(lenRho),
        vxRho(lenRho);

      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>
        memoryTransfer;
      memoryTransfer.copy(lenRho, d_rho->data(), electronChargeDensity.begin());

      xc_lda_vxc(d_funcX, lenRho, d_rho->data(), vxRho.data());
      xc_lda_vxc(d_funcC, lenRho, d_rho->data(), vcRho.data());

      int count = 0;
      for (size_type iCell = 0; iCell < electronChargeDensity.nCells(); iCell++)
        {
          for (int quadId = 0;
               quadId < electronChargeDensity.getQuadratureRuleContainer()
                          ->nCellQuadraturePoints(iCell);
               quadId++)
            {
              RealType  a = *(vxRho.data() + count) + *(vcRho.data() + count);
              RealType *b = &a;
              d_xcPotentialQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              count += 1;
            }
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
                          dim>::getLocal(Storage &cellWiseStorage) const
    {
      d_feBasisOp->computeFEMatrices(basis::realspace::LinearLocalOp::IDENTITY,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::LinearLocalOp::IDENTITY,
                                     *d_xcPotentialQuad,
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
                          dim>::evalEnergy(const utils::mpi::MPIComm &comm)
    {
      auto jxwStorage = d_feBasisDataStorage->getJxWInAllCells();

      size_type lenRho = d_electronChargeDensity->getQuadratureRuleContainer()
                           ->nQuadraturePoints();

      utils::MemoryStorage<RealType, utils::MemorySpace::HOST> ecRho(lenRho),
        exRho(lenRho);

      xc_lda_exc(d_funcX, lenRho, d_rho->data(), exRho.data());
      xc_lda_exc(d_funcC, lenRho, d_rho->data(), ecRho.data());

      int count = 0;
      for (size_type iCell = 0; iCell < d_electronChargeDensity->nCells();
           iCell++)
        {
          size_type quadId = 0;
          for (int quadId = 0;
               quadId < d_electronChargeDensity->getQuadratureRuleContainer()
                          ->nCellQuadraturePoints(iCell);
               quadId++)
            {
              RealType  a = *(exRho.data() + count) + *(ecRho.data() + count);
              RealType *b = &a;
              d_xcPotentialQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              count += 1;
            }
        }

      RealType totalEnergy =
        ExchangeCorrelationFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          memorySpace,
          dim>(*d_xcPotentialQuad,
               *d_electronChargeDensity,
               jxwStorage,
               d_linAlgOpContext,
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

  } // end of namespace ksdft
} // end of namespace dftefe
