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

#include <ksdft/Defaults.h>
#include <utils/SmearChargePotentialFunction.h>
#include <utils/SmearChargeDensityFunction.h>
#include <basis/FEBasisDofHandler.h>
namespace dftefe
{
  namespace ksdft
  {
    namespace ElectrostaticAllElectronFEInternal
    {
      template <typename ValueTypeBasisData,
                typename ValueTypeBasisCoeff,
                utils::MemorySpace memorySpace,
                size_type          dim>
      std::vector<typename ElectrostaticFE<ValueTypeBasisData,
                                           ValueTypeBasisCoeff,
                                           memorySpace,
                                           dim>::RealType>
      getIntegralFieldTimesRho(
        const quadrature::QuadratureValuesContainer<
          typename ElectrostaticFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   memorySpace,
                                   dim>::RealType,
          memorySpace> &field,
        const quadrature::QuadratureValuesContainer<
          typename ElectrostaticFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   memorySpace,
                                   dim>::RealType,
          memorySpace> &                                             rho,
        const utils::MemoryStorage<ValueTypeBasisData, memorySpace> &jxwStorage,
        size_type numLocallyOwnedCells,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm)
      {
        using RealType = typename ElectrostaticFE<ValueTypeBasisData,
                                                  ValueTypeBasisCoeff,
                                                  memorySpace,
                                                  dim>::RealType;

        quadrature::QuadratureValuesContainer<RealType, memorySpace> fieldxrho(
          field);
        quadrature::QuadratureValuesContainer<RealType, memorySpace>
          fieldxrhoxJxW(field);

        linearAlgebra::blasLapack::
          hadamardProduct<RealType, RealType, memorySpace>(
            field.nEntries(),
            field.begin(),
            rho.begin(),
            linearAlgebra::blasLapack::ScalarOp::Identity,
            linearAlgebra::blasLapack::ScalarOp::Identity,
            fieldxrho.begin(),
            *linAlgOpContext);

        std::vector<linearAlgebra::blasLapack::ScalarOp> scalarOpA(
          numLocallyOwnedCells, linearAlgebra::blasLapack::ScalarOp::Identity);
        std::vector<linearAlgebra::blasLapack::ScalarOp> scalarOpB(
          numLocallyOwnedCells, linearAlgebra::blasLapack::ScalarOp::Identity);
        std::vector<size_type> mTmp(numLocallyOwnedCells, 0);
        std::vector<size_type> nTmp(numLocallyOwnedCells, 0);
        std::vector<size_type> kTmp(numLocallyOwnedCells, 0);
        std::vector<size_type> stATmp(numLocallyOwnedCells, 0);
        std::vector<size_type> stBTmp(numLocallyOwnedCells, 0);
        std::vector<size_type> stCTmp(numLocallyOwnedCells, 0);

        for (size_type iCell = 0; iCell < numLocallyOwnedCells; iCell++)
          {
            mTmp[iCell] = 1;
            nTmp[iCell] = fieldxrho.getNumberComponents();
            kTmp[iCell] =
              fieldxrho.getQuadratureRuleContainer()->nCellQuadraturePoints(
                iCell);
            stATmp[iCell] = mTmp[iCell] * kTmp[iCell];
            stBTmp[iCell] = nTmp[iCell] * kTmp[iCell];
            stCTmp[iCell] = mTmp[iCell] * nTmp[iCell] * kTmp[iCell];
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
          memoryTransferHost;

        utils::MemoryStorage<size_type, memorySpace> mSize(
          numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> nSize(
          numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> kSize(
          numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> stA(numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> stB(numLocallyOwnedCells);
        utils::MemoryStorage<size_type, memorySpace> stC(numLocallyOwnedCells);
        memoryTransferHost.copy(numLocallyOwnedCells,
                                mSize.data(),
                                mTmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                nSize.data(),
                                nTmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                kSize.data(),
                                kTmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                stA.data(),
                                stATmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                stB.data(),
                                stBTmp.data());
        memoryTransferHost.copy(numLocallyOwnedCells,
                                stC.data(),
                                stCTmp.data());

        linearAlgebra::blasLapack::
          scaleStridedVarBatched<ValueTypeBasisData, RealType, memorySpace>(
            numLocallyOwnedCells,
            scalarOpA.data(),
            scalarOpB.data(),
            stA.data(),
            stB.data(),
            stC.data(),
            mSize.data(),
            nSize.data(),
            kSize.data(),
            jxwStorage.data(),
            fieldxrho.begin(),
            fieldxrhoxJxW.begin(),
            *linAlgOpContext);

        std::vector<RealType> value(fieldxrho.getNumberComponents());

        for (size_type iCell = 0; iCell < fieldxrhoxJxW.nCells(); iCell++)
          {
            for (size_type iComp = 0; iComp < fieldxrho.getNumberComponents();
                 iComp++)
              {
                std::vector<RealType> a(
                  fieldxrho.getQuadratureRuleContainer()->nCellQuadraturePoints(
                    iCell));
                fieldxrhoxJxW
                  .template getCellQuadValues<utils::MemorySpace::HOST>(
                    iCell, iComp, a.data());
                value[iComp] +=
                  std::accumulate(a.begin(), a.end(), (RealType)0);
              }
          }

        int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          value.data(),
          value.size(),
          utils::mpi::Types<RealType>::getMPIDatatype(),
          utils::mpi::MPISum,
          comm);

        return value;
      }
    } // namespace ElectrostaticAllElectronFEInternal

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::
      ElectrostaticAllElectronFE(
        std::vector<utils::Point> atomCoordinates,
        std::vector<double>       atomCharges,
        std::vector<double>       smearedChargeRadius,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &                                               electronChargeDensity,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeRhs,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellTimesNumVecs)
      : d_atomCharges(atomCharges)
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(atomCharges.size(), nullptr)
      , d_feBMNuclearCharge(atomCharges.size(), nullptr)
    {
      reinitBasis(atomCoordinates,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDTotalChargeRhs);

      reinitField(electronChargeDensity);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::~ElectrostaticAllElectronFE()
    {
      if (d_totalChargeDensity == nullptr)
        {
          delete d_totalChargeDensity;
          d_totalChargeDensity = nullptr;
        }
      if (d_nuclearChargesDensity == nullptr)
        {
          delete d_nuclearChargesDensity;
          d_nuclearChargesDensity = nullptr;
        }
      if (d_totalChargePotentialQuad == nullptr)
        {
          delete d_totalChargePotentialQuad;
          d_totalChargePotentialQuad = nullptr;
        }
      for (auto &i : d_nuclearChargesPotential)
        {
          if (i == nullptr)
            {
              delete i;
              i = nullptr;
            }
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::
      reinitBasis(
        std::vector<utils::Point>                         atomCoordinates,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeRhs)
    {
      d_atomCoordinates                = atomCoordinates;
      d_feBDTotalChargeRhs             = feBDTotalChargeRhs;
      d_feBMTotalCharge                = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;
      d_feBasisOp =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(
          d_feBDTotalChargeRhs,
          ksdft::PoissonProblemDefaults::MAX_CELL_TIMES_NUMVECS);


      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBDTotalChargeRhs->getQuadratureRuleContainer();

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainer, d_numComponents);

      const utils::SmearChargeDensityFunction smfunc(d_atomCoordinates,
                                                     d_atomCharges,
                                                     d_smearedChargeRadius);

      for (size_type iCell = 0; iCell < quadRuleContainer->nCells(); iCell++)
        {
          for (size_type iComp = 0; iComp < d_numComponents; iComp++)
            {
              size_type             quadId = 0;
              std::vector<RealType> a(
                quadRuleContainer->nCellQuadraturePoints(iCell));
              for (auto j : quadRuleContainer->getCellRealPoints(iCell))
                {
                  a[quadId] = (RealType)smfunc(j);
                  quadId    = quadId + 1;
                }
              RealType *b = a.data();
              d_nuclearChargesDensity
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       iComp,
                                                                       b);
            }
        }

      // create the correction quadValuesContainer
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity)
    {
      utils::throwException(
        electronChargeDensity.getQuadratureRuleContainer() ==
          d_feBDTotalChargeRhs->getQuadratureRuleContainer(),
        "The electronic and total charge rhs for poisson solve quadrature rule"
        " should be same, otherwise they cannot be added in ElectrostaticFE reinitField()");

      /*---- solve poisson problem for b+rho system ---*/

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBDTotalChargeRhs->getQuadratureRuleContainer();

      // Init the phi_el multivector
      linearAlgebra::MultiVector<ValueType, memorySpace> totalChargePotential(
        d_feBMTotalCharge->getMPIPatternP2P(),
        d_linAlgOpContext,
        d_numComponents);

      // add nuclear and electron charge densities
      d_totalChargeDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainer, d_numComponents);

      d_totalChargePotentialQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainer, d_numComponents);

      quadrature::add((RealType)1.0,
                      *d_nuclearChargesDensity,
                      (RealType)1.0,
                      electronChargeDensity,
                      *d_totalChargeDensity,
                      *d_linAlgOpContext);

      // Scale by 4\pi
      quadrature::scale((RealType)(4 * utils::mathConstants::pi),
                        *d_totalChargeDensity,
                        *d_linAlgOpContext);

      std::shared_ptr<linearAlgebra::LinearSolverFunction<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace>>
        linearSolverFunction = std::make_shared<
          electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpace,
                                                        dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBDTotalChargeRhs,
          *d_totalChargeDensity,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContext,
          ksdft::PoissonProblemDefaults::MAX_CELL_TIMES_NUMVECS);

      linearAlgebra::LinearAlgebraProfiler profiler;

      std::shared_ptr<linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpace>>
        CGSolve =
          std::make_shared<linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                         ValueTypeBasisCoeff,
                                                         memorySpace>>(
            ksdft::PoissonProblemDefaults::MAX_ITER,
            ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
            ksdft::PoissonProblemDefaults::RELATIVE_TOL,
            ksdft::PoissonProblemDefaults::DIVERGENCE_TOL,
            profiler);

      CGSolve->solve(*linearSolverFunction);

      linearSolverFunction->getSolution(totalChargePotential);

      d_feBasisOp->interpolate(totalChargePotential,
                               *d_feBMTotalCharge,
                               *d_totalChargePotentialQuad);

      // add the correction quadValuesCOntainer for potential to
      // d_totalChargePotentialQuad
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::getLocal(Storage &cellWiseStorage) const
    {
      d_feBasisOp->computeFEMatrices(basis::realspace::LinearLocalOp::IDENTITY,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::VectorMathOp::MULT,
                                     basis::realspace::LinearLocalOp::IDENTITY,
                                     *d_totalChargePotentialQuad,
                                     cellWiseStorage,
                                     *d_linAlgOpContext);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::
      nuclearPotentialSolve(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs)
    {
      // Solve poisson problem for individual atoms
      std::shared_ptr<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>
        feBDHNuclearCharge = std::dynamic_pointer_cast<
          const basis::
            FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
          feBDNuclearChargeRhs->getBasisDofHandler());
      utils::throwException(
        feBDHNuclearCharge != nullptr,
        "Could not cast BasisDofHandler of the input Field to FEBasisDofHandler "
        "in ElectrostaticAllElectronFE.evalEnergy()");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = feBDNuclearChargeRhs->getQuadratureRuleContainer();

      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        nuclearChargeDensity(quadRuleContainer, d_numComponents);

      for (unsigned int iAtom = 0; iAtom < d_atomCoordinates.size(); iAtom++)
        {
          std::shared_ptr<const utils::ScalarSpatialFunctionReal> smfunc =
            std::make_shared<const utils::SmearChargePotentialFunction>(
              d_atomCoordinates[iAtom],
              d_atomCharges[iAtom],
              d_smearedChargeRadius[iAtom]);

          d_feBMNuclearCharge[iAtom] =
            std::make_shared<basis::FEBasisManager<ValueTypeBasisCoeff,
                                                   ValueTypeBasisData,
                                                   memorySpace,
                                                   dim>>(feBDHNuclearCharge,
                                                         smfunc);

          smfunc = std::make_shared<const utils::SmearChargeDensityFunction>(
            d_atomCoordinates[iAtom],
            d_atomCharges[iAtom],
            d_smearedChargeRadius[iAtom]);

          for (size_type iCell = 0; iCell < quadRuleContainer->nCells();
               iCell++)
            {
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  size_type             quadId = 0;
                  std::vector<RealType> a(
                    quadRuleContainer->nCellQuadraturePoints(iCell));
                  for (auto j : quadRuleContainer->getCellRealPoints(iCell))
                    {
                      a[quadId] = (RealType)(*smfunc)(j);
                      quadId    = quadId + 1;
                    }
                  RealType *b = a.data();
                  nuclearChargeDensity
                    .template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                          iComp,
                                                                          b);
                }
            }

          // Scale by 4\pi
          scale((RealType)(4 * utils::mathConstants::pi),
                nuclearChargeDensity,
                *d_linAlgOpContext);

          std::shared_ptr<
            linearAlgebra::LinearSolverFunction<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpace>>
            linearSolverFunction = std::make_shared<
              electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                            ValueTypeBasisCoeff,
                                                            memorySpace,
                                                            dim>>(
              d_feBMNuclearCharge[iAtom],
              feBDNuclearChargeStiffnessMatrix,
              feBDNuclearChargeRhs,
              nuclearChargeDensity,
              ksdft::PoissonProblemDefaults::PC_TYPE,
              d_linAlgOpContext,
              ksdft::PoissonProblemDefaults::MAX_CELL_TIMES_NUMVECS);

          linearAlgebra::LinearAlgebraProfiler profiler;

          std::shared_ptr<linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace>>
            CGSolve = std::make_shared<
              linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                            ValueTypeBasisCoeff,
                                            memorySpace>>(
              ksdft::PoissonProblemDefaults::MAX_ITER,
              ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
              ksdft::PoissonProblemDefaults::RELATIVE_TOL,
              ksdft::PoissonProblemDefaults::DIVERGENCE_TOL,
              profiler);

          CGSolve->solve(*linearSolverFunction);

          d_nuclearChargesPotential[iAtom] =
            new linearAlgebra::MultiVector<ValueType, memorySpace>(
              d_feBMNuclearCharge[iAtom]->getMPIPatternP2P(),
              d_linAlgOpContext,
              d_numComponents);
          linearSolverFunction->getSolution(*d_nuclearChargesPotential[iAtom]);
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::
      evalEnergy(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs)
    {
      auto jxwStorage = d_feBDTotalChargeRhs->getJxWInAllCells();

      const size_type numLocallyOwnedCells =
        d_feBMTotalCharge->nLocallyOwnedCells();

      std::vector<RealType> totalEnergyVec =
        ElectrostaticAllElectronFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          memorySpace,
          dim>(*d_totalChargePotentialQuad,
               *d_totalChargeDensity,
               jxwStorage,
               numLocallyOwnedCells,
               d_linAlgOpContext,
               d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      RealType totalEnergy = totalEnergyVec[0];

      totalEnergy = totalEnergy * 0.5 * (1 / (4 * utils::mathConstants::pi));

      nuclearPotentialSolve(feBDNuclearChargeStiffnessMatrix,
                            feBDNuclearChargeRhs);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = feBDNuclearChargeRhs->getQuadratureRuleContainer();

      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        nuclearChargePotentialQuad(quadRuleContainer, d_numComponents);

      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        nuclearChargeDensity(quadRuleContainer, d_numComponents);

      auto jxwStorageNucl = feBDNuclearChargeRhs->getJxWInAllCells();

      const size_type numLocallyOwnedCellsNucl =
        d_feBMNuclearCharge[0]->nLocallyOwnedCells();

      RealType selfEnergy = 0;
      for (unsigned int iAtom = 0; iAtom < d_atomCoordinates.size(); iAtom++)
        {
          const utils::SmearChargeDensityFunction smfunc(
            d_atomCoordinates[iAtom],
            d_atomCharges[iAtom],
            d_smearedChargeRadius[iAtom]);

          for (size_type iCell = 0; iCell < quadRuleContainer->nCells();
               iCell++)
            {
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  size_type             quadId = 0;
                  std::vector<RealType> a(
                    quadRuleContainer->nCellQuadraturePoints(iCell));
                  for (auto j : quadRuleContainer->getCellRealPoints(iCell))
                    {
                      a[quadId] = (RealType)(smfunc)(j);
                      quadId    = quadId + 1;
                    }
                  RealType *b = a.data();
                  nuclearChargeDensity
                    .template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                          iComp,
                                                                          b);
                }
            }

          basis::FEBasisOperations<ValueTypeBasisCoeff,
                                   ValueTypeBasisData,
                                   memorySpace,
                                   dim>
            feBasisOp(feBDNuclearChargeRhs,
                      ksdft::PoissonProblemDefaults::MAX_CELL_TIMES_NUMVECS);

          feBasisOp.interpolate(*d_nuclearChargesPotential[iAtom],
                                *d_feBMNuclearCharge[iAtom],
                                nuclearChargePotentialQuad);

          std::vector<RealType> energyVec =
            ElectrostaticAllElectronFEInternal::getIntegralFieldTimesRho<
              ValueTypeBasisData,
              ValueTypeBasisCoeff,
              memorySpace,
              dim>(nuclearChargePotentialQuad,
                   nuclearChargeDensity,
                   jxwStorageNucl,
                   numLocallyOwnedCellsNucl,
                   d_linAlgOpContext,
                   d_feBMNuclearCharge[iAtom]
                     ->getMPIPatternP2P()
                     ->mpiCommunicator());

          selfEnergy += energyVec[0];

          // correction energy evaluation
        }

      selfEnergy *= 0.5;

      d_energy = totalEnergy - (RealType)selfEnergy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::evalEnergy()
    {
      auto jxwStorage = d_feBDTotalChargeRhs->getJxWInAllCells();

      const size_type numLocallyOwnedCells =
        d_feBMTotalCharge->nLocallyOwnedCells();

      std::vector<RealType> totalEnergyVec =
        ElectrostaticAllElectronFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          memorySpace,
          dim>(*d_totalChargePotentialQuad,
               *d_totalChargeDensity,
               jxwStorage,
               numLocallyOwnedCells,
               d_linAlgOpContext,
               d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      RealType totalEnergy = totalEnergyVec[0];

      totalEnergy = totalEnergy * 0.5 * (1 / (4 * utils::mathConstants::pi));

      double selfEnergy = 0;

      for (unsigned int iAtom = 0; iAtom < d_atomCoordinates.size(); iAtom++)
        {
          const utils::SmearChargePotentialFunction smfunc(
            d_atomCoordinates[iAtom],
            d_atomCharges[iAtom],
            d_smearedChargeRadius[iAtom]);

          double Ig = 10976. / (17875 * d_smearedChargeRadius[iAtom]);
          selfEnergy += 0.5 * (Ig - smfunc(d_atomCoordinates[iAtom]));

          // correction energy evaluation
        }

      d_energy = totalEnergy + (RealType)selfEnergy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename ElectrostaticFE<ValueTypeBasisData,
                             ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::RealType
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::getEnergy() const
    {
      return d_energy;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
