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
#include <utils/ConditionalOStream.h>
namespace dftefe
{
  namespace ksdft
  {
    namespace ElectrostaticLocalFEInternal
    {
      /* Assumption : field and rho have numComponents = 1 */
      template <typename ValueTypeBasisData,
                typename ValueTypeBasisCoeff,
                typename ValueTypeWaveFnBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      typename ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               ValueTypeWaveFnBasisData,
                               memorySpace,
                               dim>::RealType
      getIntegralFieldTimesRho(
        const quadrature::QuadratureValuesContainer<
          typename ElectrostaticFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   ValueTypeWaveFnBasisData,
                                   memorySpace,
                                   dim>::RealType,
          memorySpace> &field,
        const quadrature::QuadratureValuesContainer<
          typename ElectrostaticFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   ValueTypeWaveFnBasisData,
                                   memorySpace,
                                   dim>::RealType,
          memorySpace> &                                             rho,
        const utils::MemoryStorage<ValueTypeBasisData, memorySpace> &jxwStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm)
      {
        using RealType = typename ElectrostaticFE<ValueTypeBasisData,
                                                  ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
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
    } // namespace ElectrostaticLocalFEInternal

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      ElectrostaticLocalFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
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
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const bool      useDealiiMatrixFreePoissonSolve)
      : d_atomCharges(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(d_numAtoms, nullptr)
      , d_feBMNuclearCharge(d_numAtoms, nullptr)
      , d_maxCellBlock(maxCellBlock)
      , d_isDeltaRhoSolve(false)
      , d_rootCout(std::cout)
      , d_useDealiiMatrixFreePoissonSolve(useDealiiMatrixFreePoissonSolve)
    {
      int rank;
      utils::mpi::MPICommRank(
        feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(), &rank);

      d_rootCout.setCondition(rank == 0);

      reinitBasis(atomCoordinates,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDNuclearChargeRhs,
                  feBDElectronicChargeRhs,
                  feBDHamiltonian,
                  externalPotentialFunction);

      reinitField(electronChargeDensity);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      ElectrostaticLocalFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
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
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclChargeStiffnessMatrixNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclChargeRhsNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const bool      useDealiiMatrixFreePoissonSolve)
      : d_atomCharges(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(d_numAtoms, nullptr)
      , d_feBMNuclearCharge(d_numAtoms, nullptr)
      , d_maxCellBlock(maxCellBlock)
      , d_isDeltaRhoSolve(false)
      , d_rootCout(std::cout)
      , d_useDealiiMatrixFreePoissonSolve(useDealiiMatrixFreePoissonSolve)
    {
      int rank;
      utils::mpi::MPICommRank(
        feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(), &rank);

      d_rootCout.setCondition(rank == 0);

      reinitBasis(atomCoordinates,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDNuclearChargeRhs,
                  feBDElectronicChargeRhs,
                  feBDNuclChargeStiffnessMatrixNumSol,
                  feBDNuclChargeRhsNumSol,
                  feBDHamiltonian,
                  externalPotentialFunction);

      reinitField(electronChargeDensity);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      ElectrostaticLocalFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &atomicElectronChargeDensity,
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                    memorySpace>
          &atomicTotalElecPotNuclearQuad,
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                    memorySpace>
          &atomicTotalElecPotElectronicQuad,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>>
          feBMTotalCharge, // will be same as bc of totalCharge -
                           // atomicTotalCharge
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const bool      useDealiiMatrixFreePoissonSolve)
      : d_atomCoordinates(atomCoordinates)
      , d_atomCharges(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(d_numAtoms, nullptr)
      , d_feBMNuclearCharge(d_numAtoms, nullptr)
      , d_maxCellBlock(maxCellBlock)
      , d_isDeltaRhoSolve(true)
      , d_rootCout(std::cout)
      , d_useDealiiMatrixFreePoissonSolve(useDealiiMatrixFreePoissonSolve)
    {
      int rank;
      utils::mpi::MPICommRank(
        feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(), &rank);

      d_rootCout.setCondition(rank == 0);

      reinitBasis(atomCoordinates,
                  atomicElectronChargeDensity,
                  atomicTotalElecPotNuclearQuad,
                  atomicTotalElecPotElectronicQuad,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDNuclearChargeRhs,
                  feBDElectronicChargeRhs,
                  feBDHamiltonian,
                  externalPotentialFunction);

      reinitField(atomicElectronChargeDensity);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::~ElectrostaticLocalFE()
    {
      if (d_scratchDensNuclearQuad != nullptr)
        {
          delete d_scratchDensNuclearQuad;
          d_scratchDensNuclearQuad = nullptr;
        }
      if (d_nuclearChargesDensity != nullptr)
        {
          delete d_nuclearChargesDensity;
          d_nuclearChargesDensity = nullptr;
        }
      if (d_scratchPotHamQuad != nullptr)
        {
          delete d_scratchPotHamQuad;
          d_scratchPotHamQuad = nullptr;
        }
      if (d_correctionPotHamQuad != nullptr)
        {
          delete d_correctionPotHamQuad;
          d_correctionPotHamQuad = nullptr;
        }
      // if (d_correctionPotRhoQuad != nullptr)
      //   {
      //     delete d_correctionPotRhoQuad;
      //     d_correctionPotRhoQuad = nullptr;
      //   }
      if (d_totalChargePotential != nullptr)
        {
          delete d_totalChargePotential;
          d_totalChargePotential = nullptr;
        }
      for (auto &i : d_nuclearChargesPotential)
        {
          if (i != nullptr)
            {
              delete i;
              i = nullptr;
            }
        }
      d_nuclearChargesPotential.clear();
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      reinitBasis(
        const std::vector<utils::Point>                        & atomCoordinates,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclChargeStiffnessMatrixNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclChargeRhsNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction                                    
        /*std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData,
                                          memorySpace>> feBDHamiltonianElec*/)
    {
      utils::throwException(
        !d_isDeltaRhoSolve,
        "cannot call this reinitBasis() if Analytical/1D Solve rho, b and PhiTotal is used. Use different reinitBasis() instead.");
      utils::throwException(
        feBDNuclChargeRhsNumSol->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          feBDNuclChargeRhsNumSol->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            feBDNuclearChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          feBDNuclChargeRhsNumSol->getQuadratureRuleContainer() ==
            feBDNuclearChargeRhs->getQuadratureRuleContainer(),
        "The nuclearCharges RHS for both poisson solves should have same Quadrature.");

      utils::throwException(
        feBDElectronicChargeRhs->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          feBDElectronicChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            feBDHamiltonian->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          feBDElectronicChargeRhs->getQuadratureRuleContainer() ==
            feBDHamiltonian->getQuadratureRuleContainer(),
        "The  feBDElectronicChargeRHS and feBDHamiltonian should have same Quadrature.");

      d_isNumericalVSelfSolve          = true;
      d_atomCoordinates                = atomCoordinates;
      d_feBDNuclearChargeRhs           = feBDNuclearChargeRhs;
      d_feBDNuclChargeRhsNumSol        = feBDNuclChargeRhsNumSol;
      d_feBDElectronicChargeRhs        = feBDElectronicChargeRhs;
      d_feBMTotalCharge                = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;
      d_feBasisOpNuclear =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(d_feBDNuclearChargeRhs,
                                                        d_maxCellBlock,
                                                        d_numComponents);
      d_feBasisOpElectronic =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(
          d_feBDElectronicChargeRhs, d_maxCellBlock, d_numComponents);

      d_feBasisOpHamiltonian =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>>(feBDHamiltonian,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerElec =
          d_feBDElectronicChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerHam = feBDHamiltonian->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

      /*-----Getting V_effNiNj -------*/
      d_scratchPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_correctionPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);
      /*-----Getting V_effNiNj -------*/

      // create nuclear and electron charge densities
      d_scratchDensNuclearQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchPotNuclearQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchDensRhoQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerElec, d_numComponents);

      d_scratchPotRhoQuad = d_scratchPotHamQuad;
      // new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
      //   quadRuleContainerElec, d_numComponents);

      d_correctionPotRhoQuad = d_correctionPotHamQuad;
      // new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
      //   quadRuleContainerElec, d_numComponents);

      // Init the phi_el multivector
      d_totalChargePotential =
        new linearAlgebra::MultiVector<ValueType, memorySpace>(
          d_feBMTotalCharge->getMPIPatternP2P(),
          d_linAlgOpContext,
          d_numComponents);

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      const utils::SmearChargeDensityFunction smfunc(d_atomCoordinates,
                                                     d_atomCharges,
                                                     d_smearedChargeRadius);

      RealType d_totNuclearChargeQuad = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
           iCell++)
        {
          size_type           quadId = 0;
          std::vector<double> jxw    = quadRuleContainerNucl->getCellJxW(iCell);
          for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (RealType)smfunc(j);
                  d_totNuclearChargeQuad += (RealType)smfunc(j) * jxw[quadId];
                }
              RealType *b = a.data();
              d_nuclearChargesDensity
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &d_totNuclearChargeQuad,
        1,
        utils::mpi::Types<RealType>::getMPIDatatype(),
        utils::mpi::MPISum,
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      double totalAtomCharges =
        std::accumulate(d_atomCharges.begin(), d_atomCharges.end(), (double)0);
      // d_nuclearChargesDensity by totalAtomCharges/d_totNuclearChargeQuad
      quadrature::scale((RealType)std::abs(totalAtomCharges /
                                           d_totNuclearChargeQuad),
                        *d_nuclearChargesDensity,
                        *d_linAlgOpContext);

      d_rootCout << "Integral of nuclear charges over domain: "
                 << d_totNuclearChargeQuad << "\n";

      // create the correction quadValuesContainer for numerical solve

      nuclearPotentialSolve(feBDNuclChargeStiffnessMatrixNumSol,
                            feBDNuclChargeRhsNumSol);

      for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
        {
          /* Change this to feBasisOperations for electrostaic basis with same
           * quadrulecontainer as hamiltonian*/
          d_feBasisOpElectronic->interpolate(*d_nuclearChargesPotential[iAtom],
                                             *d_feBMNuclearCharge[iAtom],
                                             *d_scratchPotHamQuad);

          quadrature::add((ValueType)1.0,
                          *d_scratchPotHamQuad,
                          (ValueType)1.0,
                          *d_correctionPotHamQuad,
                          *d_correctionPotHamQuad,
                          *d_linAlgOpContext);
        }

      for (size_type iCell = 0; iCell < quadRuleContainerHam->nCells(); iCell++)
        {
          size_type quadId = 0;
          for (auto j : quadRuleContainerHam->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (ValueType)(externalPotentialFunction)(j);
                }
              RealType *b = a.data();
              d_scratchPotHamQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      quadrature::add((ValueType)-1.0,
                      *d_correctionPotHamQuad,
                      (ValueType)1.0,
                      *d_scratchPotHamQuad,
                      *d_correctionPotHamQuad,
                      *d_linAlgOpContext);

      /*
            for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
              {
                d_feBasisOpElectronic->interpolate(*d_nuclearChargesPotential[iAtom],
                                         *d_feBMNuclearCharge[iAtom],
                                         *d_correctionPotRhoQuad);

                quadrature::add((ValueType)1.0,
                                *d_correctionPotRhoQuad,
                                (ValueType)1.0,
                                *d_scratchPotRhoQuad,
                                *d_scratchPotRhoQuad,
                                *d_linAlgOpContext);
              }

            for (size_type iCell = 0; iCell < quadRuleContainerElec->nCells();
         iCell++)
              {
                size_type quadId = 0;
                for (auto j : quadRuleContainerElec->getCellRealPoints(iCell))
                  {
                    std::vector<RealType> a(d_numComponents);
                    for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                      {
                        a[iComp] = (ValueType)(externalPotentialFunction)(j);
                      }
                    RealType *b = a.data();
                    d_correctionPotRhoQuad
                      ->template
         setCellQuadValues<utils::MemorySpace::HOST>(iCell, quadId, b); quadId =
         quadId + 1;
                  }
              }

            quadrature::add((ValueType)-1.0,
                            *d_scratchPotRhoQuad,
                            (ValueType)1.0,
                            *d_correctionPotRhoQuad,
                            *d_correctionPotRhoQuad,
                            *d_linAlgOpContext);
      */

      std::map<
        std::string,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace> &>
        inpRhsMap;

      d_feBasisDataStorageRhsMap = {{"bSmear", d_feBDNuclearChargeRhs},
                                    {"rho", d_feBDElectronicChargeRhs}};
      inpRhsMap                  = {{"bSmear", *d_scratchDensNuclearQuad},
                   {"rho", *d_scratchDensRhoQuad}};

      if (!d_useDealiiMatrixFreePoissonSolve)
        d_linearSolverFunction = std::make_shared<
          electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpace,
                                                        dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContext,
          ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
          d_numComponents);
      else
        d_poissonSolverDealiiMatFree = std::make_shared<
          electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace,
                                                          dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      reinitBasis(
        const std::vector<utils::Point> &                 atomCoordinates,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction)
    {
      utils::throwException(
        !d_isDeltaRhoSolve,
        "cannot call this reinitBasis() if Analytical/1D Solve rho, b and PhiTotal is used. Use different reinitBasis() instead.");
      utils::throwException(
        feBDElectronicChargeRhs->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          feBDElectronicChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            feBDHamiltonian->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          feBDElectronicChargeRhs->getQuadratureRuleContainer() ==
            feBDHamiltonian->getQuadratureRuleContainer(),
        "The  feBDElectronicChargeRHS and feBDHamiltonian should have same Quadrature.");

      d_isNumericalVSelfSolve          = false;
      d_atomCoordinates                = atomCoordinates;
      d_feBDNuclearChargeRhs           = feBDNuclearChargeRhs;
      d_feBDElectronicChargeRhs        = feBDElectronicChargeRhs;
      d_feBMTotalCharge                = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;
      d_feBasisOpNuclear =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(d_feBDNuclearChargeRhs,
                                                        d_maxCellBlock,
                                                        d_numComponents);
      d_feBasisOpElectronic =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(
          d_feBDElectronicChargeRhs, d_maxCellBlock, d_numComponents);

      d_feBasisOpHamiltonian =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>>(feBDHamiltonian,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerElec =
          d_feBDElectronicChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerHam = feBDHamiltonian->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

      /*-----Getting V_effNiNj -------*/
      d_scratchPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_correctionPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);
      /*-----Getting V_effNiNj -------*/

      // create nuclear and electron charge densities and total charge potential
      // with correction
      d_scratchDensNuclearQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchPotNuclearQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchDensRhoQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerElec, d_numComponents);

      d_scratchPotRhoQuad = d_scratchPotHamQuad;
      // new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
      //   quadRuleContainerElec, d_numComponents);

      d_correctionPotRhoQuad = d_correctionPotHamQuad;
      // new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
      //   quadRuleContainerElec, d_numComponents);

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      // Init the phi_el multivector
      d_totalChargePotential =
        new linearAlgebra::MultiVector<ValueType, memorySpace>(
          d_feBMTotalCharge->getMPIPatternP2P(),
          d_linAlgOpContext,
          d_numComponents);

      const utils::SmearChargeDensityFunction smfunc(d_atomCoordinates,
                                                     d_atomCharges,
                                                     d_smearedChargeRadius);

      RealType d_totNuclearChargeQuad = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
           iCell++)
        {
          size_type           quadId = 0;
          std::vector<double> jxw    = quadRuleContainerNucl->getCellJxW(iCell);
          for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (RealType)smfunc(j);
                  d_totNuclearChargeQuad += (RealType)smfunc(j) * jxw[quadId];
                }
              RealType *b = a.data();
              d_nuclearChargesDensity
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &d_totNuclearChargeQuad,
        1,
        utils::mpi::Types<RealType>::getMPIDatatype(),
        utils::mpi::MPISum,
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      double totalAtomCharges =
        std::accumulate(d_atomCharges.begin(), d_atomCharges.end(), (double)0);
      // d_nuclearChargesDensity by totalAtomCharges/d_totNuclearChargeQuad
      quadrature::scale((RealType)std::abs(totalAtomCharges /
                                           d_totNuclearChargeQuad),
                        *d_nuclearChargesDensity,
                        *d_linAlgOpContext);

      d_rootCout << "Integral of nuclear charges over domain: "
                 << d_totNuclearChargeQuad << "\n";

      // create the correction quadValuesContainer for analytical solve

      const utils::SmearChargePotentialFunction smfuncPot(
        d_atomCoordinates, d_atomCharges, d_smearedChargeRadius);

      for (size_type iCell = 0; iCell < quadRuleContainerHam->nCells(); iCell++)
        {
          size_type quadId = 0;
          for (auto j : quadRuleContainerHam->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (ValueType)(externalPotentialFunction)(j) -
                             (ValueType)(smfuncPot)(j);
                }
              RealType *b = a.data();
              d_correctionPotHamQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }
      /*
            for (size_type iCell = 0; iCell < quadRuleContainerElec->nCells();
         iCell++)
              {
                size_type quadId = 0;
                for (auto j : quadRuleContainerElec->getCellRealPoints(iCell))
                  {
                    std::vector<RealType> a(d_numComponents);
                    for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                      {
                        a[iComp] = (ValueType)(externalPotentialFunction)(j) -
                                   (ValueType)(smfuncPot)(j);
                      }
                    RealType *b = a.data();
                    d_correctionPotRhoQuad
                      ->template
         setCellQuadValues<utils::MemorySpace::HOST>(iCell, quadId, b); quadId =
         quadId + 1;
                  }
              }
      */
      d_scratchDensNuclearQuad->setValue(0);
      std::map<
        std::string,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace> &>
        inpRhsMap;

      d_feBasisDataStorageRhsMap = {{"bSmear", d_feBDNuclearChargeRhs},
                                    {"rho", d_feBDElectronicChargeRhs}};
      inpRhsMap                  = {{"bSmear", *d_scratchDensNuclearQuad},
                   {"rho", *d_scratchDensRhoQuad}};

      if (!d_useDealiiMatrixFreePoissonSolve)
        d_linearSolverFunction = std::make_shared<
          electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpace,
                                                        dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContext,
          ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
          d_numComponents);
      else
        d_poissonSolverDealiiMatFree = std::make_shared<
          electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace,
                                                          dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      reinitBasis(
        const std::vector<utils::Point> &atomCoordinates,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &atomicElectronChargeDensity,
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                    memorySpace>
          &atomicTotalElecPotNuclearQuad,
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                    memorySpace>
          &atomicTotalElecPotElectronicQuad,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction)
    {
      utils::throwException(
        d_isDeltaRhoSolve,
        "cannot call this reinitBasis() if Analytical/1D Solve rho, b and PhiTotal is used. Use different reinitBasis() instead.");
      utils::throwException(
        feBDElectronicChargeRhs->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          feBDElectronicChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            feBDHamiltonian->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          feBDElectronicChargeRhs->getQuadratureRuleContainer() ==
            feBDHamiltonian->getQuadratureRuleContainer(),
        "The  feBDElectronicChargeRHS and feBDHamiltonian should have same Quadrature.");

      d_atomicTotalElecPotNuclearQuad    = &atomicTotalElecPotNuclearQuad;
      d_atomicTotalElecPotElectronicQuad = &atomicTotalElecPotElectronicQuad;
      d_atomicElectronChargeDensity      = atomicElectronChargeDensity;
      d_isNumericalVSelfSolve            = false;
      d_atomCoordinates                  = atomCoordinates;
      d_feBDNuclearChargeRhs             = feBDNuclearChargeRhs;
      d_feBDElectronicChargeRhs          = feBDElectronicChargeRhs;
      d_feBMTotalCharge                  = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix   = feBDTotalChargeStiffnessMatrix;
      d_feBasisOpNuclear =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(d_feBDNuclearChargeRhs,
                                                        d_maxCellBlock,
                                                        d_numComponents);
      d_feBasisOpElectronic =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(
          d_feBDElectronicChargeRhs, d_maxCellBlock, d_numComponents);

      d_feBasisOpHamiltonian =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>>(feBDHamiltonian,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerElec =
          d_feBDElectronicChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerHam = feBDHamiltonian->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

      /*-----Getting V_effNiNj -------*/
      d_scratchPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_correctionPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);
      /*-----Getting V_effNiNj -------*/

      // create nuclear and electron charge densities and total charge potential
      // with correction
      d_scratchDensNuclearQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchPotNuclearQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchDensRhoQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerElec, d_numComponents);

      d_scratchPotRhoQuad = d_scratchPotHamQuad;

      d_correctionPotRhoQuad = d_correctionPotHamQuad;

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainerNucl, d_numComponents);

      // Init the phi_el multivector
      d_totalChargePotential =
        new linearAlgebra::MultiVector<ValueType, memorySpace>(
          d_feBMTotalCharge->getMPIPatternP2P(),
          d_linAlgOpContext,
          d_numComponents);

      const utils::SmearChargeDensityFunction smfunc(d_atomCoordinates,
                                                     d_atomCharges,
                                                     d_smearedChargeRadius);

      RealType d_totNuclearChargeQuad = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
           iCell++)
        {
          size_type           quadId = 0;
          std::vector<double> jxw    = quadRuleContainerNucl->getCellJxW(iCell);
          for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (RealType)smfunc(j);
                  d_totNuclearChargeQuad += (RealType)smfunc(j) * jxw[quadId];
                }
              RealType *b = a.data();
              d_nuclearChargesDensity
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &d_totNuclearChargeQuad,
        1,
        utils::mpi::Types<RealType>::getMPIDatatype(),
        utils::mpi::MPISum,
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      double totalAtomCharges =
        std::accumulate(d_atomCharges.begin(), d_atomCharges.end(), (double)0);
      // d_nuclearChargesDensity by totalAtomCharges/d_totNuclearChargeQuad
      quadrature::scale((RealType)std::abs(totalAtomCharges /
                                           d_totNuclearChargeQuad),
                        *d_nuclearChargesDensity,
                        *d_linAlgOpContext);

      d_rootCout << "Integral of nuclear charges over domain: "
                 << d_totNuclearChargeQuad << "\n";

      // create the correction quadValuesContainer for analytical solve

      const utils::SmearChargePotentialFunction smfuncPot(
        d_atomCoordinates, d_atomCharges, d_smearedChargeRadius);

      for (size_type iCell = 0; iCell < quadRuleContainerHam->nCells(); iCell++)
        {
          size_type quadId = 0;
          for (auto j : quadRuleContainerHam->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (ValueType)(externalPotentialFunction)(j) -
                             (ValueType)(smfuncPot)(j);
                }
              RealType *b = a.data();
              d_correctionPotHamQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      d_scratchDensNuclearQuad->setValue(0);
      std::map<
        std::string,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace> &>
        inpRhsMap;

      d_feBasisDataStorageRhsMap = {{"deltarho", d_feBDElectronicChargeRhs}};
      inpRhsMap                  = {{"deltarho", *d_scratchDensRhoQuad}};

      if (!d_useDealiiMatrixFreePoissonSolve)
        d_linearSolverFunction = std::make_shared<
          electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpace,
                                                        dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContext,
          ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
          d_numComponents);
      else
        d_poissonSolverDealiiMatFree = std::make_shared<
          electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace,
                                                          dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity)
    {
      if (d_isDeltaRhoSolve)
        {
          d_electronChargeDensity = &electronChargeDensity;
          quadrature::add((RealType)1.0,
                          *d_electronChargeDensity,
                          (RealType)(-1.0),
                          d_atomicElectronChargeDensity,
                          *d_scratchDensRhoQuad,
                          *d_linAlgOpContext);

          /**----------Integral Delta Rho--------**/
          size_type quadId    = 0;
          double    normValue = 0.;
          auto      jxwData =
            d_scratchDensRhoQuad->getQuadratureRuleContainer()->getJxW();
          for (size_type iCell = 0; iCell < d_scratchDensRhoQuad->nCells();
               iCell++)
            {
              std::vector<RealType> a(
                d_scratchDensRhoQuad->nCellQuadraturePoints(iCell) *
                d_scratchDensRhoQuad->getNumberComponents());
              d_scratchDensRhoQuad
                ->template getCellValues<utils::MemorySpace::HOST>(iCell,
                                                                   a.data());
              for (auto j : a)
                {
                  normValue += *(jxwData.data() + quadId) * j;
                  quadId = quadId + 1;
                }
            }
          utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &normValue,
            1,
            utils::mpi::Types<double>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          d_rootCout << "Integral Delta Rho: " << normValue << "\n";
          /**----------Integral Delta Rho--------**/

          // Scale by 4\pi
          quadrature::scale((RealType)(4 * utils::mathConstants::pi),
                            *d_scratchDensRhoQuad,
                            *d_linAlgOpContext);

          /*---- solve poisson problem for delta rho system ---*/

          std::map<std::string,
                   const quadrature::QuadratureValuesContainer<RealType,
                                                               memorySpace> &>
            inpRhsMap = {{"deltarho", *d_scratchDensRhoQuad}};

          utils::Profiler p(
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(),
            "Delta Rho Poisson Solve");
          p.registerStart("Reinit");
          if (!d_useDealiiMatrixFreePoissonSolve)
            d_linearSolverFunction->reinit(d_feBMTotalCharge, inpRhsMap);
          else
            d_poissonSolverDealiiMatFree->reinit(d_feBMTotalCharge, inpRhsMap);
          p.registerEnd("Reinit");

          p.registerStart("Solve");
          if (!d_useDealiiMatrixFreePoissonSolve)
            {
              linearAlgebra::LinearAlgebraProfiler profiler;

              std::shared_ptr<
                linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
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
              CGSolve->solve(*d_linearSolverFunction);
            }
          else
            {
              d_poissonSolverDealiiMatFree->solve(
                ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
                ksdft::PoissonProblemDefaults::MAX_ITER);
            }
          p.registerEnd("Solve");
          p.print();

          if (!d_useDealiiMatrixFreePoissonSolve)
            d_linearSolverFunction->getSolution(*d_totalChargePotential);
          else
            d_poissonSolverDealiiMatFree->getSolution(*d_totalChargePotential);
        }
      else
        {
          d_electronChargeDensity = &electronChargeDensity;

          // Scale by 4\pi
          quadrature::scale((RealType)(4 * utils::mathConstants::pi),
                            electronChargeDensity,
                            *d_scratchDensRhoQuad,
                            *d_linAlgOpContext);

          // Scale by 4\pi
          quadrature::scale((RealType)(4 * utils::mathConstants::pi),
                            *d_nuclearChargesDensity,
                            *d_scratchDensNuclearQuad,
                            *d_linAlgOpContext);

          /*---- solve poisson problem for b+rho system ---*/

          std::map<std::string,
                   const quadrature::QuadratureValuesContainer<RealType,
                                                               memorySpace> &>
            inpRhsMap = {{"bSmear", *d_scratchDensNuclearQuad},
                         {"rho", *d_scratchDensRhoQuad}};

          utils::Profiler p(
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(),
            "b+rho Poisson Solve");
          p.registerStart("Reinit");
          if (!d_useDealiiMatrixFreePoissonSolve)
            d_linearSolverFunction->reinit(d_feBMTotalCharge, inpRhsMap);
          else
            d_poissonSolverDealiiMatFree->reinit(d_feBMTotalCharge, inpRhsMap);
          p.registerEnd("Reinit");

          p.registerStart("Solve");
          if (!d_useDealiiMatrixFreePoissonSolve)
            {
              linearAlgebra::LinearAlgebraProfiler profiler;

              std::shared_ptr<
                linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
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
              CGSolve->solve(*d_linearSolverFunction);
            }
          else
            {
              d_poissonSolverDealiiMatFree->solve(
                ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
                ksdft::PoissonProblemDefaults::MAX_ITER);
            }
          p.registerEnd("Solve");
          p.print();

          if (!d_useDealiiMatrixFreePoissonSolve)
            d_linearSolverFunction->getSolution(*d_totalChargePotential);
          else
            d_poissonSolverDealiiMatFree->getSolution(*d_totalChargePotential);
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::getLocal(Storage &cellWiseStorage) const
    {
      // add the correction quadValuesCOntainer for potential to
      // d_scratchPotHamQuad

      /* Change this to feBasisOperations for electrostaic basis with same
       * quadrule as hamiltonian*/
      d_feBasisOpElectronic->interpolate(*d_totalChargePotential,
                                         *d_feBMTotalCharge,
                                         *d_scratchPotHamQuad);

      if (d_isDeltaRhoSolve)
        quadrature::add((ValueType)1.0,
                        *d_atomicTotalElecPotElectronicQuad,
                        (ValueType)1.0,
                        *d_scratchPotHamQuad,
                        *d_linAlgOpContext);

      quadrature::add((ValueType)1.0,
                      *d_scratchPotHamQuad,
                      (ValueType)1.0,
                      *d_correctionPotHamQuad,
                      *d_scratchPotHamQuad,
                      *d_linAlgOpContext);

      d_feBasisOpHamiltonian->computeFEMatrices(
        basis::realspace::LinearLocalOp::IDENTITY,
        basis::realspace::VectorMathOp::MULT,
        basis::realspace::VectorMathOp::MULT,
        basis::realspace::LinearLocalOp::IDENTITY,
        *d_scratchPotHamQuad,
        cellWiseStorage,
        *d_linAlgOpContext);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
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
        "in ElectrostaticLocalFE");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          feBDNuclearChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<
        electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpace,
                                                      dim>>
        linearSolverFunctionNuclear = nullptr;

      std::shared_ptr<
        electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpace,
                                                        dim>>
        poissonSolverDealiiMatFree = nullptr;

      d_nuclearChargeQuad.clear();
      d_nuclearChargeQuad.resize(d_numAtoms, 0);
      for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
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

          d_nuclearChargeQuad[iAtom] = 0;
          for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
               iCell++)
            {
              size_type           quadId = 0;
              std::vector<double> jxw =
                quadRuleContainerNucl->getCellJxW(iCell);
              for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
                {
                  std::vector<RealType> a(d_numComponents);
                  for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                    {
                      a[iComp] = (RealType)(*smfunc)(j);
                      d_nuclearChargeQuad[iAtom] +=
                        (RealType)(*smfunc)(j)*jxw[quadId];
                    }
                  RealType *b = a.data();
                  d_scratchDensNuclearQuad
                    ->template setCellQuadValues<utils::MemorySpace::HOST>(
                      iCell, quadId, b);
                  quadId = quadId + 1;
                }
            }

          utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &d_nuclearChargeQuad[iAtom],
            1,
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          // Scale by 4\pi * d_atomCharges[iAtom]/d_nuclearChargeQuad[iAtom]
          quadrature::scale((RealType)std::abs(4 * utils::mathConstants::pi *
                                               d_atomCharges[iAtom] /
                                               d_nuclearChargeQuad[iAtom]),
                            *d_scratchDensNuclearQuad,
                            *d_linAlgOpContext);

          utils::Profiler p(
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(),
            "bNuclear Poisson Solve for Atom " + std::to_string(iAtom + 1));
          p.registerStart("Reinit");
          if (iAtom == 0)
            {
              if (!d_useDealiiMatrixFreePoissonSolve)
                linearSolverFunctionNuclear = std::make_shared<
                  electrostatics::PoissonLinearSolverFunctionFE<
                    ValueTypeBasisData,
                    ValueTypeBasisCoeff,
                    memorySpace,
                    dim>>(d_feBMNuclearCharge[iAtom],
                          feBDNuclearChargeStiffnessMatrix,
                          feBDNuclearChargeRhs,
                          *d_scratchDensNuclearQuad,
                          ksdft::PoissonProblemDefaults::PC_TYPE,
                          d_linAlgOpContext,
                          ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
                          d_numComponents);
              else
                poissonSolverDealiiMatFree = std::make_shared<
                  electrostatics::PoissonSolverDealiiMatrixFreeFE<
                    ValueTypeBasisData,
                    ValueTypeBasisCoeff,
                    memorySpace,
                    dim>>(d_feBMNuclearCharge[iAtom],
                          feBDNuclearChargeStiffnessMatrix,
                          feBDNuclearChargeRhs,
                          *d_scratchDensNuclearQuad,
                          ksdft::PoissonProblemDefaults::PC_TYPE);
            }
          else
            {
              if (!d_useDealiiMatrixFreePoissonSolve)
                linearSolverFunctionNuclear->reinit(d_feBMNuclearCharge[iAtom],
                                                    *d_scratchDensNuclearQuad);
              else
                poissonSolverDealiiMatFree->reinit(d_feBMNuclearCharge[iAtom],
                                                   *d_scratchDensNuclearQuad);
            }
          p.registerEnd("Reinit");

          p.registerStart("Solve");
          if (!d_useDealiiMatrixFreePoissonSolve)
            {
              linearAlgebra::LinearAlgebraProfiler profiler;

              std::shared_ptr<
                linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
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
              CGSolve->solve(*linearSolverFunctionNuclear);
            }
          else
            poissonSolverDealiiMatFree->solve(
              ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
              ksdft::PoissonProblemDefaults::MAX_ITER);
          p.registerEnd("Solve");
          p.print();

          d_nuclearChargesPotential[iAtom] =
            new linearAlgebra::MultiVector<ValueType, memorySpace>(
              d_feBMNuclearCharge[iAtom]->getMPIPatternP2P(),
              d_linAlgOpContext,
              d_numComponents);

          if (!d_useDealiiMatrixFreePoissonSolve)
            linearSolverFunctionNuclear->getSolution(
              *d_nuclearChargesPotential[iAtom]);
          else
            poissonSolverDealiiMatFree->getSolution(
              *d_nuclearChargesPotential[iAtom]);
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::evalEnergy()
    {
      d_feBasisOpNuclear->interpolate(*d_totalChargePotential,
                                      *d_feBMTotalCharge,
                                      *d_scratchPotNuclearQuad);

      if (d_isDeltaRhoSolve)
        quadrature::add((ValueType)1.0,
                        *d_atomicTotalElecPotNuclearQuad,
                        (ValueType)1.0,
                        *d_scratchPotNuclearQuad,
                        *d_linAlgOpContext);

      d_feBasisOpElectronic->interpolate(*d_totalChargePotential,
                                         *d_feBMTotalCharge,
                                         *d_scratchPotRhoQuad);

      if (d_isDeltaRhoSolve)
        quadrature::add((ValueType)1.0,
                        *d_atomicTotalElecPotElectronicQuad,
                        (ValueType)1.0,
                        *d_scratchPotRhoQuad,
                        *d_linAlgOpContext);

      RealType integralPhixRho =
        ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          ValueTypeWaveFnBasisData,
          memorySpace,
          dim>(*d_scratchPotNuclearQuad,
               *d_nuclearChargesDensity,
               d_feBDNuclearChargeRhs->getJxWInAllCells(),
               d_linAlgOpContext,
               d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      RealType integralPhixbSmear =
        ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          ValueTypeWaveFnBasisData,
          memorySpace,
          dim>(*d_scratchPotRhoQuad,
               *d_electronChargeDensity,
               d_feBDElectronicChargeRhs->getJxWInAllCells(),
               d_linAlgOpContext,
               d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      RealType totalEnergy = (integralPhixRho + integralPhixbSmear) * 0.5;

      // self energy computation // TODO : Do this only once in reinitBasis
      RealType selfEnergy = 0;

      if (d_isNumericalVSelfSolve)
        {
          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainerNucl =
              d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

          auto jxwStorageNucl = d_feBDNuclearChargeRhs->getJxWInAllCells();

          for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
            {
              const utils::SmearChargeDensityFunction smfunc(
                d_atomCoordinates[iAtom],
                d_atomCharges[iAtom],
                d_smearedChargeRadius[iAtom]);

              for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
                   iCell++)
                {
                  size_type           quadId = 0;
                  std::vector<double> jxw =
                    quadRuleContainerNucl->getCellJxW(iCell);
                  for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
                    {
                      std::vector<RealType> a(d_numComponents);
                      for (size_type iComp = 0; iComp < d_numComponents;
                           iComp++)
                        {
                          a[iComp] = (RealType)(smfunc)(j);
                        }
                      RealType *b = a.data();
                      d_scratchDensNuclearQuad
                        ->template setCellQuadValues<utils::MemorySpace::HOST>(
                          iCell, quadId, b);
                      quadId = quadId + 1;
                    }
                }

              // Scale by d_atomCharges[iAtom]/d_nuclearChargeQuad[iAtom]
              quadrature::scale((RealType)std::abs(d_atomCharges[iAtom] /
                                                   d_nuclearChargeQuad[iAtom]),
                                *d_scratchDensNuclearQuad,
                                *d_linAlgOpContext);

              basis::FEBasisOperations<ValueTypeBasisCoeff,
                                       ValueTypeBasisData,
                                       memorySpace,
                                       dim>
                feBasisOp(d_feBDNuclChargeRhsNumSol,
                          d_maxCellBlock,
                          d_numComponents);

              feBasisOp.interpolate(*d_nuclearChargesPotential[iAtom],
                                    *d_feBMNuclearCharge[iAtom],
                                    *d_scratchPotNuclearQuad);

              RealType selfEnergyAtom =
                ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
                  ValueTypeBasisData,
                  ValueTypeBasisCoeff,
                  ValueTypeWaveFnBasisData,
                  memorySpace,
                  dim>(*d_scratchPotNuclearQuad,
                       *d_scratchDensNuclearQuad,
                       jxwStorageNucl,
                       d_linAlgOpContext,
                       d_feBMNuclearCharge[iAtom]
                         ->getMPIPatternP2P()
                         ->mpiCommunicator());

              selfEnergy += selfEnergyAtom;
            }
          selfEnergy *= 0.5;
        }
      else
        {
          /**
                    for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
                      {
                        const utils::SmearChargePotentialFunction smfunc(
                          d_atomCoordinates[iAtom],
                          d_atomCharges[iAtom],
                          d_smearedChargeRadius[iAtom]);

                        double Ig = 10976. / (17875 *
          d_smearedChargeRadius[iAtom]); selfEnergy += (RealType)(0.5 *
          std::pow(d_atomCharges[iAtom], 2) * (Ig -
          (smfunc(d_atomCoordinates[iAtom]) / d_atomCharges[iAtom])));
                      }
                    selfEnergy *= -1;
          **/
          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainerNucl =
              d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

          auto jxwStorageNucl = d_feBDNuclearChargeRhs->getJxWInAllCells();

          for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
            {
              RealType                                atomNuclearChargeQuad = 0;
              const utils::SmearChargeDensityFunction smfunc(
                d_atomCoordinates[iAtom],
                d_atomCharges[iAtom],
                d_smearedChargeRadius[iAtom]);

              const utils::SmearChargePotentialFunction smfunc1(
                d_atomCoordinates[iAtom],
                d_atomCharges[iAtom],
                d_smearedChargeRadius[iAtom]);

              for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
                   iCell++)
                {
                  size_type           quadId = 0;
                  std::vector<double> jxw =
                    quadRuleContainerNucl->getCellJxW(iCell);
                  for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
                    {
                      std::vector<RealType> a(d_numComponents);
                      for (size_type iComp = 0; iComp < d_numComponents;
                           iComp++)
                        {
                          a[iComp] = (RealType)smfunc(j);
                          atomNuclearChargeQuad +=
                            (RealType)smfunc(j) * jxw[quadId];
                        }
                      RealType *b = a.data();
                      d_scratchDensNuclearQuad
                        ->template setCellQuadValues<utils::MemorySpace::HOST>(
                          iCell, quadId, b);

                      for (size_type iComp = 0; iComp < d_numComponents;
                           iComp++)
                        {
                          a[iComp] = smfunc1(j);
                        }
                      b = a.data();
                      d_scratchPotNuclearQuad
                        ->template setCellQuadValues<utils::MemorySpace::HOST>(
                          iCell, quadId, b);
                      quadId = quadId + 1;
                    }
                }

              utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                &atomNuclearChargeQuad,
                1,
                utils::mpi::Types<RealType>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

              // Scale by d_atomCharges[iAtom]/totalNuclearChargeAtom
              quadrature::scale((RealType)std::abs(d_atomCharges[iAtom] /
                                                   atomNuclearChargeQuad),
                                *d_scratchDensNuclearQuad,
                                *d_linAlgOpContext);

              RealType selfEnergyAtom =
                ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
                  ValueTypeBasisData,
                  ValueTypeBasisCoeff,
                  ValueTypeWaveFnBasisData,
                  memorySpace,
                  dim>(
                  *d_scratchPotNuclearQuad,
                  *d_scratchDensNuclearQuad,
                  jxwStorageNucl,
                  d_linAlgOpContext,
                  d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

              selfEnergy += selfEnergyAtom;
            }
          selfEnergy *= 0.5;
        }

      // correction energy evaluation

      RealType correctionEnergy =
        ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          ValueTypeWaveFnBasisData,
          memorySpace,
          dim>(*d_correctionPotRhoQuad,
               *d_electronChargeDensity,
               d_feBDElectronicChargeRhs->getJxWInAllCells(),
               d_linAlgOpContext,
               d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      d_energy = totalEnergy - selfEnergy + correctionEnergy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename ElectrostaticFE<ValueTypeBasisData,
                             ValueTypeBasisCoeff,
                             ValueTypeWaveFnBasisData,
                             memorySpace,
                             dim>::RealType
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::getEnergy() const
    {
      return d_energy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const quadrature::QuadratureValuesContainer<
      typename ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               ValueTypeWaveFnBasisData,
                               memorySpace,
                               dim>::ValueType,
      memorySpace> &
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::getFunctionalDerivative() const
    {
      d_feBasisOpElectronic->interpolate(*d_totalChargePotential,
                                         *d_feBMTotalCharge,
                                         *d_scratchPotHamQuad);

      if (d_isDeltaRhoSolve)
        quadrature::add((ValueType)1.0,
                        *d_atomicTotalElecPotElectronicQuad,
                        (ValueType)1.0,
                        *d_scratchPotHamQuad,
                        *d_linAlgOpContext);

      quadrature::add((ValueType)1.0,
                      *d_scratchPotHamQuad,
                      (ValueType)1.0,
                      *d_correctionPotHamQuad,
                      *d_scratchPotHamQuad,
                      *d_linAlgOpContext);

      return *d_scratchPotHamQuad;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
