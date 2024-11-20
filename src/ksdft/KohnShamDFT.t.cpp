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

namespace dftefe
{
  namespace ksdft
  {
    namespace KohnShamDFTInternal
    {
      template <typename RealType, utils::MemorySpace memorySpace>
      RealType
      computeResidualQuadData(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &outValues,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &inValues,
        quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &residualValues,
        const utils::MemoryStorage<RealType, utils::MemorySpace::HOST> &JxW,
        const bool                                   computeNorm,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const utils::mpi::MPIComm &                  mpiComm)
      {
        linearAlgebra::blasLapack::axpby<RealType, RealType, memorySpace>(
          outValues.nQuadraturePoints() * outValues.getNumberComponents(),
          1.0,
          outValues.begin(),
          -1.0,
          inValues.begin(),
          residualValues.begin(),
          linAlgOpContext);

        double normValue = 0.0;
        if (computeNorm)
          {
            size_type quadId = 0;
            for (size_type iCell = 0; iCell < residualValues.nCells(); iCell++)
              {
                std::vector<RealType> a(
                  residualValues.nCellQuadraturePoints(iCell) *
                  residualValues.getNumberComponents());
                residualValues.template getCellValues<utils::MemorySpace::HOST>(
                  iCell, a.data());
                for (auto j : a)
                  {
                    normValue += *(JxW.data() + quadId) * j * j;
                    quadId = quadId + 1;
                  }
              }
            utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              utils::mpi::MPIInPlace,
              &normValue,
              1,
              utils::mpi::Types<double>::getMPIDatatype(),
              utils::mpi::MPISum,
              mpiComm);
          }
        return std::sqrt(normValue);
      }

      template <typename RealType, utils::MemorySpace memorySpace>
      RealType
      normalizeDensityQuadData(
        quadrature::QuadratureValuesContainer<RealType, memorySpace> &inValues,
        const size_type numElectrons,
        const utils::MemoryStorage<RealType, utils::MemorySpace::HOST> &JxW,
        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext,
        const utils::mpi::MPIComm &                  mpiComm,
        bool                                         computeTotalDensity,
        bool                                         scaleDensity,
        utils::ConditionalOStream &                  rootCout)
      {
        RealType totalDensityInQuad = 0.0;
        if (computeTotalDensity || scaleDensity)
          {
            int quadId = 0;
            for (size_type iCell = 0; iCell < inValues.nCells(); iCell++)
              {
                std::vector<RealType> a(inValues.nCellQuadraturePoints(iCell) *
                                        inValues.getNumberComponents());
                inValues.template getCellValues<utils::MemorySpace::HOST>(
                  iCell, a.data());
                for (auto j : a)
                  {
                    totalDensityInQuad += j * *(JxW.data() + quadId);
                    quadId += 1;
                  }
              }
            utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
              utils::mpi::MPIInPlace,
              &totalDensityInQuad,
              1,
              utils::mpi::Types<RealType>::getMPIDatatype(),
              utils::mpi::MPISum,
              mpiComm);
          }

        if (scaleDensity)
          {
            rootCout << "Electronic Density with the rho quadrature: "
                     << totalDensityInQuad << std::endl;
            quadrature::scale((RealType)(std::abs((RealType)numElectrons /
                                                  totalDensityInQuad)),
                              inValues,
                              linAlgOpContext);
          }

        return totalDensityInQuad;
      }
    } // namespace KohnShamDFTInternal

    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    KohnShamDFT<ValueTypeElectrostaticsCoeff,
                ValueTypeElectrostaticsBasis,
                ValueTypeWaveFunctionCoeff,
                ValueTypeWaveFunctionBasis,
                memorySpace,
                dim>::
      KohnShamDFT(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
        const size_type                  numElectrons,
        const size_type                  numWantedEigenvalues,
        const double                     smearingTemperature,
        const double                     fermiEnergyTolerance,
        const double                     fracOccupancyTolerance,
        const double                     eigenSolveResidualTolerance,
        const double                     scfDensityResidualNormTolerance,
        const size_type                  maxChebyshevFilterPass,
        const size_type                  maxSCFIter,
        const bool                       evaluateEnergyEverySCF,
        const size_type                  mixingHistory,
        const double                     mixingParameter,
        const bool                       isAdaptiveAndersonMixingParameter,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensityInput,
        std::shared_ptr<
          const basis::FEBasisManager<ValueTypeElectrostaticsCoeff,
                                      ValueTypeElectrostaticsBasis,
                                      memorySpace,
                                      dim>>               feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                    ValueTypeWaveFunctionBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDTotalChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                          memorySpace>> feBDKineticHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                          memorySpace>>
          feBDElectrostaticsHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                          memorySpace>> feBDEXCHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                         linAlgOpContext,
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext)
      : d_mixingHistory(mixingHistory)
      , d_mixingParameter(mixingParameter)
      , d_isAdaptiveAndersonMixingParameter(isAdaptiveAndersonMixingParameter)
      , d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
      , d_densityInQuadValues(electronChargeDensityInput)
      , d_densityOutQuadValues(electronChargeDensityInput)
      , d_densityResidualQuadValues(electronChargeDensityInput)
      , d_numMaxSCFIter(maxSCFIter)
      , d_MContext(&MContext)
      , d_MInvContext(&MInvContext)
      , d_mpiCommDomain(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator())
      , d_mixingScheme(d_mpiCommDomain)
      , d_numWantedEigenvalues(numWantedEigenvalues)
      , d_linAlgOpContext(linAlgOpContext)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_waveFunctionSubspaceGuess(feBMWaveFn->getMPIPatternP2P(),
                                    linAlgOpContext,
                                    numWantedEigenvalues,
                                    0.0,
                                    1.0)
      , d_kohnShamWaveFunctions(&d_waveFunctionSubspaceGuess)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
    {
      utils::throwException(electronChargeDensityInput.getNumberComponents() ==
                              1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
          electronChargeDensityInput.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDElectrostaticsHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho =
          electronChargeDensityInput.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      //************* CHANGE THIS **********************
      utils::MemoryTransfer<utils::MemorySpace::HOST, utils::MemorySpace::HOST>
           memTransfer;
      auto jxwData = quadRuleContainerRho->getJxW();
      d_jxwDataHost.resize(jxwData.size());
      memTransfer.copy(jxwData.size(), d_jxwDataHost.data(), jxwData.data());

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(d_densityInQuadValues,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContext,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL);

      d_hamitonianElec =
        std::make_shared<ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>(
          atomCoordinates,
          atomCharges,
          smearedChargeRadius,
          d_densityInQuadValues,
          feBMTotalCharge,
          feBDTotalChargeStiffnessMatrix,
          feBDTotalChargeRhs,
          feBDElectrostaticsHamiltonian,
          externalPotentialFunction,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE);
      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          d_densityInQuadValues,
          feBDEXCHamiltonian,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE);
      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin.get(), d_hamitonianElec.get(), d_hamitonianXC.get()};
      // form the kohn sham operator
      d_hamitonianOperator =
        std::make_shared<KohnShamOperatorContextFE<ValueTypeOperator,
                                                   ValueTypeOperand,
                                                   ValueTypeWaveFunctionBasis,
                                                   memorySpace,
                                                   dim>>(
          *feBMWaveFn,
          hamiltonianComponentsVec,
          *linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          numWantedEigenvalues);

      // call the eigensolver

      d_lanczosGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(d_lanczosGuess, 1);

      d_waveFunctionSubspaceGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        d_waveFunctionSubspaceGuess, numWantedEigenvalues);

      // form the kohn sham operator
      d_ksEigSolve = std::make_shared<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
        numElectrons,
        smearingTemperature,
        fermiEnergyTolerance,
        fracOccupancyTolerance,
        eigenSolveResidualTolerance,
        maxChebyshevFilterPass,
        d_waveFunctionSubspaceGuess,
        d_lanczosGuess,
        d_numWantedEigenvalues,
        MContextForInv,
        MInvContext);

      d_densCalc =
        std::make_shared<DensityCalculator<ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(
          feBDEXCHamiltonian,
          *feBMWaveFn,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE);
    }

    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    KohnShamDFT<ValueTypeElectrostaticsCoeff,
                ValueTypeElectrostaticsBasis,
                ValueTypeWaveFunctionCoeff,
                ValueTypeWaveFunctionBasis,
                memorySpace,
                dim>::
      KohnShamDFT(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
        const size_type                  numElectrons,
        const size_type                  numWantedEigenvalues,
        const double                     smearingTemperature,
        const double                     fermiEnergyTolerance,
        const double                     fracOccupancyTolerance,
        const double                     eigenSolveResidualTolerance,
        const double                     scfDensityResidualNormTolerance,
        const size_type                  maxChebyshevFilterPass,
        const size_type                  maxSCFIter,
        const bool                       evaluateEnergyEverySCF,
        const size_type                  mixingHistory,
        const double                     mixingParameter,
        const bool                       isAdaptiveAndersonMixingParameter,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensityInput,
        std::shared_ptr<
          const basis::FEBasisManager<ValueTypeElectrostaticsCoeff,
                                      ValueTypeElectrostaticsBasis,
                                      memorySpace,
                                      dim>>               feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                    ValueTypeWaveFunctionBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDTotalChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                          memorySpace>> feBDKineticHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                          memorySpace>>
          feBDElectrostaticsHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                          memorySpace>> feBDEXCHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                         linAlgOpContext,
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext)
      : d_mixingHistory(mixingHistory)
      , d_mixingParameter(mixingParameter)
      , d_isAdaptiveAndersonMixingParameter(isAdaptiveAndersonMixingParameter)
      , d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
      , d_densityInQuadValues(electronChargeDensityInput)
      , d_densityOutQuadValues(electronChargeDensityInput)
      , d_densityResidualQuadValues(electronChargeDensityInput)
      , d_numMaxSCFIter(maxSCFIter)
      , d_MContext(&MContext)
      , d_MInvContext(&MInvContext)
      , d_mpiCommDomain(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator())
      , d_mixingScheme(d_mpiCommDomain)
      , d_numWantedEigenvalues(numWantedEigenvalues)
      , d_linAlgOpContext(linAlgOpContext)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_waveFunctionSubspaceGuess(feBMWaveFn->getMPIPatternP2P(),
                                    linAlgOpContext,
                                    numWantedEigenvalues,
                                    0.0,
                                    1.0)
      , d_kohnShamWaveFunctions(&d_waveFunctionSubspaceGuess)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
    {
      utils::throwException(electronChargeDensityInput.getNumberComponents() ==
                              1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
          electronChargeDensityInput.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho =
          electronChargeDensityInput.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      //************* CHANGE THIS **********************
      utils::MemoryTransfer<utils::MemorySpace::HOST, utils::MemorySpace::HOST>
           memTransfer;
      auto jxwData = quadRuleContainerRho->getJxW();
      d_jxwDataHost.resize(jxwData.size());
      memTransfer.copy(jxwData.size(), d_jxwDataHost.data(), jxwData.data());

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(d_densityInQuadValues,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContext,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL);

      d_hamitonianElec =
        std::make_shared<ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>(
          atomCoordinates,
          atomCharges,
          smearedChargeRadius,
          d_densityInQuadValues,
          feBMTotalCharge,
          feBDTotalChargeStiffnessMatrix,
          feBDTotalChargeRhs,
          feBDNuclearChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectrostaticsHamiltonian,
          externalPotentialFunction,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE);
      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          d_densityInQuadValues,
          feBDEXCHamiltonian,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE);
      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin.get(), d_hamitonianElec.get(), d_hamitonianXC.get()};
      // form the kohn sham operator
      d_hamitonianOperator =
        std::make_shared<KohnShamOperatorContextFE<ValueTypeOperator,
                                                   ValueTypeOperand,
                                                   ValueTypeWaveFunctionBasis,
                                                   memorySpace,
                                                   dim>>(
          *feBMWaveFn,
          hamiltonianComponentsVec,
          *linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          numWantedEigenvalues);

      // call the eigensolver

      d_lanczosGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(d_lanczosGuess, 1);

      d_waveFunctionSubspaceGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        d_waveFunctionSubspaceGuess, numWantedEigenvalues);

      // form the kohn sham operator
      d_ksEigSolve = std::make_shared<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
        numElectrons,
        smearingTemperature,
        fermiEnergyTolerance,
        fracOccupancyTolerance,
        eigenSolveResidualTolerance,
        maxChebyshevFilterPass,
        d_waveFunctionSubspaceGuess,
        d_lanczosGuess,
        d_numWantedEigenvalues,
        MContextForInv,
        MInvContext);

      d_densCalc =
        std::make_shared<DensityCalculator<ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(
          feBDEXCHamiltonian,
          *feBMWaveFn,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE);
    }

    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    KohnShamDFT<ValueTypeElectrostaticsCoeff,
                ValueTypeElectrostaticsBasis,
                ValueTypeWaveFunctionCoeff,
                ValueTypeWaveFunctionBasis,
                memorySpace,
                dim>::solve()
    {
      d_isSolved = true;
      d_hamitonianElec->evalEnergy();
      RealType elecEnergy = d_hamitonianElec->getEnergy();
      d_rootCout << "Electrostatic energy with guess density: " << elecEnergy
                 << "\n";

      d_mixingScheme.addMixingVariable(
        mixingVariable::rho,
        d_jxwDataHost,
        true, // call MPI REDUCE while computing dot products
        d_mixingParameter,
        d_isAdaptiveAndersonMixingParameter);

      //
      // Begin SCF iteration
      //
      unsigned int scfIter = 0;
      double       norm    = 1.0;
      d_rootCout << "Starting SCF iterations....\n";
      while ((norm > d_SCFTol) && (scfIter < d_numMaxSCFIter))
        {
          d_rootCout
            << "************************Begin Self-Consistent-Field Iteration: "
            << std::setw(2) << scfIter + 1 << " ***********************\n";

          // mix the densities with  Anderson mix if scf > 0
          // Update the history of mixing variables

          if (scfIter > 0)
            {
              norm = KohnShamDFTInternal::computeResidualQuadData(
                d_densityOutQuadValues,
                d_densityInQuadValues,
                d_densityResidualQuadValues,
                d_jxwDataHost,
                true,
                *d_linAlgOpContext,
                d_mpiCommDomain);

              d_mixingScheme.template addVariableToInHist<memorySpace>(
                mixingVariable::rho,
                d_densityInQuadValues.begin(),
                d_densityInQuadValues.nQuadraturePoints());

              d_mixingScheme.template addVariableToResidualHist<memorySpace>(
                mixingVariable::rho,
                d_densityResidualQuadValues.begin(),
                d_densityResidualQuadValues.nQuadraturePoints());

              // Delete old history if it exceeds a pre-described
              // length
              d_mixingScheme.popOldHistory(d_mixingHistory);

              // Compute the mixing coefficients
              d_mixingScheme.computeAndersonMixingCoeff(
                std::vector<mixingVariable>{mixingVariable::rho},
                *d_linAlgOpContext);

              // update the mixing variables
              // get next input density
              d_mixingScheme.template mixVariable<memorySpace>(
                mixingVariable::rho,
                d_densityInQuadValues.begin(),
                d_densityInQuadValues.nQuadraturePoints());
            }

          // reinit the components of hamiltonian
          if (scfIter > 0)
            {
              // normalize electroncharge density each scf
              RealType totalDensityInQuad =
                KohnShamDFTInternal::normalizeDensityQuadData(
                  d_densityInQuadValues,
                  d_numElectrons,
                  d_jxwDataHost,
                  *d_linAlgOpContext,
                  d_mpiCommDomain,
                  true,
                  false,
                  d_rootCout);

              d_rootCout << "Electron density in : " << totalDensityInQuad
                         << "\n";

              d_hamitonianElec->reinitField(d_densityInQuadValues);
              d_hamitonianXC->reinitField(d_densityInQuadValues);
              std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
                d_hamitonianKin.get(),
                d_hamitonianElec.get(),
                d_hamitonianXC.get()};
              d_hamitonianOperator->reinit(*d_feBMWaveFn,
                                           hamiltonianComponentsVec);
            }

          // reinit the chfsi bounds
          if (scfIter > 0)
            {
              d_ksEigSolve->reinitBounds(
                d_kohnShamEnergies[0],
                d_kohnShamEnergies[d_numWantedEigenvalues - 1]);
            }

          // Linear Eigen Solve
          linearAlgebra::EigenSolverError err =
            d_ksEigSolve->solve(*d_hamitonianOperator,
                                d_kohnShamEnergies,
                                *d_kohnShamWaveFunctions,
                                true,
                                *d_MContext,
                                *d_MInvContext);

          d_occupation = d_ksEigSolve->getFractionalOccupancy();

          std::vector<RealType> eigSolveResNorm =
            d_ksEigSolve->getEigenSolveResidualNorm();

          /*
          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainer =
          d_feBDEXCHamiltonian->getQuadratureRuleContainer();

              std::shared_ptr<const
            basis::FEBasisOperations<ValueTypeWaveFunctionCoeff,
                                                                  ValueTypeWaveFunctionBasis,
                                                                  memorySpace,
                                                                  dim>>
          feBasisOp = std::make_shared<const
            basis::FEBasisOperations<ValueTypeWaveFunctionCoeff,
                                                                  ValueTypeWaveFunctionBasis,
                                                                  memorySpace,
                                                                  dim>>(
                    d_feBDEXCHamiltonian, 50,
          d_kohnShamWaveFunctions->getNumberComponents());

            quadrature::QuadratureValuesContainer<ValueType, memorySpace>
              waveFuncQuad( quadRuleContainer,
          d_kohnShamWaveFunctions->getNumberComponents());

              feBasisOp->interpolate(*d_kohnShamWaveFunctions,
                                        *d_feBMWaveFn,
                                        waveFuncQuad);

              double denSum = 0;
              for(dftefe::size_type i = 0 ; i < waveFuncQuad.nCells() ; i++)
              {
                std::vector<double> jxwCell = quadRuleContainer->getCellJxW(i);
                for(int j = 0 ; j < jxwCell.size() ; j++)
                  {
                    std::vector<double>
          a(d_kohnShamWaveFunctions->getNumberComponents(), 0);
                    waveFuncQuad.template
          getCellQuadValues<utils::MemorySpace::HOST>(i, j, a.data()); for(int k
          = 0 ; k < a.size() ; k++) denSum += jxwCell[j] * std::abs(a[k]) *
          std::abs(a[k]);
                  }
              }

              utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                &denSum,
                1,
                utils::mpi::Types<double>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_kohnShamWaveFunctions->getMPIPatternP2P()->mpiCommunicator());

              std::cout << "Wavefn sum: "<< denSum << std::endl;

              feBasisOp->interpolate(d_ksEigSolve->getOrthogonalizedFilteredSubspace(),
                                        *d_feBMWaveFn,
                                        waveFuncQuad);

              denSum = 0;
              for(dftefe::size_type i = 0 ; i < waveFuncQuad.nCells() ; i++)
              {
                std::vector<double> jxwCell = quadRuleContainer->getCellJxW(i);
                for(int j = 0 ; j < jxwCell.size() ; j++)
                  {
                    std::vector<double>
          a(d_kohnShamWaveFunctions->getNumberComponents(), 0);
                    waveFuncQuad.template
          getCellQuadValues<utils::MemorySpace::HOST>(i, j, a.data()); for(int k
          = 0 ; k < a.size() ; k++) denSum += jxwCell[j] * std::abs(a[k]) *
          std::abs(a[k]);
                  }
              }

              utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                &denSum,
                1,
                utils::mpi::Types<double>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_kohnShamWaveFunctions->getMPIPatternP2P()->mpiCommunicator());

              std::cout << "getOrthogonalizedFilteredSubspace sum: "<< denSum <<
          std::endl;

              feBasisOp->interpolate(d_ksEigSolve->getFilteredSubspace(),
                                        *d_feBMWaveFn,
                                        waveFuncQuad);

              denSum = 0;
              for(dftefe::size_type i = 0 ; i < waveFuncQuad.nCells() ; i++)
              {
                std::vector<double> jxwCell = quadRuleContainer->getCellJxW(i);
                for(int j = 0 ; j < jxwCell.size() ; j++)
                  {
                    std::vector<double>
          a(d_kohnShamWaveFunctions->getNumberComponents(), 0);
                    waveFuncQuad.template
          getCellQuadValues<utils::MemorySpace::HOST>(i, j, a.data()); for(int k
          = 0 ; k < a.size() ; k++) denSum += jxwCell[j] * std::abs(a[k]) *
          std::abs(a[k]);
                  }
              }

              utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                &denSum,
                1,
                utils::mpi::Types<double>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_kohnShamWaveFunctions->getMPIPatternP2P()->mpiCommunicator());

              std::cout << "getFilteredSubspace sum: "<< denSum << std::endl;
          */

          // compute output rho
          d_densCalc->computeRho(d_occupation,
                                 *d_kohnShamWaveFunctions,
                                 d_densityOutQuadValues);

          RealType totalDensityInQuad =
            KohnShamDFTInternal::normalizeDensityQuadData(
              d_densityOutQuadValues,
              d_numElectrons,
              d_jxwDataHost,
              *d_linAlgOpContext,
              d_mpiCommDomain,
              true,
              false,
              d_rootCout);

          d_rootCout << "Electron density out : " << totalDensityInQuad << "\n";

          // check residual in density if else
          if (d_evaluateEnergyEverySCF)
            {
              d_hamitonianElec->reinitField(d_densityOutQuadValues);
              d_hamitonianKin->evalEnergy(
                d_occupation,
                *d_feBMWaveFn,
                *d_kohnShamWaveFunctions,
                KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE);
              RealType kinEnergy = d_hamitonianKin->getEnergy();
              d_rootCout << "Kinetic energy: " << kinEnergy << "\n";

              d_hamitonianElec->evalEnergy();
              RealType elecEnergy = d_hamitonianElec->getEnergy();
              d_rootCout << "Electrostatic energy: " << elecEnergy << "\n";

              d_hamitonianXC->reinitField(d_densityOutQuadValues);
              d_hamitonianXC->evalEnergy(d_mpiCommDomain);
              RealType xcEnergy = d_hamitonianXC->getEnergy();
              d_rootCout << "LDA EXC energy: " << xcEnergy << "\n";

              // calculate band energy
              RealType bandEnergy = 0;
              for (size_type i = 0; i < d_occupation.size(); i++)
                {
                  bandEnergy += 2 * d_occupation[i] * d_kohnShamEnergies[i];
                }

              d_rootCout << "Band energy: " << bandEnergy << "\n";

              RealType totalEnergy = kinEnergy + elecEnergy + xcEnergy;

              d_rootCout << "Ground State Energy: " << totalEnergy << "\n";

              d_groundStateEnergy = totalEnergy;
            }

          if (scfIter > 0)
            d_rootCout << "Density Residual Norm : " << norm << "\n";
          scfIter += 1;
        }

      if (!d_evaluateEnergyEverySCF)
        {
          d_hamitonianElec->reinitField(d_densityOutQuadValues);
          d_hamitonianKin->evalEnergy(
            d_occupation,
            *d_feBMWaveFn,
            *d_kohnShamWaveFunctions,
            KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE);
          RealType kinEnergy = d_hamitonianKin->getEnergy();
          d_rootCout << "Kinetic energy: " << kinEnergy << "\n";

          d_hamitonianElec->evalEnergy();
          RealType elecEnergy = d_hamitonianElec->getEnergy();
          d_rootCout << "Electrostatic energy: " << elecEnergy << "\n";

          d_hamitonianXC->reinitField(d_densityOutQuadValues);
          d_hamitonianXC->evalEnergy(d_mpiCommDomain);
          RealType xcEnergy = d_hamitonianXC->getEnergy();
          d_rootCout << "LDA EXC energy: " << xcEnergy << "\n";

          // calculate band energy
          RealType bandEnergy = 0;
          for (size_type i = 0; i < d_occupation.size(); i++)
            {
              bandEnergy += 2 * d_occupation[i] * d_kohnShamEnergies[i];
            }

          d_rootCout << "Band energy: " << bandEnergy << "\n";

          RealType totalEnergy = kinEnergy + elecEnergy + xcEnergy;

          d_rootCout << "Ground State Energy: " << totalEnergy << "\n";

          d_groundStateEnergy = totalEnergy;
        }
    }

    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    double
    KohnShamDFT<ValueTypeElectrostaticsCoeff,
                ValueTypeElectrostaticsBasis,
                ValueTypeWaveFunctionCoeff,
                ValueTypeWaveFunctionBasis,
                memorySpace,
                dim>::getGroundStateEnergy()
    {
      utils::throwException(
        d_isSolved,
        "Cannot call ksdft getGroundStateEnergy() before solving the KS problem.");
      return d_groundStateEnergy;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
