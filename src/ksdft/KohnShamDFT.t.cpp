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
                for (size_type iComp = 0;
                     iComp < residualValues.getNumberComponents();
                     iComp++)
                  {
                    std::vector<RealType> a(
                      residualValues.nCellQuadraturePoints(iCell));
                    residualValues
                      .template getCellQuadValues<utils::MemorySpace::HOST>(
                        iCell, iComp, a.data());
                    for (auto j : a)
                      {
                        normValue += *(JxW.data() + quadId) * j * j;
                        quadId = quadId + 1;
                      }
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
        const size_type                  chebyshevPolynomialDegree,
        const size_type                  maxChebyshevFilterPass,
        const size_type                  maxSCFIter,
        const bool                       evaluateEnergyEverySCF,
        const size_type                  mixingHistory,
        const double                     mixingParameter,
        const double                     adaptAndersonMixingParameter,
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
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                         linAlgOpContext,
        const size_type  cellBlockSize,
        const size_type  waveFunctionBatchSize,
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext)
      : d_mixingHistory(mixingHistory)
      , d_mixingParameter(mixingParameter)
      , d_adaptAndersonMixingParameter(adaptAndersonMixingParameter)
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
      , d_kohnShamWaveFunctions(feBMWaveFn->getMPIPatternP2P(),
                                linAlgOpContext,
                                d_numWantedEigenvalues)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_waveFunctionBatchSize(waveFunctionBatchSize)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
    {
      utils::throwException(electronChargeDensityInput.getNumberComponents() ==
                              1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDHamiltonian->getQuadratureRuleContainer() ==
          electronChargeDensityInput.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDHamiltonian and electronChargeDensity should be same.");

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(feBDHamiltonian,
                                                         linAlgOpContext,
                                                         cellBlockSize);

      d_hamitonianElec = std::make_shared<
        ElectrostaticAllElectronFE<ValueTypeElectrostaticsBasis,
                                   ValueTypeElectrostaticsCoeff,
                                   ValueTypeWaveFunctionBasis,
                                   memorySpace,
                                   dim>>(atomCoordinates,
                                         atomCharges,
                                         smearedChargeRadius,
                                         d_densityInQuadValues,
                                         feBMTotalCharge,
                                         feBDTotalChargeStiffnessMatrix,
                                         feBDTotalChargeRhs,
                                         feBDNuclearChargeStiffnessMatrix,
                                         feBDNuclearChargeRhs,
                                         feBDHamiltonian,
                                         externalPotentialFunction,
                                         linAlgOpContext,
                                         cellBlockSize);
      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(d_densityInQuadValues,
                                                     feBDHamiltonian,
                                                     linAlgOpContext,
                                                     cellBlockSize);
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
          cellBlockSize);

      // call the eigensolver

      linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace>
        waveFunctionSubspaceGuess(feBMWaveFn->getMPIPatternP2P(),
                                  linAlgOpContext,
                                  numWantedEigenvalues,
                                  0.0,
                                  1.0);

      linearAlgebra::Vector<ValueTypeWaveFunctionCoeff, memorySpace>
        lanczosGuess(feBMWaveFn->getMPIPatternP2P(), linAlgOpContext, 0.0, 1.0);

      lanczosGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(lanczosGuess, 1);

      waveFunctionSubspaceGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        waveFunctionSubspaceGuess, numWantedEigenvalues);

      // form the kohn sham operator
      d_ksEigSolve = std::make_shared<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
        numElectrons,
        smearingTemperature,
        fermiEnergyTolerance,
        fracOccupancyTolerance,
        eigenSolveResidualTolerance,
        chebyshevPolynomialDegree,
        maxChebyshevFilterPass,
        waveFunctionSubspaceGuess,
        lanczosGuess,
        waveFunctionBatchSize,
        MContextForInv,
        MInvContext);

      d_densCalc =
        std::make_shared<DensityCalculator<ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(feBDHamiltonian,
                                                 *feBMWaveFn,
                                                 linAlgOpContext,
                                                 cellBlockSize,
                                                 waveFunctionBatchSize);

      //************* CHANGE THIS **********************
      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace> memTransfer;
      auto jxwData = feBDHamiltonian->getJxWInAllCells();
      d_jxwDataHost.resize(jxwData.size());
      memTransfer.copy(jxwData.size(), d_jxwDataHost.data(), jxwData.data());
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
      d_mixingScheme.addMixingVariable(
        mixingVariable::rho,
        d_jxwDataHost,
        true, // call MPI REDUCE while computing dot products
        d_mixingParameter,
        d_adaptAndersonMixingParameter);

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
              d_hamitonianElec->reinitField(d_densityInQuadValues);
              d_hamitonianXC->reinitField(d_densityInQuadValues);
              std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
                d_hamitonianKin.get(),
                d_hamitonianElec.get(),
                d_hamitonianXC.get()};
              d_hamitonianOperator->reinit(*d_feBMWaveFn,
                                           hamiltonianComponentsVec);
            }

          // Linear Eigen Solve
          linearAlgebra::EigenSolverError err =
            d_ksEigSolve->solve(*d_hamitonianOperator,
                                d_kohnShamEnergies,
                                d_kohnShamWaveFunctions,
                                true,
                                *d_MContext,
                                *d_MInvContext);

          d_rootCout << err.msg << "\n";
          d_rootCout << "*****************The EigenValues are: "
                        "******************\n";
          for (auto &i : d_kohnShamEnergies)
            d_rootCout << i << ", ";
          d_rootCout << "\n";

          d_occupation = d_ksEigSolve->getFractionalOccupancy();

          // compute output rho
          d_densCalc->computeRho(d_occupation,
                                 d_kohnShamWaveFunctions,
                                 d_densityOutQuadValues);

          // check residual in density if else
          if (d_evaluateEnergyEverySCF)
            {
              d_hamitonianElec->reinitField(d_densityOutQuadValues);
              d_hamitonianKin->evalEnergy(d_occupation,
                                          *d_feBMWaveFn,
                                          d_kohnShamWaveFunctions,
                                          d_waveFunctionBatchSize);
              RealType kinEnergy = d_hamitonianKin->getEnergy();
              d_rootCout << "Kinetic energy: " << kinEnergy << "\n";

              d_hamitonianElec->evalEnergy();
              RealType elecEnergy = d_hamitonianElec->getEnergy();
              d_rootCout << "Electrostatic energy: " << elecEnergy << "\n";

              d_hamitonianXC->evalEnergy(d_mpiCommDomain);
              RealType xcEnergy = d_hamitonianXC->getEnergy();
              d_rootCout << "LDA EXC energy: " << xcEnergy << "\n";

              // calculate band energy
              RealType bandEnergy;
              for (size_type i = 0; i < d_occupation.size(); i++)
                {
                  bandEnergy += 2 * d_occupation[i] * d_kohnShamEnergies[i];
                }

              d_rootCout << "Band energy: " << bandEnergy << "\n";

              RealType totalEnergy = kinEnergy + elecEnergy + xcEnergy;

              d_rootCout << "Ground State Energy: " << totalEnergy << "\n";
            }

          d_rootCout << "Density Residual Norm : " << norm << "\n";
          scfIter += 1;
        }

      if (!d_evaluateEnergyEverySCF)
        {
          d_hamitonianElec->reinitField(d_densityOutQuadValues);
          d_hamitonianKin->evalEnergy(d_occupation,
                                      *d_feBMWaveFn,
                                      d_kohnShamWaveFunctions,
                                      d_waveFunctionBatchSize);
          RealType kinEnergy = d_hamitonianKin->getEnergy();
          d_rootCout << "Kinetic energy: " << kinEnergy << "\n";

          d_hamitonianElec->evalEnergy();
          RealType elecEnergy = d_hamitonianElec->getEnergy();
          d_rootCout << "Electrostatic energy: " << elecEnergy << "\n";

          d_hamitonianXC->evalEnergy(d_mpiCommDomain);
          RealType xcEnergy = d_hamitonianXC->getEnergy();
          d_rootCout << "LDA EXC energy: " << xcEnergy << "\n";

          // calculate band energy
          RealType bandEnergy;
          for (size_type i = 0; i < d_occupation.size(); i++)
            {
              bandEnergy += 2 * d_occupation[i] * d_kohnShamEnergies[i];
            }

          d_rootCout << "Band energy: " << bandEnergy << "\n";

          RealType totalEnergy = kinEnergy + elecEnergy + xcEnergy;

          d_rootCout << "Ground State Energy: " << totalEnergy << "\n";
        }
    }
  } // end of namespace ksdft
} // end of namespace dftefe