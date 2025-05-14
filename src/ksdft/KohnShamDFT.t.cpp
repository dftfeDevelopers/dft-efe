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
#include <utils/PointChargePotentialFunction.h>
#include <boost/math/distributions/normal.hpp>

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

      template <typename ValueType, utils::MemorySpace memorySpace>
      void
      generateRandNormDistMultivec(
        linearAlgebra::MultiVector<ValueType, memorySpace> &multiVectorGuess)
      {
        int rank;
        utils::mpi::MPICommRank(
          multiVectorGuess.getMPIPatternP2P()->mpiCommunicator(), &rank);
        boost::math::normal normDist;
        std::mt19937        randomIntGenerator(rank);
        ValueType *         temp = multiVectorGuess.data();
        for (unsigned int i = 0;
             i < multiVectorGuess.localSize() * multiVectorGuess.numVectors();
             ++i)
          {
            double z = (-0.5 + ((double)randomIntGenerator() -
                                (double)randomIntGenerator.min()) /
                                 ((double)randomIntGenerator.max() -
                                  (double)randomIntGenerator.min())) *
                       3.0;
            double value = boost::math::pdf(normDist, z);
            if (randomIntGenerator() % 2 == 0)
              value = -1.0 * value;

            temp[i] = (ValueType)value;
          }

        // const basis::BasisDofHandler &basisDofHandler =
        // feBMWaveFn->getBasisDofHandler();

        // const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
        // ValueTypeWaveFunctionBasis, memorySpace, dim>
        //   &feDofHandlerWF = dynamic_cast<
        //     const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
        //     ValueTypeWaveFunctionBasis, memorySpace, dim> &>(
        //     basisDofHandler);

        // global_size_type numGlobalEnrichmentIds = 0;
        // if(&feDofHandlerWF != nullptr)
        // {
        //   global_size_type globalEnrichmentStartId =
        //   feDofHandlerWF.getGlobalRanges()[1].first; numGlobalEnrichmentIds =
        //   feDofHandlerWF.getGlobalRanges()[1].second -
        //   globalEnrichmentStartId;

        //   for(global_size_type enrichId = 0 ; enrichId <
        //   numGlobalEnrichmentIds ; enrichId ++)
        //   {
        //     for(size_type i = 0 ; i < multiVectorGuess.localSize() ; i++)
        //     {
        //       for(size_type j = 0 ; j < multiVectorGuess.numVectors() ; j++)
        //       {
        //         if(feBMWaveFn->localToGlobalIndex(i) == enrichId +
        //         globalEnrichmentStartId && j == enrichId)
        //         {
        //           *(multiVectorGuess.data() + i *
        //           multiVectorGuess.numVectors() + j) = (ValueType)1.0;
        //         }
        //       }
        //     }
        //   }
        // }
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
                                          memorySpace>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDElectronicChargeRhs,
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
        const OpContext &MInvContext,
        bool             isResidualChebyshevFilter)
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
                                    numWantedEigenvalues)
      , d_kohnShamWaveFunctions(&d_waveFunctionSubspaceGuess)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
    {
      if (dynamic_cast<
            const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim> *>(
            &feBMWaveFn->getBasisDofHandler()) != nullptr)
        if (dynamic_cast<
              const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim> *>(
              &feBMWaveFn->getBasisDofHandler()) != nullptr)
          d_isOEFEBasis = true;
        else
          d_isOEFEBasis = false;

      KohnShamDFTInternal::generateRandNormDistMultivec(
        d_waveFunctionSubspaceGuess);
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

      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE);

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
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
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
      d_p.registerEnd("Hamiltonian Components Initilization");

      d_hamiltonianElectroExc =
        std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                                            ValueTypeElectrostaticsBasis,
                                            ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim>>(d_hamitonianElec,
                                                  d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin, d_hamiltonianElectroExc};

      d_p.registerStart("Hamiltonian Operator Creation");
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
      d_p.registerEnd("Hamiltonian Operator Creation");

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
        isResidualChebyshevFilter,
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

      if (dynamic_cast<const utils::PointChargePotentialFunction *>(
            &externalPotentialFunction) != nullptr)
        d_isPSPCalculation = false;
      else
        d_isPSPCalculation = true;

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       d_waveFunctionSubspaceGuess,
                       d_lanczosGuess,
                       false,
                       d_numWantedEigenvalues,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }
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
                                          memorySpace>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>>
          feBDNuclChargeStiffnessMatrixNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDNuclChargeRhsNumSol,
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
        const OpContext &MInvContext,
        bool             isResidualChebyshevFilter)
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
                                    numWantedEigenvalues)
      , d_kohnShamWaveFunctions(&d_waveFunctionSubspaceGuess)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
    {
      if (dynamic_cast<
            const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim> *>(
            &feBMWaveFn->getBasisDofHandler()) != nullptr)
        d_isOEFEBasis = true;
      else
        d_isOEFEBasis = false;

      KohnShamDFTInternal::generateRandNormDistMultivec(
        d_waveFunctionSubspaceGuess);
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

      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE);

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
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDNuclChargeStiffnessMatrixNumSol,
          feBDNuclChargeRhsNumSol,
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
      d_p.registerEnd("Hamiltonian Components Initilization");

      d_hamiltonianElectroExc =
        std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                                            ValueTypeElectrostaticsBasis,
                                            ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim>>(d_hamitonianElec,
                                                  d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin, d_hamiltonianElectroExc};

      d_p.registerStart("Hamiltonian Operator Creation");
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
      d_p.registerEnd("Hamiltonian Operator Creation");
      d_p.print();

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
        isResidualChebyshevFilter,
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

      if (dynamic_cast<const utils::PointChargePotentialFunction *>(
            &externalPotentialFunction) != nullptr)
        d_isPSPCalculation = false;
      else
        d_isPSPCalculation = true;

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       d_waveFunctionSubspaceGuess,
                       d_lanczosGuess,
                       false,
                       d_numWantedEigenvalues,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }
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
        /* Atom related info */
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
        const size_type                  numElectrons,
        /* SCF related info */
        const size_type numWantedEigenvalues,
        const double    smearingTemperature,
        const double    fermiEnergyTolerance,
        const double    fracOccupancyTolerance,
        const double    eigenSolveResidualTolerance,
        const double    scfDensityResidualNormTolerance,
        const size_type maxChebyshevFilterPass,
        const size_type maxSCFIter,
        const bool      evaluateEnergyEverySCF,
        /* Mixing related info */
        const size_type mixingHistory,
        const double    mixingParameter,
        const bool      isAdaptiveAndersonMixingParameter,
        /* Basis related info */
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensityInput,
        /* Atomic potential for delta rho */
        const quadrature::QuadratureValuesContainer<
          ValueTypeElectrostaticsCoeff,
          memorySpace> &atomicTotalElecPotNuclearQuad,
        const quadrature::QuadratureValuesContainer<
          ValueTypeElectrostaticsCoeff,
          memorySpace> &atomicTotalElecPotElectronicQuad,
        /* Field boundary */
        std::shared_ptr<
          const basis::FEBasisManager<ValueTypeElectrostaticsCoeff,
                                      ValueTypeElectrostaticsBasis,
                                      memorySpace,
                                      dim>>               feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                    ValueTypeWaveFunctionBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        /* Field data storages poisson solves*/
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDElectronicChargeRhs,
        /* Field data storages eigen solve*/
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
        /* PSP/AE related info */
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        /* linAgOperations Context*/
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        /* basis overlap related info */
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext,
        bool             isResidualChebyshevFilter)
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
                                    numWantedEigenvalues)
      , d_kohnShamWaveFunctions(&d_waveFunctionSubspaceGuess)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
    {
      if (dynamic_cast<
            const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim> *>(
            &feBMWaveFn->getBasisDofHandler()) != nullptr)
        d_isOEFEBasis = true;
      else
        d_isOEFEBasis = false;

      KohnShamDFTInternal::generateRandNormDistMultivec(
        d_waveFunctionSubspaceGuess);
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

      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE);

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
          atomicTotalElecPotNuclearQuad,
          atomicTotalElecPotElectronicQuad,
          feBMTotalCharge,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
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
      d_p.registerEnd("Hamiltonian Components Initilization");

      d_hamiltonianElectroExc =
        std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                                            ValueTypeElectrostaticsBasis,
                                            ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim>>(d_hamitonianElec,
                                                  d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin, d_hamiltonianElectroExc};

      d_p.registerStart("Hamiltonian Operator Creation");
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
      d_p.registerEnd("Hamiltonian Operator Creation");

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
        isResidualChebyshevFilter,
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

      if (dynamic_cast<const utils::PointChargePotentialFunction *>(
            &externalPotentialFunction) != nullptr)
        d_isPSPCalculation = false;
      else
        d_isPSPCalculation = true;

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       d_waveFunctionSubspaceGuess,
                       d_lanczosGuess,
                       false,
                       d_numWantedEigenvalues,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }
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
        const std::vector<std::string> & atomSymbolVec,
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
                                          memorySpace>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDElectronicChargeRhs,
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
        const std::map<std::string, std::string> &      atomSymbolToPSPFilename,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                         linAlgOpContext,
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext,
        bool             isResidualChebyshevFilter)
      : d_mixingHistory(mixingHistory)
      , d_mixingParameter(mixingParameter)
      , d_isAdaptiveAndersonMixingParameter(isAdaptiveAndersonMixingParameter)
      , d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
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
                                    numWantedEigenvalues)
      , d_kohnShamWaveFunctions(&d_waveFunctionSubspaceGuess)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
    {

      const std::vector<std::string> metadataNames =
        atoms::AtomSphDataPSPDefaults::METADATANAMES;
      std::vector<std::string> fieldNamesPSP = {"vlocal", "rhoatom"};

      d_atomSphericalDataContainerPSP =
        std::make_shared<atoms::AtomSphericalDataContainer>(
          atoms::AtomSphericalDataType::PSEUDOPOTENTIAL,
          atomSymbolToPSPFilename,
          fieldNamesPSP,
          metadataNames);

      d_isONCVNonLocPSP = false, d_isNlcc = false;
      for (int atomSymbolId = 0; atomSymbolId < atomSymbolVec.size();
           atomSymbolId++)
        {
          int numProj = 0;
          utils::stringOps::strToInt(
            d_atomSphericalDataContainerPSP->getMetadata(
              atomSymbolVec[atomSymbolId], "number_of_proj"),
            numProj);
          if (numProj > 0)
            {
              d_isONCVNonLocPSP      = true;
              bool coreCorrect = false;
              utils::stringOps::strToBool(
                d_atomSphericalDataContainerPSP->getMetadata(
                  atomSymbolVec[atomSymbolId], "core_correction"),
                coreCorrect);
              if (coreCorrect)
                {
                  d_isNlcc = true;
                  break;
                }
            }
        }

      if (d_isONCVNonLocPSP)
        {
          d_atomSphericalDataContainerPSP->addFieldName("beta");
          if (d_isNlcc)
            {
              d_atomSphericalDataContainerPSP->addFieldName("nlcc");
            }
        }

        d_densityInQuadValues = quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      d_densityOutQuadValues = d_densityInQuadValues;
      d_densityResidualQuadValues = d_densityInQuadValues;

      if (dynamic_cast<
            const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim> *>(
            &feBMWaveFn->getBasisDofHandler()) != nullptr)
        d_isOEFEBasis = true;
      else
        d_isOEFEBasis = false;

      KohnShamDFTInternal::generateRandNormDistMultivec(
        d_waveFunctionSubspaceGuess);
      utils::throwException(d_densityInQuadValues.getNumberComponents() ==
                              1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
        d_densityInQuadValues.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDElectrostaticsHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho =
        d_densityInQuadValues.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      const atoms::AtomSevereFunction<dim> rho(
          d_atomSphericalDataContainerPSP,
          atomSymbolVec,
          atomCoordinates,
          "rhoatom",
          0,
          1);

      for (size_type iCell = 0; iCell < d_densityInQuadValues.nCells(); iCell++)
      {
            size_type             quadId = 0;
            std::vector<RealType> a(
              d_densityInQuadValues.nCellQuadraturePoints(iCell));
            a = (rho)(quadRuleContainerRho->getCellRealPoints(iCell));
            RealType *b = a.data();
            d_densityInQuadValues.template 
              setCellValues<utils::MemorySpace::HOST>(iCell, b);
      }
      
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

      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE);

      d_hamitonianElec =
        std::make_shared<ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                                   ValueTypeElectrostaticsCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
          atomCoordinates,
          atomCharges,
          atomSymbolVec,
          d_atomSphericalDataContainerPSP,
          smearedChargeRadius,
          d_densityInQuadValues,
          feBMTotalCharge,
          feBMWaveFn,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDElectrostaticsHamiltonian,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          numWantedEigenvalues);

      if(d_isNlcc && d_isONCVNonLocPSP)
      {
        d_coreCorrDensUPF = quadrature::QuadratureValuesContainer<RealType, memorySpace>(
            feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

        d_coreCorrectedDensity = quadrature::QuadratureValuesContainer<RealType, memorySpace>(
                    feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);
        
        const atoms::AtomSevereFunction<dim> rhoCoreCorrection(
          d_atomSphericalDataContainerPSP,
          atomSymbolVec,
          atomCoordinates,
          "nlcc",
          0,
          1);

        for (size_type iCell = 0; iCell < d_densityInQuadValues.nCells(); iCell++)
        {
              size_type             quadId = 0;
              std::vector<RealType> a(
                d_densityInQuadValues.nCellQuadraturePoints(iCell));
              a = (rhoCoreCorrection)(quadRuleContainerRho->getCellRealPoints(iCell));
              RealType *b = a.data();
              d_coreCorrDensUPF.template 
                setCellValues<utils::MemorySpace::HOST>(iCell, b);
        }
        quadrature::add((ValueType)1.0,
                        d_densityInQuadValues,
                        (ValueType)1.0,
                        d_coreCorrDensUPF,
                        d_coreCorrectedDensity,
                        *d_linAlgOpContext);

      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          d_coreCorrectedDensity,
          feBDEXCHamiltonian,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE);                        
      }
      else
      {
      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          d_densityInQuadValues,
          feBDEXCHamiltonian,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE);
      }
      d_p.registerEnd("Hamiltonian Components Initilization");

      d_hamiltonianElectroExc =
        std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                                            ValueTypeElectrostaticsBasis,
                                            ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim>>(d_hamitonianElec,
                                                  d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin, d_hamiltonianElectroExc};

      d_p.registerStart("Hamiltonian Operator Creation");
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
      d_p.registerEnd("Hamiltonian Operator Creation");

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
        isResidualChebyshevFilter,
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

      d_isPSPCalculation = true;

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       d_waveFunctionSubspaceGuess,
                       d_lanczosGuess,
                       false,
                       d_numWantedEigenvalues,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }
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
        const std::vector<std::string> & atomSymbolVec,
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
        const utils::ScalarSpatialFunctionReal &atomicTotalElectroPotentialFunction,
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
                                          memorySpace>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpace>> feBDElectronicChargeRhs,
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
        const std::map<std::string, std::string> &      atomSymbolToPSPFilename,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                         linAlgOpContext,
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext,
        bool             isResidualChebyshevFilter)
      : d_mixingHistory(mixingHistory)
      , d_mixingParameter(mixingParameter)
      , d_isAdaptiveAndersonMixingParameter(isAdaptiveAndersonMixingParameter)
      , d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
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
                                    numWantedEigenvalues)
      , d_kohnShamWaveFunctions(&d_waveFunctionSubspaceGuess)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
    {

      const std::vector<std::string> metadataNames =
        atoms::AtomSphDataPSPDefaults::METADATANAMES;
      std::vector<std::string> fieldNamesPSP = {"vlocal", "rhoatom"};

      d_atomSphericalDataContainerPSP =
        std::make_shared<atoms::AtomSphericalDataContainer>(
          atoms::AtomSphericalDataType::PSEUDOPOTENTIAL,
          atomSymbolToPSPFilename,
          fieldNamesPSP,
          metadataNames);

      d_isONCVNonLocPSP = false, d_isNlcc = false;
      for (int atomSymbolId = 0; atomSymbolId < atomSymbolVec.size();
           atomSymbolId++)
        {
          int numProj = 0;
          utils::stringOps::strToInt(
            d_atomSphericalDataContainerPSP->getMetadata(
              atomSymbolVec[atomSymbolId], "number_of_proj"),
            numProj);
          if (numProj > 0)
            {
              d_isONCVNonLocPSP      = true;
              bool coreCorrect = false;
              utils::stringOps::strToBool(
                d_atomSphericalDataContainerPSP->getMetadata(
                  atomSymbolVec[atomSymbolId], "core_correction"),
                coreCorrect);
              if (coreCorrect)
                {
                  d_isNlcc = true;
                  break;
                }
            }
        }

      if (d_isONCVNonLocPSP)
        {
          d_atomSphericalDataContainerPSP->addFieldName("beta");
          if (d_isNlcc)
            {
              d_atomSphericalDataContainerPSP->addFieldName("nlcc");
            }
        }
   
       d_densityInQuadValues = quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);
  
        d_densityOutQuadValues = d_densityInQuadValues;
        d_densityResidualQuadValues = d_densityInQuadValues;

      if (dynamic_cast<
            const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim> *>(
            &feBMWaveFn->getBasisDofHandler()) != nullptr)
        d_isOEFEBasis = true;
      else
        d_isOEFEBasis = false;

      KohnShamDFTInternal::generateRandNormDistMultivec(
        d_waveFunctionSubspaceGuess);
      utils::throwException(d_densityInQuadValues.getNumberComponents() ==
                              1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
        d_densityInQuadValues.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDElectrostaticsHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho =
        d_densityInQuadValues.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      const atoms::AtomSevereFunction<dim> rho(
        d_atomSphericalDataContainerPSP,
        atomSymbolVec,
        atomCoordinates,
        "rhoatom",
        0,
        1);

      for (size_type iCell = 0; iCell < d_densityInQuadValues.nCells(); iCell++)
      {
            size_type             quadId = 0;
            std::vector<RealType> a(
              d_densityInQuadValues.nCellQuadraturePoints(iCell));
            a = (rho)(quadRuleContainerRho->getCellRealPoints(iCell));
            RealType *b = a.data();
            d_densityInQuadValues.template 
              setCellValues<utils::MemorySpace::HOST>(iCell, b);
      }

      quadrature::QuadratureValuesContainer<
        ValueTypeElectrostaticsCoeff,
        memorySpace> atomicTotalElecPotNuclearQuad(feBDNuclearChargeRhs->getQuadratureRuleContainer(), 1, 0.0);
      
      quadrature::QuadratureValuesContainer<
        ValueTypeElectrostaticsCoeff,
        memorySpace> atomicTotalElecPotElectronicQuad(feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      for (size_type iCell = 0; iCell < atomicTotalElecPotNuclearQuad.nCells(); iCell++)
      {
            size_type             quadId = 0;
            std::vector<ValueTypeElectrostaticsCoeff> a(
              atomicTotalElecPotNuclearQuad.nCellQuadraturePoints(iCell));
            a = (atomicTotalElectroPotentialFunction)(feBDNuclearChargeRhs->getQuadratureRuleContainer()->getCellRealPoints(iCell));
            ValueTypeElectrostaticsCoeff *b = a.data();
            atomicTotalElecPotNuclearQuad.template 
              setCellValues<utils::MemorySpace::HOST>(iCell, b);
      }

      for (size_type iCell = 0; iCell < atomicTotalElecPotElectronicQuad.nCells(); iCell++)
      {
            size_type             quadId = 0;
            std::vector<ValueTypeElectrostaticsCoeff> a(
              atomicTotalElecPotElectronicQuad.nCellQuadraturePoints(iCell));
            a = (atomicTotalElectroPotentialFunction)(quadRuleContainerRho->getCellRealPoints(iCell));
            ValueTypeElectrostaticsCoeff *b = a.data();
            atomicTotalElecPotElectronicQuad.template 
              setCellValues<utils::MemorySpace::HOST>(iCell, b);
      }

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

      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE);

      d_hamitonianElec =
        std::make_shared<ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                                   ValueTypeElectrostaticsCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
          atomCoordinates,
          atomCharges,
          atomSymbolVec,
          d_atomSphericalDataContainerPSP,
          smearedChargeRadius,
          d_densityInQuadValues,
          atomicTotalElecPotNuclearQuad,
          atomicTotalElecPotElectronicQuad,
          feBMTotalCharge,
          feBMWaveFn,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDElectrostaticsHamiltonian,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          numWantedEigenvalues);

      if(d_isNlcc && d_isONCVNonLocPSP)
      {
        d_coreCorrDensUPF = quadrature::QuadratureValuesContainer<RealType, memorySpace>(
            feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

        d_coreCorrectedDensity = quadrature::QuadratureValuesContainer<RealType, memorySpace>(
                    feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);
        
        const atoms::AtomSevereFunction<dim> rhoCoreCorrection(
          d_atomSphericalDataContainerPSP,
          atomSymbolVec,
          atomCoordinates,
          "nlcc",
          0,
          1);

        for (size_type iCell = 0; iCell < d_densityInQuadValues.nCells(); iCell++)
        {
              size_type             quadId = 0;
              std::vector<RealType> a(
                d_densityInQuadValues.nCellQuadraturePoints(iCell));
              a = (rhoCoreCorrection)(quadRuleContainerRho->getCellRealPoints(iCell));
              RealType *b = a.data();
              d_coreCorrDensUPF.template 
                setCellValues<utils::MemorySpace::HOST>(iCell, b);
        }
        quadrature::add((ValueType)1.0,
                        d_densityInQuadValues,
                        (ValueType)1.0,
                        d_coreCorrDensUPF,
                        d_coreCorrectedDensity,
                        *d_linAlgOpContext);

      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          d_coreCorrectedDensity,
          feBDEXCHamiltonian,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE);                        
      }
      else
      {
      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          d_densityInQuadValues,
          feBDEXCHamiltonian,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE);
      }
      d_p.registerEnd("Hamiltonian Components Initilization");

      d_hamiltonianElectroExc =
        std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                                            ValueTypeElectrostaticsBasis,
                                            ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim>>(d_hamitonianElec,
                                                  d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin, d_hamiltonianElectroExc};

      d_p.registerStart("Hamiltonian Operator Creation");
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
      d_p.registerEnd("Hamiltonian Operator Creation");

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
        isResidualChebyshevFilter,
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

      d_isPSPCalculation = true;

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       d_waveFunctionSubspaceGuess,
                       d_lanczosGuess,
                       false,
                       d_numWantedEigenvalues,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }
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

      if (auto hamiltonian = std::dynamic_pointer_cast<
            ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                 ValueTypeElectrostaticsCoeff,
                                 ValueTypeWaveFunctionBasis,
                                 memorySpace,
                                 dim>>(d_hamitonianElec))
        {
          hamiltonian->evalEnergy();
        }
      else if (auto hamiltonian = std::dynamic_pointer_cast<
                 ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                           ValueTypeElectrostaticsCoeff,
                                           ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(d_hamitonianElec))
        {
          hamiltonian->evalEnergy(d_occupation, *d_kohnShamWaveFunctions);
        }

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
      while (((norm > d_SCFTol) && (scfIter < d_numMaxSCFIter)))
        {
          d_p.reset();
          d_rootCout
            << "************************Begin Self-Consistent-Field Iteration: "
            << std::setw(2) << scfIter + 1 << " ***********************\n";

          // mix the densities with  Anderson mix if scf > 0
          // Update the history of mixing variables

          d_p.registerStart("Density Mixing");
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
          d_p.registerEnd("Density Mixing");

          d_p.registerStart("Hamiltonian Reinit");
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

              if (auto hamiltonian = std::dynamic_pointer_cast<
                    ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                         ValueTypeElectrostaticsCoeff,
                                         ValueTypeWaveFunctionBasis,
                                         memorySpace,
                                         dim>>(d_hamitonianElec))
                {
                  hamiltonian->reinitField(d_densityInQuadValues);
                }
              else if (auto hamiltonian = std::dynamic_pointer_cast<
                         ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                                   ValueTypeElectrostaticsCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(d_hamitonianElec))
                {
                  hamiltonian->reinitField(d_densityInQuadValues);
                }

              if(d_isNlcc && d_isONCVNonLocPSP)
              {
                quadrature::add((ValueType)1.0,
                                d_densityInQuadValues,
                                (ValueType)1.0,
                                d_coreCorrDensUPF,
                                d_coreCorrectedDensity,
                                *d_linAlgOpContext);
                d_hamitonianXC->reinitField(d_coreCorrectedDensity);
              }
              else
                d_hamitonianXC->reinitField(d_densityInQuadValues);

              d_hamiltonianElectroExc->reinit(d_hamitonianElec, d_hamitonianXC);

              std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
                d_hamitonianKin, d_hamiltonianElectroExc};

              d_hamitonianOperator->reinit(*d_feBMWaveFn,
                                           hamiltonianComponentsVec);
            }
          d_p.registerEnd("Hamiltonian Reinit");

          // reinit the chfsi bounds
          if (scfIter > 0)
            {
              d_ksEigSolve->reinitBounds(
                d_kohnShamEnergies[0],
                d_kohnShamEnergies[d_numWantedEigenvalues - 1]);
            }

          if (scfIter == 0 && d_isPSPCalculation)
            d_ksEigSolve->setChebyPolyScalingFactor(1.34);

          d_p.registerStart("EigenSolve");
          // Linear Eigen Solve
          linearAlgebra::EigenSolverError err =
            d_ksEigSolve->solve(*d_hamitonianOperator,
                                d_kohnShamEnergies,
                                *d_kohnShamWaveFunctions,
                                true,
                                *d_MContext,
                                *d_MInvContext);
          d_p.registerEnd("EigenSolve");

          d_occupation = d_ksEigSolve->getFractionalOccupancy();

          std::vector<RealType> eigSolveResNorm =
            d_ksEigSolve->getEigenSolveResidualNorm();

          /*
          ============== DEBUG : Integral \psi and \psi_orthonormalized =
          Numelectrons========= std::shared_ptr<const
          quadrature::QuadratureRuleContainer> quadRuleContainer =
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
           ============== DEBUG : Integral \psi and \psi_orthonormalized =
          Numelectrons=========
          */

          d_p.registerStart("Density Compute");
          // compute output rho
          d_densCalc->computeRho(d_occupation,
                                 *d_kohnShamWaveFunctions,
                                 d_densityOutQuadValues);
          d_p.registerEnd("Density Compute");
          d_p.print();

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
              if (auto hamiltonian = std::dynamic_pointer_cast<
                    ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                         ValueTypeElectrostaticsCoeff,
                                         ValueTypeWaveFunctionBasis,
                                         memorySpace,
                                         dim>>(d_hamitonianElec))
                {
                  hamiltonian->reinitField(d_densityOutQuadValues);
                }
              else if (auto hamiltonian = std::dynamic_pointer_cast<
                         ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                                   ValueTypeElectrostaticsCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(d_hamitonianElec))
                {
                  hamiltonian->reinitField(d_densityOutQuadValues);
                }

              d_hamitonianKin->evalEnergy(d_occupation,
                                          *d_feBMWaveFn,
                                          *d_kohnShamWaveFunctions);
              RealType kinEnergy = d_hamitonianKin->getEnergy();
              d_rootCout << "Kinetic energy: " << kinEnergy << "\n";

              if (auto hamiltonian = std::dynamic_pointer_cast<
                    ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                         ValueTypeElectrostaticsCoeff,
                                         ValueTypeWaveFunctionBasis,
                                         memorySpace,
                                         dim>>(d_hamitonianElec))
                {
                  hamiltonian->evalEnergy();
                }
              else if (auto hamiltonian = std::dynamic_pointer_cast<
                         ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                                   ValueTypeElectrostaticsCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(d_hamitonianElec))
                {
                  hamiltonian->evalEnergy(d_occupation,
                                          *d_kohnShamWaveFunctions);
                }

              RealType elecEnergy = d_hamitonianElec->getEnergy();
              d_rootCout << "Electrostatic energy: " << elecEnergy << "\n";

              if(d_isNlcc && d_isONCVNonLocPSP)
              {
                quadrature::add((ValueType)1.0,
                                d_densityOutQuadValues,
                                (ValueType)1.0,
                                d_coreCorrDensUPF,
                                d_coreCorrectedDensity,
                                *d_linAlgOpContext);
                d_hamitonianXC->reinitField(d_coreCorrectedDensity);
              }
              else
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
          if (auto hamiltonian = std::dynamic_pointer_cast<
                ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                     ValueTypeElectrostaticsCoeff,
                                     ValueTypeWaveFunctionBasis,
                                     memorySpace,
                                     dim>>(d_hamitonianElec))
            {
              hamiltonian->reinitField(d_densityOutQuadValues);
            }
          else if (auto hamiltonian = std::dynamic_pointer_cast<
                     ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                               ValueTypeElectrostaticsCoeff,
                                               ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(d_hamitonianElec))
            {
              hamiltonian->reinitField(d_densityOutQuadValues);
            }

          d_hamitonianKin->evalEnergy(d_occupation,
                                      *d_feBMWaveFn,
                                      *d_kohnShamWaveFunctions);
          RealType kinEnergy = d_hamitonianKin->getEnergy();
          d_rootCout << "Kinetic energy: " << kinEnergy << "\n";

          if (auto hamiltonian = std::dynamic_pointer_cast<
                ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                     ValueTypeElectrostaticsCoeff,
                                     ValueTypeWaveFunctionBasis,
                                     memorySpace,
                                     dim>>(d_hamitonianElec))
            {
              hamiltonian->evalEnergy();
            }
          else if (auto hamiltonian = std::dynamic_pointer_cast<
                     ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                               ValueTypeElectrostaticsCoeff,
                                               ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(d_hamitonianElec))
            {
              hamiltonian->evalEnergy(d_occupation, *d_kohnShamWaveFunctions);
            }

          RealType elecEnergy = d_hamitonianElec->getEnergy();
          d_rootCout << "Electrostatic energy: " << elecEnergy << "\n";

          if(d_isNlcc && d_isONCVNonLocPSP)
          {
            quadrature::add((ValueType)1.0,
                            d_densityOutQuadValues,
                            (ValueType)1.0,
                            d_coreCorrDensUPF,
                            d_coreCorrectedDensity,
                            *d_linAlgOpContext);
            d_hamitonianXC->reinitField(d_coreCorrectedDensity);
          }
          else
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
