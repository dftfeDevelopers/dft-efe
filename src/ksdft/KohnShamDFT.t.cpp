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
      double
      computeEntropicEnergy(const std::vector<double> &partialOccupancies,
                            const double               temperature)
      {
        double          entropy        = 0.0;
        const size_type numEigenValues = partialOccupancies.size();

        for (size_type i = 0; i < numEigenValues; ++i)
          {
            double partialOccupancy = partialOccupancies[i];

            double fTimeslogf, oneminusfTimeslogoneminusf;

            if (std::abs(partialOccupancy - 1.0) <= 1e-07 ||
                std::abs(partialOccupancy) <= 1e-07)
              {
                fTimeslogf                 = 0.0;
                oneminusfTimeslogoneminusf = 0.0;
              }
            else
              {
                fTimeslogf = partialOccupancy * log(partialOccupancy);
                oneminusfTimeslogoneminusf =
                  (1.0 - partialOccupancy) * log(1.0 - partialOccupancy);
              }
            entropy += -2.0 * Constants::BOLTZMANN_CONST_HARTREE *
                       (fTimeslogf + oneminusfTimeslogoneminusf);
          }

        return temperature * entropy;
      }

      template <typename RealType>
      RealType
      computeResidualQuadData(
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &outValues,
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &inValues,
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &residualValues,
        const std::vector<RealType> &JxW,
        const bool                                   computeNorm,
        linearAlgebra::LinAlgOpContext<memorySpaceHost> &linAlgOpContext,
        const utils::mpi::MPIComm &                  mpiComm)
      {
        linearAlgebra::blasLapack::axpby<RealType, RealType, memorySpaceHost>(
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

      template <typename RealType>
      RealType
      normalizeDensityQuadData(
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost> &inValues,
        const size_type numElectrons,
        const std::vector<RealType> &JxW,
        linearAlgebra::LinAlgOpContext<memorySpaceHost> &linAlgOpContext,
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
        utils::MemoryStorage<ValueType, memorySpaceHost> multiVectorGuessHost
          (multiVectorGuess.localSize() * multiVectorGuess.numVectors(), ValueType());
        int rank;
        utils::mpi::MPICommRank(
          multiVectorGuess.getMPIPatternP2P()->mpiCommunicator(), &rank);
        boost::math::normal normDist;
        std::mt19937        randomIntGenerator(rank);
        ValueType *         temp = multiVectorGuessHost.data();
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

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
          memoryTransfer;

        memoryTransfer.copy(multiVectorGuessHost.size(),
                            multiVectorGuess.data(),
                            multiVectorGuessHost.data());

      }
    } // namespace KohnShamDFTInternal

    // used if analytical vself canellation route taken
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
        const double &                   smearedChargeRadius,
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
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &electronChargeDensityInput,
        /* Basis related info */
        /* Field boundary */
        std::shared_ptr<
          const basis::FEBasisManager<ValueTypeElectrostaticsCoeff,
                                      ValueTypeElectrostaticsBasis,
                                      memorySpaceHost,
                                      dim>>               feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                    ValueTypeWaveFunctionBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        /* Field data storages poisson solves*/
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDElectronicChargeRhs,
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
      , d_linAlgOpContextHost(linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_kohnShamWaveFunctions(feBMWaveFn->getMPIPatternP2P(),
                                linAlgOpContext,
                                numWantedEigenvalues,
                                (ValueType)0.0)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
      , d_isONCVNonLocPSP(false)
      , d_isNlcc(false)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
    {
      d_p.registerStart("Pre Init Checks");
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
        d_kohnShamWaveFunctions);
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
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(d_densityInQuadValues,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";
      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues > KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues);

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

      size_type waveFnBatch =
        numWantedEigenvalues > KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues;

      d_p.registerStart("Hamiltonian Operator Creation");
      // form the kohn sham operator
      d_hamitonianOperator =
        std::make_shared<KohnShamOperatorContextFE<ValueTypeElectrostaticsCoeff,
                                                   ValueTypeElectrostaticsBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   memorySpace,
                                                   dim>>(
          *feBMWaveFn,
          hamiltonianComponentsVec,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          waveFnBatch);
      d_p.registerEnd("Hamiltonian Operator Creation");

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      d_lanczosGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(d_lanczosGuess, 1);

      d_kohnShamWaveFunctions.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        d_kohnShamWaveFunctions, numWantedEigenvalues);

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize = KSDFTDefaults::SCALAPACK_BLOCK_SIZE;
      d_elpaScala = std::make_shared<linearAlgebra::ElpaScalapackManager>(
        d_mpiCommDomain,
        scalapackParalProcs,
        useELPA,
        scalapackBlockSize,
        useELPADeviceKernel);

      d_elpaScala->processGridELPASetup(numWantedEigenvalues);
      utils::mpi::MPIBarrier(d_mpiCommDomain);

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       numWantedEigenvalues,
                       d_lanczosGuess,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }

      // form the kohn sham operator
      d_ksEigSolve = std::make_shared<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
        numElectrons,
        smearingTemperature,
        fermiEnergyTolerance,
        fracOccupancyTolerance,
        eigenSolveResidualTolerance,
        maxChebyshevFilterPass,
        numWantedEigenvalues,
        d_lanczosGuess,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext);

      d_p.registerEnd("KS EigenSolver Init");

      d_densCalc =
        std::make_shared<DensityCalculator<ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(
          feBDEXCHamiltonian,
          *feBMWaveFn,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          KSDFTDefaults::MAX_DENSCOMP_WAVEFN_BATCH_SIZE);

      if (dynamic_cast<const utils::PointChargePotentialFunction *>(
            &externalPotentialFunction) != nullptr)
        d_isPSPCalculation = false;
      else
        d_isPSPCalculation = true;
      d_p.print();
    }

    // used if numerical poisson solve vself canellation route taken
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
        const double &                   smearedChargeRadius,
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
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &electronChargeDensityInput,
        /* Basis related info */
        /* Field boundary */
        std::shared_ptr<
          const basis::FEBasisManager<ValueTypeElectrostaticsCoeff,
                                      ValueTypeElectrostaticsBasis,
                                      memorySpaceHost,
                                      dim>>               feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                    ValueTypeWaveFunctionBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        /* Field data storages poisson solves */
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDNuclChargeStiffnessMatrixNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDNuclChargeRhsNumSol,
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
      , d_linAlgOpContextHost(linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_kohnShamWaveFunctions(feBMWaveFn->getMPIPatternP2P(),
                                linAlgOpContext,
                                numWantedEigenvalues,
                                (ValueType)0.0)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
      , d_isONCVNonLocPSP(false)
      , d_isNlcc(false)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
    {
      d_p.registerStart("Pre Init Checks");
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
        d_kohnShamWaveFunctions);
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
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(d_densityInQuadValues,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues > KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues);

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

      size_type waveFnBatch =
        numWantedEigenvalues > KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues;

      d_p.registerStart("Hamiltonian Operator Creation");
      // form the kohn sham operator
      d_hamitonianOperator =
        std::make_shared<KohnShamOperatorContextFE<ValueTypeElectrostaticsCoeff,
                                                   ValueTypeElectrostaticsBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   memorySpace,
                                                   dim>>(
          *feBMWaveFn,
          hamiltonianComponentsVec,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          waveFnBatch);
      d_p.registerEnd("Hamiltonian Operator Creation");
      d_p.print();

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      d_lanczosGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(d_lanczosGuess, 1);

      d_kohnShamWaveFunctions.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        d_kohnShamWaveFunctions, numWantedEigenvalues);

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize = KSDFTDefaults::SCALAPACK_BLOCK_SIZE;
      d_elpaScala = std::make_shared<linearAlgebra::ElpaScalapackManager>(
        d_mpiCommDomain,
        scalapackParalProcs,
        useELPA,
        scalapackBlockSize,
        useELPADeviceKernel);

      d_elpaScala->processGridELPASetup(numWantedEigenvalues);
      utils::mpi::MPIBarrier(d_mpiCommDomain);

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       numWantedEigenvalues,
                       d_lanczosGuess,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }

      // form the kohn sham operator
      d_ksEigSolve = std::make_shared<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
        numElectrons,
        smearingTemperature,
        fermiEnergyTolerance,
        fracOccupancyTolerance,
        eigenSolveResidualTolerance,
        maxChebyshevFilterPass,
        numWantedEigenvalues,
        d_lanczosGuess,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext);

      d_p.registerEnd("KS EigenSolver Init");

      d_densCalc =
        std::make_shared<DensityCalculator<ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(
          feBDEXCHamiltonian,
          *feBMWaveFn,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          KSDFTDefaults::MAX_DENSCOMP_WAVEFN_BATCH_SIZE);

      if (dynamic_cast<const utils::PointChargePotentialFunction *>(
            &externalPotentialFunction) != nullptr)
        d_isPSPCalculation = false;
      else
        d_isPSPCalculation = true;
      d_p.print();
    }

      // used if delta rho approach is taken with phi total from 1D KS solve
      // with analytical vself energy cancellation    
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
        const std::vector<std::string> & atomSymbolVec,
        const double &                   smearedChargeRadius,
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
        const utils::ScalarSpatialFunctionReal
          &atomicTotalElectroPotentialFunction,
        const utils::ScalarSpatialFunctionReal
          &atomicElectronicChargeDensityFunction,
        /* Field boundary */
        std::shared_ptr<
          const basis::FEBasisManager<ValueTypeElectrostaticsCoeff,
                                      ValueTypeElectrostaticsBasis,
                                      memorySpaceHost,
                                      dim>>               feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                    ValueTypeWaveFunctionBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        /* Field data storages poisson solves*/
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDElectronicChargeRhs,
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
        bool isResidualChebyshevFilter,
        /* TCI related info */
        const atoms::TCIADataParams &params)
      : d_mixingHistory(mixingHistory)
      , d_mixingParameter(mixingParameter)
      , d_isAdaptiveAndersonMixingParameter(isAdaptiveAndersonMixingParameter)
      , d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
      // , d_densityInQuadValues(electronChargeDensityInput)
      // , d_densityResidualQuadValues(electronChargeDensityInput)
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
      , d_kohnShamWaveFunctions(feBMWaveFn->getMPIPatternP2P(),
                                linAlgOpContext,
                                numWantedEigenvalues,
                                (ValueType)0.0)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
      , d_isONCVNonLocPSP(false)
      , d_isNlcc(false)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
    {
      d_p.registerStart("Pre Init Checks");
      if (dynamic_cast<
            const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim> *>(
            &feBMWaveFn->getBasisDofHandler()) != nullptr)
        d_isOEFEBasis = true;
      else
        d_isOEFEBasis = false;

      d_densityInQuadValues =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      KohnShamDFTInternal::generateRandNormDistMultivec(
        d_kohnShamWaveFunctions);
      utils::throwException(d_densityInQuadValues.getNumberComponents() == 1,
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

      //************* CHANGE THIS **********************
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      RealType *quadValueIter = d_densityInQuadValues.begin();
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
                quadRuleContainerVal = quadRuleContainerRho;
      size_type cumulativeQuadInCell = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerVal->nCells(); iCell++)
        {
          size_type numQuadInCell =
            quadRuleContainerVal->nCellQuadraturePoints(iCell);
          std::vector<RealType> valInCellQuad =
            (atomicElectronicChargeDensityFunction)(
              quadRuleContainerVal->getCellRealPoints(iCell));
          for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
            {
              quadValueIter[cumulativeQuadInCell + iQuad] =
                valInCellQuad[iQuad];
            }
          cumulativeQuadInCell += numQuadInCell;
        }

      d_densityResidualQuadValues = d_densityInQuadValues;

      d_densityOutQuadValues = d_densityInQuadValues;
      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(d_densityInQuadValues,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues > KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues);

      std::unordered_map<std::string, std::shared_ptr<atoms::AtomTCIASpline>>
        fieldToTCIASplineMap = {};
      if (params.folderName != "")
        {
          d_rootCout
            << "\nTCIA Data provided , using that for atomic data energy contributions.\n";

          fieldToTCIASplineMap["rhoAtom-phiAtom"] =
            std::make_shared<atoms::AtomTCIASpline>("rhoAtom-phiAtom",
                                                    params,
                                                    atomSymbolVec,
                                                    std::vector<std::string>{
                                                      "S"},
                                                    1000);

          fieldToTCIASplineMap["rhoAtom-vlocCorrection"] =
            std::make_shared<atoms::AtomTCIASpline>("rhoAtom-vlocCorrection",
                                                    params,
                                                    atomSymbolVec,
                                                    std::vector<std::string>{
                                                      "S"},
                                                    1000);

          fieldToTCIASplineMap["bSmear-phiAtom"] =
            std::make_shared<atoms::AtomTCIASpline>("bSmear-phiAtom",
                                                    params,
                                                    atomSymbolVec,
                                                    std::vector<std::string>{
                                                      "S"},
                                                    1000);

          bool useEZZCorr = false;
          for (auto i : fieldToTCIASplineMap)
            {
              double smearedChargeRadiusZZCorr =
                i.second->smearedChargeRadiusZZCorr();
              double smearedChargeRadius = i.second->smearedChargeRadius();
              if (std::abs(smearedChargeRadiusZZCorr - smearedChargeRadius) >
                  1e-12)
                {
                  useEZZCorr = true;
                  d_rootCout
                    << "\nOne of the smeared charge radiuses is > 0.7, using the energy correction due to spreaded nuclear charges.\n\n";
                  break;
                }
            }

          if (useEZZCorr)
            fieldToTCIASplineMap["sumBZZCorrBSmear-diffVZZCorrVSmear"] =
              std::make_shared<atoms::AtomTCIASpline>(
                "sumBZZCorrBSmear-diffVZZCorrVSmear",
                params,
                std::vector<std::string>{"DefaultAtom"},
                std::vector<std::string>{"S"},
                1000);
        }
      else
        {
          d_rootCout
            << "\nTCIA Data not provided , using bSmear quad rule for atomic data energy contributions.\n\n";
        }

      d_hamitonianElec =
        std::make_shared<ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>(
          atomCoordinates,
          atomSymbolVec,
          atomCharges,
          smearedChargeRadius,
          // d_densityOutQuadValues,  /*NOTE: Atomic density input should not be
          // normalized*/ atomicTotalElecPotNuclearQuad,
          // atomicTotalElecPotElectronicQuad,
          atomicTotalElectroPotentialFunction,
          atomicElectronicChargeDensityFunction,
          feBMTotalCharge,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDElectrostaticsHamiltonian,
          externalPotentialFunction,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          fieldToTCIASplineMap);
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

      size_type waveFnBatch =
        numWantedEigenvalues > KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues;

      d_p.registerStart("Hamiltonian Operator Creation");
      // form the kohn sham operator
      d_hamitonianOperator =
        std::make_shared<KohnShamOperatorContextFE<ValueTypeElectrostaticsCoeff,
                                                   ValueTypeElectrostaticsBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   memorySpace,
                                                   dim>>(
          *feBMWaveFn,
          hamiltonianComponentsVec,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          waveFnBatch);
      d_p.registerEnd("Hamiltonian Operator Creation");

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      d_lanczosGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(d_lanczosGuess, 1);

      d_kohnShamWaveFunctions.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        d_kohnShamWaveFunctions, numWantedEigenvalues);

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize = KSDFTDefaults::SCALAPACK_BLOCK_SIZE;
      d_elpaScala = std::make_shared<linearAlgebra::ElpaScalapackManager>(
        d_mpiCommDomain,
        scalapackParalProcs,
        useELPA,
        scalapackBlockSize,
        useELPADeviceKernel);

      d_elpaScala->processGridELPASetup(numWantedEigenvalues);
      utils::mpi::MPIBarrier(d_mpiCommDomain);
      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       numWantedEigenvalues,
                       d_lanczosGuess,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }

      // form the kohn sham operator
      d_ksEigSolve = std::make_shared<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
        numElectrons,
        smearingTemperature,
        fermiEnergyTolerance,
        fracOccupancyTolerance,
        eigenSolveResidualTolerance,
        maxChebyshevFilterPass,
        numWantedEigenvalues,
        d_lanczosGuess,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext);

      d_p.registerEnd("KS EigenSolver Init");

      d_densCalc =
        std::make_shared<DensityCalculator<ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(
          feBDEXCHamiltonian,
          *feBMWaveFn,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          KSDFTDefaults::MAX_DENSCOMP_WAVEFN_BATCH_SIZE);

      if (dynamic_cast<const utils::PointChargePotentialFunction *>(
            &externalPotentialFunction) != nullptr)
        d_isPSPCalculation = false;
      else
        d_isPSPCalculation = true;
      d_p.print();
    }

      //// used if analytical vself canellation route taken with PSP
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
        const std::vector<std::string> & atomSymbolVec,
        const double &                   smearedChargeRadius,
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
        /* Field boundary */
        std::shared_ptr<
          const basis::FEBasisManager<ValueTypeElectrostaticsCoeff,
                                      ValueTypeElectrostaticsBasis,
                                      memorySpaceHost,
                                      dim>>               feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                    ValueTypeWaveFunctionBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        /* Field data storages poisson solves*/
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDElectronicChargeRhs,
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
        /* PSP related info */
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                          memorySpace>>
          feBDAtomCenterNonLocalOperator,
        const std::map<std::string, std::string> &atomSymbolToPSPFilename,
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
      , d_numMaxSCFIter(maxSCFIter)
      , d_MContext(&MContext)
      , d_MInvContext(&MInvContext)
      , d_mpiCommDomain(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator())
      , d_mixingScheme(d_mpiCommDomain)
      , d_numWantedEigenvalues(numWantedEigenvalues)
      , d_linAlgOpContext(linAlgOpContext)
      , d_linAlgOpContextHost(linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_kohnShamWaveFunctions(feBMWaveFn->getMPIPatternP2P(),
                                linAlgOpContext,
                                numWantedEigenvalues,
                                (ValueType)0.0)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
    {
      d_p.registerStart("Pre Init Checks");
      const std::vector<std::string> metadataNames =
        atoms::AtomSphDataPSPDefaults::METADATANAMES;
      std::vector<std::string> fieldNamesPSP = {"vlocal", "rhoatom"};

      d_atomSphericalDataContainerPSP =
        std::make_shared<atoms::AtomSphericalDataContainer>(
          atoms::AtomSphericalDataType::PSEUDOPOTENTIAL,
          atomSymbolToPSPFilename,
          fieldNamesPSP,
          metadataNames);

      for (int i = 0; i < atomSymbolVec.size(); i++)
        {
          if (std::abs(std::stod(d_atomSphericalDataContainerPSP->getMetadata(
                atomSymbolVec[i], "z_valence"))) -
                std::abs(atomCharges[i]) >
              1e-12)
            {
              utils::throwException(
                false,
                "The input basis file Z does not match with that given in input.");
            }
        }

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
              d_isONCVNonLocPSP = true;
              bool coreCorrect  = false;
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
          else
            {
              if (d_isONCVNonLocPSP)
                {
                  utils::throwException(
                    false,
                    "All the Atoms should have nonLocal Components in PSP.");
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

      d_densityInQuadValues =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      d_densityOutQuadValues      = d_densityInQuadValues;
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
        d_kohnShamWaveFunctions);
      utils::throwException(d_densityInQuadValues.getNumberComponents() == 1,
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

      const atoms::AtomSevereFunction rho(d_atomSphericalDataContainerPSP,
                                               atomSymbolVec,
                                               atomCoordinates,
                                               "rhoatom",
                                               0,
                                               1);

      RealType *quadValueIter = d_densityInQuadValues.begin();
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
                quadRuleContainerVal = quadRuleContainerRho;
      size_type cumulativeQuadInCell = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerVal->nCells(); iCell++)
        {
          size_type numQuadInCell =
            quadRuleContainerVal->nCellQuadraturePoints(iCell);
          std::vector<RealType> valInCellQuad =
            (rho)(quadRuleContainerVal->getCellRealPoints(iCell));
          for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
            {
              quadValueIter[cumulativeQuadInCell + iQuad] =
                valInCellQuad[iQuad];
            }
          cumulativeQuadInCell += numQuadInCell;
        }

      //************* CHANGE THIS **********************
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(d_densityInQuadValues,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues > KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues);

      size_type waveFnBatch =
        numWantedEigenvalues > KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues;

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
          feBDAtomCenterNonLocalOperator,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          waveFnBatch);

      if (d_isNlcc && d_isONCVNonLocPSP)
        {
          d_coreCorrDensUPF =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

          d_coreCorrectedDensity =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

          const atoms::AtomSevereFunction rhoCoreCorrection(
            d_atomSphericalDataContainerPSP,
            atomSymbolVec,
            atomCoordinates,
            "nlcc",
            0,
            1);

          RealType *quadValueIter = d_coreCorrDensUPF.begin();
          std::shared_ptr<const quadrature::QuadratureRuleContainer>
                    quadRuleContainerVal = quadRuleContainerRho;
          size_type cumulativeQuadInCell = 0;
          for (size_type iCell = 0; iCell < quadRuleContainerVal->nCells();
               iCell++)
            {
              size_type numQuadInCell =
                quadRuleContainerVal->nCellQuadraturePoints(iCell);
              std::vector<RealType> valInCellQuad = (rhoCoreCorrection)(
                quadRuleContainerVal->getCellRealPoints(iCell));
              for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
                {
                  quadValueIter[cumulativeQuadInCell + iQuad] =
                    valInCellQuad[iQuad];
                }
              cumulativeQuadInCell += numQuadInCell;
            }
          quadrature::add((ValueType)1.0,
                          d_densityInQuadValues,
                          (ValueType)1.0,
                          d_coreCorrDensUPF,
                          d_coreCorrectedDensity,
                          *d_linAlgOpContextHost);

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
        std::make_shared<KohnShamOperatorContextFE<ValueTypeElectrostaticsCoeff,
                                                   ValueTypeElectrostaticsBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   memorySpace,
                                                   dim>>(
          *feBMWaveFn,
          hamiltonianComponentsVec,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          waveFnBatch);
      d_p.registerEnd("Hamiltonian Operator Creation");

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      d_lanczosGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(d_lanczosGuess, 1);

      d_kohnShamWaveFunctions.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        d_kohnShamWaveFunctions, numWantedEigenvalues);

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize = KSDFTDefaults::SCALAPACK_BLOCK_SIZE;
      d_elpaScala = std::make_shared<linearAlgebra::ElpaScalapackManager>(
        d_mpiCommDomain,
        scalapackParalProcs,
        useELPA,
        scalapackBlockSize,
        useELPADeviceKernel);

      d_elpaScala->processGridELPASetup(numWantedEigenvalues);
      utils::mpi::MPIBarrier(d_mpiCommDomain);

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       numWantedEigenvalues,
                       d_lanczosGuess,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }

      // form the kohn sham operator
      d_ksEigSolve = std::make_shared<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
        numElectrons,
        smearingTemperature,
        fermiEnergyTolerance,
        fracOccupancyTolerance,
        eigenSolveResidualTolerance,
        maxChebyshevFilterPass,
        numWantedEigenvalues,
        d_lanczosGuess,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext);

      d_p.registerEnd("KS EigenSolver Init");

      d_densCalc =
        std::make_shared<DensityCalculator<ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(
          feBDEXCHamiltonian,
          *feBMWaveFn,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          KSDFTDefaults::MAX_DENSCOMP_WAVEFN_BATCH_SIZE);

      d_isPSPCalculation = true;
      d_p.print();
    }

      // used if delta rho with PSP approach is taken with phi total from 1D KS
      // solve with analytical vself energy cancellation
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
        const std::vector<std::string> & atomSymbolVec,
        const double &                   smearedChargeRadius,
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
        /* Atomic Field for delta rho ; Here vTotal atomic scalar sp fn.*/
        const utils::ScalarSpatialFunctionReal
          &atomicTotalElectroPotentialFunction,
        const utils::ScalarSpatialFunctionReal
          &atomicElectronicChargeDensityFunction,
        /* Field boundary */
        std::shared_ptr<
          const basis::FEBasisManager<ValueTypeElectrostaticsCoeff,
                                      ValueTypeElectrostaticsBasis,
                                      memorySpaceHost,
                                      dim>>               feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                    ValueTypeWaveFunctionBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        /* Field data storages poisson solves*/
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>> feBDElectronicChargeRhs,
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
        /* PSP related info */
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                          memorySpace>>
          feBDAtomCenterNonLocalOperator,
        const std::map<std::string, std::string> &atomSymbolToPSPFilename,
        /* linAgOperations Context*/
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        /* basis overlap related info */
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext,
        bool             isResidualChebyshevFilter,
        /* TCI related info */
        const atoms::TCIADataParams &params)
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
      , d_linAlgOpContextHost(linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_kohnShamWaveFunctions(feBMWaveFn->getMPIPatternP2P(),
                                linAlgOpContext,
                                numWantedEigenvalues,
                                (ValueType)0.0)
      , d_lanczosGuess(feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       0.0,
                       1.0)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_occupation(numWantedEigenvalues, 0)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
    {
      utils::Profiler<utils::MemorySpace::HOST> p(
        feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
        "Pre Init Checks");
      d_p.registerStart("Pre Init Checks");
      p.registerStart("atomSphericalDataContainerPSP create");
      const std::vector<std::string> metadataNames =
        atoms::AtomSphDataPSPDefaults::METADATANAMES;
      std::vector<std::string> fieldNamesPSP = {"vlocal"};

      d_atomSphericalDataContainerPSP =
        std::make_shared<atoms::AtomSphericalDataContainer>(
          atoms::AtomSphericalDataType::PSEUDOPOTENTIAL,
          atomSymbolToPSPFilename,
          fieldNamesPSP,
          metadataNames);

      for (int i = 0; i < atomSymbolVec.size(); i++)
        {
          if (std::abs(std::stod(d_atomSphericalDataContainerPSP->getMetadata(
                atomSymbolVec[i], "z_valence"))) -
                std::abs(atomCharges[i]) >
              1e-12)
            {
              utils::throwException(
                false,
                "The input basis file Z does not match with that given in input.");
            }
        }

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
              d_isONCVNonLocPSP = true;
              bool coreCorrect  = false;
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
          else
            {
              if (d_isONCVNonLocPSP)
                {
                  utils::throwException(
                    false,
                    "All the Atoms should have nonLocal Components in PSP.");
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
      p.registerEnd("atomSphericalDataContainerPSP create");
      p.registerStart("generateRandNormDistMultivec");

      d_densityInQuadValues =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

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
        d_kohnShamWaveFunctions);
      utils::throwException(d_densityInQuadValues.getNumberComponents() == 1,
                            "Electron density should have only one component.");
      p.registerEnd("generateRandNormDistMultivec");
      p.registerStart("rhoAtFunc");      
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

      RealType *quadValueIter = d_densityInQuadValues.begin();
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
                quadRuleContainerVal = quadRuleContainerRho;
      size_type cumulativeQuadInCell = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerVal->nCells(); iCell++)
        {
          size_type numQuadInCell =
            quadRuleContainerVal->nCellQuadraturePoints(iCell);
          std::vector<RealType> valInCellQuad =
            (atomicElectronicChargeDensityFunction)(
              quadRuleContainerVal->getCellRealPoints(iCell));
          for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
            {
              quadValueIter[cumulativeQuadInCell + iQuad] =
                valInCellQuad[iQuad];
            }
          cumulativeQuadInCell += numQuadInCell;
        }

      p.registerEnd("rhoAtFunc");
      //************* CHANGE THIS **********************
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      d_densityOutQuadValues = d_densityInQuadValues;
      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(d_densityInQuadValues,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";
      p.print();
      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization Kinetic Op");

      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues > KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues);
      d_p.registerEnd("Hamiltonian Components Initilization Kinetic Op");
      d_p.registerStart("Hamiltonian Components Initilization Electrostatic Op");

      size_type waveFnBatch =
        numWantedEigenvalues > KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults::MAX_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues;

      std::unordered_map<std::string, std::shared_ptr<atoms::AtomTCIASpline>>
        fieldToTCIASplineMap = {};
      if (params.folderName != "")
        {
          d_rootCout
            << "\nTCIA Data provided , using that for atomic data energy contributions.\n";

          fieldToTCIASplineMap["rhoAtom-phiAtom"] =
            std::make_shared<atoms::AtomTCIASpline>("rhoAtom-phiAtom",
                                                    params,
                                                    atomSymbolVec,
                                                    std::vector<std::string>{
                                                      "S"},
                                                    1000);

          fieldToTCIASplineMap["rhoAtom-vlocCorrection"] =
            std::make_shared<atoms::AtomTCIASpline>("rhoAtom-vlocCorrection",
                                                    params,
                                                    atomSymbolVec,
                                                    std::vector<std::string>{
                                                      "S"},
                                                    1000);

          fieldToTCIASplineMap["bSmear-phiAtom"] =
            std::make_shared<atoms::AtomTCIASpline>("bSmear-phiAtom",
                                                    params,
                                                    atomSymbolVec,
                                                    std::vector<std::string>{
                                                      "S"},
                                                    1000);

          bool useEZZCorr = false;
          for (auto i : fieldToTCIASplineMap)
            {
              double smearedChargeRadiusZZCorr =
                i.second->smearedChargeRadiusZZCorr();
              double smearedChargeRadius = i.second->smearedChargeRadius();

              if (std::abs(smearedChargeRadiusZZCorr - smearedChargeRadius) >
                  1e-12)
                {
                  useEZZCorr = true;
                  d_rootCout
                    << "\nOne of the smeared charge radiuses is > 0.7, using the energy correction due to spreaded nuclear charges.\n\n";
                  break;
                }
            }

          if (useEZZCorr)
            fieldToTCIASplineMap["sumBZZCorrBSmear-diffVZZCorrVSmear"] =
              std::make_shared<atoms::AtomTCIASpline>(
                "sumBZZCorrBSmear-diffVZZCorrVSmear",
                params,
                std::vector<std::string>{"DefaultAtom"},
                std::vector<std::string>{"S"},
                1000);

          for (auto i : fieldToTCIASplineMap)
            {
              double smearedChargeRadiusZZCorr =
                i.second->smearedChargeRadiusZZCorr();
              if (std::abs(smearedChargeRadiusZZCorr - smearedChargeRadius) >
                  1e-12)
                {
                  utils::throwException(
                    false,
                    "The TCIA data smearedChargeRadiusZZCorr " +
                      std::to_string(smearedChargeRadiusZZCorr) +
                      " does not match with input smearedChargeRadius " +
                      std::to_string(smearedChargeRadius));
                }
            }

          for (int atomSymbolId = 0; atomSymbolId < atomSymbolVec.size();
               atomSymbolId++)
            {
              if (d_atomSphericalDataContainerPSP->getMetadata(
                    atomSymbolVec[atomSymbolId], "pseudo_type") !=
                  fieldToTCIASplineMap["bSmear-phiAtom"]->getVLocInfo(
                    atomSymbolVec[atomSymbolId], "pseudo_type"))
                {
                  utils::throwException(
                    false,
                    "The PSP upf file pseudo_type does not match the UPF file used"
                    " for TCIA data generation for atomSymbol " +
                      atomSymbolVec[atomSymbolId] + ".");
                }
              if (d_atomSphericalDataContainerPSP->getMetadata(
                    atomSymbolVec[atomSymbolId], "z_valence") !=
                  fieldToTCIASplineMap["bSmear-phiAtom"]->getVLocInfo(
                    atomSymbolVec[atomSymbolId], "z_valence"))
                {
                  utils::throwException(
                    false,
                    "The PSP upf file z_valence does not match the UPF file used"
                    " for TCIA data generation for atomSymbol " +
                      atomSymbolVec[atomSymbolId] + ".");
                }
              if (d_atomSphericalDataContainerPSP->getMetadata(
                    atomSymbolVec[atomSymbolId], "total_psenergy") !=
                  fieldToTCIASplineMap["bSmear-phiAtom"]->getVLocInfo(
                    atomSymbolVec[atomSymbolId], "total_psenergy"))
                {
                  utils::throwException(
                    false,
                    "The PSP upf file total_psenergy does not match the UPF file used"
                    " for TCIA data generation for atomSymbol " +
                      atomSymbolVec[atomSymbolId] + ".");
                }
            }
        }
      else
        {
          d_rootCout
            << "\nTCIA Data not provided , using bSmear quad rule for atomic data energy contributions.\n\n";
        }

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
          // d_densityOutQuadValues, /*NOTE: Atomic density input should not be
          // normalized*/ 
          atomicTotalElectroPotentialFunction,
          atomicElectronicChargeDensityFunction,
          feBMTotalCharge,
          feBMWaveFn,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDElectrostaticsHamiltonian,
          feBDAtomCenterNonLocalOperator,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          waveFnBatch,
          fieldToTCIASplineMap);
        d_p.registerEnd("Hamiltonian Components Initilization Electrostatic Op");
        d_p.registerStart("Hamiltonian Components Initilization Exc Op");

      if (d_isNlcc && d_isONCVNonLocPSP)
        {
          d_coreCorrDensUPF =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

          d_coreCorrectedDensity =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

          const atoms::AtomSevereFunction rhoCoreCorrection(
            d_atomSphericalDataContainerPSP,
            atomSymbolVec,
            atomCoordinates,
            "nlcc",
            0,
            1);

          RealType *quadValueIter = d_coreCorrDensUPF.begin();
          std::shared_ptr<const quadrature::QuadratureRuleContainer>
                    quadRuleContainerVal = quadRuleContainerRho;
          size_type cumulativeQuadInCell = 0;
          for (size_type iCell = 0; iCell < quadRuleContainerVal->nCells();
               iCell++)
            {
              size_type numQuadInCell =
                quadRuleContainerVal->nCellQuadraturePoints(iCell);
              std::vector<RealType> valInCellQuad = (rhoCoreCorrection)(
                quadRuleContainerVal->getCellRealPoints(iCell));
              for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
                {
                  quadValueIter[cumulativeQuadInCell + iQuad] =
                    valInCellQuad[iQuad];
                }
              cumulativeQuadInCell += numQuadInCell;
            }
          quadrature::add((ValueType)1.0,
                          d_densityInQuadValues,
                          (ValueType)1.0,
                          d_coreCorrDensUPF,
                          d_coreCorrectedDensity,
                          *d_linAlgOpContextHost);

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

      d_p.registerEnd("Hamiltonian Components Initilization Exc Op");
      d_p.registerStart("Hamiltonian Operator Creation");
      // form the kohn sham operator
      d_hamitonianOperator =
        std::make_shared<KohnShamOperatorContextFE<ValueTypeElectrostaticsCoeff,
                                                   ValueTypeElectrostaticsBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   memorySpace,
                                                   dim>>(
          *feBMWaveFn,
          hamiltonianComponentsVec,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          waveFnBatch);
      d_p.registerEnd("Hamiltonian Operator Creation");

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      d_lanczosGuess.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(d_lanczosGuess, 1);

      d_kohnShamWaveFunctions.updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        d_kohnShamWaveFunctions, numWantedEigenvalues);

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize = KSDFTDefaults::SCALAPACK_BLOCK_SIZE;
      d_elpaScala = std::make_shared<linearAlgebra::ElpaScalapackManager>(
        d_mpiCommDomain,
        scalapackParalProcs,
        useELPA,
        scalapackBlockSize,
        useELPADeviceKernel);

      d_elpaScala->processGridELPASetup(numWantedEigenvalues);
      utils::mpi::MPIBarrier(d_mpiCommDomain);

      if (isResidualChebyshevFilter)
        {
          KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>
            ksEigSolve(numElectrons,
                       smearingTemperature,
                       fermiEnergyTolerance,
                       fracOccupancyTolerance,
                       eigenSolveResidualTolerance,
                       1,
                       numWantedEigenvalues,
                       d_lanczosGuess,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           d_kohnShamWaveFunctions,
                           false,
                           *d_MContext,
                           *d_MInvContext);
        }

      // form the kohn sham operator
      d_ksEigSolve = std::make_shared<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>(
        numElectrons,
        smearingTemperature,
        fermiEnergyTolerance,
        fracOccupancyTolerance,
        eigenSolveResidualTolerance,
        maxChebyshevFilterPass,
        numWantedEigenvalues,
        d_lanczosGuess,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext);

      d_p.registerEnd("KS EigenSolver Init");

      d_densCalc =
        std::make_shared<DensityCalculator<ValueTypeWaveFunctionBasis,
                                           ValueTypeWaveFunctionCoeff,
                                           memorySpace,
                                           dim>>(
          feBDEXCHamiltonian,
          *feBMWaveFn,
          linAlgOpContext,
          KSDFTDefaults::CELL_BATCH_SIZE,
          KSDFTDefaults::MAX_DENSCOMP_WAVEFN_BATCH_SIZE);

      d_isPSPCalculation = true;
      d_p.print();
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
                dim>::~KohnShamDFT()
    {
      d_elpaScala->elpaDeallocateHandles();
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
      d_pTotal.reset();
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
          hamiltonian->evalEnergy(d_occupation, d_kohnShamWaveFunctions);
        }

      RealType elecEnergy = d_hamitonianElec->getEnergy();
      d_rootCout << "Electrostatic energy with guess density: " << elecEnergy
                 << "\n";

      utils::MemoryStorage<RealType, memorySpaceHost> jxwDataHost(d_jxwDataHost.size());
      jxwDataHost.copyFrom(d_jxwDataHost);

      d_mixingScheme.addMixingVariable(
        mixingVariable::rho,
        jxwDataHost,
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
          utils::printCurrentMemoryUsage(d_mpiCommDomain, "SCF beginning");
          d_p.reset();
          d_rootCout
            << "************************Begin Self-Consistent-Field Iteration: "
            << std::setw(2) << scfIter + 1 << " ***********************\n";

          // mix the densities with  Anderson mix if scf > 0
          // Update the history of mixing variables

          if (scfIter > 0)
            {
              d_p.registerStart("Density Mixing");
              d_pTotal.registerStart("Density Mixing");
              norm = KohnShamDFTInternal::computeResidualQuadData(
                d_densityOutQuadValues,
                d_densityInQuadValues,
                d_densityResidualQuadValues,
                d_jxwDataHost,
                true,
                *d_linAlgOpContextHost,
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
                *dftefe::linearAlgebra::LinAlgOpContextDefaults::
                  LINALG_OP_CONTXT_HOST);

              // update the mixing variables
              // get next input density
              d_mixingScheme.template mixVariable<memorySpace>(
                mixingVariable::rho,
                d_densityInQuadValues.begin(),
                d_densityInQuadValues.nQuadraturePoints());
              d_pTotal.registerEnd("Density Mixing");
              d_p.registerEnd("Density Mixing");
            }

          // reinit the components of hamiltonian
          if (scfIter > 0)
            {
              d_pTotal.registerStart("Hamiltonian Reinit");
              d_p.registerStart("Hamiltonian Reinit");
              // normalize electroncharge density each scf
              RealType totalDensityInQuad =
                KohnShamDFTInternal::normalizeDensityQuadData(
                  d_densityInQuadValues,
                  d_numElectrons,
                  d_jxwDataHost,
                  *d_linAlgOpContextHost,
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

              if (d_isNlcc && d_isONCVNonLocPSP)
                {
                  quadrature::add((ValueType)1.0,
                                  d_densityInQuadValues,
                                  (ValueType)1.0,
                                  d_coreCorrDensUPF,
                                  d_coreCorrectedDensity,
                                  *d_linAlgOpContextHost);
                  d_hamitonianXC->reinitField(d_coreCorrectedDensity);
                }
              else
                d_hamitonianXC->reinitField(d_densityInQuadValues);

              d_hamiltonianElectroExc->reinit(d_hamitonianElec, d_hamitonianXC);

              std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
                d_hamitonianKin, d_hamiltonianElectroExc};

              d_hamitonianOperator->reinit(*d_feBMWaveFn,
                                           hamiltonianComponentsVec);
              d_p.registerEnd("Hamiltonian Reinit");
              d_pTotal.registerEnd("Hamiltonian Reinit");
            }

          d_p.registerStart("EigenSolve");
          d_pTotal.registerStart("EigenSolve");

          // reinit the chfsi bounds
          if (scfIter > 0)
            {
              d_ksEigSolve->reinitBounds(
                d_kohnShamEnergies[0],
                d_kohnShamEnergies[d_numWantedEigenvalues - 1]);
            }

          if (scfIter == 0 && d_isPSPCalculation)
            d_ksEigSolve->setChebyPolyScalingFactor(1.34);

          // Linear Eigen Solve
          linearAlgebra::EigenSolverError err =
            d_ksEigSolve->solve(*d_hamitonianOperator,
                                d_kohnShamEnergies,
                                d_kohnShamWaveFunctions,
                                true,
                                *d_MContext,
                                *d_MInvContext);

          d_occupation = d_ksEigSolve->getFractionalOccupancy();

          std::vector<RealType> eigSolveResNorm =
            d_ksEigSolve->getEigenSolveResidualNorm();

          d_pTotal.registerEnd("EigenSolve");
          d_p.registerEnd("EigenSolve");            

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

              feBasisOp->interpolate(d_kohnShamWaveFunctions,
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
          d_pTotal.registerStart("Density Compute");
          // compute output rho
          d_densCalc->computeRho(d_occupation,
                                 d_kohnShamWaveFunctions,
                                 d_densityOutQuadValues);

          RealType totalDensityInQuad =
            KohnShamDFTInternal::normalizeDensityQuadData(
              d_densityOutQuadValues,
              d_numElectrons,
              d_jxwDataHost,
              *d_linAlgOpContextHost,
              d_mpiCommDomain,
              true,
              false,
              d_rootCout);

          d_rootCout << "Electron density out : " << totalDensityInQuad << "\n";
          d_pTotal.registerEnd("Density Compute");
          d_p.registerEnd("Density Compute");
          d_p.print();

          // check residual in density if else
          if (d_evaluateEnergyEverySCF)
            {
              d_pTotal.registerStart("Energy Compute");
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
                                          d_kohnShamWaveFunctions);
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
                                          d_kohnShamWaveFunctions);
                }

              RealType elecEnergy = d_hamitonianElec->getEnergy();
              d_rootCout << "Electrostatic energy: " << elecEnergy << "\n";

              if (d_isNlcc && d_isONCVNonLocPSP)
                {
                  quadrature::add((ValueType)1.0,
                                  d_densityOutQuadValues,
                                  (ValueType)1.0,
                                  d_coreCorrDensUPF,
                                  d_coreCorrectedDensity,
                                  *d_linAlgOpContextHost);
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

              RealType entEnergy = KohnShamDFTInternal::computeEntropicEnergy(
                d_occupation, d_smearingTemperature);

              d_rootCout << "Entropic Energy: " << entEnergy << "\n";

              d_rootCout << "Free Energy: " << totalEnergy - entEnergy << "\n";

              d_freeEnergy = totalEnergy - entEnergy;
              d_pTotal.registerEnd("Energy Compute");
            }

          if (scfIter > 0)
            d_rootCout << "Density Residual Norm : " << norm << "\n";

          scfIter += 1;
        }

      if (!d_evaluateEnergyEverySCF)
        {
          d_pTotal.registerStart("Energy Compute");
          int rank;
          utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
          utils::ConditionalOStream rootCout(std::cout, rank == 0, 16, true);

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
                                      d_kohnShamWaveFunctions);
          RealType kinEnergy = d_hamitonianKin->getEnergy();
          rootCout << "Kinetic energy: " << kinEnergy << "\n";

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
              hamiltonian->evalEnergy(d_occupation, d_kohnShamWaveFunctions);
            }

          RealType elecEnergy = d_hamitonianElec->getEnergy();
          rootCout << "Electrostatic energy: " << elecEnergy << "\n";

          if (d_isNlcc && d_isONCVNonLocPSP)
            {
              quadrature::add((ValueType)1.0,
                              d_densityOutQuadValues,
                              (ValueType)1.0,
                              d_coreCorrDensUPF,
                              d_coreCorrectedDensity,
                              *d_linAlgOpContextHost);
              d_hamitonianXC->reinitField(d_coreCorrectedDensity);
            }
          else
            d_hamitonianXC->reinitField(d_densityOutQuadValues);

          d_hamitonianXC->evalEnergy(d_mpiCommDomain);
          RealType xcEnergy = d_hamitonianXC->getEnergy();
          rootCout << "LDA EXC energy: " << xcEnergy << "\n";

          // calculate band energy
          RealType bandEnergy = 0;
          for (size_type i = 0; i < d_occupation.size(); i++)
            {
              bandEnergy += 2 * d_occupation[i] * d_kohnShamEnergies[i];
            }

          rootCout << "Band energy: " << bandEnergy << "\n";

          RealType totalEnergy = kinEnergy + elecEnergy + xcEnergy;

          rootCout << "Ground State Energy: " << totalEnergy << "\n";

          d_groundStateEnergy = totalEnergy;

          RealType entEnergy =
            KohnShamDFTInternal::computeEntropicEnergy(d_occupation,
                                                       d_smearingTemperature);

          rootCout << "Entropic Energy: " << entEnergy << "\n";

          rootCout << "Free Energy: " << totalEnergy - entEnergy << "\n";

          d_freeEnergy = totalEnergy - entEnergy;
          d_pTotal.registerEnd("Energy Compute");
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
                dim>::getFreeEnergy()
    {
      utils::throwException(
        d_isSolved,
        "Cannot call ksdft getFreeEnergy() before solving the KS problem.");
      return d_freeEnergy;
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
                dim>::printTotalInScopeTimings()
    {
      d_ksEigSolve->printTotalInScopeTimings();
      d_pTotal.print();
    }
  } // end of namespace ksdft
} // end of namespace dftefe
