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
                            const double               temperature,
                            const double               spinFactor)
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
            entropy += -spinFactor * Constants::BOLTZMANN_CONST_HARTREE *
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
          &                                              residualValues,
        const std::vector<RealType> &                    JxW,
        const bool                                       computeNorm,
        linearAlgebra::LinAlgOpContext<memorySpaceHost> &linAlgOpContext,
        const utils::mpi::MPIComm &                      mpiComm)
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
              utils::mpi::Types<RealType>::getMPIDatatype(),
              utils::mpi::MPISum,
              mpiComm);
          }
        return std::sqrt(normValue);
      }

      template <typename RealType>
      RealType
      normalizeDensityQuadData(
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &                                              inValues,
        const size_type                                  numElectrons,
        const std::vector<RealType> &                    JxW,
        linearAlgebra::LinAlgOpContext<memorySpaceHost> &linAlgOpContext,
        const utils::mpi::MPIComm &                      mpiComm,
        bool                                             computeTotalDensity,
        bool                                             scaleDensity,
        utils::ConditionalOStream &                      rootCout)
      {
        RealType totalDensityInQuad = 0.0;
        if (computeTotalDensity || scaleDensity)
          {
            size_type quadId = 0;
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
        utils::MemoryStorage<ValueType, memorySpaceHost> multiVectorGuessHost(
          multiVectorGuess.localSize() * multiVectorGuess.numVectors(),
          ValueType());
        int rank;
        utils::mpi::MPICommRank(
          multiVectorGuess.getMPIPatternP2P()->mpiCommunicator(), &rank);
        boost::math::normal normDist;
        std::mt19937        randomIntGenerator(rank);
        ValueType *         temp = multiVectorGuessHost.data();
        for (size_type i = 0;
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

      std::set<DensityDescrAttr>
      getDescrAttributes(const std::string &xcType)
      {
        if (xcType.rfind("GGA", 0) == 0)
          return {DensityDescrAttr::Val, DensityDescrAttr::Grad};
        return {DensityDescrAttr::Val};
      }

      template <typename RealType>
      std::unordered_map<DensityDescrAttr,
                         std::vector<quadrature::QuadratureValuesContainer<
                           RealType,
                           utils::MemorySpace::HOST>>>
      buildDescrMap(const std::map<DensityDescrAttr,
                                   quadrature::QuadratureValuesContainer<
                                     RealType,
                                     utils::MemorySpace::HOST>> &descrInput,
                   const SpinMode                                 spinMode)
      {
        size_type ncomp = 1;
        if (spinMode == SpinMode::Collinear)
          ncomp = 2;
        else if (spinMode == SpinMode::NonCollinear)
          ncomp = 4;

        std::unordered_map<DensityDescrAttr,
                           std::vector<quadrature::QuadratureValuesContainer<
                             RealType,
                             utils::MemorySpace::HOST>>>
          descrMap;
        for (const auto &[attr, qvc] : descrInput)
          {
            descrMap[attr].resize(
              ncomp,
              quadrature::QuadratureValuesContainer<RealType,
                                                    utils::MemorySpace::HOST>(
                qvc.getQuadratureRuleContainer(),
                qvc.getNumberComponents(),
                static_cast<RealType>(0.0)));
            descrMap[attr][0] = qvc;
          }
        return descrMap;
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
        // const quadrature::QuadratureValuesContainer<RealType,
        // memorySpaceHost>
        //   &electronChargeDensityInput,
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicElectronicChargeDensityFunction,
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
                                          memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDElectronicChargeRhs,
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
        /* exc type */
        const std::string &xcType,
        /* linAgOperations Context*/
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        /* basis overlap related info */
        const OpContext &           MContextForInv,
        const OpContext &           MContext,
        const OpContext &           MInvContext,
        bool                        isResidualChebyshevFilter,
        const std::vector<double> & atomMagZFactors,
        SpinMode                    spinMode)
      : d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
      , d_numMaxSCFIter(maxSCFIter)
      , d_MContext(&MContext)
      , d_MInvContext(&MInvContext)
      , d_mpiCommDomain(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator())
      , d_mixingScheme(d_mpiCommDomain)
      , d_numWantedEigenvalues(numWantedEigenvalues)
      , d_linAlgOpContext(linAlgOpContext)
      , d_linAlgOpContextHost(
          linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_spinMode(spinMode)
      , d_isONCVNonLocPSP(false)
      , d_isNlcc(false)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
      , d_xcType(xcType)
    {
      std::unique_ptr<
        linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace>>
                                       wfnPtr;
      const size_type numSpacesS = (d_spinMode == SpinMode::Unpolarized) ? 1 : 2;
      std::vector<std::vector<double>> occupancies = {
        std::vector<double>(numSpacesS * numWantedEigenvalues, 0.0)};
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

      auto densIn =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      if (d_spinMode == SpinMode::Collinear)
        wfnPtr = std::make_unique<linearAlgebra::MultiVectorProductSpaceBlocked<
          ValueTypeWaveFunctionCoeff,
          memorySpace>>(feBMWaveFn->getMPIPatternP2P(),
                        linAlgOpContext,
                        2,
                        numWantedEigenvalues,
                        (ValueTypeWaveFunctionCoeff)0.0);
      else if (d_spinMode == SpinMode::NonCollinear)
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          2,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);
      else
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          1,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);

      KohnShamDFTInternal::generateRandNormDistMultivec(*wfnPtr);
      wfnPtr->updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        *wfnPtr, wfnPtr->numVectors());

      d_rdm1Spectral = std::make_shared<RDM1FE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
        feBDEXCHamiltonian,
        *feBMWaveFn,
        linAlgOpContext,
        d_mpiCommDomain,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
        KSDFTDefaults<memorySpace>::MAX_DENSCOMP_WAVEFN_BATCH_SIZE,
        spinMode);

      utils::throwException(densIn.getNumberComponents() == 1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
          densIn.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDElectrostaticsHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho = densIn.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      //************* CHANGE THIS **********************
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      d_rdm1Mix = std::make_shared<RDM1Mixing<
        linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff>,
        memorySpace>>(
        d_mixingScheme,
        mixingHistory,
        d_jxwDataHost,
        mixingParameter,
        isAdaptiveAndersonMixingParameter,
        d_spinMode,
        xcType,
        linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST,
        d_mpiCommDomain);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerVal = quadRuleContainerRho;

      utils::MemoryStorage<RealType, memorySpace> densityInQuadValuesMemspace(
        quadRuleContainerVal->nQuadraturePoints());

      atomicElectronicChargeDensityFunction.evaluate(
        quadRuleContainerVal->nQuadraturePoints(),
        atoms::AtomSuperpositionFuncType::Identity,
        quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
        densityInQuadValuesMemspace.data(),
        1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));

      utils::MemoryTransfer<memorySpaceHost, memorySpace> memTrans;
      memTrans.copy(densityInQuadValuesMemspace.size(),
                    densIn.begin(),
                    densityInQuadValuesMemspace.data());

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(densIn,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      std::map<DensityDescrAttr,
               quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>
        initDescrMap;
      initDescrMap[DensityDescrAttr::Val] = densIn;

      if (KohnShamDFTInternal::getDescrAttributes(xcType).count(
            DensityDescrAttr::Grad) > 0)
        {
          auto gradIn =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              quadRuleContainerVal, dim, (RealType)0.0);
          utils::MemoryStorage<RealType, memorySpace> gradInQuadValuesMemspace(
            quadRuleContainerVal->nQuadraturePoints() * dim);
          atomicElectronicChargeDensityFunction.evaluate(
            quadRuleContainerVal->nQuadraturePoints(),
            atoms::AtomSuperpositionFuncType::Grad,
            quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
            gradInQuadValuesMemspace.data(),
            1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));
          memTrans.copy(gradInQuadValuesMemspace.size(),
                        gradIn.begin(),
                        gradInQuadValuesMemspace.data());
          initDescrMap[DensityDescrAttr::Grad] = gradIn;
        }

      {
        auto initMixDescrMap =
          KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode);
        if (d_spinMode != SpinMode::Unpolarized && !atomMagZFactors.empty())
          {
            auto &          spinDensVal = initMixDescrMap[DensityDescrAttr::Val];
            const size_type numQuad     = spinDensVal[0].nQuadraturePoints();
            const double *  quadRealPointsHost =
              spinDensVal[0]
                .getQuadratureRuleContainer()
                ->template getRealPointsPtr<utils::MemorySpace::HOST>();
            const double densNormFactor =
              std::abs(static_cast<double>(numElectrons) /
                       static_cast<double>(totalDensityInQuad));
            std::vector<double> magZInQuadValues(numQuad, 0.0);
            atomicElectronicChargeDensityFunction.evaluateHost(
              numQuad,
              atoms::AtomSuperpositionFuncType::Identity,
              quadRealPointsHost,
              magZInQuadValues.data(),
              densNormFactor /
                (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
              atomMagZFactors);
            for (size_type i = 0; i < numQuad; ++i)
              spinDensVal[1].data()[i] = magZInQuadValues[i];
            if (xcType.rfind("GGA", 0) == 0)
              {
                auto &          spinDensGrad =
                  initMixDescrMap[DensityDescrAttr::Grad];
                std::vector<double> magZGradInQuadValues(numQuad * dim, 0.0);
                atomicElectronicChargeDensityFunction.evaluateHost(
                  numQuad,
                  atoms::AtomSuperpositionFuncType::Grad,
                  quadRealPointsHost,
                  magZGradInQuadValues.data(),
                  1.0 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
                  atomMagZFactors);
                for (size_type i = 0; i < numQuad * dim; ++i)
                  spinDensGrad[1].data()[i] = magZGradInQuadValues[i];
              }
          }
        d_rdm1Mix->setDescriptors(initMixDescrMap, {});
      }

      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues,
        spinMode);

      d_hamitonianElec =
        std::make_shared<ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>(
          atomCoordinates,
          atomCharges,
          smearedChargeRadius,
          densIn,
          feBMTotalCharge,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDElectrostaticsHamiltonian,
          externalPotentialFunction,
          linAlgOpContext,
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          true,
          spinMode);

      d_rdm1Spectral->setDescriptors(
        KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode), {});

      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          xcType,
          *d_rdm1Spectral,
          linAlgOpContext,
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE);
      d_p.registerEnd("Hamiltonian Components Initilization");

      // d_hamiltonianElectroExc =
      //   std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
      //                                       ValueTypeElectrostaticsBasis,
      //                                       ValueTypeWaveFunctionCoeff,
      //                                       ValueTypeWaveFunctionBasis,
      //                                       memorySpace,
      //                                       dim>>(d_hamitonianElec,
      //                                             d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin,
        d_hamitonianElec,
        d_hamitonianXC /* d_hamiltonianElectroExc*/};

      size_type waveFnBatch =
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE :
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
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          waveFnBatch,
          true,
          spinMode);
      d_p.registerEnd("Hamiltonian Operator Creation");

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults<memorySpace>::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize =
        KSDFTDefaults<memorySpace>::SCALAPACK_BLOCK_SIZE;
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
                       feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext,
                      true, /*isGHEP*/
                      linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
                      false,     /*storeIntermediateSubspaces*/
                      true,     /*useSameScratchInEigenSolver*/
                      spinMode);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *wfnPtr,
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
        feBMWaveFn->getMPIPatternP2P(),
        linAlgOpContext,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

      d_rdm1Spectral->setSpectral(std::move(wfnPtr),
                                  occupancies,
                                  numWantedEigenvalues);

      d_p.registerEnd("KS EigenSolver Init");

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
        /* Electron density related info */
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicElectronicChargeDensityFunction,
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
                                          memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDNuclChargeStiffnessMatrixNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDNuclChargeRhsNumSol,
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
        /* exc type */
        const std::string &xcType,
        /* linAgOperations Context*/
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        /* basis overlap related info */
        const OpContext &           MContextForInv,
        const OpContext &           MContext,
        const OpContext &           MInvContext,
        bool                        isResidualChebyshevFilter,
        const std::vector<double> & atomMagZFactors,
        SpinMode                    spinMode)
      : d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
      , d_numMaxSCFIter(maxSCFIter)
      , d_MContext(&MContext)
      , d_MInvContext(&MInvContext)
      , d_mpiCommDomain(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator())
      , d_mixingScheme(d_mpiCommDomain)
      , d_numWantedEigenvalues(numWantedEigenvalues)
      , d_linAlgOpContext(linAlgOpContext)
      , d_linAlgOpContextHost(
          linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_spinMode(spinMode)
      , d_isONCVNonLocPSP(false)
      , d_isNlcc(false)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
      , d_xcType(xcType)
    {
      std::unique_ptr<
        linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace>>
                                       wfnPtr;
      const size_type numSpacesS = (d_spinMode == SpinMode::Unpolarized) ? 1 : 2;
      std::vector<std::vector<double>> occupancies = {
        std::vector<double>(numSpacesS * numWantedEigenvalues, 0.0)};

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

      auto densIn =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      if (d_spinMode == SpinMode::Collinear)
        wfnPtr = std::make_unique<linearAlgebra::MultiVectorProductSpaceBlocked<
          ValueTypeWaveFunctionCoeff,
          memorySpace>>(feBMWaveFn->getMPIPatternP2P(),
                        linAlgOpContext,
                        2,
                        numWantedEigenvalues,
                        (ValueTypeWaveFunctionCoeff)0.0);
      else if (d_spinMode == SpinMode::NonCollinear)
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          2,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);
      else
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          1,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);

      KohnShamDFTInternal::generateRandNormDistMultivec(*wfnPtr);
      wfnPtr->updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        *wfnPtr, wfnPtr->numVectors());

      d_rdm1Spectral = std::make_shared<RDM1FE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
        feBDEXCHamiltonian,
        *feBMWaveFn,
        linAlgOpContext,
        d_mpiCommDomain,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
        KSDFTDefaults<memorySpace>::MAX_DENSCOMP_WAVEFN_BATCH_SIZE,
        spinMode);

      utils::throwException(densIn.getNumberComponents() == 1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
          densIn.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho = densIn.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      //************* CHANGE THIS **********************
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      d_rdm1Mix = std::make_shared<RDM1Mixing<
        linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff>,
        memorySpace>>(
        d_mixingScheme,
        mixingHistory,
        d_jxwDataHost,
        mixingParameter,
        isAdaptiveAndersonMixingParameter,
        d_spinMode,
        xcType,
        linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST,
        d_mpiCommDomain);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerVal = quadRuleContainerRho;

      utils::MemoryStorage<RealType, memorySpace> densityInQuadValuesMemspace(
        quadRuleContainerVal->nQuadraturePoints());

      atomicElectronicChargeDensityFunction.evaluate(
        quadRuleContainerVal->nQuadraturePoints(),
        atoms::AtomSuperpositionFuncType::Identity,
        quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
        densityInQuadValuesMemspace.data(),
        1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));

      utils::MemoryTransfer<memorySpaceHost, memorySpace> memTrans;
      memTrans.copy(densityInQuadValuesMemspace.size(),
                    densIn.begin(),
                    densityInQuadValuesMemspace.data());

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(densIn,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      std::map<DensityDescrAttr,
               quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>
        initDescrMap;
      initDescrMap[DensityDescrAttr::Val] = densIn;

      if (KohnShamDFTInternal::getDescrAttributes(xcType).count(
            DensityDescrAttr::Grad) > 0)
        {
          auto gradIn =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              quadRuleContainerVal, dim, (RealType)0.0);
          utils::MemoryStorage<RealType, memorySpace> gradInQuadValuesMemspace(
            quadRuleContainerVal->nQuadraturePoints() * dim);
          atomicElectronicChargeDensityFunction.evaluate(
            quadRuleContainerVal->nQuadraturePoints(),
            atoms::AtomSuperpositionFuncType::Grad,
            quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
            gradInQuadValuesMemspace.data(),
            1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));
          memTrans.copy(gradInQuadValuesMemspace.size(),
                        gradIn.begin(),
                        gradInQuadValuesMemspace.data());
          initDescrMap[DensityDescrAttr::Grad] = gradIn;
        }

      {
        auto initMixDescrMap =
          KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode);
        if (d_spinMode != SpinMode::Unpolarized && !atomMagZFactors.empty())
          {
            auto &          spinDensVal = initMixDescrMap[DensityDescrAttr::Val];
            const size_type numQuad     = spinDensVal[0].nQuadraturePoints();
            const double *  quadRealPointsHost =
              spinDensVal[0]
                .getQuadratureRuleContainer()
                ->template getRealPointsPtr<utils::MemorySpace::HOST>();
            const double densNormFactor =
              std::abs(static_cast<double>(numElectrons) /
                       static_cast<double>(totalDensityInQuad));
            std::vector<double> magZInQuadValues(numQuad, 0.0);
            atomicElectronicChargeDensityFunction.evaluateHost(
              numQuad,
              atoms::AtomSuperpositionFuncType::Identity,
              quadRealPointsHost,
              magZInQuadValues.data(),
              densNormFactor /
                (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
              atomMagZFactors);
            for (size_type i = 0; i < numQuad; ++i)
              spinDensVal[1].data()[i] = magZInQuadValues[i];
            if (xcType.rfind("GGA", 0) == 0)
              {
                auto &          spinDensGrad =
                  initMixDescrMap[DensityDescrAttr::Grad];
                std::vector<double> magZGradInQuadValues(numQuad * dim, 0.0);
                atomicElectronicChargeDensityFunction.evaluateHost(
                  numQuad,
                  atoms::AtomSuperpositionFuncType::Grad,
                  quadRealPointsHost,
                  magZGradInQuadValues.data(),
                  1.0 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
                  atomMagZFactors);
                for (size_type i = 0; i < numQuad * dim; ++i)
                  spinDensGrad[1].data()[i] = magZGradInQuadValues[i];
              }
          }
        d_rdm1Mix->setDescriptors(initMixDescrMap, {});
      }

      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues,
        spinMode);

      d_hamitonianElec =
        std::make_shared<ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>(
          atomCoordinates,
          atomCharges,
          smearedChargeRadius,
          densIn,
          feBMTotalCharge,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDNuclChargeStiffnessMatrixNumSol,
          feBDNuclChargeRhsNumSol,
          feBDElectrostaticsHamiltonian,
          externalPotentialFunction,
          linAlgOpContext,
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          true,
          spinMode);

      d_rdm1Spectral->setDescriptors(
        KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode), {});

      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          xcType,
          *d_rdm1Spectral,
          linAlgOpContext,
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE);

      d_p.registerEnd("Hamiltonian Components Initilization");

      // d_hamiltonianElectroExc =
      //   std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
      //                                       ValueTypeElectrostaticsBasis,
      //                                       ValueTypeWaveFunctionCoeff,
      //                                       ValueTypeWaveFunctionBasis,
      //                                       memorySpace,
      //                                       dim>>(d_hamitonianElec,
      //                                             d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin,
        d_hamitonianElec,
        d_hamitonianXC /* d_hamiltonianElectroExc*/};

      size_type waveFnBatch =
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE :
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
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          waveFnBatch,
          true,
          spinMode);
      d_p.registerEnd("Hamiltonian Operator Creation");
      d_p.print();

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults<memorySpace>::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize =
        KSDFTDefaults<memorySpace>::SCALAPACK_BLOCK_SIZE;
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
                       feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *wfnPtr,
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
        feBMWaveFn->getMPIPatternP2P(),
        linAlgOpContext,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

      d_rdm1Spectral->setSpectral(std::move(wfnPtr),
                                  occupancies,
                                  numWantedEigenvalues);

      d_p.registerEnd("KS EigenSolver Init");

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
        /* Atomic Field for delta rho ; Here vTotal atomic scalar sp fn.*/
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicTotalElectroPotentialFunction,
        const atoms::AtomSuperpositionFunction<memorySpace>
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
                                          memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDElectronicChargeRhs,
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
        /* exc type */
        const std::string &xcType,
        /* linAgOperations Context*/
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        /* basis overlap related info */
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext,
        bool                         isResidualChebyshevFilter,
        /* TCI related info */
        const atoms::TCIADataParams &params,
        const std::vector<double> &  atomMagZFactors,
        SpinMode                     spinMode)
      : d_feBMWaveFn(feBMWaveFn)
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
      , d_linAlgOpContextHost(
          linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_spinMode(spinMode)
      , d_isONCVNonLocPSP(false)
      , d_isNlcc(false)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
      , d_xcType(xcType)
    {
      std::unique_ptr<
        linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace>>
                                       wfnPtr;
      const size_type numSpacesS = (d_spinMode == SpinMode::Unpolarized) ? 1 : 2;
      std::vector<std::vector<double>> occupancies = {
        std::vector<double>(numSpacesS * numWantedEigenvalues, 0.0)};
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

      auto densIn =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      if (d_spinMode == SpinMode::Collinear)
        wfnPtr = std::make_unique<linearAlgebra::MultiVectorProductSpaceBlocked<
          ValueTypeWaveFunctionCoeff,
          memorySpace>>(feBMWaveFn->getMPIPatternP2P(),
                        linAlgOpContext,
                        2,
                        numWantedEigenvalues,
                        (ValueTypeWaveFunctionCoeff)0.0);
      else if (d_spinMode == SpinMode::NonCollinear)
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          2,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);
      else
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          1,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);

      KohnShamDFTInternal::generateRandNormDistMultivec(*wfnPtr);
      wfnPtr->updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        *wfnPtr, wfnPtr->numVectors());

      d_rdm1Spectral = std::make_shared<RDM1FE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
        feBDEXCHamiltonian,
        *feBMWaveFn,
        linAlgOpContext,
        d_mpiCommDomain,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
        KSDFTDefaults<memorySpace>::MAX_DENSCOMP_WAVEFN_BATCH_SIZE,
        spinMode);

      utils::throwException(densIn.getNumberComponents() == 1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
          densIn.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDElectrostaticsHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho = densIn.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      //************* CHANGE THIS **********************
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      d_rdm1Mix = std::make_shared<RDM1Mixing<
        linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff>,
        memorySpace>>(
        d_mixingScheme,
        mixingHistory,
        d_jxwDataHost,
        mixingParameter,
        isAdaptiveAndersonMixingParameter,
        d_spinMode,
        xcType,
        linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST,
        d_mpiCommDomain);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerVal = quadRuleContainerRho;

      utils::MemoryStorage<RealType, memorySpace> densityInQuadValuesMemspace(
        quadRuleContainerVal->nQuadraturePoints());

      atomicElectronicChargeDensityFunction.evaluate(
        quadRuleContainerVal->nQuadraturePoints(),
        atoms::AtomSuperpositionFuncType::Identity,
        quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
        densityInQuadValuesMemspace.data(),
        1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));

      utils::MemoryTransfer<memorySpaceHost, memorySpace> memTrans;
      memTrans.copy(densityInQuadValuesMemspace.size(),
                    densIn.begin(),
                    densityInQuadValuesMemspace.data());

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(densIn,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      std::map<DensityDescrAttr,
               quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>
        initDescrMap;
      initDescrMap[DensityDescrAttr::Val] = densIn;

      if (KohnShamDFTInternal::getDescrAttributes(xcType).count(
            DensityDescrAttr::Grad) > 0)
        {
          auto gradIn =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              quadRuleContainerVal, dim, (RealType)0.0);
          utils::MemoryStorage<RealType, memorySpace> gradInQuadValuesMemspace(
            quadRuleContainerVal->nQuadraturePoints() * dim);
          atomicElectronicChargeDensityFunction.evaluate(
            quadRuleContainerVal->nQuadraturePoints(),
            atoms::AtomSuperpositionFuncType::Grad,
            quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
            gradInQuadValuesMemspace.data(),
            1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));
          memTrans.copy(gradInQuadValuesMemspace.size(),
                        gradIn.begin(),
                        gradInQuadValuesMemspace.data());
          initDescrMap[DensityDescrAttr::Grad] = gradIn;
        }

      {
        auto initMixDescrMap =
          KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode);
        if (d_spinMode != SpinMode::Unpolarized && !atomMagZFactors.empty())
          {
            auto &          spinDensVal = initMixDescrMap[DensityDescrAttr::Val];
            const size_type numQuad     = spinDensVal[0].nQuadraturePoints();
            const double *  quadRealPointsHost =
              spinDensVal[0]
                .getQuadratureRuleContainer()
                ->template getRealPointsPtr<utils::MemorySpace::HOST>();
            const double densNormFactor =
              std::abs(static_cast<double>(numElectrons) /
                       static_cast<double>(totalDensityInQuad));
            std::vector<double> magZInQuadValues(numQuad, 0.0);
            atomicElectronicChargeDensityFunction.evaluateHost(
              numQuad,
              atoms::AtomSuperpositionFuncType::Identity,
              quadRealPointsHost,
              magZInQuadValues.data(),
              densNormFactor /
                (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
              atomMagZFactors);
            for (size_type i = 0; i < numQuad; ++i)
              spinDensVal[1].data()[i] = magZInQuadValues[i];
            if (xcType.rfind("GGA", 0) == 0)
              {
                auto &          spinDensGrad =
                  initMixDescrMap[DensityDescrAttr::Grad];
                std::vector<double> magZGradInQuadValues(numQuad * dim, 0.0);
                atomicElectronicChargeDensityFunction.evaluateHost(
                  numQuad,
                  atoms::AtomSuperpositionFuncType::Grad,
                  quadRealPointsHost,
                  magZGradInQuadValues.data(),
                  1.0 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
                  atomMagZFactors);
                for (size_type i = 0; i < numQuad * dim; ++i)
                  spinDensGrad[1].data()[i] = magZGradInQuadValues[i];
              }
          }
        d_rdm1Mix->setDescriptors(initMixDescrMap, {});
      }

      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues,
        spinMode);

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
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          fieldToTCIASplineMap,
          true,
          false,
          spinMode);

      d_rdm1Spectral->setDescriptors(
        KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode), {});

      d_hamitonianXC =
        std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
          xcType,
          *d_rdm1Spectral,
          linAlgOpContext,
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE);

      d_p.registerEnd("Hamiltonian Components Initilization");

      // d_hamiltonianElectroExc =
      //   std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
      //                                       ValueTypeElectrostaticsBasis,
      //                                       ValueTypeWaveFunctionCoeff,
      //                                       ValueTypeWaveFunctionBasis,
      //                                       memorySpace,
      //                                       dim>>(d_hamitonianElec,
      //                                             d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin,
        d_hamitonianElec,
        d_hamitonianXC /* d_hamiltonianElectroExc*/};

      size_type waveFnBatch =
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE :
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
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          waveFnBatch,
          true,
          spinMode);
      d_p.registerEnd("Hamiltonian Operator Creation");

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults<memorySpace>::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize =
        KSDFTDefaults<memorySpace>::SCALAPACK_BLOCK_SIZE;
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
                       feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *wfnPtr,
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
        feBMWaveFn->getMPIPatternP2P(),
        linAlgOpContext,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

      d_rdm1Spectral->setSpectral(std::move(wfnPtr),
                                  occupancies,
                                  numWantedEigenvalues);

      d_p.registerEnd("KS EigenSolver Init");

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
        /* Desnity Input*/
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicElectronicChargeDensityFunction,
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
                                          memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDElectronicChargeRhs,
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
        /* exc type */
        const std::string &xcType,
        /* linAgOperations Context*/
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        /* basis overlap related info */
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext,
        bool                        isResidualChebyshevFilter,
        const std::vector<double> & atomMagZFactors,
        SpinMode                    spinMode)
      : d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
      , d_numMaxSCFIter(maxSCFIter)
      , d_MContext(&MContext)
      , d_MInvContext(&MInvContext)
      , d_mpiCommDomain(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator())
      , d_mixingScheme(d_mpiCommDomain)
      , d_numWantedEigenvalues(numWantedEigenvalues)
      , d_linAlgOpContext(linAlgOpContext)
      , d_linAlgOpContextHost(
          linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_spinMode(spinMode)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
      , d_xcType(xcType)
    {
      std::unique_ptr<
        linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace>>
                                       wfnPtr;
      const size_type numSpacesS = (d_spinMode == SpinMode::Unpolarized) ? 1 : 2;
      std::vector<std::vector<double>> occupancies = {
        std::vector<double>(numSpacesS * numWantedEigenvalues, 0.0)};

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

      for (size_type i = 0; i < atomSymbolVec.size(); i++)
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
      for (size_type atomSymbolId = 0; atomSymbolId < atomSymbolVec.size();
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

      auto densIn =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      if (dynamic_cast<
            const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim> *>(
            &feBMWaveFn->getBasisDofHandler()) != nullptr)
        d_isOEFEBasis = true;
      else
        d_isOEFEBasis = false;

      if (d_spinMode == SpinMode::Collinear)
        wfnPtr = std::make_unique<linearAlgebra::MultiVectorProductSpaceBlocked<
          ValueTypeWaveFunctionCoeff,
          memorySpace>>(feBMWaveFn->getMPIPatternP2P(),
                        linAlgOpContext,
                        2,
                        numWantedEigenvalues,
                        (ValueTypeWaveFunctionCoeff)0.0);
      else if (d_spinMode == SpinMode::NonCollinear)
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          2,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);
      else
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          1,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);

      KohnShamDFTInternal::generateRandNormDistMultivec(*wfnPtr);
      wfnPtr->updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        *wfnPtr, wfnPtr->numVectors());

      d_rdm1Spectral = std::make_shared<RDM1FE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
        feBDEXCHamiltonian,
        *feBMWaveFn,
        linAlgOpContext,
        d_mpiCommDomain,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
        KSDFTDefaults<memorySpace>::MAX_DENSCOMP_WAVEFN_BATCH_SIZE,
        spinMode);

      utils::throwException(densIn.getNumberComponents() == 1,
                            "Electron density should have only one component.");

      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
          densIn.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDElectrostaticsHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho = densIn.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      // --------TODO : use eval()-----
      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerVal = quadRuleContainerRho;

      utils::MemoryStorage<RealType, memorySpace> densityInQuadValuesMemspace(
        quadRuleContainerVal->nQuadraturePoints());

      atomicElectronicChargeDensityFunction.evaluate(
        quadRuleContainerVal->nQuadraturePoints(),
        atoms::AtomSuperpositionFuncType::Identity,
        quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
        densityInQuadValuesMemspace.data(),
        1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));

      utils::MemoryTransfer<memorySpaceHost, memorySpace> memTrans;
      memTrans.copy(densityInQuadValuesMemspace.size(),
                    densIn.begin(),
                    densityInQuadValuesMemspace.data());

      //************* CHANGE THIS **********************
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      d_rdm1Mix = std::make_shared<RDM1Mixing<
        linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff>,
        memorySpace>>(
        d_mixingScheme,
        mixingHistory,
        d_jxwDataHost,
        mixingParameter,
        isAdaptiveAndersonMixingParameter,
        d_spinMode,
        xcType,
        linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST,
        d_mpiCommDomain);

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(densIn,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";

      std::map<DensityDescrAttr,
               quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>
        initDescrMap;
      initDescrMap[DensityDescrAttr::Val] = densIn;

      if (KohnShamDFTInternal::getDescrAttributes(xcType).count(
            DensityDescrAttr::Grad) > 0)
        {
          auto gradIn =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              quadRuleContainerVal, dim, (RealType)0.0);
          utils::MemoryStorage<RealType, memorySpace> gradInQuadValuesMemspace(
            quadRuleContainerVal->nQuadraturePoints() * dim);
          atomicElectronicChargeDensityFunction.evaluate(
            quadRuleContainerVal->nQuadraturePoints(),
            atoms::AtomSuperpositionFuncType::Grad,
            quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
            gradInQuadValuesMemspace.data(),
            1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));
          memTrans.copy(gradInQuadValuesMemspace.size(),
                        gradIn.begin(),
                        gradInQuadValuesMemspace.data());
          initDescrMap[DensityDescrAttr::Grad] = gradIn;
        }

      {
        auto initMixDescrMap =
          KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode);
        if (d_spinMode != SpinMode::Unpolarized && !atomMagZFactors.empty())
          {
            auto &          spinDensVal = initMixDescrMap[DensityDescrAttr::Val];
            const size_type numQuad     = spinDensVal[0].nQuadraturePoints();
            const double *  quadRealPointsHost =
              spinDensVal[0]
                .getQuadratureRuleContainer()
                ->template getRealPointsPtr<utils::MemorySpace::HOST>();
            const double densNormFactor =
              std::abs(static_cast<double>(numElectrons) /
                       static_cast<double>(totalDensityInQuad));
            std::vector<double> magZInQuadValues(numQuad, 0.0);
            atomicElectronicChargeDensityFunction.evaluateHost(
              numQuad,
              atoms::AtomSuperpositionFuncType::Identity,
              quadRealPointsHost,
              magZInQuadValues.data(),
              densNormFactor /
                (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
              atomMagZFactors);
            for (size_type i = 0; i < numQuad; ++i)
              spinDensVal[1].data()[i] = magZInQuadValues[i];
            if (xcType.rfind("GGA", 0) == 0)
              {
                auto &          spinDensGrad =
                  initMixDescrMap[DensityDescrAttr::Grad];
                std::vector<double> magZGradInQuadValues(numQuad * dim, 0.0);
                atomicElectronicChargeDensityFunction.evaluateHost(
                  numQuad,
                  atoms::AtomSuperpositionFuncType::Grad,
                  quadRealPointsHost,
                  magZGradInQuadValues.data(),
                  1.0 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
                  atomMagZFactors);
                for (size_type i = 0; i < numQuad * dim; ++i)
                  spinDensGrad[1].data()[i] = magZGradInQuadValues[i];
              }
          }
        d_rdm1Mix->setDescriptors(initMixDescrMap, {});
      }

      d_p.registerEnd("Pre Init Checks");
      d_p.registerStart("Hamiltonian Components Initilization");
      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues,
        spinMode);

      size_type waveFnBatch =
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE :
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
          densIn,
          feBMTotalCharge,
          feBMWaveFn,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDElectrostaticsHamiltonian,
          feBDAtomCenterNonLocalOperator,
          linAlgOpContext,
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          waveFnBatch,
          true,
          spinMode);

      d_rdm1Spectral->setDescriptors(
        KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode), {});

      if (d_isNlcc && d_isONCVNonLocPSP)
        d_hamitonianXC =
          std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                 ValueTypeWaveFunctionCoeff,
                                                 memorySpace,
                                                 dim>>(
            xcType,
            *d_rdm1Spectral,
            linAlgOpContext,
            KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
            d_atomSphericalDataContainerPSP,
            atomSymbolVec,
            atomCoordinates);
      else
        d_hamitonianXC =
          std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                 ValueTypeWaveFunctionCoeff,
                                                 memorySpace,
                                                 dim>>(
            xcType,
            *d_rdm1Spectral,
            linAlgOpContext,
            KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE);

      d_p.registerEnd("Hamiltonian Components Initilization");

      // d_hamiltonianElectroExc =
      //   std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
      //                                       ValueTypeElectrostaticsBasis,
      //                                       ValueTypeWaveFunctionCoeff,
      //                                       ValueTypeWaveFunctionBasis,
      //                                       memorySpace,
      //                                       dim>>(d_hamitonianElec,
      //                                             d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin,
        d_hamitonianElec,
        d_hamitonianXC /* d_hamiltonianElectroExc*/};

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
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          waveFnBatch,
          true,
          spinMode);
      d_p.registerEnd("Hamiltonian Operator Creation");

      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults<memorySpace>::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize =
        KSDFTDefaults<memorySpace>::SCALAPACK_BLOCK_SIZE;
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
                       feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *wfnPtr,
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
        feBMWaveFn->getMPIPatternP2P(),
        linAlgOpContext,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

      d_rdm1Spectral->setSpectral(std::move(wfnPtr),
                                  occupancies,
                                  numWantedEigenvalues);

      d_p.registerEnd("KS EigenSolver Init");

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
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicTotalElectroPotentialFunction,
        const atoms::AtomSuperpositionFunction<memorySpace>
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
                                          memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeElectrostaticsBasis,
                                          memorySpaceHost>>
          feBDElectronicChargeRhs,
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
        /* exc type */
        const std::string &xcType,
        /* linAgOperations Context*/
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext,
        /* basis overlap related info */
        const OpContext &MContextForInv,
        const OpContext &MContext,
        const OpContext &MInvContext,
        bool                         isResidualChebyshevFilter,
        /* TCI related info */
        const atoms::TCIADataParams &params,
        const std::vector<double> &  atomMagZFactors,
        SpinMode                     spinMode)
      : d_feBMWaveFn(feBMWaveFn)
      , d_evaluateEnergyEverySCF(evaluateEnergyEverySCF)
      , d_numMaxSCFIter(maxSCFIter)
      , d_MContext(&MContext)
      , d_MInvContext(&MInvContext)
      , d_mpiCommDomain(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator())
      , d_mixingScheme(d_mpiCommDomain)
      , d_numWantedEigenvalues(numWantedEigenvalues)
      , d_linAlgOpContext(linAlgOpContext)
      , d_linAlgOpContextHost(
          linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_kohnShamEnergies(numWantedEigenvalues, 0.0)
      , d_SCFTol(scfDensityResidualNormTolerance)
      , d_rootCout(std::cout)
      , d_numElectrons(numElectrons)
      , d_feBDEXCHamiltonian(feBDEXCHamiltonian)
      , d_isSolved(false)
      , d_groundStateEnergy(0)
      , d_freeEnergy(0)
      , d_smearingTemperature(smearingTemperature)
      , d_p(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Kohn Sham DFT")
      , d_isResidualChebyshevFilter(isResidualChebyshevFilter)
      , d_spinMode(spinMode)
      , d_pTotal(feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(),
                 "Kohn Sham DFT Solve time")
      , d_xcType(xcType)
    {
      std::unique_ptr<
        linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace>>
                                       wfnPtr;
      const size_type numSpacesS = (d_spinMode == SpinMode::Unpolarized) ? 1 : 2;
      std::vector<std::vector<double>> occupancies = {std::vector<double>(numSpacesS * numWantedEigenvalues, 0.0)};
      utils::Profiler<utils::MemorySpace::HOST> p(
        feBMWaveFn->getMPIPatternP2P()->mpiCommunicator(), "Pre Init Checks");
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

      for (size_type i = 0; i < atomSymbolVec.size(); i++)
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
      for (size_type atomSymbolId = 0; atomSymbolId < atomSymbolVec.size();
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

      auto densIn =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      if (dynamic_cast<
            const basis::EFEBasisDofHandler<ValueTypeWaveFunctionCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim> *>(
            &feBMWaveFn->getBasisDofHandler()) != nullptr)
        d_isOEFEBasis = true;
      else
        d_isOEFEBasis = false;

      if (d_spinMode == SpinMode::Collinear)
        wfnPtr = std::make_unique<linearAlgebra::MultiVectorProductSpaceBlocked<
          ValueTypeWaveFunctionCoeff,
          memorySpace>>(feBMWaveFn->getMPIPatternP2P(),
                        linAlgOpContext,
                        2,
                        numWantedEigenvalues,
                        (ValueTypeWaveFunctionCoeff)0.0);
      else if (d_spinMode == SpinMode::NonCollinear)
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          2,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);
      else
        wfnPtr = std::make_unique<
          linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFunctionCoeff,
                                                memorySpace>>(
          feBMWaveFn->getMPIPatternP2P(),
          linAlgOpContext,
          1,
          numWantedEigenvalues,
          (ValueTypeWaveFunctionCoeff)0.0);

      KohnShamDFTInternal::generateRandNormDistMultivec(*wfnPtr);
      wfnPtr->updateGhostValues();
      feBMWaveFn->getConstraints().distributeParentToChild(
        *wfnPtr, wfnPtr->numVectors());

      d_rdm1Spectral = std::make_shared<RDM1FE<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(
        feBDEXCHamiltonian,
        *feBMWaveFn,
        linAlgOpContext,
        d_mpiCommDomain,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
        KSDFTDefaults<memorySpace>::MAX_DENSCOMP_WAVEFN_BATCH_SIZE,
        spinMode);

      utils::throwException(densIn.getNumberComponents() == 1,
                            "Electron density should have only one component.");
      p.registerEnd("generateRandNormDistMultivec");
      p.registerStart("rhoAtFunc");
      utils::throwException(
        feBDEXCHamiltonian->getQuadratureRuleContainer() ==
          densIn.getQuadratureRuleContainer(),
        "The QuadratureRuleContainer for feBDElectrostaticsHamiltonian and electronChargeDensity should be same.");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerRho = densIn.getQuadratureRuleContainer();

      int rank;
      utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
      d_rootCout.setCondition(rank == 0);

      // --------TODO : use eval()-----

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerVal = quadRuleContainerRho;

      utils::MemoryStorage<RealType, memorySpace> densityInQuadValuesMemspace(
        quadRuleContainerVal->nQuadraturePoints());

      atomicElectronicChargeDensityFunction.evaluate(
        quadRuleContainerVal->nQuadraturePoints(),
        atoms::AtomSuperpositionFuncType::Identity,
        quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
        densityInQuadValuesMemspace.data(),
        1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));

      utils::MemoryTransfer<memorySpaceHost, memorySpace> memTrans;
      memTrans.copy(densityInQuadValuesMemspace.size(),
                    densIn.begin(),
                    densityInQuadValuesMemspace.data());

      p.registerEnd("rhoAtFunc");
      //************* CHANGE THIS **********************
      d_jxwDataHost = quadRuleContainerRho->getJxW();

      d_rdm1Mix = std::make_shared<RDM1Mixing<
        linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff>,
        memorySpace>>(
        d_mixingScheme,
        mixingHistory,
        d_jxwDataHost,
        mixingParameter,
        isAdaptiveAndersonMixingParameter,
        d_spinMode,
        xcType,
        linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST,
        d_mpiCommDomain);

      // normalize electroncharge density
      RealType totalDensityInQuad =
        KohnShamDFTInternal::normalizeDensityQuadData(densIn,
                                                      numElectrons,
                                                      d_jxwDataHost,
                                                      *d_linAlgOpContextHost,
                                                      d_mpiCommDomain,
                                                      true,
                                                      true,
                                                      d_rootCout);

      d_rootCout << "Electron density in : " << totalDensityInQuad << "\n";
      p.print();

      std::map<DensityDescrAttr,
               quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>
        initDescrMap;
      initDescrMap[DensityDescrAttr::Val] = densIn;

      if (KohnShamDFTInternal::getDescrAttributes(xcType).count(
            DensityDescrAttr::Grad) > 0)
        {
          auto gradIn =
            quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
              quadRuleContainerVal, dim, (RealType)0.0);
          utils::MemoryStorage<RealType, memorySpace> gradInQuadValuesMemspace(
            quadRuleContainerVal->nQuadraturePoints() * dim);
          atomicElectronicChargeDensityFunction.evaluate(
            quadRuleContainerVal->nQuadraturePoints(),
            atoms::AtomSuperpositionFuncType::Grad,
            quadRuleContainerVal->template getRealPointsPtr<memorySpace>(),
            gradInQuadValuesMemspace.data(),
            1 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)));
          memTrans.copy(gradInQuadValuesMemspace.size(),
                        gradIn.begin(),
                        gradInQuadValuesMemspace.data());
          initDescrMap[DensityDescrAttr::Grad] = gradIn;
        }

      {
        auto initMixDescrMap =
          KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode);
        if (d_spinMode != SpinMode::Unpolarized && !atomMagZFactors.empty())
          {
            auto &          spinDensVal = initMixDescrMap[DensityDescrAttr::Val];
            const size_type numQuad     = spinDensVal[0].nQuadraturePoints();
            const double *  quadRealPointsHost =
              spinDensVal[0]
                .getQuadratureRuleContainer()
                ->template getRealPointsPtr<utils::MemorySpace::HOST>();
            const double densNormFactor =
              std::abs(static_cast<double>(numElectrons) /
                       static_cast<double>(totalDensityInQuad));
            std::vector<double> magZInQuadValues(numQuad, 0.0);
            atomicElectronicChargeDensityFunction.evaluateHost(
              numQuad,
              atoms::AtomSuperpositionFuncType::Identity,
              quadRealPointsHost,
              magZInQuadValues.data(),
              densNormFactor /
                (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
              atomMagZFactors);
            for (size_type i = 0; i < numQuad; ++i)
              spinDensVal[1].data()[i] = magZInQuadValues[i];
            if (xcType.rfind("GGA", 0) == 0)
              {
                auto &          spinDensGrad =
                  initMixDescrMap[DensityDescrAttr::Grad];
                std::vector<double> magZGradInQuadValues(numQuad * dim, 0.0);
                atomicElectronicChargeDensityFunction.evaluateHost(
                  numQuad,
                  atoms::AtomSuperpositionFuncType::Grad,
                  quadRealPointsHost,
                  magZGradInQuadValues.data(),
                  1.0 / (atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0)),
                  atomMagZFactors);
                for (size_type i = 0; i < numQuad * dim; ++i)
                  spinDensGrad[1].data()[i] = magZGradInQuadValues[i];
              }
          }
        d_rdm1Mix->setDescriptors(initMixDescrMap, {});
      }

      d_p.registerEnd("Pre Init Checks");

      utils::printCurrentMemoryUsage<memorySpace>(d_mpiCommDomain,
                                                  "After PreInit Checks");

      d_p.registerStart("Hamiltonian Components Initilization Kinetic Op");

      d_hamitonianKin = std::make_shared<KineticFE<ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(
        feBDKineticHamiltonian,
        linAlgOpContext,
        KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE_GRAD_EVAL,
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_KINENG_WAVEFN_BATCH_SIZE :
          numWantedEigenvalues,
        spinMode);
      d_p.registerEnd("Hamiltonian Components Initilization Kinetic Op");
      utils::printCurrentMemoryUsage<memorySpace>(d_mpiCommDomain,
                                                  "After KinEngy Init");
      d_p.registerStart(
        "Hamiltonian Components Initilization Electrostatic Op");

      size_type waveFnBatch =
        numWantedEigenvalues >
            KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE ?
          KSDFTDefaults<memorySpace>::MAX_WAVEFN_BATCH_SIZE :
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

          for (size_type atomSymbolId = 0; atomSymbolId < atomSymbolVec.size();
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
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          waveFnBatch,
          fieldToTCIASplineMap,
          true,
          spinMode);
      d_p.registerEnd("Hamiltonian Components Initilization Electrostatic Op");
      utils::printCurrentMemoryUsage<memorySpace>(d_mpiCommDomain,
                                                  "After Elec Init");
      d_p.registerStart("Hamiltonian Components Initilization Exc Op");

      d_rdm1Spectral->setDescriptors(
        KohnShamDFTInternal::buildDescrMap(initDescrMap, d_spinMode), {});

      if (d_isNlcc && d_isONCVNonLocPSP)
        d_hamitonianXC =
          std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                 ValueTypeWaveFunctionCoeff,
                                                 memorySpace,
                                                 dim>>(
            xcType,
            *d_rdm1Spectral,
            linAlgOpContext,
            KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
            d_atomSphericalDataContainerPSP,
            atomSymbolVec,
            atomCoordinates);
      else
        d_hamitonianXC =
          std::make_shared<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                 ValueTypeWaveFunctionCoeff,
                                                 memorySpace,
                                                 dim>>(
            xcType,
            *d_rdm1Spectral,
            linAlgOpContext,
            KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE);

      // d_hamiltonianElectroExc =
      //   std::make_shared<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
      //                                       ValueTypeElectrostaticsBasis,
      //                                       ValueTypeWaveFunctionCoeff,
      //                                       ValueTypeWaveFunctionBasis,
      //                                       memorySpace,
      //                                       dim>>(d_hamitonianElec,
      //                                             d_hamitonianXC);

      std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
        d_hamitonianKin,
        d_hamitonianElec,
        d_hamitonianXC /* d_hamiltonianElectroExc*/};

      d_p.registerEnd("Hamiltonian Components Initilization Exc Op");
      utils::printCurrentMemoryUsage<memorySpace>(d_mpiCommDomain,
                                                  "After Exc Init");
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
          KSDFTDefaults<memorySpace>::CELL_BATCH_SIZE,
          waveFnBatch,
          true,
          spinMode);
      d_p.registerEnd("Hamiltonian Operator Creation");
      utils::printCurrentMemoryUsage<memorySpace>(
        d_mpiCommDomain, "After Hamiltonian Operator Init");
      d_p.registerStart("KS EigenSolver Init");
      // call the eigensolver

      if (elpa_init(ELPA_API_VERSION) != ELPA_OK)
        {
          utils::throwException(false,
                                ("Error: ELPA API version not supported."));
        }

      const bool      useELPA             = true;
      const bool      useELPADeviceKernel = false;
      const size_type scalapackParalProcs =
        KSDFTDefaults<memorySpace>::SCALAPACK_PARAL_PROCS;
      const size_type scalapackBlockSize =
        KSDFTDefaults<memorySpace>::SCALAPACK_BLOCK_SIZE;
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
                       feBMWaveFn->getMPIPatternP2P(),
                       linAlgOpContext,
                       *d_elpaScala,
                       false,
                       waveFnBatch,
                       MContextForInv,
                       MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

          ksEigSolve.setChebyshevPolynomialDegree(1);

          ksEigSolve.solve(*d_hamitonianOperator,
                           d_kohnShamEnergies,
                           *wfnPtr,
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
        feBMWaveFn->getMPIPatternP2P(),
        linAlgOpContext,
        *d_elpaScala,
        isResidualChebyshevFilter,
        waveFnBatch,
        MContextForInv,
        MInvContext,
        true, /*isGHEP*/
        linearAlgebra::OrthogonalizationType::CHOLESKY_GRAMSCHMIDT, /*orthoType */
        false,     /*storeIntermediateSubspaces*/
        true,     /*useSameScratchInEigenSolver*/
        spinMode);

      d_rdm1Spectral->setSpectral(std::move(wfnPtr),
                                  occupancies,
                                  numWantedEigenvalues);

      d_p.registerEnd("KS EigenSolver Init");
      utils::printCurrentMemoryUsage<memorySpace>(d_mpiCommDomain,
                                                  "After KS EigenSolver Init");

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

      std::unique_ptr<
        linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace>>
                                         wfnPtr;
      std::vector<std::vector<RealType>> occupancies;
      size_type                          nKSOrbs;

      d_rdm1Spectral->getSpectral(wfnPtr, occupancies, nKSOrbs);

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
          hamiltonian->evalEnergy(occupancies[0], *wfnPtr);
        }

      d_rdm1Spectral->setSpectral(std::move(wfnPtr), occupancies, nKSOrbs);

      RealType elecEnergy = d_hamitonianElec->getEnergy();
      d_rootCout << "Electrostatic energy with guess density: " << elecEnergy
                 << "\n";

      //
      // Begin SCF iteration
      //
      size_type scfIter  = 0;
      double    norm     = 1.0;
      double    magNorm  = 0.0;
      d_rootCout << "Starting SCF iterations....\n";

      std::unordered_map<
        DensityDescrAttr,
        std::vector<
          quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>>
        densAttrIn, densAttrOut;
      std::unordered_map<
        WfcDescrAttr,
        std::vector<
          quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>>>
        wfcAttrIn, wfcAttrOut;

      while (((norm > d_SCFTol) && (scfIter < d_numMaxSCFIter)))
        {
          utils::printCurrentMemoryUsage<memorySpace>(d_mpiCommDomain,
                                                      "SCF beginning");
          d_p.reset();
          d_rootCout
            << "************************Begin Self-Consistent-Field Iteration: "
            << std::setw(2) << scfIter + 1 << " ***********************\n";

          // reinit the components of hamiltonian with mixed density (scfIter >
          // 0)
          if (scfIter > 0)
            {
              d_pTotal.registerStart("Hamiltonian Reinit");
              d_p.registerStart("Hamiltonian Reinit");

              d_rdm1Mix->getDescriptors(KohnShamDFTInternal::getDescrAttributes(
                                          d_xcType),
                                        {},
                                        densAttrIn,
                                        wfcAttrIn);
              auto &densIn = densAttrIn.at(DensityDescrAttr::Val)[0];

              RealType totalDensityInQuad =
                KohnShamDFTInternal::normalizeDensityQuadData(
                  densIn,
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
                  hamiltonian->reinitField(densIn);
                }
              else if (auto hamiltonian = std::dynamic_pointer_cast<
                         ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                                   ValueTypeElectrostaticsCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(d_hamitonianElec))
                {
                  hamiltonian->reinitField(densIn);
                }

              d_hamitonianXC->reinitField(*d_rdm1Mix);

              // d_hamiltonianElectroExc->reinit(d_hamitonianElec,
              // d_hamitonianXC);

              std::vector<HamiltonianPtrVariant> hamiltonianComponentsVec{
                d_hamitonianKin,
                d_hamitonianElec,
                d_hamitonianXC /* d_hamiltonianElectroExc*/};

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
              RealType topEig = d_kohnShamEnergies[d_numWantedEigenvalues - 1];
              size_type numSpacesS = (d_spinMode == SpinMode::Unpolarized) ? 1 : 2;
              for (size_type s = 1; s < numSpacesS; ++s)
                topEig = std::max(topEig,
                                  d_kohnShamEnergies[(s + 1) * d_numWantedEigenvalues - 1]);
              d_ksEigSolve->reinitBounds(d_kohnShamEnergies[0], topEig);
              // d_ksEigSolve->reinitBounds(
              //   d_kohnShamEnergies[0],
              //   d_kohnShamEnergies[d_numWantedEigenvalues - 1]);
            }

          if (scfIter == 0 && d_isPSPCalculation)
            d_ksEigSolve->setChebyPolyScalingFactor(1.34);

          // Linear Eigen Solve
          d_rdm1Spectral->getSpectral(wfnPtr, occupancies, nKSOrbs);

          linearAlgebra::EigenSolverError err =
            d_ksEigSolve->solve(*d_hamitonianOperator,
                                d_kohnShamEnergies,
                                *wfnPtr,
                                true,
                                *d_MContext,
                                *d_MInvContext);

          occupancies = {d_ksEigSolve->getFractionalOccupancy()};

          d_rdm1Spectral->setSpectral(std::move(wfnPtr), occupancies, nKSOrbs);

          std::vector<RealType> eigSolveResNorm =
            d_ksEigSolve->getEigenSolveResidualNorm();

          d_pTotal.registerEnd("EigenSolve");
          d_p.registerEnd("EigenSolve");

          d_p.registerStart("Density Compute");
          d_pTotal.registerStart("Density Compute");

          d_rdm1Spectral->getDescriptors(
            KohnShamDFTInternal::getDescrAttributes(d_xcType),
            {},
            densAttrOut,
            wfcAttrOut);
          auto &densOut = densAttrOut.at(DensityDescrAttr::Val)[0];

          RealType totalDensityOutQuad =
            KohnShamDFTInternal::normalizeDensityQuadData(
              densOut,
              d_numElectrons,
              d_jxwDataHost,
              *d_linAlgOpContextHost,
              d_mpiCommDomain,
              true,
              false,
              d_rootCout);

          d_rootCout << "Electron density out : " << totalDensityOutQuad
                     << "\n";

          if (d_spinMode == SpinMode::Collinear)
            {
              auto &          magDensOut = densAttrOut.at(DensityDescrAttr::Val)[1];
              const size_type numQuad    = magDensOut.nQuadraturePoints();
              RealType        netMag = 0.0, absMag = 0.0;
              for (size_type i = 0; i < numQuad; ++i)
                {
                  const RealType mz = magDensOut.data()[i];
                  netMag += mz * d_jxwDataHost[i];
                  absMag += std::abs(mz) * d_jxwDataHost[i];
                }
              utils::mpi::MPIAllreduce<memorySpaceHost>(
                utils::mpi::MPIInPlace,
                &netMag,
                1,
                utils::mpi::Types<RealType>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_mpiCommDomain);
              utils::mpi::MPIAllreduce<memorySpaceHost>(
                utils::mpi::MPIInPlace,
                &absMag,
                1,
                utils::mpi::Types<RealType>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_mpiCommDomain);
              d_rootCout << "Net magnetization     : " << netMag << "\n";
              d_rootCout << "Absolute magnetization: " << absMag << "\n";
            }

          d_pTotal.registerEnd("Density Compute");
          d_p.registerEnd("Density Compute");

          d_p.registerStart("Density Mixing");
          d_pTotal.registerStart("Density Mixing");

          // Mix for NEXT iteration
          d_rdm1Mix->setRDM1(d_rdm1Spectral);

          if (scfIter > 0)
            {
              auto &densIn = densAttrIn.at(DensityDescrAttr::Val)[0];
              quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
                densityResidualQuadValues(densIn);
              norm = KohnShamDFTInternal::computeResidualQuadData(
                densOut,
                densIn,
                densityResidualQuadValues,
                d_jxwDataHost,
                true,
                *d_linAlgOpContextHost,
                d_mpiCommDomain);

              if (d_spinMode != SpinMode::Unpolarized)
                {
                  auto &magDensOut = densAttrOut.at(DensityDescrAttr::Val)[1];
                  auto &magDensIn  = densAttrIn.at(DensityDescrAttr::Val)[1];
                  quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
                    magResidualQuadValues(magDensIn);
                  magNorm = KohnShamDFTInternal::computeResidualQuadData(
                    magDensOut,
                    magDensIn,
                    magResidualQuadValues,
                    d_jxwDataHost,
                    true,
                    *d_linAlgOpContextHost,
                    d_mpiCommDomain);
                }
            }

          d_rdm1Mix->mix();

          d_pTotal.registerEnd("Density Mixing");
          d_p.registerEnd("Density Mixing");

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
                  hamiltonian->reinitField(densOut);
                }
              else if (auto hamiltonian = std::dynamic_pointer_cast<
                         ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                                   ValueTypeElectrostaticsCoeff,
                                                   ValueTypeWaveFunctionBasis,
                                                   ValueTypeWaveFunctionCoeff,
                                                   memorySpace,
                                                   dim>>(d_hamitonianElec))
                {
                  hamiltonian->reinitField(densOut);
                }

              d_rdm1Spectral->getSpectral(wfnPtr, occupancies, nKSOrbs);
              d_hamitonianKin->evalEnergy(occupancies[0],
                                          *d_feBMWaveFn,
                                          *wfnPtr);

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
                  hamiltonian->evalEnergy(occupancies[0], *wfnPtr);
                }

              d_rdm1Spectral->setSpectral(std::move(wfnPtr),
                                          occupancies,
                                          nKSOrbs);

              RealType elecEnergy = d_hamitonianElec->getEnergy();
              d_rootCout << "Electrostatic energy: " << elecEnergy << "\n";

              d_hamitonianXC->evalEnergy(*d_rdm1Spectral, d_mpiCommDomain);
              RealType xcEnergy = d_hamitonianXC->getEnergy();
              d_rootCout << " EXC energy: " << xcEnergy << "\n";

              // calculate band energy
              const RealType bandEnergySpinFactor =
                (d_spinMode == SpinMode::Unpolarized) ? (RealType)2 : (RealType)1;
              RealType bandEnergy = 0;
              for (size_type i = 0; i < occupancies[0].size(); i++)
                {
                  bandEnergy +=
                    bandEnergySpinFactor * occupancies[0][i] * d_kohnShamEnergies[i];
                }

              d_rootCout << "Band energy: " << bandEnergy << "\n";

              RealType totalEnergy = kinEnergy + elecEnergy + xcEnergy;

              d_rootCout << "Ground State Energy: " << totalEnergy << "\n";

              d_groundStateEnergy = totalEnergy;

              RealType entEnergy = KohnShamDFTInternal::computeEntropicEnergy(
                occupancies[0],
                d_smearingTemperature,
                (d_spinMode == SpinMode::Unpolarized) ? 2.0 : 1.0);

              d_rootCout << "Entropic Energy: " << entEnergy << "\n";

              d_rootCout << "Free Energy: " << totalEnergy - entEnergy << "\n";

              d_freeEnergy = totalEnergy - entEnergy;
              d_pTotal.registerEnd("Energy Compute");
            }

          if (scfIter > 0)
            {
              d_rootCout << "ANDERSON mixing, L2 norm of electron-density difference: "
                         << norm << "\n";
              if (d_spinMode != SpinMode::Unpolarized)
                d_rootCout
                  << "ANDERSON mixing, L2 norm of magnetization-density difference: "
                  << magNorm << "\n";
            }

          d_p.print();

          scfIter += 1;
        }

      if (d_spinMode == SpinMode::Collinear)
        {
          auto &          magDensFinal = densAttrOut.at(DensityDescrAttr::Val)[1];
          const size_type numQuad      = magDensFinal.nQuadraturePoints();
          RealType        netMag = 0.0, absMag = 0.0;
          for (size_type i = 0; i < numQuad; ++i)
            {
              const RealType mz = magDensFinal.data()[i];
              netMag += mz * d_jxwDataHost[i];
              absMag += std::abs(mz) * d_jxwDataHost[i];
            }
          utils::mpi::MPIAllreduce<memorySpaceHost>(
            utils::mpi::MPIInPlace,
            &netMag,
            1,
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_mpiCommDomain);
          utils::mpi::MPIAllreduce<memorySpaceHost>(
            utils::mpi::MPIInPlace,
            &absMag,
            1,
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_mpiCommDomain);
          d_rootCout << "Final net magnetization     : " << netMag << "\n";
          d_rootCout << "Final absolute magnetization: " << absMag << "\n";
        }

      if (!d_evaluateEnergyEverySCF)
        {
          d_pTotal.registerStart("Energy Compute");
          int rank;
          utils::mpi::MPICommRank(d_mpiCommDomain, &rank);
          utils::ConditionalOStream rootCout(std::cout, rank == 0, 16, true);

          auto &densOut = densAttrOut.at(DensityDescrAttr::Val)[0];

          if (auto hamiltonian = std::dynamic_pointer_cast<
                ElectrostaticLocalFE<ValueTypeElectrostaticsBasis,
                                     ValueTypeElectrostaticsCoeff,
                                     ValueTypeWaveFunctionBasis,
                                     memorySpace,
                                     dim>>(d_hamitonianElec))
            {
              hamiltonian->reinitField(densOut);
            }
          else if (auto hamiltonian = std::dynamic_pointer_cast<
                     ElectrostaticONCVNonLocFE<ValueTypeElectrostaticsBasis,
                                               ValueTypeElectrostaticsCoeff,
                                               ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff,
                                               memorySpace,
                                               dim>>(d_hamitonianElec))
            {
              hamiltonian->reinitField(densOut);
            }

          d_rdm1Spectral->getSpectral(wfnPtr, occupancies, nKSOrbs);
          d_hamitonianKin->evalEnergy(occupancies[0], *d_feBMWaveFn, *wfnPtr);
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
              hamiltonian->evalEnergy(occupancies[0], *wfnPtr);
            }
          d_rdm1Spectral->setSpectral(std::move(wfnPtr), occupancies, nKSOrbs);

          RealType elecEnergy = d_hamitonianElec->getEnergy();
          rootCout << "Electrostatic energy: " << elecEnergy << "\n";

          d_hamitonianXC->evalEnergy(*d_rdm1Spectral, d_mpiCommDomain);
          RealType xcEnergy = d_hamitonianXC->getEnergy();
          rootCout << "EXC energy: " << xcEnergy << "\n";

          // calculate band energy
          const RealType bandEnergySpinFactor =
            (d_spinMode == SpinMode::Unpolarized) ? (RealType)2 : (RealType)1;
          RealType bandEnergy = 0;
          for (size_type i = 0; i < occupancies[0].size(); i++)
            {
              bandEnergy +=
                bandEnergySpinFactor * occupancies[0][i] * d_kohnShamEnergies[i];
            }

          rootCout << "Band energy: " << bandEnergy << "\n";

          RealType totalEnergy = kinEnergy + elecEnergy + xcEnergy;

          rootCout << "Ground State Energy: " << totalEnergy << "\n";

          d_groundStateEnergy = totalEnergy;

          RealType entEnergy = KohnShamDFTInternal::computeEntropicEnergy(
            occupancies[0],
            d_smearingTemperature,
            (d_spinMode == SpinMode::Unpolarized) ? 2.0 : 1.0);

          rootCout << "Entropic Energy: " << entEnergy << "\n";

          rootCout << "Free Energy: " << totalEnergy - entEnergy << "\n";

          d_freeEnergy = totalEnergy - entEnergy;
          d_pTotal.registerEnd("Energy Compute");
        }

      /*
      /////============== DEBUG : Integral \psi and \psi_orthonormalized =
      Numelectrons========= /////

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
      quadRuleContainer = d_feBDEXCHamiltonian->getQuadratureRuleContainer();

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
      ///// ============== DEBUG : Integral \psi and \psi_orthonormalized =
      Numelectrons ========= /////
      */
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
