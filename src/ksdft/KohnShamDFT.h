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

#ifndef dftefeKohnShamDFT_h
#define dftefeKohnShamDFT_h

#include <variant>
#include <ksdft/ElectrostaticLocalFE.h>
#include <ksdft/ElectrostaticONCVNonLocFE.h>
#include <ksdft/KineticFE.h>
#include <ksdft/ExchangeCorrelationFE.h>
#include <ksdft/KohnShamOperatorContextFE.h>
#include <ksdft/ElectrostaticExcFE.h>
#include <ksdft/KohnShamEigenSolver.h>
#include <ksdft/DensityCalculator.h>
#include <ksdft/RDM1FE.h>
#include <utils/ConditionalOStream.h>
#include <ksdft/MixingScheme.h>
#include <ksdft/RDM1Mixing.h>
#include <utils/Profiler.h>
#include <linearAlgebra/ScalapackTemplates.h>
#include <linearAlgebra/MultiVectorProductSpace.h>
#include <linearAlgebra/MultiVectorProductSpaceBlocked.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class KohnShamDFT
    {
    public:
      using HamiltonianPtrVariant = std::variant<
        std::shared_ptr<Hamiltonian<float, memorySpace>>,
        std::shared_ptr<Hamiltonian<double, memorySpace>>,
        std::shared_ptr<Hamiltonian<std::complex<float>, memorySpace>>,
        std::shared_ptr<Hamiltonian<std::complex<double>, memorySpace>>>;

      using ValueTypeOperator =
        linearAlgebra::blasLapack::scalar_type<ValueTypeElectrostaticsBasis,
                                               ValueTypeWaveFunctionBasis>;
      using ValueTypeOperand =
        linearAlgebra::blasLapack::scalar_type<ValueTypeElectrostaticsCoeff,
                                               ValueTypeWaveFunctionCoeff>;
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperator>;
      using RealType  = linearAlgebra::blasLapack::real_type<ValueType>;
      using OpContext = typename linearAlgebra::HermitianIterativeEigenSolver<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace>::OpContext;

    public:
      // used if analytical vself canellation route taken
      KohnShamDFT(
        /* Atom related info */
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
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
        const OpContext &MContextForInv =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MInvContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        bool                        isResidualChebyshevFilter = true,
        const std::vector<double> & atomMagMomentsVec         = {},
        SpinMode                    spinMode                   = SpinMode::Unpolarized);


      // used if numerical poisson solve vself canellation route taken
      KohnShamDFT(
        /* Atom related info */
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
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
        const OpContext &MContextForInv =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MInvContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        bool                        isResidualChebyshevFilter = true,
        const std::vector<double> & atomMagMomentsVec         = {},
        SpinMode                    spinMode                   = SpinMode::Unpolarized);

      // used if delta rho approach is taken with phi total from 1D KS solve
      // with analytical vself energy cancellation
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
        const OpContext &MContextForInv =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MInvContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        bool                         isResidualChebyshevFilter = true,
        /* TCI related info */
        const atoms::TCIADataParams &params            = TCIADataDefaults::TCIA_PARAMS,
        const std::vector<double> &  atomMagMomentsVec = {},
        SpinMode                     spinMode           = SpinMode::Unpolarized);

      //// used if analytical vself canellation route taken with PSP
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
        const OpContext &MContextForInv =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MInvContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        bool                        isResidualChebyshevFilter = true,
        const std::vector<double> & atomMagMomentsVec         = {},
        SpinMode                    spinMode                   = SpinMode::Unpolarized);


      // used if delta rho with PSP approach is taken with phi total from 1D KS
      // solve with analytical vself energy cancellation
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
        const OpContext &MContextForInv =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        const OpContext &MInvContext =
          linearAlgebra::IdentityOperatorContext<ValueTypeOperator,
                                                 ValueTypeOperand,
                                                 memorySpace>(),
        bool isResidualChebyshevFilter = false,
        /* TCI related info */
        const atoms::TCIADataParams &params           = TCIADataDefaults::TCIA_PARAMS,
        const std::vector<double> &  atomMagMomentsVec = {},
        SpinMode                     spinMode           = SpinMode::Unpolarized);

      ~KohnShamDFT();

      void
      solve();

      double
      getGroundStateEnergy();

      double
      getFreeEnergy();

      void
      printTotalInScopeTimings();

      const std::shared_ptr<KohnShamOperatorContextFE<ValueTypeElectrostaticsCoeff,
                                                      ValueTypeElectrostaticsBasis,
                                                      ValueTypeWaveFunctionCoeff,
                                                      ValueTypeWaveFunctionBasis,
                                                      memorySpace,
                                                      dim>> &
      getHamiltonianOperator() const
      {
        return d_hamitonianOperator;
      }

      const std::shared_ptr<KineticFE<ValueTypeWaveFunctionBasis,
                                      ValueTypeWaveFunctionCoeff,
                                      memorySpace,
                                      dim>> &
      getHamitonianKin() const
      {
        return d_hamitonianKin;
      }

      const std::shared_ptr<ElectrostaticFE<ValueTypeElectrostaticsBasis,
                                            ValueTypeElectrostaticsCoeff,
                                            ValueTypeWaveFunctionBasis,
                                            memorySpace,
                                            dim>> &
      getHamitonianElec() const
      {
        return d_hamitonianElec;
      }

      const std::shared_ptr<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                  ValueTypeWaveFunctionCoeff,
                                                  memorySpace,
                                                  dim>> &
      getHamitonianXC() const
      {
        return d_hamitonianXC;
      }

    private:
      SpinMode              d_spinMode;
      const size_type       d_numWantedEigenvalues;
      const double          d_SCFTol;
      std::vector<RealType> d_jxwDataHost;
      std::shared_ptr<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>
        d_ksEigSolve;
      std::shared_ptr<RDM1Spectral<
        linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff>,
        memorySpace>>
        d_rdm1Spectral;
      std::shared_ptr<RDM1Mixing<
        linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFunctionBasis,
                                               ValueTypeWaveFunctionCoeff>,
        memorySpace>>
        d_rdm1Mix;
      std::shared_ptr<KohnShamOperatorContextFE<ValueTypeElectrostaticsCoeff,
                                                ValueTypeElectrostaticsBasis,
                                                ValueTypeWaveFunctionCoeff,
                                                ValueTypeWaveFunctionBasis,
                                                memorySpace,
                                                dim>>
        d_hamitonianOperator;
      std::shared_ptr<ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                            ValueTypeWaveFunctionCoeff,
                                            memorySpace,
                                            dim>>
        d_hamitonianXC;
      std::shared_ptr<ElectrostaticFE<ValueTypeElectrostaticsBasis,
                                      ValueTypeElectrostaticsCoeff,
                                      ValueTypeWaveFunctionBasis,
                                      memorySpace,
                                      dim>>
        d_hamitonianElec;
      std::shared_ptr<KineticFE<ValueTypeWaveFunctionBasis,
                                ValueTypeWaveFunctionCoeff,
                                memorySpace,
                                dim>>
        d_hamitonianKin;

      std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFunctionCoeff,
                                                  ValueTypeWaveFunctionBasis,
                                                  memorySpace,
                                                  dim>>
                                       d_feBMWaveFn;
      std::vector<RealType>            d_kohnShamEnergies;
      utils::ConditionalOStream        d_rootCout;
      bool                             d_evaluateEnergyEverySCF;
      size_type                        d_numMaxSCFIter;
      const OpContext *                d_MContext, *d_MInvContext;
      const utils::mpi::MPIComm &      d_mpiCommDomain;
      MixingScheme<RealType, RealType> d_mixingScheme;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                      d_linAlgOpContext;
      const size_type d_numElectrons;

      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                        memorySpace>>
        d_feBDEXCHamiltonian;

      RealType                     d_groundStateEnergy;
      bool                         d_isSolved;
      utils::Profiler<memorySpace> d_p, d_pTotal;
      bool                         d_isPSPCalculation;

      std::shared_ptr<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                                         ValueTypeElectrostaticsBasis,
                                         ValueTypeWaveFunctionCoeff,
                                         ValueTypeWaveFunctionBasis,
                                         memorySpace,
                                         dim>>
           d_hamiltonianElectroExc;
      bool d_isResidualChebyshevFilter;
      bool d_isOEFEBasis;

      std::shared_ptr<atoms::AtomSphericalDataContainer>
        d_atomSphericalDataContainerPSP;

      bool d_isONCVNonLocPSP, d_isNlcc;

      std::shared_ptr<linearAlgebra::ElpaScalapackManager> d_elpaScala;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpaceHost>>
                  d_linAlgOpContextHost;
      double      d_smearingTemperature, d_freeEnergy;
      std::string d_xcType;

    }; // end of KohnShamDFT
  }    // end of namespace ksdft
} // end of namespace dftefe
#include "KohnShamDFT.t.cpp"
#endif // dftefeKohnShamDFT_h
