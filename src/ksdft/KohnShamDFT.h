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
#include <utils/ConditionalOStream.h>
#include <ksdft/MixingScheme.h>
#include <utils/Profiler.h>

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
        /* Electron density related info */
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensityInput,
        /* Basis related info */
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
        bool isResidualChebyshevFilter = true);


      // used if numerical poisson solve vself canellation route taken
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
        /* Electron density related info */
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensityInput,
        /* Basis related info */
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
        /* Field data storages poisson solves */
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
        bool isResidualChebyshevFilter = true);

      // used if delta rho approach is taken with phi total from 1D KS solve
      // with analytical vself energy cancellation
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
        /* Atomic Field for delta rho */
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
        bool isResidualChebyshevFilter = true);

      //// used if analytical vself canellation route taken with PSP
      KohnShamDFT(
        /* Atom related info */
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> & atomSymbolVec,
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
        /* Electron density related info */
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensityInput,
        /* Basis related info */
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
        const std::map<std::string, std::string> &atomSymbolToPSPFilename,
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
        bool isResidualChebyshevFilter = true);


      // used if delta rho with PSP approach is taken with phi total from 1D KS
      // solve with analytical vself energy cancellation
      KohnShamDFT(
        /* Atom related info */
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> & atomSymbolVec,
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
        /* Atomic Field for delta rho */
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
        const std::map<std::string, std::string> &atomSymbolToPSPFilename,
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
        bool isResidualChebyshevFilter = true);

      ~KohnShamDFT() = default;

      void
      solve();

      double
      getGroundStateEnergy();

    private:
      const size_type       d_numWantedEigenvalues;
      std::vector<RealType> d_occupation;
      const double          d_SCFTol;
      utils::MemoryStorage<RealType, utils::MemorySpace::HOST> d_jxwDataHost;
      std::shared_ptr<
        KohnShamEigenSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>>
        d_ksEigSolve;
      std::shared_ptr<DensityCalculator<ValueTypeWaveFunctionBasis,
                                        ValueTypeWaveFunctionCoeff,
                                        memorySpace,
                                        dim>>
        d_densCalc;
      std::shared_ptr<KohnShamOperatorContextFE<ValueTypeOperator,
                                                ValueTypeOperand,
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
      std::vector<RealType> d_kohnShamEnergies;
      linearAlgebra::MultiVector<ValueType, memorySpace>
        *                       d_kohnShamWaveFunctions;
      utils::ConditionalOStream d_rootCout;
      size_type                 d_mixingHistory;
      double                    d_mixingParameter;
      bool                      d_isAdaptiveAndersonMixingParameter;
      bool                      d_evaluateEnergyEverySCF;
      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        d_densityInQuadValues, d_densityOutQuadValues,
        d_densityResidualQuadValues;
      size_type                        d_numMaxSCFIter;
      const OpContext *                d_MContext, *d_MInvContext;
      const utils::mpi::MPIComm &      d_mpiCommDomain;
      MixingScheme<RealType, RealType> d_mixingScheme;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;
      linearAlgebra::Vector<ValueTypeWaveFunctionCoeff, memorySpace>
        d_lanczosGuess;
      linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace>
                      d_waveFunctionSubspaceGuess;
      const size_type d_numElectrons;

      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeWaveFunctionBasis,
                                        memorySpace>>
        d_feBDEXCHamiltonian;

      RealType        d_groundStateEnergy;
      bool            d_isSolved;
      utils::Profiler d_p;
      bool            d_isPSPCalculation;

      std::shared_ptr<ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                                         ValueTypeElectrostaticsBasis,
                                         ValueTypeWaveFunctionCoeff,
                                         ValueTypeWaveFunctionBasis,
                                         memorySpace,
                                         dim>>
           d_hamiltonianElectroExc;
      bool d_isResidualChebyshevFilter;
      bool d_isOEFEBasis;

    }; // end of KohnShamDFT
  }    // end of namespace ksdft
} // end of namespace dftefe
#include "KohnShamDFT.t.cpp"
#endif // dftefeKohnShamDFT_h
