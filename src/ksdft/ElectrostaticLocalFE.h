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

#ifndef dftefeElectrostaticLocalFE_h
#define dftefeElectrostaticLocalFE_h

#include <utils/MemorySpaceType.h>
#include <linearAlgebra/MultiVector.h>
#include <ksdft/ElectrostaticFE.h>
#include <basis/FEBasisDataStorage.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <basis/FEBasisManager.h>
#include <linearAlgebra/LinearSolverFunction.h>
#include <electrostatics/PoissonLinearSolverFunctionFE.h>
#include <linearAlgebra/LinearAlgebraProfiler.h>
#include <linearAlgebra/CGLinearSolver.h>
#include <utils/ConditionalOStream.h>
#include <electrostatics/PoissonSolverDealiiMatrixFreeFE.h>
#include <atoms/AtomTCIASpline.h>
#include <atoms/AtomSuperpositionFunction.h>
#include "Defaults.h"
#include <ksdft/KSAttributes.h>
#include <ksdft/HamiltonianSpinBlockCopyKernels.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class ElectrostaticLocalFE
      : public ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               ValueTypeWaveFnBasisData,
                               memorySpace,
                               dim>
    {
    public:
      using ValueType = typename ElectrostaticFE<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff,
                                                 ValueTypeWaveFnBasisData,
                                                 memorySpace,
                                                 dim>::ValueType;
      using Storage   = typename ElectrostaticFE<ValueTypeBasisData,
                                               ValueTypeBasisCoeff,
                                               ValueTypeWaveFnBasisData,
                                               memorySpace,
                                               dim>::Storage;
      using RealType  = typename ElectrostaticFE<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                ValueTypeWaveFnBasisData,
                                                memorySpace,
                                                dim>::RealType;

    public:
      /**
       * @brief Constructor
       */
      // used if analytical vself canellation route taken
      ElectrostaticLocalFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const double &                   smearedChargeRadius,
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &                                               electronChargeDensity,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpaceHost,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        bool            useDealiiMatrixFreePoissonSolve = true,
        SpinMode        spinMode = SpinMode::Unpolarized);

      // used if numerical poisson solve vself canellation route taken
      ElectrostaticLocalFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const double &                   smearedChargeRadius,
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &                                               electronChargeDensity,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpaceHost,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclChargeStiffnessMatrixNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclChargeRhsNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        bool            useDealiiMatrixFreePoissonSolve = true,
        SpinMode        spinMode = SpinMode::Unpolarized);

      // used if delta rho approach is taken with phi total from 1D KS solve
      // with analytical vself energy cancellation
      ElectrostaticLocalFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<std::string> & atomSymbols,
        const std::vector<double> &      atomCharges,
        const double &                   smearedChargeRadius,
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicTotalElectroPotentialFunction,
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicElectronicChargeDensityFunction,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpaceHost,
                                                    dim>>
          feBMTotalCharge, // will be same as bc of totalCharge -
                           // atomicTotalCharge
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const std::unordered_map<std::string,
                                 std::shared_ptr<atoms::AtomTCIASpline>>
                   fieldToTCIASplineMap            = {},
        const bool useDealiiMatrixFreePoissonSolve = true,
        const bool calculateIntegralDeltaRho       = false,
        SpinMode   spinMode                        = SpinMode::Unpolarized);


      ~ElectrostaticLocalFE();

      // used if analytical vself canellation route taken
      void
      reinitBasis(
        const std::vector<utils::Point> &                 atomCoordinates,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpaceHost,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction);

      // used if numerical poisson solve vself canellation route taken
      void
      reinitBasis(
        const std::vector<utils::Point> &                 atomCoordinates,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpaceHost,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclChargeStiffnessMatrixNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclChargeRhsNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction);

      // used if delta rho approach is taken with phi total from 1D KS solve
      // with analytical vself energy cancellation
      void
      reinitBasis(
        const std::vector<utils::Point> &atomCoordinates,
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicTotalElectroPotentialFunction,
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicElectronicChargeDensityFunction,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpaceHost,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction);

      void
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &electronChargeDensity);

      void
      getLocal(Storage &cellWiseStorage) const override;

      void
      evalEnergy();

      RealType
      getEnergy() const override;

      std::vector<quadrature::QuadratureValuesContainer<ValueType, memorySpace>>
      getFunctionalDerivative() const override;

      void
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeWaveFnBasisData, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeWaveFnBasisData, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const override;

      bool
      hasLocalComponent() const override;

      bool
      hasNonLocalComponent() const override;

    private:
      /* Solves the nuclear potential problem, gets \sum \integral b_sm*V_sm ,
       * gets \sum \integral V_sm * rho, \sum V_smAtRhoQuadPts
       */
      void
      nuclearPotentialSolve(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeRhs);

      void
      computeNuclearSelfEnergy();

      void
      deleteStorages();

      bool                      d_useDealiiMatrixFreePoissonSolve;
      const bool                d_isCalculateIntegralDeltaRho;
      bool                      d_isNumericalVSelfSolve;
      bool                      d_isDeltaRhoSolve;
      const size_type           d_maxCellBlock;
      const size_type           d_numComponents;
      std::vector<utils::Point> d_atomCoordinates;
      const size_type           d_numAtoms;
      const std::vector<double> d_atomCharges;
      const double              d_smearedChargeRadius;
      RealType                  d_energy;
      RealType                  d_nuclearSelfEnergy;

      // Causing memory errors: Change these to smart pointers
      quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
        *d_nuclearChargesDensity;
      const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
        *d_electronChargeDensity;
      quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                            memorySpaceHost>
        *d_atomicTotalElecPotElectronicQuad;
      quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
        d_atomicElectronChargeDensity /*,d_atomicElectronChargeDensityNucQuad*/;
      quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>
        *d_correctionPotHamQuad;
      quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>
        *d_correctionPotRhoQuad;

      quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
        *d_scratchDensNuclearQuad;
      quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
        *d_scratchDensRhoQuad;
      quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>
        *d_scratchPotHamQuad;
      quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>
        *d_scratchPotRhoQuad;
      quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>
        *d_scratchPotNuclearQuad;

      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        *d_potentialHamQuadMemspace;

      linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpaceHost>
        *d_totalChargePotential;
      std::vector<linearAlgebra::MultiVector<ValueType, memorySpaceHost> *>
        d_nuclearChargesPotential;

      std::vector<std::shared_ptr<basis::FEBasisManager<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpaceHost,
                                                        dim>>>
        d_feBMNuclearCharge;

      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpaceHost,
                                                     dim>>
        d_feBasisOpNuclear;
      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpaceHost,
                                                     dim>>
        d_feBasisOpElectronic;
      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeWaveFnBasisData,
                                                     memorySpace,
                                                     dim>>
        d_feBasisOpHamiltonian;

      std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpaceHost,
                                                  dim>>
        d_feBMTotalCharge;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
        d_feBDTotalChargeStiffnessMatrix;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
        d_feBDElectronicChargeRhs;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
        d_feBDNuclearChargeRhs;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
        d_feBDNuclChargeRhsNumSol;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpaceHost>>
                            d_linAlgOpContextHost;
      std::vector<RealType> d_nuclearChargeQuad;
      size_type             d_cellTimesNumVecPoisson;
      std::shared_ptr<
        electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpaceHost,
                                                      dim>>
        d_linearSolverFunction;
      std::shared_ptr<
        electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpace,
                                                        dim>>
               d_poissonSolverDealiiMatFree;
      RealType d_totNuclearChargeQuad;

      utils::ConditionalOStream d_rootCout;

      std::map<
        std::string,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>>
        d_feBasisDataStorageRhsMap;

      RealType d_integralPhiAtxbSmear, d_intRhoAtPhiAt,
        d_correctionEnergyAtomic;

      std::vector<std::string> d_atomSymbolVec;
      const std::unordered_map<std::string,
                               std::shared_ptr<atoms::AtomTCIASpline>>
             d_fieldToTCIASplineMap;
      bool   d_isTCIEnabled;
      double d_integralDiffVZZCorrVSmearxSumBZZCorrBSmear;
      double d_integralAtRho;

      size_type              d_S;
      SpinStorageLayout      d_layout;
      std::vector<size_type> d_numCellDofs;
      size_type              d_basisOverlapSize;
      mutable Storage        d_elecCellWiseTemp;

    }; // end of class ElectrostaticLocalFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/ElectrostaticLocalFE.t.cpp>
#endif // dftefeElectrostaticLocalFE_h
