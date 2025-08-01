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
        bool            useDealiiMatrixFreePoissonSolve = true);

      // used if numerical poisson solve vself canellation route taken
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
        bool            useDealiiMatrixFreePoissonSolve = true);

      // used if delta rho approach is taken with phi total from 1D KS solve
      // with analytical vself energy cancellation
      ElectrostaticLocalFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<double> &      smearedChargeRadius,
        // const quadrature::QuadratureValuesContainer<RealType, memorySpace>
        //   &atomicElectronChargeDensity,
        // const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
        //                                             memorySpace>
        //   &atomicTotalElecPotNuclearQuad,
        // const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
        //                                             memorySpace>
        //   &atomicTotalElecPotElectronicQuad,
        const utils::ScalarSpatialFunctionReal
          &atomicTotalElectroPotentialFunction,
        const utils::ScalarSpatialFunctionReal
          &atomicElectronicChargeDensityFunction,
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
        const bool      useDealiiMatrixFreePoissonSolve = true,
        const bool      calculateIntegralDeltaRho       = false);


      ~ElectrostaticLocalFE();

      // used if analytical vself canellation route taken
      void
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
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction);

      // used if numerical poisson solve vself canellation route taken
      void
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
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclChargeStiffnessMatrixNumSol,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
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
        // const quadrature::QuadratureValuesContainer<RealType, memorySpace>
        //   &atomicElectronChargeDensity,
        // const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
        //                                             memorySpace>
        //   &atomicTotalElecPotNuclearQuad,
        // const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
        //                                             memorySpace>
        //   &atomicTotalElecPotElectronicQuad,
        const utils::ScalarSpatialFunctionReal
          &atomicTotalElectroPotentialFunction,
        const utils::ScalarSpatialFunctionReal
          &atomicElectronicChargeDensityFunction,
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
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction);

      void
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity);

      void
      getLocal(Storage &cellWiseStorage) const override;

      void
      evalEnergy();

      RealType
      getEnergy() const override;

      const quadrature::QuadratureValuesContainer<ValueType, memorySpace> &
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
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs);

      void
      computeNuclearSelfEnergy();

      void
      deleteStorages();

      const bool                d_useDealiiMatrixFreePoissonSolve;
      const bool                d_isCalculateIntegralDeltaRho;
      bool                      d_isNumericalVSelfSolve;
      bool                      d_isDeltaRhoSolve;
      const size_type           d_maxCellBlock;
      const size_type           d_numComponents;
      std::vector<utils::Point> d_atomCoordinates;
      const size_type           d_numAtoms;
      const std::vector<double> d_atomCharges;
      const std::vector<double> d_smearedChargeRadius;
      RealType                  d_energy;
      RealType                  d_nuclearSelfEnergy;

      // Causing memory errors: Change these to smart pointers
      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        *d_nuclearChargesDensity;
      const quadrature::QuadratureValuesContainer<RealType, memorySpace>
        *d_electronChargeDensity;
      quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff, memorySpace>
        *d_atomicTotalElecPotElectronicQuad;
      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        d_atomicElectronChargeDensity /*,d_atomicElectronChargeDensityNucQuad*/;
      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        *d_correctionPotHamQuad;
      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        *d_correctionPotRhoQuad;

      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        *d_scratchDensNuclearQuad;
      quadrature::QuadratureValuesContainer<RealType, memorySpace>
        *d_scratchDensRhoQuad;
      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        *d_scratchPotHamQuad;
      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        *d_scratchPotRhoQuad;
      quadrature::QuadratureValuesContainer<ValueType, memorySpace>
        *d_scratchPotNuclearQuad;

      linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
        *d_totalChargePotential;
      std::vector<linearAlgebra::MultiVector<ValueType, memorySpace> *>
        d_nuclearChargesPotential;

      std::vector<std::shared_ptr<basis::FEBasisManager<ValueTypeBasisCoeff,
                                                        ValueTypeBasisData,
                                                        memorySpace,
                                                        dim>>>
        d_feBMNuclearCharge;

      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>>
        d_feBasisOpNuclear;
      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>>
        d_feBasisOpElectronic;
      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeWaveFnBasisData,
                                                     memorySpace,
                                                     dim>>
        d_feBasisOpHamiltonian;

      std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>
        d_feBMTotalCharge;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        d_feBDTotalChargeStiffnessMatrix;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        d_feBDElectronicChargeRhs;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        d_feBDNuclearChargeRhs;
      std::shared_ptr<
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
        d_feBDNuclChargeRhsNumSol;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                            d_linAlgOpContext;
      std::vector<RealType> d_nuclearChargeQuad;
      size_type             d_cellTimesNumVecPoisson;
      std::shared_ptr<
        electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpace,
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
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>>
        d_feBasisDataStorageRhsMap;

      RealType d_integralPhiAtxbSmear, d_intRhoAtPhiAt,
        d_correctionEnergyAtomic;

    }; // end of class ElectrostaticLocalFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/ElectrostaticLocalFE.t.cpp>
#endif // dftefeElectrostaticLocalFE_h
