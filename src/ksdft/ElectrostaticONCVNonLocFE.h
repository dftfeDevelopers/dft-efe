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

#ifndef dftefeElectrostaticONCVNonLocFE_h
#define dftefeElectrostaticONCVNonLocFE_h

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
#include <basis/AtomCenterNonLocalOpContextFE.h>
#include <atoms/AtomSevereFunction.h>
#include <ksdft/ElectrostaticLocalFE.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
                  typename ValueTypeBasisCoeff,
                  typename ValueTypeWaveFnBasis,
                  typename ValueTypeWaveFnCoeff,
                  utils::MemorySpace memorySpace,
                  size_type          dim>
    class ElectrostaticONCVNonLocFE
      : public ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               ValueTypeWaveFnBasis,
                               memorySpace,
                               dim>
    {
    public:
      using ValueType = typename ElectrostaticFE<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff,
                                                 ValueTypeWaveFnBasis,
                                                 memorySpace,
                                                 dim>::ValueType;
      using Storage   = typename ElectrostaticFE<ValueTypeBasisData,
                                               ValueTypeBasisCoeff,
                                               ValueTypeWaveFnBasis,
                                               memorySpace,
                                               dim>::Storage;
      using RealType  = typename ElectrostaticFE<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                ValueTypeWaveFnBasis,
                                                memorySpace,
                                                dim>::RealType;

    public:
      /**
       * @brief Constructor
       */
      // used if analytical vself canellation route taken
      ElectrostaticONCVNonLocFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> & atomSymbolVec,
        const std::map<std::string, std::string> &atomSymbolToPSPFilename,
        const std::vector<double> &      smearedChargeRadius,
         const quadrature::QuadratureValuesContainer<RealType, memorySpace>
           &                                               electronChargeDensity,
         std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>> feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,                                                     
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
           const basis::FEBasisDataStorage<ValueTypeWaveFnBasis,
                                           memorySpace>> feBDHamiltonian,
         std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                         linAlgOpContext,
         const size_type maxCellBlock,
         const size_type maxWaveFnBlock,
         const bool      useDealiiMatrixFreePoissonSolve = true);

      // used if delta rho approach is taken with phi total from 1D KS solve with analytical vself energy cancellation
      ElectrostaticONCVNonLocFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> & atomSymbolVec,
        const std::map<std::string, std::string> &atomSymbolToPSPFilename,        
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
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,                              
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis,
                                          memorySpace>> feBDHamiltonian,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxWaveFnBlock,        
        const bool      useDealiiMatrixFreePoissonSolve = true);

      ~ElectrostaticONCVNonLocFE() = default;

       // // used if analytical vself canellation route taken
      void
      reinitBasis(
         const std::vector<utils::Point>                        & atomCoordinates,
         std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>> feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis,
                                          memorySpace>> feBDHamiltonian);

      // used if delta rho approach is taken with phi total from 1D KS solve with analytical vself energy cancellation
      void
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
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,                                                    
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis,
                                          memorySpace>> feBDHamiltonian);                                        

      void
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity);

      void
      getLocal(Storage &cellWiseStorage) const override;

      void
      applyNonLocal(linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &X, 
                    linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &Y,
                    bool updateGhostX,
                    bool updateGhostY) const override;

      bool
      hasLocalComponent() const override;

      bool
      hasNonLocalComponent() const override;

      void
      evalEnergy(const std::vector<RealType> &                  occupation,
        linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &X);

      RealType
      getEnergy() const override;

      const quadrature::QuadratureValuesContainer<ValueType, memorySpace> &
      getFunctionalDerivative() const override;

    private:

      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>> d_linAlgOpContext;
      std::shared_ptr<atoms::AtomSphericalDataContainer> d_atomSphericalDataContainerPSP;
      std::shared_ptr<const atoms::AtomSevereFunction<dim>> d_atomVLocFunction;
      std::shared_ptr<const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                      ValueTypeWaveFnCoeff,
                                                      memorySpace,
                                                      dim>> d_atomNonLocOpContext;
      std::shared_ptr<ElectrostaticLocalFE<ValueTypeBasisData,
                                                    ValueTypeBasisCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> d_electrostaticLocal;
      RealType d_energy;
      const size_type d_numComponents;
      utils::ConditionalOStream d_rootCout;
      const utils::mpi::MPIComm d_mpiComm;
      const size_type                         d_maxCellBlock;
      const size_type                         d_maxWaveFnBlock;
      const std::vector<std::string>          d_atomSymbolVec;


    }; // end of class ElectrostaticONCVNonLocFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/ElectrostaticONCVNonLocFE.t.cpp>
#endif // dftefeElectrostaticONCVNonLocFE_h
