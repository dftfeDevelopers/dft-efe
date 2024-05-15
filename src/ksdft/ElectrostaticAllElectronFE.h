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

#ifndef dftefeElectrostaticAllElectronFE_h
#define dftefeElectrostaticAllElectronFE_h

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

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class ElectrostaticAllElectronFE
      : public ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>
    {
    public:
      using ValueType = typename ElectrostaticFE<ValueTypeBasisData,
                                                 ValueTypeBasisCoeff,
                                                 memorySpace,
                                                 dim>::ValueType;
      using Storage   = typename ElectrostaticFE<ValueTypeBasisData,
                                               ValueTypeBasisCoeff,
                                               memorySpace,
                                               dim>::Storage;
      using RealType  = typename ElectrostaticFE<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpace,
                                                dim>::RealType;

    public:
      /**
       * @brief Constructor
       */
      ElectrostaticAllElectronFE(
        std::vector<utils::Point> atomCoordinates,
        std::vector<double>       atomCharges,
        std::vector<double>       smearedChargeRadius,
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
          feBDTotalChargeRhs,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellTimesNumVecs);

      void
      reinitBasis(
        std::vector<utils::Point>                         atomCoordinates,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeRhs);

      void
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity);

      void
      getLocal(Storage &cellWiseStorage) const override;

      void
      evalEnergy(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs);

      void
      evalEnergy();

      RealType
      getEnergy() const override;

    private:
      RealType
      nuclearSelfEnergy(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs);

      const size_type           d_numComponents;
      std::vector<utils::Point> d_atomCoordinates;
      const std::vector<double> d_atomCharges;
      const std::vector<double> d_smearedChargeRadius;
      RealType                  d_energy;
      std::shared_ptr<
        quadrature::QuadratureValuesContainer<RealType, memorySpace>>
        d_totalChargeDensity;
      std::shared_ptr<
        quadrature::QuadratureValuesContainer<RealType, memorySpace>>
        d_nuclearChargesDensity;
      std::shared_ptr<
        quadrature::QuadratureValuesContainer<ValueType, memorySpace>>
        d_totalChargePotentialQuad;
      std::shared_ptr<const basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>>
        d_feBasisOp;
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
        d_feBDTotalChargeRhs;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;
    }; // end of class ElectrostaticAllElectronFE
  }    // end of namespace ksdft
} // end of namespace dftefe
#include <ksdft/ElectrostaticAllElectronFE.t.cpp>
#endif // dftefeElectrostaticAllElectronFE_h
