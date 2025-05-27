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

#include <ksdft/Defaults.h>
#include <basis/FEBasisDofHandler.h>
#include <utils/ConditionalOStream.h>
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
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::
      ElectrostaticONCVNonLocFE(
        const std::vector<utils::Point> &         atomCoordinates,
        const std::vector<double> &               atomCharges,
        const std::vector<std::string> &          atomSymbolVec,
        const std::shared_ptr<atoms::AtomSphericalDataContainer> atomSphericalDataContainerPSP,
        const std::vector<double> &               smearedChargeRadius,
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDAtomCenterNonLocalOperator,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxWaveFnBlock,
        const bool      useDealiiMatrixFreePoissonSolve)
      : d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_rootCout(std::cout)
      , d_mpiComm(feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator())
      , d_atomSymbolVec(atomSymbolVec)
      , d_maxCellBlock(maxCellBlock)
      , d_maxWaveFnBlock(maxWaveFnBlock)
      , d_energy((RealType)0)
      , d_atomSphericalDataContainerPSP(atomSphericalDataContainerPSP)
    {
      int rank;
      utils::mpi::MPICommRank(d_mpiComm, &rank);

      d_rootCout.setCondition(rank == 0);

      d_isNonLocPSP = false;
      for(auto i : d_atomSphericalDataContainerPSP->getFieldNames())
      {
        if(i == "beta")
        {
          d_isNonLocPSP = true;
          break;
        }
      }

      if (d_isNonLocPSP)
      {
      d_atomNonLocOpContext = std::make_shared<
        const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                   ValueTypeWaveFnCoeff,
                                                   memorySpace,
                                                   dim>>(
        *feBMWaveFn,
        *feBDAtomCenterNonLocalOperator,
        d_atomSphericalDataContainerPSP,
        ElectroHamiltonianDefaults::ATOM_PARTITION_TOL_BETA,
        atomSymbolVec,
        atomCoordinates,
        maxCellBlock,
        maxWaveFnBlock,
        linAlgOpContext,
        d_mpiComm);
      }

      d_atomVLocFunction =
        std::make_shared<const atoms::AtomSevereFunction<dim>>(
          d_atomSphericalDataContainerPSP,
          atomSymbolVec,
          atomCoordinates,
          "vlocal",
          0,
          1);

      d_electrostaticLocal =
        std::make_shared<ElectrostaticLocalFE<ValueTypeBasisData,
                                                    ValueTypeBasisCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>>(
          atomCoordinates,
          atomCharges,
          smearedChargeRadius,
          electronChargeDensity,
          feBMTotalCharge,
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDHamiltonian,
          *d_atomVLocFunction,
          linAlgOpContext,
          maxCellBlock,
          useDealiiMatrixFreePoissonSolve);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::
      ElectrostaticONCVNonLocFE(
        const std::vector<utils::Point> &         atomCoordinates,
        const std::vector<double> &               atomCharges,
        const std::vector<std::string> &          atomSymbolVec,
        const std::shared_ptr<atoms::AtomSphericalDataContainer> atomSphericalDataContainerPSP,
        const std::vector<double> &               smearedChargeRadius,
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDAtomCenterNonLocalOperator,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxWaveFnBlock,
        const bool      useDealiiMatrixFreePoissonSolve)
      : d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_rootCout(std::cout)
      , d_mpiComm(feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator())
      , d_atomSymbolVec(atomSymbolVec)
      , d_maxCellBlock(maxCellBlock)
      , d_maxWaveFnBlock(maxWaveFnBlock)
      , d_energy((RealType)0)
      , d_atomSphericalDataContainerPSP(atomSphericalDataContainerPSP)
    {
      int rank;
      utils::mpi::MPICommRank(d_mpiComm, &rank);

      d_rootCout.setCondition(rank == 0);

      d_isNonLocPSP = false;
      for(auto i : d_atomSphericalDataContainerPSP->getFieldNames())
      {
        if(i == "beta")
        {
          d_isNonLocPSP = true;
          break;
        }
      }
      
      if (d_isNonLocPSP)
      {
      d_atomNonLocOpContext = std::make_shared<
        const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                   ValueTypeWaveFnCoeff,
                                                   memorySpace,
                                                   dim>>(
        *feBMWaveFn,
        *feBDAtomCenterNonLocalOperator,
        d_atomSphericalDataContainerPSP,
        ElectroHamiltonianDefaults::ATOM_PARTITION_TOL_BETA,
        atomSymbolVec,
        atomCoordinates,
        maxCellBlock,
        maxWaveFnBlock,
        linAlgOpContext,
        d_mpiComm);
      }

      d_atomVLocFunction =
        std::make_shared<const atoms::AtomSevereFunction<dim>>(
          d_atomSphericalDataContainerPSP,
          atomSymbolVec,
          atomCoordinates,
          "vlocal",
          0,
          1);

      d_electrostaticLocal =
        std::make_shared<ElectrostaticLocalFE<ValueTypeBasisData,
                                                    ValueTypeBasisCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>>(
          atomCoordinates,
          atomCharges,
          smearedChargeRadius,
          atomicElectronChargeDensity,
          atomicTotalElecPotNuclearQuad,
          atomicTotalElecPotElectronicQuad,
          feBMTotalCharge, // will be same as bc of totalCharge -
                           // atomicTotalCharge
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDHamiltonian,
          *d_atomVLocFunction,
          linAlgOpContext,
          maxCellBlock,
          useDealiiMatrixFreePoissonSolve);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::
      reinitBasis(
        const std::vector<utils::Point> &                 atomCoordinates,
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDAtomCenterNonLocalOperator)
    {
      if(d_isNonLocPSP)
      {
      d_atomNonLocOpContext = std::make_shared<
        const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                   ValueTypeWaveFnCoeff,
                                                   memorySpace,
                                                   dim>>(
        *feBMWaveFn,
        *feBDAtomCenterNonLocalOperator,
        d_atomSphericalDataContainerPSP,
        ElectroHamiltonianDefaults::ATOM_PARTITION_TOL_BETA,
        d_atomSymbolVec,
        atomCoordinates,
        d_maxCellBlock,
        d_maxWaveFnBlock,
        d_linAlgOpContext,
        d_mpiComm);
      }

      d_atomVLocFunction =
        std::make_shared<const atoms::AtomSevereFunction<dim>>(
          d_atomSphericalDataContainerPSP,
          d_atomSymbolVec,
          atomCoordinates,
          "vlocal",
          0,
          1);

      d_electrostaticLocal->reinitBasis(atomCoordinates,
                                        feBMTotalCharge,
                                        feBDTotalChargeStiffnessMatrix,
                                        feBDNuclearChargeRhs,
                                        feBDElectronicChargeRhs,
                                        feBDHamiltonian,
                                        *d_atomVLocFunction);
    }


    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDAtomCenterNonLocalOperator)
    {
      if(d_isNonLocPSP)
      {
      d_atomNonLocOpContext = std::make_shared<
        const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                   ValueTypeWaveFnCoeff,
                                                   memorySpace,
                                                   dim>>(
        *feBMWaveFn,
        *feBDAtomCenterNonLocalOperator,
        d_atomSphericalDataContainerPSP,
        ElectroHamiltonianDefaults::ATOM_PARTITION_TOL_BETA,
        d_atomSymbolVec,
        atomCoordinates,
        d_maxCellBlock,
        d_maxWaveFnBlock,
        d_linAlgOpContext,
        d_mpiComm);
      }

      d_atomVLocFunction =
        std::make_shared<const atoms::AtomSevereFunction<dim>>(
          d_atomSphericalDataContainerPSP,
          d_atomSymbolVec,
          atomCoordinates,
          "vlocal",
          0,
          1);

      d_electrostaticLocal->reinitBasis(atomCoordinates,
                                        atomicElectronChargeDensity,
                                        atomicTotalElecPotNuclearQuad,
                                        atomicTotalElecPotElectronicQuad,
                                        feBMTotalCharge,
                                        feBDTotalChargeStiffnessMatrix,
                                        feBDNuclearChargeRhs,
                                        feBDElectronicChargeRhs,
                                        feBDHamiltonian,
                                        *d_atomVLocFunction);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity)
    {
      d_electrostaticLocal->reinitField(electronChargeDensity);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::getLocal(Storage &cellWiseStorage) const
    {
      d_electrostaticLocal->getLocal(cellWiseStorage);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const
    {
      if(d_isNonLocPSP)
        d_atomNonLocOpContext->apply(X, Y, updateGhostX, updateGhostY);
      else
      utils::throwException(
        false,
        "applyNonLocal cannot be called as number of Projectors in UPF is = 0");
    }


    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::hasLocalComponent() const
    {
      return true;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::hasNonLocalComponent() const
    {
      return d_isNonLocPSP ? true : false ;
    }


    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticONCVNonLocFE<
      ValueTypeBasisData,
      ValueTypeBasisCoeff,
      ValueTypeWaveFnBasis,
      ValueTypeWaveFnCoeff,
      memorySpace,
      dim>::evalEnergy(const std::vector<RealType> &            occupation,
                       linearAlgebra::MultiVector<ValueTypeWaveFnCoeff,
                                                  memorySpace> &X)
    {
      d_energy = (RealType)0;
      d_electrostaticLocal->evalEnergy();
      d_energy = d_electrostaticLocal->getEnergy();

      RealType nonLocEnergy = (RealType)0;
      if(d_isNonLocPSP)
      {
      linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> psiBatch(
        X.getMPIPatternP2P(),
        d_linAlgOpContext,
        d_maxWaveFnBlock,
        ValueTypeWaveFnCoeff());

      linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> Y(
        psiBatch, (ValueTypeWaveFnCoeff)0);
      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

      for (size_type psiStartId = 0; psiStartId < X.getNumberComponents();
           psiStartId += d_maxWaveFnBlock)
        {
          const size_type psiEndId =
            std::min(psiStartId + d_maxWaveFnBlock, X.getNumberComponents());
          const size_type numPsiInBatch = psiEndId - psiStartId;

          std::vector<RealType> occupationInBatch(numPsiInBatch, (RealType)0);
          RealType              energyBatchSum = 0;

          std::copy(occupation.begin() + psiStartId,
                    occupation.begin() + psiEndId,
                    occupationInBatch.begin());

          std::vector<linearAlgebra::blasLapack::
                        scalar_type<ValueTypeWaveFnCoeff, ValueTypeWaveFnCoeff>>
            dotProds(numPsiInBatch);

          if (numPsiInBatch < d_maxWaveFnBlock)
            {
              linearAlgebra::MultiVector<ValueType, memorySpace> psiBatchSmall(
                X.getMPIPatternP2P(),
                d_linAlgOpContext,
                numPsiInBatch,
                ValueType());

              linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace>
                YSmall(psiBatchSmall, (ValueTypeWaveFnCoeff)0);

              for (size_type iSize = 0; iSize < X.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    psiBatchSmall.data() +
                                      numPsiInBatch * iSize,
                                    X.data() + iSize * X.getNumberComponents() +
                                      psiStartId);

              d_atomNonLocOpContext->apply(psiBatchSmall, YSmall, true, true);
              linearAlgebra::dot(psiBatchSmall,
                                 YSmall,
                                 dotProds,
                                 linearAlgebra::blasLapack::ScalarOp::Conj,
                                 linearAlgebra::blasLapack::ScalarOp::Identity);
            }
          else
            {
              for (size_type iSize = 0; iSize < X.localSize(); iSize++)
                memoryTransfer.copy(numPsiInBatch,
                                    psiBatch.data() + numPsiInBatch * iSize,
                                    X.data() + iSize * X.getNumberComponents() +
                                      psiStartId);

              d_atomNonLocOpContext->apply(psiBatch, Y, true, true);
              linearAlgebra::dot(psiBatch,
                                 Y,
                                 dotProds,
                                 linearAlgebra::blasLapack::ScalarOp::Conj,
                                 linearAlgebra::blasLapack::ScalarOp::Identity);
            }

          for (int i = 0; i < dotProds.size(); i++)
            nonLocEnergy += dotProds[i] * 2 * occupationInBatch[i];
        }
        }
      d_rootCout << "\nNonLocal PSP Energy: " << nonLocEnergy << "\n\n";
      d_energy += nonLocEnergy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename ElectrostaticFE<ValueTypeBasisData,
                             ValueTypeBasisCoeff,
                             ValueTypeWaveFnBasis,
                             memorySpace,
                             dim>::RealType
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::getEnergy() const
    {
      return d_energy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const quadrature::QuadratureValuesContainer<
      typename ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               ValueTypeWaveFnBasis,
                               memorySpace,
                               dim>::ValueType,
      memorySpace> &
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::getFunctionalDerivative() const
    {
      return d_electrostaticLocal->getFunctionalDerivative();
    }

  } // end of namespace ksdft
} // end of namespace dftefe
