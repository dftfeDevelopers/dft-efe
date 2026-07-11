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
#include <linearAlgebra/MultiVectorProductSpace.h>
#include <linearAlgebra/MultiVectorOps.h>
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
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> & atomSymbolVec,
        const std::shared_ptr<atoms::AtomSphericalDataContainer>
                      atomSphericalDataContainerPSP,
        const double &smearedChargeRadius,
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &                                               electronChargeDensity,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpaceHost,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDAtomCenterNonLocalOperator,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxWaveFnBlock,
        const bool      useDealiiMatrixFreePoissonSolve,
        SpinMode        spinMode)
      : d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_rootCout(std::cout)
      , d_mpiComm(feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator())
      , d_atomSymbolVec(atomSymbolVec)
      , d_maxCellBlock(maxCellBlock)
      , d_maxWaveFnBlock(maxWaveFnBlock)
      , d_energy((RealType)0)
      , d_atomSphericalDataContainerPSP(atomSphericalDataContainerPSP)
      , d_spinMode(spinMode)
    {
      int rank;
      utils::mpi::MPICommRank(d_mpiComm, &rank);

      d_rootCout.setCondition(rank == 0);

      d_isNonLocPSP = false;
      for (auto i : d_atomSphericalDataContainerPSP->getFieldNames())
        {
          if (i == "beta")
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
        std::make_shared<const atoms::AtomSevereFunction<memorySpace>>(
          d_atomSphericalDataContainerPSP,
          atomSymbolVec,
          atomCoordinates,
          "vlocal",
          0,
          1,
          1,
          d_linAlgOpContext.get());

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
          useDealiiMatrixFreePoissonSolve,
          spinMode);
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
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> & atomSymbolVec,
        const std::shared_ptr<atoms::AtomSphericalDataContainer>
                      atomSphericalDataContainerPSP,
        const double &smearedChargeRadius,
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
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDAtomCenterNonLocalOperator,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxWaveFnBlock,
        const std::unordered_map<std::string,
                                 std::shared_ptr<atoms::AtomTCIASpline>>
                   fieldToTCIASplineMap,
        const bool useDealiiMatrixFreePoissonSolve,
        SpinMode   spinMode)
      : d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_rootCout(std::cout)
      , d_mpiComm(feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator())
      , d_atomSymbolVec(atomSymbolVec)
      , d_maxCellBlock(maxCellBlock)
      , d_maxWaveFnBlock(maxWaveFnBlock)
      , d_energy((RealType)0)
      , d_atomSphericalDataContainerPSP(atomSphericalDataContainerPSP)
      , d_spinMode(spinMode)
    {
      int rank;
      utils::mpi::MPICommRank(d_mpiComm, &rank);

      d_rootCout.setCondition(rank == 0);

      d_isNonLocPSP = false;
      for (auto i : d_atomSphericalDataContainerPSP->getFieldNames())
        {
          if (i == "beta")
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
        std::make_shared<const atoms::AtomSevereFunction<memorySpace>>(
          d_atomSphericalDataContainerPSP,
          atomSymbolVec,
          atomCoordinates,
          "vlocal",
          0,
          1,
          1,
          d_linAlgOpContext.get());

      ////-------DEBUG V_Local print---------------------
      // for(int i = 0 ; i < 2000 ; i++)
      //   d_rootCout << i*0.01 << "\t" <<
      //   (*d_atomVLocFunction)(dftefe::utils::Point({i*0.01,0,0}))<<std::endl;
      ////-------DEBUG V_Local print---------------------

      d_electrostaticLocal =
        std::make_shared<ElectrostaticLocalFE<ValueTypeBasisData,
                                              ValueTypeBasisCoeff,
                                              ValueTypeWaveFnBasis,
                                              memorySpace,
                                              dim>>(
          atomCoordinates,
          atomSymbolVec,
          atomCharges,
          smearedChargeRadius,
          // atomicElectronChargeDensity,
          // atomicTotalElecPotNuclearQuad,
          // atomicTotalElecPotElectronicQuad,
          atomicTotalElectroPotentialFunction,
          atomicElectronicChargeDensityFunction,
          feBMTotalCharge, // will be same as bc of totalCharge -
                           // atomicTotalCharge
          feBDTotalChargeStiffnessMatrix,
          feBDNuclearChargeRhs,
          feBDElectronicChargeRhs,
          feBDHamiltonian,
          *d_atomVLocFunction,
          linAlgOpContext,
          maxCellBlock,
          fieldToTCIASplineMap,
          useDealiiMatrixFreePoissonSolve,
          false,
          spinMode);
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
                                                    memorySpaceHost,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDAtomCenterNonLocalOperator)
    {
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
            d_atomSymbolVec,
            atomCoordinates,
            d_maxCellBlock,
            d_maxWaveFnBlock,
            d_linAlgOpContext,
            d_mpiComm);
        }

      d_atomVLocFunction =
        std::make_shared<const atoms::AtomSevereFunction<memorySpace>>(
          d_atomSphericalDataContainerPSP,
          d_atomSymbolVec,
          atomCoordinates,
          "vlocal",
          0,
          1,
          1,
          d_linAlgOpContext.get());

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
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicTotalElectroPotentialFunction,
        const atoms::AtomSuperpositionFunction<memorySpace>
          &atomicElectronicChargeDensityFunction,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpaceHost,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
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
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDHamiltonian,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis, memorySpace>>
          feBDAtomCenterNonLocalOperator)
    {
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
            d_atomSymbolVec,
            atomCoordinates,
            d_maxCellBlock,
            d_maxWaveFnBlock,
            d_linAlgOpContext,
            d_mpiComm);
        }

      d_atomVLocFunction =
        std::make_shared<const atoms::AtomSevereFunction<memorySpace>>(
          d_atomSphericalDataContainerPSP,
          d_atomSymbolVec,
          atomCoordinates,
          "vlocal",
          0,
          1,
          1,
          d_linAlgOpContext.get());

      d_electrostaticLocal->reinitBasis(atomCoordinates,
                                        // atomicElectronChargeDensity,
                                        // atomicTotalElecPotNuclearQuad,
                                        // atomicTotalElecPotElectronicQuad,
                                        atomicTotalElectroPotentialFunction,
                                        atomicElectronicChargeDensityFunction,
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
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
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
      if (d_isNonLocPSP)
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
      return d_isNonLocPSP ? true : false;
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
      const RealType spinFactor =
        (d_spinMode == SpinMode::Unpolarized) ? (RealType)2 : (RealType)1;
      if (d_isNonLocPSP)
        {
          const linearAlgebra::MultiVectorProductSpace<ValueTypeWaveFnCoeff,
                                                       memorySpace> *Xps =
            static_cast<const linearAlgebra::MultiVectorProductSpace<
              ValueTypeWaveFnCoeff,
              memorySpace> *>(&X);

          const size_type numSpaces      = Xps->numSpaces();
          const size_type numVecPerSpace = Xps->numVectorsPerSpace();
          const size_type batchPerSpin   = d_maxWaveFnBlock / numSpaces;

          if (d_mpiPatternP2P == nullptr ||
              !d_mpiPatternP2P->isCompatible(*X.getMPIPatternP2P()))
            {
              d_mpiPatternP2P = X.getMPIPatternP2P();
              d_psiBatch      = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                X.getLinAlgOpContext(),
                d_maxWaveFnBlock,
                ValueTypeWaveFnCoeff());
              d_YBatch = std::make_shared<
                linearAlgebra::MultiVector<ValueType, memorySpace>>(
                d_mpiPatternP2P,
                X.getLinAlgOpContext(),
                d_maxWaveFnBlock,
                ValueTypeWaveFnCoeff());
            }

          const size_type smallSpin  = numVecPerSpace % batchPerSpin;
          const size_type smallTotal = numSpaces * smallSpin;
          if (numVecPerSpace > batchPerSpin && smallSpin != 0)
            {
              if (d_psiBatchSmall == nullptr ||
                  d_psiBatchSmall->getNumberComponents() != smallTotal)
                {
                  d_psiBatchSmall = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    d_mpiPatternP2P,
                    X.getLinAlgOpContext(),
                    smallTotal,
                    ValueTypeWaveFnCoeff());
                  d_YBatchSmall = std::make_shared<
                    linearAlgebra::MultiVector<ValueType, memorySpace>>(
                    d_mpiPatternP2P,
                    X.getLinAlgOpContext(),
                    smallTotal,
                    ValueTypeWaveFnCoeff());
                }
            }

          for (size_type psiStartId = 0; psiStartId < numVecPerSpace;
               psiStartId += batchPerSpin)
            {
              const size_type numPsiInBatch =
                std::min(psiStartId + batchPerSpin, numVecPerSpace) -
                psiStartId;
              const size_type numPsiInBatchTotal = numSpaces * numPsiInBatch;

              std::vector<RealType> occupationInBatch(numPsiInBatchTotal,
                                                      (RealType)0);
              for (size_type s = 0; s < numSpaces; ++s)
                std::copy(
                  occupation.begin() + s * numVecPerSpace + psiStartId,
                  occupation.begin() + s * numVecPerSpace + psiStartId +
                    numPsiInBatch,
                  occupationInBatch.begin() + s * numPsiInBatch);

              std::vector<
                linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFnCoeff,
                                                       ValueTypeWaveFnCoeff>>
                dotProds(numPsiInBatchTotal);

              if (numPsiInBatch < batchPerSpin)
                {
                  linearAlgebra::MultiVectorOps::copyToBatch(
                    *Xps,
                    psiStartId,
                    numPsiInBatch,
                    *d_psiBatchSmall,
                    *X.getLinAlgOpContext());

                  d_atomNonLocOpContext->apply(*d_psiBatchSmall,
                                               *d_YBatchSmall,
                                               true,
                                               true);
                  linearAlgebra::dot(
                    *d_psiBatchSmall,
                    *d_YBatchSmall,
                    dotProds,
                    linearAlgebra::blasLapack::ScalarOp::Conj,
                    linearAlgebra::blasLapack::ScalarOp::Identity);
                }
              else
                {
                  linearAlgebra::MultiVectorOps::copyToBatch(
                    *Xps,
                    psiStartId,
                    numPsiInBatch,
                    *d_psiBatch,
                    *X.getLinAlgOpContext());

                  d_atomNonLocOpContext->apply(*d_psiBatch,
                                               *d_YBatch,
                                               true,
                                               true);
                  linearAlgebra::dot(
                    *d_psiBatch,
                    *d_YBatch,
                    dotProds,
                    linearAlgebra::blasLapack::ScalarOp::Conj,
                    linearAlgebra::blasLapack::ScalarOp::Identity);
                }

              for (size_type i = 0; i < numPsiInBatchTotal; ++i)
                nonLocEnergy +=
                  dotProds[i] * spinFactor * occupationInBatch[i];
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
    std::vector<quadrature::QuadratureValuesContainer<
      typename ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               ValueTypeWaveFnBasis,
                               memorySpace,
                               dim>::ValueType,
      memorySpace>>
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::getFunctionalDerivative() const
    {
      return d_electrostaticLocal->getFunctionalDerivative();
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasis,
              typename ValueTypeWaveFnCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    std::shared_ptr<
      const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                 ValueTypeWaveFnCoeff,
                                                 memorySpace,
                                                 dim>>
    ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              ValueTypeWaveFnCoeff,
                              memorySpace,
                              dim>::getAtomCenterNonLocalOpContextFE() const
    {
      return d_atomNonLocOpContext;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
