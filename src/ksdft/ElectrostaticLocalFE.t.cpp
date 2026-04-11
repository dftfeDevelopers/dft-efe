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
#include <utils/SmearChargePotentialFunction.h>
#include <utils/SmearChargeDensityFunction.h>
#include <basis/FEBasisDofHandler.h>
#include <utils/ConditionalOStream.h>
#include <atoms/SphericalHarmonicFunctions.h>
#include <basis/EFEBasisDofHandler.h>
namespace dftefe
{
  namespace ksdft
  {
    namespace ElectrostaticLocalFEInternal
    {
      /* Assumption : field and rho have numComponents = 1 */
      template <typename ValueTypeBasisData,
                typename ValueTypeBasisCoeff,
                typename ValueTypeWaveFnBasisData,
                utils::MemorySpace memorySpace,
                size_type          dim>
      typename ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               ValueTypeWaveFnBasisData,
                               memorySpace,
                               dim>::RealType
      getIntegralFieldTimesRho(
        const quadrature::QuadratureValuesContainer<
          typename ElectrostaticFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   ValueTypeWaveFnBasisData,
                                   memorySpace,
                                   dim>::RealType,
          memorySpaceHost> &field,
        const quadrature::QuadratureValuesContainer<
          typename ElectrostaticFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   ValueTypeWaveFnBasisData,
                                   memorySpace,
                                   dim>::RealType,
          memorySpaceHost> &                                             rho,
        const utils::MemoryStorage<ValueTypeBasisData, memorySpaceHost> &jxwStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm)
      {
        using RealType = typename ElectrostaticFE<ValueTypeBasisData,
                                                  ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>::RealType;

        RealType        value                = 0;
        const RealType *fieldIter            = field.begin();
        const RealType *rhoIter              = rho.begin();
        const RealType *jxwStorageIter       = jxwStorage.data();
        size_type       cumulativeQuadInCell = 0;

        for (size_type iCell = 0; iCell < field.nCells(); iCell++)
          {
            size_type numQuadInCell = field.nCellQuadraturePoints(iCell);
            for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
              {
                const RealType jxwVal =
                  jxwStorageIter[cumulativeQuadInCell + iQuad];
                const RealType fieldVal =
                  fieldIter[cumulativeQuadInCell + iQuad];
                const RealType rhoVal = rhoIter[cumulativeQuadInCell + iQuad];
                value += rhoVal * fieldVal * jxwVal;
              }
            cumulativeQuadInCell += numQuadInCell;
          }

        // quadrature::QuadratureValuesContainer<RealType, memorySpace>
        // fieldxrho(
        //   field);

        // linearAlgebra::blasLapack::
        //   hadamardProduct<RealType, RealType, memorySpace>(
        //     field.nEntries(),
        //     field.begin(),
        //     rho.begin(),
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     fieldxrho.begin(),
        //     *linAlgOpContext);

        // linearAlgebra::blasLapack::
        //   hadamardProduct<RealType, RealType, memorySpace>(
        //     fieldxrho.nEntries(),
        //     fieldxrho.begin(),
        //     jxwStorage.data(),
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     linearAlgebra::blasLapack::ScalarOp::Identity,
        //     fieldxrho.begin(),
        //     *linAlgOpContext);

        // for (size_type iCell = 0; iCell < fieldxrho.nCells(); iCell++)
        //   {
        //     std::vector<RealType> a(
        //       fieldxrho.getQuadratureRuleContainer()->nCellQuadraturePoints(
        //         iCell));
        //     fieldxrho.template getCellValues<utils::MemorySpace::HOST>(
        //       iCell, a.data());
        //     value += std::accumulate(a.begin(), a.end(), (RealType)0);
        //   }

        int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          utils::mpi::MPIInPlace,
          &value,
          1,
          utils::mpi::Types<RealType>::getMPIDatatype(),
          utils::mpi::MPISum,
          comm);

        return value;
      }
    } // namespace ElectrostaticLocalFEInternal

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
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
        bool            useDealiiMatrixFreePoissonSolve)
      : d_atomCharges(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(d_numAtoms, nullptr)
      , d_feBMNuclearCharge(d_numAtoms, nullptr)
      , d_maxCellBlock(maxCellBlock)
      , d_isDeltaRhoSolve(false)
      , d_rootCout(std::cout)
      , d_useDealiiMatrixFreePoissonSolve(useDealiiMatrixFreePoissonSolve)
      , d_scratchDensNuclearQuad(nullptr)
      , d_nuclearChargesDensity(nullptr)
      , d_scratchPotHamQuad(nullptr)
      , d_correctionPotHamQuad(nullptr)
      , d_totalChargePotential(nullptr)
      , d_atomicTotalElecPotElectronicQuad(nullptr)
      , d_isCalculateIntegralDeltaRho(false)
      , d_isTCIEnabled(false)
      , d_linAlgOpContextHost(linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST)
      , d_potentialHamQuadMemspace(nullptr)
    {
      int rank;
      utils::mpi::MPICommRank(
        feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(), &rank);

      d_rootCout.setCondition(rank == 0);

      const basis::BasisDofHandler &basisDofHandler =
        feBMTotalCharge->getBasisDofHandler();

      if (dynamic_cast<const basis::EFEBasisDofHandler<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpaceHost,
                                                       dim> *>(
            &basisDofHandler) != nullptr)
        {
          d_useDealiiMatrixFreePoissonSolve = false;
        }

      reinitBasis(atomCoordinates,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDNuclearChargeRhs,
                  feBDElectronicChargeRhs,
                  feBDHamiltonian,
                  externalPotentialFunction);

      reinitField(electronChargeDensity);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
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
        bool            useDealiiMatrixFreePoissonSolve)
      : d_atomCharges(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(d_numAtoms, nullptr)
      , d_feBMNuclearCharge(d_numAtoms, nullptr)
      , d_maxCellBlock(maxCellBlock)
      , d_isDeltaRhoSolve(false)
      , d_rootCout(std::cout)
      , d_useDealiiMatrixFreePoissonSolve(useDealiiMatrixFreePoissonSolve)
      , d_scratchDensNuclearQuad(nullptr)
      , d_nuclearChargesDensity(nullptr)
      , d_scratchPotHamQuad(nullptr)
      , d_correctionPotHamQuad(nullptr)
      , d_totalChargePotential(nullptr)
      , d_atomicTotalElecPotElectronicQuad(nullptr)
      , d_isCalculateIntegralDeltaRho(false)
      , d_isTCIEnabled(false)
      , d_potentialHamQuadMemspace(nullptr)
    {
      int rank;
      utils::mpi::MPICommRank(
        feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(), &rank);

      d_rootCout.setCondition(rank == 0);

      const basis::BasisDofHandler &basisDofHandler =
        feBMTotalCharge->getBasisDofHandler();

      if (dynamic_cast<const basis::EFEBasisDofHandler<ValueTypeBasisCoeff,
                                                       ValueTypeBasisData,
                                                       memorySpaceHost,
                                                       dim> *>(
            &basisDofHandler) != nullptr)
        {
          d_useDealiiMatrixFreePoissonSolve = false;
        }

      reinitBasis(atomCoordinates,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDNuclearChargeRhs,
                  feBDElectronicChargeRhs,
                  feBDNuclChargeStiffnessMatrixNumSol,
                  feBDNuclChargeRhsNumSol,
                  feBDHamiltonian,
                  externalPotentialFunction);

      reinitField(electronChargeDensity);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      ElectrostaticLocalFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<std::string> & atomSymbols,
        const std::vector<double> &      atomCharges,
        const double &                   smearedChargeRadius,
        const utils::ScalarSpatialFunctionReal
          &atomicTotalElectroPotentialFunction,
        const utils::ScalarSpatialFunctionReal
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
                   fieldToTCIASplineMap,
        const bool useDealiiMatrixFreePoissonSolve,
        const bool calculateIntegralDeltaRho)
      : d_atomCoordinates(atomCoordinates)
      , d_atomCharges(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(d_numAtoms, nullptr)
      , d_feBMNuclearCharge(d_numAtoms, nullptr)
      , d_maxCellBlock(maxCellBlock)
      , d_isDeltaRhoSolve(true)
      , d_rootCout(std::cout)
      , d_useDealiiMatrixFreePoissonSolve(useDealiiMatrixFreePoissonSolve)
      , d_scratchDensNuclearQuad(nullptr)
      , d_nuclearChargesDensity(nullptr)
      , d_scratchPotHamQuad(nullptr)
      , d_correctionPotHamQuad(nullptr)
      , d_totalChargePotential(nullptr)
      , d_atomicTotalElecPotElectronicQuad(nullptr)
      , d_isCalculateIntegralDeltaRho(calculateIntegralDeltaRho)
      , d_atomSymbolVec(atomSymbols)
      , d_fieldToTCIASplineMap(fieldToTCIASplineMap)
      , d_isTCIEnabled(!d_fieldToTCIASplineMap.empty() ? true : false)
      , d_integralAtRho(0.)
      , d_potentialHamQuadMemspace(nullptr)
    {
      int rank;
      utils::mpi::MPICommRank(
        feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(), &rank);

      d_rootCout.setCondition(rank == 0);

      reinitBasis(atomCoordinates,
                  atomicTotalElectroPotentialFunction,
                  atomicElectronicChargeDensityFunction,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDNuclearChargeRhs,
                  feBDElectronicChargeRhs,
                  feBDHamiltonian,
                  externalPotentialFunction);

      reinitField(d_atomicElectronChargeDensity);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::~ElectrostaticLocalFE()
    {
      deleteStorages();
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::deleteStorages()
    {
      if (d_scratchDensNuclearQuad != nullptr)
        {
          delete d_scratchDensNuclearQuad;
          d_scratchDensNuclearQuad = nullptr;
        }
      if (d_nuclearChargesDensity != nullptr)
        {
          delete d_nuclearChargesDensity;
          d_nuclearChargesDensity = nullptr;
        }
      if (d_scratchPotHamQuad != nullptr)
        {
          delete d_scratchPotHamQuad;
          d_scratchPotHamQuad = nullptr;
        }
      if (d_potentialHamQuadMemspace != nullptr)
        {
          delete d_potentialHamQuadMemspace;
          d_potentialHamQuadMemspace = nullptr;
        }
      if (d_correctionPotHamQuad != nullptr)
        {
          delete d_correctionPotHamQuad;
          d_correctionPotHamQuad = nullptr;
        }
      // if (d_correctionPotRhoQuad != nullptr)
      //   {
      //     delete d_correctionPotRhoQuad;
      //     d_correctionPotRhoQuad = nullptr;
      //   }
      if (d_totalChargePotential != nullptr)
        {
          delete d_totalChargePotential;
          d_totalChargePotential = nullptr;
        }
      for (auto &i : d_nuclearChargesPotential)
        {
          if (i != nullptr)
            {
              delete i;
              i = nullptr;
            }
        }
      if (d_atomicTotalElecPotElectronicQuad != nullptr)
        {
          delete d_atomicTotalElecPotElectronicQuad;
          d_atomicTotalElecPotElectronicQuad = nullptr;
        }
      d_nuclearChargesPotential.clear();
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      reinitBasis(
        const std::vector<utils::Point>                        & atomCoordinates,
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
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction
        /*std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData,
                                          memorySpace>> feBDHamiltonianElec*/)
    {
      deleteStorages();
      utils::throwException(
        !d_isDeltaRhoSolve,
        "cannot call this reinitBasis() if Analytical/1D Solve rho, b and PhiTotal is used. Use different reinitBasis() instead.");
      utils::throwException(
        feBDNuclChargeRhsNumSol->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          feBDNuclChargeRhsNumSol->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            feBDNuclearChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          feBDNuclChargeRhsNumSol->getQuadratureRuleContainer() ==
            feBDNuclearChargeRhs->getQuadratureRuleContainer(),
        "The nuclearCharges RHS for both poisson solves should have same Quadrature.");

      utils::throwException(
        feBDElectronicChargeRhs->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          feBDElectronicChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            feBDHamiltonian->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          feBDElectronicChargeRhs->getQuadratureRuleContainer() ==
            feBDHamiltonian->getQuadratureRuleContainer(),
        "The  feBDElectronicChargeRHS and feBDHamiltonian should have same Quadrature.");

      d_isNumericalVSelfSolve          = true;
      d_atomCoordinates                = atomCoordinates;
      d_feBDNuclearChargeRhs           = feBDNuclearChargeRhs;
      d_feBDNuclChargeRhsNumSol        = feBDNuclChargeRhsNumSol;
      d_feBDElectronicChargeRhs        = feBDElectronicChargeRhs;
      d_feBMTotalCharge                = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;
      d_feBasisOpNuclear =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpaceHost,
                                                  dim>>(d_feBDNuclearChargeRhs,
                                                        d_maxCellBlock,
                                                        d_numComponents);
      d_feBasisOpElectronic =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpaceHost,
                                                  dim>>(
          d_feBDElectronicChargeRhs, d_maxCellBlock, d_numComponents);

      d_feBasisOpHamiltonian =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>>(feBDHamiltonian,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerElec =
          d_feBDElectronicChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerHam = feBDHamiltonian->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

      /*-----Getting V_effNiNj -------*/
      d_scratchPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerHam, d_numComponents);

      d_potentialHamQuadMemspace =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_correctionPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerHam, d_numComponents);
      /*-----Getting V_effNiNj -------*/

      // create nuclear and electron charge densities
      d_scratchDensNuclearQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchPotNuclearQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchDensRhoQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerElec, d_numComponents);

      d_scratchPotRhoQuad = d_scratchPotHamQuad;
      // new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
      //   quadRuleContainerElec, d_numComponents);

      d_correctionPotRhoQuad = d_correctionPotHamQuad;
      // new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
      //   quadRuleContainerElec, d_numComponents);

      // Init the phi_el multivector
      d_totalChargePotential =
        new linearAlgebra::MultiVector<ValueType, memorySpaceHost>(
          d_feBMTotalCharge->getMPIPatternP2P(),
          d_linAlgOpContextHost,
          d_numComponents);

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      const utils::SmearChargeDensityFunction smfunc(d_atomCoordinates,
                                                     d_atomCharges,
                                                     d_smearedChargeRadius);

      RealType d_totNuclearChargeQuad = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
           iCell++)
        {
          size_type           quadId = 0;
          std::vector<double> jxw    = quadRuleContainerNucl->getCellJxW(iCell);
          for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (RealType)smfunc(j);
                  d_totNuclearChargeQuad += (RealType)smfunc(j) * jxw[quadId];
                }
              RealType *b = a.data();
              d_nuclearChargesDensity
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &d_totNuclearChargeQuad,
        1,
        utils::mpi::Types<RealType>::getMPIDatatype(),
        utils::mpi::MPISum,
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      double totalAtomCharges =
        std::accumulate(d_atomCharges.begin(), d_atomCharges.end(), (double)0);
      // d_nuclearChargesDensity by totalAtomCharges/d_totNuclearChargeQuad
      quadrature::scale((RealType)std::abs(totalAtomCharges /
                                           d_totNuclearChargeQuad),
                        *d_nuclearChargesDensity,
                        *d_linAlgOpContextHost);

      d_rootCout << "Integral of nuclear charges over domain: "
                 << d_totNuclearChargeQuad << "\n";

      // create the correction quadValuesContainer for numerical solve

      nuclearPotentialSolve(feBDNuclChargeStiffnessMatrixNumSol,
                            feBDNuclChargeRhsNumSol);

      for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
        {
          /* Change this to feBasisOperations for electrostaic basis with same
           * quadrulecontainer as hamiltonian*/
          d_feBasisOpElectronic->interpolate(*d_nuclearChargesPotential[iAtom],
                                             *d_feBMNuclearCharge[iAtom],
                                             *d_scratchPotHamQuad);

          quadrature::add((ValueType)1.0,
                          *d_scratchPotHamQuad,
                          (ValueType)1.0,
                          *d_correctionPotHamQuad,
                          *d_correctionPotHamQuad,
                          *d_linAlgOpContextHost);
        }

      for (size_type iCell = 0; iCell < quadRuleContainerHam->nCells(); iCell++)
        {
          size_type quadId = 0;
          for (auto j : quadRuleContainerHam->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (ValueType)(externalPotentialFunction)(j);
                }
              RealType *b = a.data();
              d_scratchPotHamQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      quadrature::add((ValueType)-1.0,
                      *d_correctionPotHamQuad,
                      (ValueType)1.0,
                      *d_scratchPotHamQuad,
                      *d_correctionPotHamQuad,
                      *d_linAlgOpContextHost);

      /*
            for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
              {
                d_feBasisOpElectronic->interpolate(*d_nuclearChargesPotential[iAtom],
                                         *d_feBMNuclearCharge[iAtom],
                                         *d_correctionPotRhoQuad);

                quadrature::add((ValueType)1.0,
                                *d_correctionPotRhoQuad,
                                (ValueType)1.0,
                                *d_scratchPotRhoQuad,
                                *d_scratchPotRhoQuad,
                                *d_linAlgOpContextHost);
              }

            for (size_type iCell = 0; iCell < quadRuleContainerElec->nCells();
         iCell++)
              {
                size_type quadId = 0;
                for (auto j : quadRuleContainerElec->getCellRealPoints(iCell))
                  {
                    std::vector<RealType> a(d_numComponents);
                    for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                      {
                        a[iComp] = (ValueType)(externalPotentialFunction)(j);
                      }
                    RealType *b = a.data();
                    d_correctionPotRhoQuad
                      ->template
         setCellQuadValues<utils::MemorySpace::HOST>(iCell, quadId, b); quadId =
         quadId + 1;
                  }
              }

            quadrature::add((ValueType)-1.0,
                            *d_scratchPotRhoQuad,
                            (ValueType)1.0,
                            *d_correctionPotRhoQuad,
                            *d_correctionPotRhoQuad,
                            *d_linAlgOpContextHost);
      */

      computeNuclearSelfEnergy();

      std::map<
        std::string,
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost> &>
        inpRhsMap;

      d_feBasisDataStorageRhsMap = {{"bSmear", d_feBDNuclearChargeRhs},
                                    {"rho", d_feBDElectronicChargeRhs}};
      inpRhsMap                  = {{"bSmear", *d_scratchDensNuclearQuad},
                   {"rho", *d_scratchDensRhoQuad}};

      if (!d_useDealiiMatrixFreePoissonSolve)
        d_linearSolverFunction = std::make_shared<
          electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpaceHost,
                                                        dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContextHost,
          ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
          d_numComponents);
      else
        d_poissonSolverDealiiMatFree = std::make_shared<
          electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace,
                                                          dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContext);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
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
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction)
    {
      deleteStorages();
      utils::throwException(
        !d_isDeltaRhoSolve,
        "cannot call this reinitBasis() if Analytical/1D Solve rho, b and PhiTotal is used. Use different reinitBasis() instead.");
      utils::throwException(
        feBDElectronicChargeRhs->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          feBDElectronicChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            feBDHamiltonian->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          feBDElectronicChargeRhs->getQuadratureRuleContainer() ==
            feBDHamiltonian->getQuadratureRuleContainer(),
        "The  feBDElectronicChargeRHS and feBDHamiltonian should have same Quadrature.");

      d_isNumericalVSelfSolve          = false;
      d_atomCoordinates                = atomCoordinates;
      d_feBDNuclearChargeRhs           = feBDNuclearChargeRhs;
      d_feBDElectronicChargeRhs        = feBDElectronicChargeRhs;
      d_feBMTotalCharge                = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;
      d_feBasisOpNuclear =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpaceHost,
                                                  dim>>(d_feBDNuclearChargeRhs,
                                                        d_maxCellBlock,
                                                        d_numComponents);
      d_feBasisOpElectronic =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpaceHost,
                                                  dim>>(
          d_feBDElectronicChargeRhs, d_maxCellBlock, d_numComponents);

      d_feBasisOpHamiltonian =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>>(feBDHamiltonian,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerElec =
          d_feBDElectronicChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerHam = feBDHamiltonian->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

      /*-----Getting V_effNiNj -------*/
      d_scratchPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerHam, d_numComponents);

      d_potentialHamQuadMemspace =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_correctionPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerHam, d_numComponents);
      /*-----Getting V_effNiNj -------*/

      // create nuclear and electron charge densities and total charge potential
      // with correction
      d_scratchDensNuclearQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchPotNuclearQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchDensRhoQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerElec, d_numComponents);

      d_scratchPotRhoQuad = d_scratchPotHamQuad;
      // new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
      //   quadRuleContainerElec, d_numComponents);

      d_correctionPotRhoQuad = d_correctionPotHamQuad;
      // new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
      //   quadRuleContainerElec, d_numComponents);

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      // Init the phi_el multivector
      d_totalChargePotential =
        new linearAlgebra::MultiVector<ValueType, memorySpaceHost>(
          d_feBMTotalCharge->getMPIPatternP2P(),
          d_linAlgOpContextHost,
          d_numComponents);

      const utils::SmearChargeDensityFunction smfunc(d_atomCoordinates,
                                                     d_atomCharges,
                                                     d_smearedChargeRadius);

      RealType d_totNuclearChargeQuad = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
           iCell++)
        {
          size_type           quadId = 0;
          std::vector<double> jxw    = quadRuleContainerNucl->getCellJxW(iCell);
          for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (RealType)smfunc(j);
                  d_totNuclearChargeQuad += (RealType)smfunc(j) * jxw[quadId];
                }
              RealType *b = a.data();
              d_nuclearChargesDensity
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &d_totNuclearChargeQuad,
        1,
        utils::mpi::Types<RealType>::getMPIDatatype(),
        utils::mpi::MPISum,
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      double totalAtomCharges =
        std::accumulate(d_atomCharges.begin(), d_atomCharges.end(), (double)0);
      // d_nuclearChargesDensity by totalAtomCharges/d_totNuclearChargeQuad
      quadrature::scale((RealType)std::abs(totalAtomCharges /
                                           d_totNuclearChargeQuad),
                        *d_nuclearChargesDensity,
                        *d_linAlgOpContextHost);

      d_rootCout << "Integral of nuclear charges over domain: "
                 << d_totNuclearChargeQuad << "\n";

      // create the correction quadValuesContainer for analytical solve

      const utils::SmearChargePotentialFunction smfuncPot(
        d_atomCoordinates, d_atomCharges, d_smearedChargeRadius);

      for (size_type iCell = 0; iCell < quadRuleContainerHam->nCells(); iCell++)
        {
          size_type quadId = 0;
          for (auto j : quadRuleContainerHam->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (ValueType)(externalPotentialFunction)(j) -
                             (ValueType)(smfuncPot)(j);
                }
              RealType *b = a.data();
              d_correctionPotHamQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }
      /*
            for (size_type iCell = 0; iCell < quadRuleContainerElec->nCells();
         iCell++)
              {
                size_type quadId = 0;
                for (auto j : quadRuleContainerElec->getCellRealPoints(iCell))
                  {
                    std::vector<RealType> a(d_numComponents);
                    for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                      {
                        a[iComp] = (ValueType)(externalPotentialFunction)(j) -
                                   (ValueType)(smfuncPot)(j);
                      }
                    RealType *b = a.data();
                    d_correctionPotRhoQuad
                      ->template
         setCellQuadValues<utils::MemorySpace::HOST>(iCell, quadId, b); quadId =
         quadId + 1;
                  }
              }
      */

      computeNuclearSelfEnergy();

      d_scratchDensNuclearQuad->setValue(0);
      std::map<
        std::string,
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost> &>
        inpRhsMap;

      d_feBasisDataStorageRhsMap = {{"bSmear", d_feBDNuclearChargeRhs},
                                    {"rho", d_feBDElectronicChargeRhs}};
      inpRhsMap                  = {{"bSmear", *d_scratchDensNuclearQuad},
                   {"rho", *d_scratchDensRhoQuad}};

      if (!d_useDealiiMatrixFreePoissonSolve)
        d_linearSolverFunction = std::make_shared<
          electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpaceHost,
                                                        dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContextHost,
          ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
          d_numComponents);
      else
        d_poissonSolverDealiiMatFree = std::make_shared<
          electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace,
                                                          dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContext);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      reinitBasis(
        const std::vector<utils::Point> &atomCoordinates,
        const utils::ScalarSpatialFunctionReal
          &atomicTotalElectroPotentialFunction,
        const utils::ScalarSpatialFunctionReal
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
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction)
    {
      deleteStorages();
      utils::throwException(
        d_isDeltaRhoSolve,
        "cannot call this reinitBasis() if Analytical/1D Solve rho, b and PhiTotal is used. Use different reinitBasis() instead.");
      utils::throwException(
        feBDElectronicChargeRhs->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          feBDElectronicChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            feBDHamiltonian->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          feBDElectronicChargeRhs->getQuadratureRuleContainer() ==
            feBDHamiltonian->getQuadratureRuleContainer(),
        "The  feBDElectronicChargeRHS and feBDHamiltonian should have same Quadrature.");

      //utils::Profiler<utils::MemorySpace::HOST> p(feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      // d_atomicTotalElecPotElectronicQuad = &atomicTotalElecPotElectronicQuad;
      // d_atomicElectronChargeDensity      = atomicElectronChargeDensity;

      d_isNumericalVSelfSolve          = false;
      d_atomCoordinates                = atomCoordinates;
      d_feBDNuclearChargeRhs           = feBDNuclearChargeRhs;
      d_feBDElectronicChargeRhs        = feBDElectronicChargeRhs;
      d_feBMTotalCharge                = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;
      d_feBasisOpNuclear =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpaceHost,
                                                  dim>>(d_feBDNuclearChargeRhs,
                                                        d_maxCellBlock,
                                                        d_numComponents);
      d_feBasisOpElectronic =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpaceHost,
                                                  dim>>(
          d_feBDElectronicChargeRhs, d_maxCellBlock, d_numComponents);

      d_feBasisOpHamiltonian =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>>(feBDHamiltonian,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerElec =
          d_feBDElectronicChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerHam = feBDHamiltonian->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

      /*-----Getting V_effNiNj -------*/
      d_scratchPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerHam, d_numComponents);

      d_potentialHamQuadMemspace =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);
          
      d_correctionPotHamQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerHam, d_numComponents);
      /*-----Getting V_effNiNj -------*/
      // create nuclear and electron charge densities and total charge potential
      // with correction
      d_scratchDensNuclearQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchPotNuclearQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      d_scratchDensRhoQuad =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerElec, d_numComponents);

      d_scratchPotRhoQuad = d_scratchPotHamQuad;

      d_correctionPotRhoQuad = d_correctionPotHamQuad;

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          quadRuleContainerNucl, d_numComponents);

      // Init the phi_el multivector
      d_totalChargePotential =
        new linearAlgebra::MultiVector<ValueType, memorySpaceHost>(
          d_feBMTotalCharge->getMPIPatternP2P(),
          d_linAlgOpContextHost,
          d_numComponents);

      //----- Atomic storages init ----
      d_atomicElectronChargeDensity =
        quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      // d_atomicElectronChargeDensityNucQuad =
      //   quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>(
      //     feBDNuclearChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      d_atomicTotalElecPotElectronicQuad =
        new quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                  memorySpaceHost>(
          feBDElectronicChargeRhs->getQuadratureRuleContainer(), 1, 0.0);

      //----- Atomic storages init ----

      utils::Profiler<utils::MemorySpace::HOST> p(
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(),
        "Electrostiaitcs Reinit Basis");
      p.registerStart("Quad Eval for rhoAtFunc, vTotAtFunc , smfuncDens , externalPotentialFunction , smfuncPot  + TCI + numSelf");

      RealType *quadValueIter1 = d_atomicElectronChargeDensity.begin();
      ValueTypeBasisCoeff *quadValueIter2 =
        d_atomicTotalElecPotElectronicQuad->begin();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerVal =
          feBDElectronicChargeRhs->getQuadratureRuleContainer();

      size_type cumulativeQuadInCell = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerVal->nCells(); iCell++)
        {
          size_type numQuadInCell =
            quadRuleContainerVal->nCellQuadraturePoints(iCell);

          std::vector<RealType> valInCellQuad1 =
            (atomicElectronicChargeDensityFunction)(
              quadRuleContainerVal->getCellRealPoints(iCell));

          std::vector<ValueTypeBasisCoeff> valInCellQuad2 =
            (atomicTotalElectroPotentialFunction)(
              quadRuleContainerVal->getCellRealPoints(iCell));

          for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
            {
              quadValueIter1[cumulativeQuadInCell + iQuad] =
                valInCellQuad1[iQuad];
              quadValueIter2[cumulativeQuadInCell + iQuad] =
                valInCellQuad2[iQuad];
            }
          cumulativeQuadInCell += numQuadInCell;
        }

      size_type quadId = 0;
      auto      jxwData =
        d_atomicElectronChargeDensity.getQuadratureRuleContainer()->getJxW();
      for (size_type iCell = 0; iCell < d_atomicElectronChargeDensity.nCells();
           iCell++)
        {
          std::vector<RealType> a(
            d_atomicElectronChargeDensity.nCellQuadraturePoints(iCell) *
            d_atomicElectronChargeDensity.getNumberComponents());
          d_atomicElectronChargeDensity
            .template getCellValues<utils::MemorySpace::HOST>(iCell, a.data());
          for (auto j : a)
            {
              d_integralAtRho += *(jxwData.data() + quadId) * j;
              quadId = quadId + 1;
            }
        }
      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &d_integralAtRho,
        1,
        utils::mpi::Types<double>::getMPIDatatype(),
        utils::mpi::MPISum,
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      d_rootCout << "Integral Atomic Rho over domain: " << d_integralAtRho
                 << "\n";

      const utils::SmearChargeDensityFunction smfuncDens(d_atomCoordinates,
                                                         d_atomCharges,
                                                         d_smearedChargeRadius);

      // quadValueIter1 = d_atomicElectronChargeDensityNucQuad.begin();
      RealType *quadValueIter3 = d_nuclearChargesDensity->begin();

      quadRuleContainerVal = feBDNuclearChargeRhs->getQuadratureRuleContainer();

      RealType totNuclearChargeQuad = 0;
      cumulativeQuadInCell          = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerVal->nCells(); iCell++)
        {
          size_type numQuadInCell =
            quadRuleContainerVal->nCellQuadraturePoints(iCell);
          std::vector<double> jxw = quadRuleContainerVal->getCellJxW(iCell);

          // std::vector<RealType> valInCellQuad1 =
          //   (atomicElectronicChargeDensityFunction)(
          //     quadRuleContainerVal->getCellRealPoints(iCell));

          std::vector<RealType> valInCellQuad3 =
            (smfuncDens)(quadRuleContainerVal->getCellRealPoints(iCell));

          for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
            {
              // quadValueIter1[cumulativeQuadInCell + iQuad] =
              //   valInCellQuad1[iQuad];
              quadValueIter3[cumulativeQuadInCell + iQuad] =
                valInCellQuad3[iQuad];
              totNuclearChargeQuad += valInCellQuad3[iQuad] * jxw[iQuad];
            }
          cumulativeQuadInCell += numQuadInCell;
        }

      utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &totNuclearChargeQuad,
        1,
        utils::mpi::Types<RealType>::getMPIDatatype(),
        utils::mpi::MPISum,
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      double totalAtomCharges =
        std::accumulate(d_atomCharges.begin(), d_atomCharges.end(), (double)0);
      // d_nuclearChargesDensity by totalAtomCharges/totNuclearChargeQuad
      quadrature::scale((RealType)std::abs(totalAtomCharges /
                                           totNuclearChargeQuad),
                        *d_nuclearChargesDensity,
                        *d_linAlgOpContextHost);

      d_rootCout << "Integral of nuclear charges over domain: "
                 << totNuclearChargeQuad << "\n";

      const utils::SmearChargePotentialFunction smfuncPot(
        d_atomCoordinates, d_atomCharges, d_smearedChargeRadius);

      quadValueIter2       = d_correctionPotHamQuad->begin();
      quadRuleContainerVal = quadRuleContainerHam;

      cumulativeQuadInCell = 0;
      for (size_type iCell = 0; iCell < quadRuleContainerVal->nCells(); iCell++)
        {
          size_type numQuadInCell =
            quadRuleContainerVal->nCellQuadraturePoints(iCell);

          std::vector<ValueTypeBasisCoeff> valInCellQuad1 =
            (externalPotentialFunction)(
              quadRuleContainerVal->getCellRealPoints(iCell));

          std::vector<ValueTypeBasisCoeff> valInCellQuad2 =
            (smfuncPot)(quadRuleContainerVal->getCellRealPoints(iCell));

          for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
            {
              quadValueIter2[cumulativeQuadInCell + iQuad] =
                valInCellQuad1[iQuad] - valInCellQuad2[iQuad];
            }
          cumulativeQuadInCell += numQuadInCell;
        }

      computeNuclearSelfEnergy();

      /* Compute energy of d_integralPhiAtxbSmear , d_intRhoAtPhiAt ,
       * d_correctionEnergyAtomic*/

      d_integralPhiAtxbSmear                       = 0;
      d_intRhoAtPhiAt                              = 0;
      d_correctionEnergyAtomic                     = 0;
      d_integralDiffVZZCorrVSmearxSumBZZCorrBSmear = 0;

      if (!d_isTCIEnabled)
        {
          auto jxwStorageNucl = d_feBDNuclearChargeRhs->getJxWInAllCells();

          RealType        value              = 0;
          const RealType *jxwStorageIter     = jxwStorageNucl.data();
          const RealType *nuclChargeDensIter = d_nuclearChargesDensity->data();
          // const RealType *atomicElecChargeDensIter =
          // d_atomicElectronChargeDensityNucQuad.data();
          cumulativeQuadInCell = 0;

          for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
               iCell++)
            {
              size_type numQuadInCell =
                quadRuleContainerNucl->nCellQuadraturePoints(iCell);

              std::vector<RealType> atomicTotalElecPot =
                (atomicTotalElectroPotentialFunction)(
                  quadRuleContainerNucl->getCellRealPoints(iCell));
              std::vector<RealType> vext = (externalPotentialFunction)(
                quadRuleContainerNucl->getCellRealPoints(iCell));
              std::vector<RealType> vsmear =
                (smfuncPot)(quadRuleContainerNucl->getCellRealPoints(iCell));
              std::vector<RealType> atomicRho =
                (atomicElectronicChargeDensityFunction)(
                  quadRuleContainerNucl->getCellRealPoints(iCell));

              for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
                {
                  d_integralPhiAtxbSmear +=
                    nuclChargeDensIter[cumulativeQuadInCell + iQuad] *
                    atomicTotalElecPot[iQuad] *
                    jxwStorageIter[cumulativeQuadInCell + iQuad];
                  d_intRhoAtPhiAt +=
                    atomicRho[iQuad] * atomicTotalElecPot[iQuad] *
                    jxwStorageIter[cumulativeQuadInCell + iQuad];
                  d_correctionEnergyAtomic +=
                    atomicRho[iQuad] * (vext[iQuad] - vsmear[iQuad]) *
                    jxwStorageIter[cumulativeQuadInCell + iQuad];
                }
              cumulativeQuadInCell += numQuadInCell;
            }

          int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &d_integralPhiAtxbSmear,
            1,
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &d_intRhoAtPhiAt,
            1,
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &d_correctionEnergyAtomic,
            1,
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());
        }
      else
        {
          double ylm00 = atoms::Clm(0, 0) * atoms::Dm(0) * atoms::Qm(0, 0);

          std::shared_ptr<atoms::AtomTCIASpline> tciSpRhoAtPhiAt,
            tciSpRhoAtPhiCorr, tciSpBSmearPhiAt,
            tciSpSumBZZCorrBSmearDiffVZZCorrVSmear;
          auto it = d_fieldToTCIASplineMap.find("rhoAtom-phiAtom");
          if (it != d_fieldToTCIASplineMap.end())
            {
              tciSpRhoAtPhiAt = it->second;
            }
          else
            {
              utils::throwException(
                false,
                "Could not find the field rhoAtom-phiAtom in fieldToTCIASplineMap.");
            }

          it = d_fieldToTCIASplineMap.find("rhoAtom-vlocCorrection");
          if (it != d_fieldToTCIASplineMap.end())
            {
              tciSpRhoAtPhiCorr = it->second;
            }
          else
            {
              utils::throwException(
                false,
                "Could not find the field rhoAtom-vlocCorrection in fieldToTCIASplineMap.");
            }

          it = d_fieldToTCIASplineMap.find("bSmear-phiAtom");
          if (it != d_fieldToTCIASplineMap.end())
            {
              tciSpBSmearPhiAt = it->second;
            }
          else
            {
              utils::throwException(
                false,
                "Could not find the field bSmear-phiAtom in fieldToTCIASplineMap.");
            }

          bool useEZZCorr = false;
          it =
            d_fieldToTCIASplineMap.find("sumBZZCorrBSmear-diffVZZCorrVSmear");
          if (it != d_fieldToTCIASplineMap.end())
            {
              useEZZCorr                             = true;
              tciSpSumBZZCorrBSmearDiffVZZCorrVSmear = it->second;
            }

          for (int iAtom = 0; iAtom < atomCoordinates.size(); iAtom++)
            {
              for (int jAtom = 0; jAtom < atomCoordinates.size(); jAtom++)
                {
                  double r, theta, phi;
                  atoms::convertCartesianToSpherical((atomCoordinates[iAtom] -
                                                       atomCoordinates[jAtom]),
                                                     r,
                                                     theta,
                                                     phi,
                                                     1e-12);
                  std::string atomSymbolPair =
                    d_atomSymbolVec[iAtom] + "-" + d_atomSymbolVec[jAtom];
                  if (r < tciSpRhoAtPhiAt->maxRadialGrid())
                    {
                      d_intRhoAtPhiAt +=
                        0.5 *
                        (*tciSpRhoAtPhiAt->getSpline(atomSymbolPair, "S"))(r) *
                        (1 / (ylm00 * ylm00));
                    }
                  if (r < tciSpRhoAtPhiCorr->maxRadialGrid())
                    {
                      // vext - vsmear
                      d_correctionEnergyAtomic +=
                        0.5 *
                        (*tciSpRhoAtPhiCorr->getSpline(atomSymbolPair, "S"))(
                          r) *
                        (1 / (ylm00 * ylm00));
                    }
                  if (r < tciSpBSmearPhiAt->maxRadialGrid())
                    {
                      d_integralPhiAtxbSmear +=
                        0.5 * std::abs(d_atomCharges[iAtom]) *
                        (*tciSpBSmearPhiAt->getSpline(d_atomSymbolVec[jAtom],
                                                      "S"))(r) *
                        (1 / (ylm00 * ylm00));
                    }
                  if (useEZZCorr)
                    {
                      if (r < tciSpSumBZZCorrBSmearDiffVZZCorrVSmear
                                ->maxRadialGrid())
                        {
                          d_integralDiffVZZCorrVSmearxSumBZZCorrBSmear +=
                            0.5 * std::abs(d_atomCharges[iAtom]) *
                            std::abs(d_atomCharges[jAtom]) *
                            (*tciSpSumBZZCorrBSmearDiffVZZCorrVSmear->getSpline(
                              "DefaultAtom", "S"))(r) *
                            (1 / (ylm00 * ylm00));
                        }
                    }
                }
            }
        }

      d_rootCout << "Atomic Energies delta rho: " << d_intRhoAtPhiAt << "\t"
                 << d_correctionEnergyAtomic << "\t" << d_integralPhiAtxbSmear
                 << "\t" << d_integralDiffVZZCorrVSmearxSumBZZCorrBSmear
                 << "\n";

      p.registerEnd("Quad Eval for rhoAtFunc, vTotAtFunc , smfuncDens , externalPotentialFunction , smfuncPot  + TCI + numSelf");
      p.registerStart("Poisson Solve Object Creation");

      d_scratchDensNuclearQuad->setValue(0);
      std::map<
        std::string,
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost> &>
        inpRhsMap;

      d_feBasisDataStorageRhsMap = {{"deltarho", d_feBDElectronicChargeRhs}};
      inpRhsMap                  = {{"deltarho", *d_scratchDensRhoQuad}};

      if (!d_useDealiiMatrixFreePoissonSolve)
        d_linearSolverFunction = std::make_shared<
          electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpaceHost,
                                                        dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContextHost,
          ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
          d_numComponents);
      else
        d_poissonSolverDealiiMatFree = std::make_shared<
          electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace,
                                                          dim>>(
          d_feBMTotalCharge,
          d_feBDTotalChargeStiffnessMatrix,
          d_feBasisDataStorageRhsMap,
          inpRhsMap,
          ksdft::PoissonProblemDefaults::PC_TYPE,
          d_linAlgOpContext);
      p.registerEnd("Poisson Solve Object Creation");
      p.print();
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      reinitField(
        const quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          &electronChargeDensity)
    {
      if (d_isDeltaRhoSolve)
        {
          // /**----------Integral Delta Rho--------**/
          // size_type quadId    = 0;
          // double    normValue = 0.;
          // auto      jxwData =
          //   electronChargeDensity.getQuadratureRuleContainer()->getJxW();
          // for (size_type iCell = 0; iCell < electronChargeDensity.nCells();
          //      iCell++)
          //   {
          //     std::vector<RealType> a(
          //       electronChargeDensity.nCellQuadraturePoints(iCell) *
          //       electronChargeDensity.getNumberComponents());
          //     electronChargeDensity
          //       .template getCellValues<utils::MemorySpace::HOST>(
          //         iCell, a.data());
          //     for (auto j : a)
          //       {
          //         normValue += *(jxwData.data() + quadId) * j;
          //         quadId = quadId + 1;
          //       }
          //   }
          // utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
          //   utils::mpi::MPIInPlace,
          //   &normValue,
          //   1,
          //   utils::mpi::Types<double>::getMPIDatatype(),
          //   utils::mpi::MPISum,
          //   d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          // d_rootCout << "Integral Rho: " << normValue << "\n";

          d_electronChargeDensity = &electronChargeDensity;

          // quadrature::QuadratureValuesContainer<RealType, memorySpaceHost>
          // electronChargeDensityScaled(
          //           electronChargeDensity);

          // quadrature::scale((RealType)std::abs(d_integralAtRho/normValue),
          //                   electronChargeDensityScaled,
          //                   *d_linAlgOpContextHost);

          quadrature::add((RealType)1.0,
                          *d_electronChargeDensity,
                          (RealType)(-1.0),
                          d_atomicElectronChargeDensity,
                          *d_scratchDensRhoQuad,
                          *d_linAlgOpContextHost);

          if (d_isCalculateIntegralDeltaRho)
            {
              /**----------Integral Delta Rho--------**/
              size_type quadId             = 0;
              double    normValue          = 0.;
              size_type quadIdKS           = 0.;
              double    normValueRhoKS     = 0.;
              size_type quadIdAtomic       = 0.;
              double    normValueRhoAtomic = 0.;
              auto      jxwData =
                d_scratchDensRhoQuad->getQuadratureRuleContainer()->getJxW();
              for (size_type iCell = 0; iCell < d_scratchDensRhoQuad->nCells();
                   iCell++)
                {
                  std::vector<RealType> a(
                    d_scratchDensRhoQuad->nCellQuadraturePoints(iCell) *
                    d_scratchDensRhoQuad->getNumberComponents());
                  d_scratchDensRhoQuad
                    ->template getCellValues<utils::MemorySpace::HOST>(
                      iCell, a.data());
                  for (auto j : a)
                    {
                      normValue += *(jxwData.data() + quadId) * j;
                      quadId = quadId + 1;
                    }
                  d_electronChargeDensity
                    ->template getCellValues<utils::MemorySpace::HOST>(
                      iCell, a.data());
                  for (auto j : a)
                    {
                      normValueRhoKS += *(jxwData.data() + quadIdKS) * j;
                      quadIdKS = quadIdKS + 1;
                    }
                  d_atomicElectronChargeDensity
                    .template getCellValues<utils::MemorySpace::HOST>(iCell,
                                                                      a.data());
                  for (auto j : a)
                    {
                      normValueRhoAtomic +=
                        *(jxwData.data() + quadIdAtomic) * j;
                      quadIdAtomic = quadIdAtomic + 1;
                    }
                }
              utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                &normValue,
                1,
                utils::mpi::Types<double>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());
              utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                &normValueRhoKS,
                1,
                utils::mpi::Types<double>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());
              utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                &normValueRhoAtomic,
                1,
                utils::mpi::Types<double>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());
              d_rootCout << "Integral Delta Rho: " << normValue << "\n";
              d_rootCout << "KS Rho: " << normValueRhoKS << "\n";
              d_rootCout << "Atomic Rho: " << normValueRhoAtomic << "\n";
              /**----------Integral Delta Rho--------**/
            }

          // Scale by 4\pi
          quadrature::scale((RealType)(4 * utils::mathConstants::pi),
                            *d_scratchDensRhoQuad,
                            *d_linAlgOpContextHost);

          /*---- solve poisson problem for delta rho system ---*/

          std::map<std::string,
                   const quadrature::QuadratureValuesContainer<RealType,
                                                               memorySpaceHost> &>
            inpRhsMap = {{"deltarho", *d_scratchDensRhoQuad}};

          utils::Profiler<utils::MemorySpace::HOST> p(
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(),
            "Delta Rho Poisson Solve");
          p.registerStart("Reinit");
          if (!d_useDealiiMatrixFreePoissonSolve)
            d_linearSolverFunction->reinit(d_feBMTotalCharge, inpRhsMap);
          else
            d_poissonSolverDealiiMatFree->reinit(d_feBMTotalCharge, inpRhsMap);
          p.registerEnd("Reinit");

          p.registerStart("Solve");
          if (!d_useDealiiMatrixFreePoissonSolve)
            {
              linearAlgebra::LinearAlgebraProfiler profiler;

              std::shared_ptr<
                linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpaceHost>>
                CGSolve = std::make_shared<
                  linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpaceHost>>(
                  ksdft::PoissonProblemDefaults::MAX_ITER,
                  ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
                  ksdft::PoissonProblemDefaults::RELATIVE_TOL,
                  ksdft::PoissonProblemDefaults::DIVERGENCE_TOL,
                  profiler);
              CGSolve->solve(*d_linearSolverFunction);
            }
          else
            {
              d_poissonSolverDealiiMatFree->solve(
                ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
                ksdft::PoissonProblemDefaults::MAX_ITER);
            }
          p.registerEnd("Solve");
          p.print();

          if (!d_useDealiiMatrixFreePoissonSolve)
            d_linearSolverFunction->getSolution(*d_totalChargePotential);
          else
            d_poissonSolverDealiiMatFree->getSolution(*d_totalChargePotential);
        }
      else
        {
          d_electronChargeDensity = &electronChargeDensity;

          // Scale by 4\pi
          quadrature::scale((RealType)(4 * utils::mathConstants::pi),
                            electronChargeDensity,
                            *d_scratchDensRhoQuad,
                            *d_linAlgOpContextHost);

          // Scale by 4\pi
          quadrature::scale((RealType)(4 * utils::mathConstants::pi),
                            *d_nuclearChargesDensity,
                            *d_scratchDensNuclearQuad,
                            *d_linAlgOpContextHost);

          /*---- solve poisson problem for b+rho system ---*/

          std::map<std::string,
                   const quadrature::QuadratureValuesContainer<RealType,
                                                               memorySpaceHost> &>
            inpRhsMap = {{"bSmear", *d_scratchDensNuclearQuad},
                         {"rho", *d_scratchDensRhoQuad}};

          utils::Profiler<utils::MemorySpace::HOST> p(
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(),
            "b+rho Poisson Solve");
          p.registerStart("Reinit");
          if (!d_useDealiiMatrixFreePoissonSolve)
            d_linearSolverFunction->reinit(d_feBMTotalCharge, inpRhsMap);
          else
            d_poissonSolverDealiiMatFree->reinit(d_feBMTotalCharge, inpRhsMap);
          p.registerEnd("Reinit");

          p.registerStart("Solve");
          if (!d_useDealiiMatrixFreePoissonSolve)
            {
              linearAlgebra::LinearAlgebraProfiler profiler;

              std::shared_ptr<
                linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpaceHost>>
                CGSolve = std::make_shared<
                  linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpaceHost>>(
                  ksdft::PoissonProblemDefaults::MAX_ITER,
                  ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
                  ksdft::PoissonProblemDefaults::RELATIVE_TOL,
                  ksdft::PoissonProblemDefaults::DIVERGENCE_TOL,
                  profiler);
              CGSolve->solve(*d_linearSolverFunction);
            }
          else
            {
              d_poissonSolverDealiiMatFree->solve(
                ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
                ksdft::PoissonProblemDefaults::MAX_ITER);
            }
          p.registerEnd("Solve");
          p.print();

          if (!d_useDealiiMatrixFreePoissonSolve)
            d_linearSolverFunction->getSolution(*d_totalChargePotential);
          else
            d_poissonSolverDealiiMatFree->getSolution(*d_totalChargePotential);
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::getLocal(Storage &cellWiseStorage) const
    {
      // add the correction quadValuesCOntainer for potential to
      // d_scratchPotHamQuad

      /* Change this to feBasisOperations for electrostaic basis with same
       * quadrule as hamiltonian*/
      d_feBasisOpElectronic->interpolate(*d_totalChargePotential,
                                         *d_feBMTotalCharge,
                                         *d_scratchPotHamQuad);

      if (d_isDeltaRhoSolve)
        quadrature::add((ValueType)1.0,
                        *d_atomicTotalElecPotElectronicQuad,
                        (ValueType)1.0,
                        *d_scratchPotHamQuad,
                        *d_linAlgOpContextHost);

      quadrature::add((ValueType)1.0,
                      *d_scratchPotHamQuad,
                      (ValueType)1.0,
                      *d_correctionPotHamQuad,
                      *d_scratchPotHamQuad,
                      *d_linAlgOpContextHost);

      utils::MemoryTransfer<memorySpace, memorySpaceHost>
        memoryTransfer;

      memoryTransfer.copy(d_scratchPotHamQuad->nEntries(),
                          d_potentialHamQuadMemspace->data(),
                          d_scratchPotHamQuad->data());

      d_feBasisOpHamiltonian->computeFEMatrices(
        basis::realspace::LinearLocalOp::IDENTITY,
        basis::realspace::VectorMathOp::MULT,
        basis::realspace::VectorMathOp::MULT,
        basis::realspace::LinearLocalOp::IDENTITY,
        *d_potentialHamQuadMemspace,
        cellWiseStorage,
        *d_linAlgOpContext);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      nuclearPotentialSolve(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpaceHost>>
          feBDNuclearChargeRhs)
    {
      // Solve poisson problem for individual atoms
      std::shared_ptr<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpaceHost, dim>>
        feBDHNuclearCharge = std::dynamic_pointer_cast<
          const basis::
            FEBasisDofHandler<ValueTypeBasisCoeff, memorySpaceHost, dim>>(
          feBDNuclearChargeRhs->getBasisDofHandler());
      utils::throwException(
        feBDHNuclearCharge != nullptr,
        "Could not cast BasisDofHandler of the input Field to FEBasisDofHandler "
        "in ElectrostaticLocalFE");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          feBDNuclearChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<
        electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpaceHost,
                                                      dim>>
        linearSolverFunctionNuclear = nullptr;

      std::shared_ptr<
        electrostatics::PoissonSolverDealiiMatrixFreeFE<ValueTypeBasisData,
                                                        ValueTypeBasisCoeff,
                                                        memorySpaceHost,
                                                        dim>>
        poissonSolverDealiiMatFree = nullptr;

      d_nuclearChargeQuad.clear();
      d_nuclearChargeQuad.resize(d_numAtoms, 0);
      for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
        {
          std::shared_ptr<const utils::ScalarSpatialFunctionReal> smfunc =
            std::make_shared<const utils::SmearChargePotentialFunction>(
              d_atomCoordinates[iAtom],
              d_atomCharges[iAtom],
              d_smearedChargeRadius);

          d_feBMNuclearCharge[iAtom] =
            std::make_shared<basis::FEBasisManager<ValueTypeBasisCoeff,
                                                   ValueTypeBasisData,
                                                   memorySpaceHost,
                                                   dim>>(feBDHNuclearCharge,
                                                         smfunc);

          smfunc = std::make_shared<const utils::SmearChargeDensityFunction>(
            d_atomCoordinates[iAtom],
            d_atomCharges[iAtom],
            d_smearedChargeRadius);

          d_nuclearChargeQuad[iAtom] = 0;
          for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
               iCell++)
            {
              size_type           quadId = 0;
              std::vector<double> jxw =
                quadRuleContainerNucl->getCellJxW(iCell);
              for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
                {
                  std::vector<RealType> a(d_numComponents);
                  for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                    {
                      a[iComp] = (RealType)(*smfunc)(j);
                      d_nuclearChargeQuad[iAtom] +=
                        (RealType)(*smfunc)(j)*jxw[quadId];
                    }
                  RealType *b = a.data();
                  d_scratchDensNuclearQuad
                    ->template setCellQuadValues<utils::MemorySpace::HOST>(
                      iCell, quadId, b);
                  quadId = quadId + 1;
                }
            }

          utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            &d_nuclearChargeQuad[iAtom],
            1,
            utils::mpi::Types<RealType>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          // Scale by 4\pi * d_atomCharges[iAtom]/d_nuclearChargeQuad[iAtom]
          quadrature::scale((RealType)std::abs(4 * utils::mathConstants::pi *
                                               d_atomCharges[iAtom] /
                                               d_nuclearChargeQuad[iAtom]),
                            *d_scratchDensNuclearQuad,
                            *d_linAlgOpContextHost);

          utils::Profiler<utils::MemorySpace::HOST> p(
            d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(),
            "bNuclear Poisson Solve for Atom " + std::to_string(iAtom + 1));
          p.registerStart("Reinit");
          if (iAtom == 0)
            {
              if (!d_useDealiiMatrixFreePoissonSolve)
                linearSolverFunctionNuclear = std::make_shared<
                  electrostatics::PoissonLinearSolverFunctionFE<
                    ValueTypeBasisData,
                    ValueTypeBasisCoeff,
                    memorySpaceHost,
                    dim>>(d_feBMNuclearCharge[iAtom],
                          feBDNuclearChargeStiffnessMatrix,
                          feBDNuclearChargeRhs,
                          *d_scratchDensNuclearQuad,
                          ksdft::PoissonProblemDefaults::PC_TYPE,
                          d_linAlgOpContextHost,
                          ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
                          d_numComponents);
              else
                poissonSolverDealiiMatFree = std::make_shared<
                  electrostatics::PoissonSolverDealiiMatrixFreeFE<
                    ValueTypeBasisData,
                    ValueTypeBasisCoeff,
                    memorySpaceHost,
                    dim>>(d_feBMNuclearCharge[iAtom],
                          feBDNuclearChargeStiffnessMatrix,
                          feBDNuclearChargeRhs,
                          *d_scratchDensNuclearQuad,
                          ksdft::PoissonProblemDefaults::PC_TYPE,
                          d_linAlgOpContextHost);
            }
          else
            {
              if (!d_useDealiiMatrixFreePoissonSolve)
                linearSolverFunctionNuclear->reinit(d_feBMNuclearCharge[iAtom],
                                                    *d_scratchDensNuclearQuad);
              else
                poissonSolverDealiiMatFree->reinit(d_feBMNuclearCharge[iAtom],
                                                   *d_scratchDensNuclearQuad);
            }
          p.registerEnd("Reinit");

          p.registerStart("Solve");
          if (!d_useDealiiMatrixFreePoissonSolve)
            {
              linearAlgebra::LinearAlgebraProfiler profiler;

              std::shared_ptr<
                linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpaceHost>>
                CGSolve = std::make_shared<
                  linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpaceHost>>(
                  ksdft::PoissonProblemDefaults::MAX_ITER,
                  ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
                  ksdft::PoissonProblemDefaults::RELATIVE_TOL,
                  ksdft::PoissonProblemDefaults::DIVERGENCE_TOL,
                  profiler);
              CGSolve->solve(*linearSolverFunctionNuclear);
            }
          else
            poissonSolverDealiiMatFree->solve(
              ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
              ksdft::PoissonProblemDefaults::MAX_ITER);
          p.registerEnd("Solve");
          p.print();

          d_nuclearChargesPotential[iAtom] =
            new linearAlgebra::MultiVector<ValueType, memorySpaceHost>(
              d_feBMNuclearCharge[iAtom]->getMPIPatternP2P(),
              d_linAlgOpContextHost,
              d_numComponents);

          if (!d_useDealiiMatrixFreePoissonSolve)
            linearSolverFunctionNuclear->getSolution(
              *d_nuclearChargesPotential[iAtom]);
          else
            poissonSolverDealiiMatFree->getSolution(
              *d_nuclearChargesPotential[iAtom]);
        }
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::computeNuclearSelfEnergy()
    {
      // self energy computation
      RealType selfEnergy = 0;

      if (d_isNumericalVSelfSolve)
        {
          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainerNucl =
              d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

          auto jxwStorageNucl = d_feBDNuclearChargeRhs->getJxWInAllCells();

          for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
            {
              const utils::SmearChargeDensityFunction smfunc(
                d_atomCoordinates[iAtom],
                d_atomCharges[iAtom],
                d_smearedChargeRadius);

              for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
                   iCell++)
                {
                  size_type           quadId = 0;
                  std::vector<double> jxw =
                    quadRuleContainerNucl->getCellJxW(iCell);
                  for (auto j : quadRuleContainerNucl->getCellRealPoints(iCell))
                    {
                      std::vector<RealType> a(d_numComponents);
                      for (size_type iComp = 0; iComp < d_numComponents;
                           iComp++)
                        {
                          a[iComp] = (RealType)(smfunc)(j);
                        }
                      RealType *b = a.data();
                      d_scratchDensNuclearQuad
                        ->template setCellQuadValues<utils::MemorySpace::HOST>(
                          iCell, quadId, b);
                      quadId = quadId + 1;
                    }
                }

              // Scale by d_atomCharges[iAtom]/d_nuclearChargeQuad[iAtom]
              quadrature::scale((RealType)std::abs(d_atomCharges[iAtom] /
                                                   d_nuclearChargeQuad[iAtom]),
                                *d_scratchDensNuclearQuad,
                                *d_linAlgOpContextHost);

              basis::FEBasisOperations<ValueTypeBasisCoeff,
                                       ValueTypeBasisData,
                                       memorySpaceHost,
                                       dim>
                feBasisOp(d_feBDNuclChargeRhsNumSol,
                          d_maxCellBlock,
                          d_numComponents);

              feBasisOp.interpolate(*d_nuclearChargesPotential[iAtom],
                                    *d_feBMNuclearCharge[iAtom],
                                    *d_scratchPotNuclearQuad);

              RealType selfEnergyAtom =
                ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
                  ValueTypeBasisData,
                  ValueTypeBasisCoeff,
                  ValueTypeWaveFnBasisData,
                  memorySpaceHost,
                  dim>(*d_scratchPotNuclearQuad,
                       *d_scratchDensNuclearQuad,
                       jxwStorageNucl,
                       d_linAlgOpContextHost,
                       d_feBMNuclearCharge[iAtom]
                         ->getMPIPatternP2P()
                         ->mpiCommunicator());

              selfEnergy += selfEnergyAtom;
            }
          selfEnergy *= 0.5;
        }
      else
        {
          if (d_isTCIEnabled)
            {
              for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
                {
                  double rc = d_fieldToTCIASplineMap.begin()
                                ->second->smearedChargeRadius();
                  const utils::SmearChargePotentialFunction smfunc(
                    d_atomCoordinates[iAtom], d_atomCharges[iAtom], rc);

                  double Ig = 10976. / (17875 * rc);
                  selfEnergy +=
                    (RealType)(0.5 * std::pow(d_atomCharges[iAtom], 2) *
                               (Ig - (smfunc(d_atomCoordinates[iAtom]) /
                                      d_atomCharges[iAtom])));
                }
              selfEnergy *= -1;
            }
          else
            {
              // for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
              //   {
              //     const utils::SmearChargePotentialFunction smfunc(
              //       d_atomCoordinates[iAtom],
              //       d_atomCharges[iAtom],
              //       d_smearedChargeRadius);

              //     double Ig = 10976. / (17875 * d_smearedChargeRadius);
              //     selfEnergy +=
              //       (RealType)(0.5 * std::pow(d_atomCharges[iAtom], 2) *
              //                  (Ig - (smfunc(d_atomCoordinates[iAtom]) /
              //                         d_atomCharges[iAtom])));
              //   }
              // selfEnergy *= -1;

              std::vector<std::shared_ptr<utils::SmearChargeDensityFunction>>
                smfuncDens(0);
              std::vector<std::shared_ptr<utils::SmearChargePotentialFunction>>
                smfuncPot(0);

              std::shared_ptr<const quadrature::QuadratureRuleContainer>
                quadRuleContainerNucl =
                  d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

              auto jxwStorageNucl = d_feBDNuclearChargeRhs->getJxWInAllCells();

              std::vector<RealType> selfEnergyAtom(d_numAtoms, 0),
                atomNuclearChargeQuad(d_numAtoms, 0);

              for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
                {
                  smfuncDens.push_back(
                    std::make_shared<utils::SmearChargeDensityFunction>(
                      d_atomCoordinates[iAtom],
                      d_atomCharges[iAtom],
                      d_smearedChargeRadius));

                  smfuncPot.push_back(
                    std::make_shared<utils::SmearChargePotentialFunction>(
                      d_atomCoordinates[iAtom],
                      d_atomCharges[iAtom],
                      d_smearedChargeRadius));
                }

              RealType        value                = 0;
              const RealType *jxwStorageIter       = jxwStorageNucl.data();
              size_type       cumulativeQuadInCell = 0;

              for (size_type iCell = 0; iCell < quadRuleContainerNucl->nCells();
                   iCell++)
                {
                  size_type numQuadInCell =
                    quadRuleContainerNucl->nCellQuadraturePoints(iCell);
                  for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
                    {
                      std::vector<RealType> chargeDens = (*smfuncDens[iAtom])(
                        quadRuleContainerNucl->getCellRealPoints(iCell));
                      std::vector<RealType> chargePot = (*smfuncPot[iAtom])(
                        quadRuleContainerNucl->getCellRealPoints(iCell));

                      for (size_type iQuad = 0; iQuad < numQuadInCell; iQuad++)
                        {
                          selfEnergyAtom[iAtom] +=
                            chargeDens[iQuad] * chargePot[iQuad] *
                            jxwStorageIter[cumulativeQuadInCell + iQuad];
                          atomNuclearChargeQuad[iAtom] +=
                            chargeDens[iQuad] *
                            jxwStorageIter[cumulativeQuadInCell + iQuad];
                        }
                    }
                  cumulativeQuadInCell += numQuadInCell;
                }

              int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                selfEnergyAtom.data(),
                d_numAtoms,
                utils::mpi::Types<RealType>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

              mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                atomNuclearChargeQuad.data(),
                d_numAtoms,
                utils::mpi::Types<RealType>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

              for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
                {
                  selfEnergy += selfEnergyAtom[iAtom] *
                                std::abs(d_atomCharges[iAtom] /
                                         atomNuclearChargeQuad[iAtom]);
                }

              selfEnergy *= 0.5;
            }
        }
      d_nuclearSelfEnergy = selfEnergy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::evalEnergy()
    {
      d_energy = (RealType)0;

      RealType totalEnergy = 0;

      d_feBasisOpNuclear->interpolate(*d_totalChargePotential,
                                      *d_feBMTotalCharge,
                                      *d_scratchPotNuclearQuad);

      if (!d_isDeltaRhoSolve)
        {
          d_feBasisOpElectronic->interpolate(*d_totalChargePotential,
                                             *d_feBMTotalCharge,
                                             *d_scratchPotRhoQuad);

          RealType integralPhixRho =
            ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
              ValueTypeBasisData,
              ValueTypeBasisCoeff,
              ValueTypeWaveFnBasisData,
              memorySpaceHost,
              dim>(*d_scratchPotRhoQuad,
                   *d_electronChargeDensity,
                   d_feBDElectronicChargeRhs->getJxWInAllCells(),
                   d_linAlgOpContextHost,
                   d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          RealType integralPhixbSmear =
            ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
              ValueTypeBasisData,
              ValueTypeBasisCoeff,
              ValueTypeWaveFnBasisData,
              memorySpaceHost,
              dim>(*d_scratchPotNuclearQuad,
                   *d_nuclearChargesDensity,
                   d_feBDNuclearChargeRhs->getJxWInAllCells(),
                   d_linAlgOpContextHost,
                   d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          totalEnergy = (integralPhixRho + integralPhixbSmear) * 0.5;
        }
      else
        {
          RealType integralDelPhixbSmear =
            ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
              ValueTypeBasisData,
              ValueTypeBasisCoeff,
              ValueTypeWaveFnBasisData,
              memorySpaceHost,
              dim>(*d_scratchPotNuclearQuad,
                   *d_nuclearChargesDensity,
                   d_feBDNuclearChargeRhs->getJxWInAllCells(),
                   d_linAlgOpContextHost,
                   d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          // RealType intRhoAtDelPhi =
          //   ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
          //     ValueTypeBasisData,
          //     ValueTypeBasisCoeff,
          //     ValueTypeWaveFnBasisData,
          //     memorySpace,
          //     dim>(*d_scratchPotNuclearQuad,
          //          d_atomicElectronChargeDensityNucQuad,
          //          d_feBDNuclearChargeRhs->getJxWInAllCells(),
          //          d_linAlgOpContextHost,
          //          d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          quadrature::add((ValueType)1.0,
                          *d_electronChargeDensity,
                          (ValueType)-1.0,
                          d_atomicElectronChargeDensity,
                          *d_scratchDensRhoQuad,
                          *d_linAlgOpContextHost);

          d_feBasisOpElectronic->interpolate(*d_totalChargePotential,
                                             *d_feBMTotalCharge,
                                             *d_scratchPotRhoQuad);

          RealType intRhoAtDelPhi =
            ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
              ValueTypeBasisData,
              ValueTypeBasisCoeff,
              ValueTypeWaveFnBasisData,
              memorySpaceHost,
              dim>(*d_scratchPotRhoQuad,
                   d_atomicElectronChargeDensity,
                   d_feBDElectronicChargeRhs->getJxWInAllCells(),
                   d_linAlgOpContextHost,
                   d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          quadrature::add((ValueType)1.0,
                          *d_atomicTotalElecPotElectronicQuad,
                          (ValueType)1.0,
                          *d_scratchPotRhoQuad,
                          *d_linAlgOpContextHost);

          RealType intDelRhoPhiTot =
            ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
              ValueTypeBasisData,
              ValueTypeBasisCoeff,
              ValueTypeWaveFnBasisData,
              memorySpaceHost,
              dim>(*d_scratchPotRhoQuad,
                   *d_scratchDensRhoQuad,
                   d_feBDElectronicChargeRhs->getJxWInAllCells(),
                   d_linAlgOpContextHost,
                   d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          totalEnergy = (d_integralPhiAtxbSmear + integralDelPhixbSmear +
                         d_intRhoAtPhiAt + intRhoAtDelPhi + intDelRhoPhiTot +
                         d_integralDiffVZZCorrVSmearxSumBZZCorrBSmear) *
                        0.5;

          // d_rootCout << "integralPhiAtxbSmear : " << d_integralPhiAtxbSmear
          // << "\n"; d_rootCout << "integralDelPhixbSmear : " <<
          // integralDelPhixbSmear << "\n"; d_rootCout << "intRhoAtPhiAt : " <<
          // d_intRhoAtPhiAt << "\n"; d_rootCout << "intRhoAtDelPhi : " <<
          // intRhoAtDelPhi << "\n"; d_rootCout << "intDelRhoPhiTot : " <<
          // intDelRhoPhiTot << "\n";
        }

      // correction energy evaluation

      RealType correctionEnergy = 0;

      if (d_isDeltaRhoSolve)
        {
          RealType correctionEnergyDelta =
            ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
              ValueTypeBasisData,
              ValueTypeBasisCoeff,
              ValueTypeWaveFnBasisData,
              memorySpaceHost,
              dim>(*d_correctionPotRhoQuad,
                   *d_scratchDensRhoQuad, // delRho from above
                   d_feBDElectronicChargeRhs->getJxWInAllCells(),
                   d_linAlgOpContextHost,
                   d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

          correctionEnergy = correctionEnergyDelta + d_correctionEnergyAtomic;

          // d_rootCout << "correctionEnergyAtomic: " <<
          // d_correctionEnergyAtomic << "\n"; d_rootCout <<
          // "correctionEnergyDelta: " << correctionEnergyDelta << "\n";
        }
      else
        {
          correctionEnergy =
            ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
              ValueTypeBasisData,
              ValueTypeBasisCoeff,
              ValueTypeWaveFnBasisData,
              memorySpaceHost,
              dim>(*d_correctionPotRhoQuad,
                   *d_electronChargeDensity,
                   d_feBDElectronicChargeRhs->getJxWInAllCells(),
                   d_linAlgOpContextHost,
                   d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());
        }

      d_energy = totalEnergy - d_nuclearSelfEnergy + correctionEnergy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename ElectrostaticFE<ValueTypeBasisData,
                             ValueTypeBasisCoeff,
                             ValueTypeWaveFnBasisData,
                             memorySpace,
                             dim>::RealType
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::getEnergy() const
    {
      return d_energy;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const quadrature::QuadratureValuesContainer<
      typename ElectrostaticFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               ValueTypeWaveFnBasisData,
                               memorySpace,
                               dim>::ValueType,
      memorySpace> &
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::getFunctionalDerivative() const
    {
      d_feBasisOpElectronic->interpolate(*d_totalChargePotential,
                                         *d_feBMTotalCharge,
                                         *d_scratchPotHamQuad);

      if (d_isDeltaRhoSolve)
        quadrature::add((ValueType)1.0,
                        *d_atomicTotalElecPotElectronicQuad,
                        (ValueType)1.0,
                        *d_scratchPotHamQuad,
                        *d_linAlgOpContextHost);

      quadrature::add((ValueType)1.0,
                      *d_scratchPotHamQuad,
                      (ValueType)1.0,
                      *d_correctionPotHamQuad,
                      *d_scratchPotHamQuad,
                      *d_linAlgOpContextHost);

      utils::MemoryTransfer<memorySpace, memorySpaceHost>
        memoryTransfer;

      memoryTransfer.copy(d_scratchPotHamQuad->nEntries(),
                          d_potentialHamQuadMemspace->data(),
                          d_scratchPotHamQuad->data()); 

      return *d_potentialHamQuadMemspace;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::
      applyNonLocal(
        linearAlgebra::MultiVector<ValueTypeWaveFnBasisData, memorySpace> &X,
        linearAlgebra::MultiVector<ValueTypeWaveFnBasisData, memorySpace> &Y,
        bool updateGhostX,
        bool updateGhostY) const
    {
      utils::throwException(
        false,
        "Non-Local component not present to call in ElectrostaticLocalFE.h");
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::hasLocalComponent() const
    {
      return true;
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              typename ValueTypeWaveFnBasisData,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ElectrostaticLocalFE<ValueTypeBasisData,
                         ValueTypeBasisCoeff,
                         ValueTypeWaveFnBasisData,
                         memorySpace,
                         dim>::hasNonLocalComponent() const
    {
      return false;
    }

  } // end of namespace ksdft
} // end of namespace dftefe
