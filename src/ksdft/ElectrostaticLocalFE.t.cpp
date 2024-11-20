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
          memorySpace> &field,
        const quadrature::QuadratureValuesContainer<
          typename ElectrostaticFE<ValueTypeBasisData,
                                   ValueTypeBasisCoeff,
                                   ValueTypeWaveFnBasisData,
                                   memorySpace,
                                   dim>::RealType,
          memorySpace> &                                             rho,
        const utils::MemoryStorage<ValueTypeBasisData, memorySpace> &jxwStorage,
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
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock)
      : d_atomCharges(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(d_numAtoms, nullptr)
      , d_feBMNuclearCharge(d_numAtoms, nullptr)
      , d_externalPotentialFunction(externalPotentialFunction)
      , d_maxCellBlock(maxCellBlock)
      , d_numericalCancelScratch1(nullptr)
      , d_numericalCancelScratch2(nullptr)
    {
      reinitBasis(atomCoordinates,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDTotalChargeRhs,
                  feBDHamiltonian);

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
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian,
        const utils::ScalarSpatialFunctionReal &externalPotentialFunction,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock)
      : d_atomCharges(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_smearedChargeRadius(smearedChargeRadius)
      , d_linAlgOpContext(linAlgOpContext)
      , d_numComponents(1)
      , d_energy((RealType)0)
      , d_nuclearChargesPotential(d_numAtoms, nullptr)
      , d_feBMNuclearCharge(d_numAtoms, nullptr)
      , d_externalPotentialFunction(externalPotentialFunction)
      , d_maxCellBlock(maxCellBlock)
      , d_numericalCancelScratch1(nullptr)
      , d_numericalCancelScratch2(nullptr)
    {
      reinitBasis(atomCoordinates,
                  feBMTotalCharge,
                  feBDTotalChargeStiffnessMatrix,
                  feBDTotalChargeRhs,
                  feBDNuclearChargeStiffnessMatrix,
                  feBDNuclearChargeRhs,
                  feBDHamiltonian);

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
                         dim>::~ElectrostaticLocalFE()
    {
      if (d_totalChargeDensity != nullptr)
        {
          delete d_totalChargeDensity;
          d_totalChargeDensity = nullptr;
        }
      if (d_nuclearChargesDensity != nullptr)
        {
          delete d_nuclearChargesDensity;
          d_nuclearChargesDensity = nullptr;
        }
      if (d_totalAuxChargePotentialQuad != nullptr)
        {
          delete d_totalAuxChargePotentialQuad;
          d_totalAuxChargePotentialQuad = nullptr;
        }
      if (d_nuclearCorrectionPotQuad != nullptr)
        {
          delete d_nuclearCorrectionPotQuad;
          d_nuclearCorrectionPotQuad = nullptr;
        }
      if (d_totalAuxChargePotAtbSmearQuad != nullptr)
        {
          delete d_totalAuxChargePotAtbSmearQuad;
          d_totalAuxChargePotAtbSmearQuad = nullptr;
        }
      if (d_nuclearCorrectionPotAtRhoQuad != nullptr)
        {
          delete d_nuclearCorrectionPotAtRhoQuad;
          d_nuclearCorrectionPotAtRhoQuad = nullptr;
        }
      if (d_totalChargePotential != nullptr)
        {
          delete d_totalChargePotential;
          d_totalChargePotential = nullptr;
        }
      if (d_totalChargePotentialQuad != nullptr)
        {
          delete d_totalChargePotentialQuad;
          d_totalChargePotentialQuad = nullptr;
        }
      if (d_numericalCancelScratch1 != nullptr)
        {
          delete d_numericalCancelScratch1;
          d_numericalCancelScratch1 = nullptr;
        }
      if (d_numericalCancelScratch2 != nullptr)
        {
          delete d_numericalCancelScratch2;
          d_numericalCancelScratch2 = nullptr;
        }
      for (auto &i : d_nuclearChargesPotential)
        {
          if (i != nullptr)
            {
              delete i;
              i = nullptr;
            }
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
          feBDTotalChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian
        /*std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData,
                                          memorySpace>> feBDHamiltonianElec*/)
    {
      d_isNumerical                    = true;
      d_atomCoordinates                = atomCoordinates;
      d_feBDTotalChargeRhs             = feBDTotalChargeRhs;
      d_feBDNuclearChargeRhs           = feBDNuclearChargeRhs;
      d_feBMTotalCharge                = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;
      d_feBasisOp =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(d_feBDTotalChargeRhs,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      d_feBasisOpHamiltonian =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>>(feBDHamiltonian,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBDTotalChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerHam = feBDHamiltonian->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerNucl =
          d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

      // create nuclear and electron charge densities
      d_totalChargeDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainer, d_numComponents);

      d_totalAuxChargePotentialQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_totalChargePotentialQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_nuclearCorrectionPotQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_totalAuxChargePotAtbSmearQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainer, d_numComponents);

      d_nuclearCorrectionPotAtRhoQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainer, d_numComponents);

      d_numericalCancelScratch1 =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainer, d_numComponents);

      d_numericalCancelScratch2 =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainer, d_numComponents);

      // Init the phi_el multivector
      d_totalChargePotential =
        new linearAlgebra::MultiVector<ValueType, memorySpace>(
          d_feBMTotalCharge->getMPIPatternP2P(),
          d_linAlgOpContext,
          d_numComponents);

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainer, d_numComponents);

      const utils::SmearChargeDensityFunction smfunc(d_atomCoordinates,
                                                     d_atomCharges,
                                                     d_smearedChargeRadius);

      RealType d_totNuclearChargeQuad = 0;
      for (size_type iCell = 0; iCell < quadRuleContainer->nCells(); iCell++)
        {
          size_type           quadId = 0;
          std::vector<double> jxw    = quadRuleContainer->getCellJxW(iCell);
          for (auto j : quadRuleContainer->getCellRealPoints(iCell))
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
                        *d_linAlgOpContext);

      int rank;
      utils::mpi::MPICommRank(
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(), &rank);

      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);
      rootCout << "Integral of nuclear charges over domain: "
               << d_totNuclearChargeQuad << "\n";

      // create the correction quadValuesContainer for numerical solve

      nuclearPotentialSolve(feBDNuclearChargeStiffnessMatrix,
                            feBDNuclearChargeRhs);

      for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
        {
          /* Change this to feBasisOperations for electrostaic basis with same
           * quadrulecontainer as hamiltonian*/
          d_feBasisOp->interpolate(*d_nuclearChargesPotential[iAtom],
                                   *d_feBMNuclearCharge[iAtom],
                                   *d_totalAuxChargePotentialQuad);

          quadrature::add((ValueType)1.0,
                          *d_totalAuxChargePotentialQuad,
                          (ValueType)1.0,
                          *d_totalChargePotentialQuad,
                          *d_totalChargePotentialQuad,
                          *d_linAlgOpContext);
        }

      for (size_type iCell = 0; iCell < quadRuleContainerHam->nCells(); iCell++)
        {
          size_type quadId = 0;
          for (auto j : quadRuleContainerHam->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (ValueType)(d_externalPotentialFunction)(j);
                }
              RealType *b = a.data();
              d_totalAuxChargePotentialQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      quadrature::add((ValueType)-1.0,
                      *d_totalChargePotentialQuad,
                      (ValueType)1.0,
                      *d_totalAuxChargePotentialQuad,
                      *d_nuclearCorrectionPotQuad,
                      *d_linAlgOpContext);

      d_totalAuxChargePotentialQuad->setValue(0);
      d_totalChargePotentialQuad->setValue(0);

      // Do this for energy evaluation with b+rho quad

      for (size_type iCell = 0; iCell < quadRuleContainer->nCells(); iCell++)
        {
          size_type quadId = 0;
          for (auto j : quadRuleContainer->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (ValueType)(d_externalPotentialFunction)(j);
                }
              RealType *b = a.data();
              d_numericalCancelScratch1
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
        {
          d_feBasisOp->interpolate(*d_nuclearChargesPotential[iAtom],
                                   *d_feBMNuclearCharge[iAtom],
                                   *d_nuclearCorrectionPotAtRhoQuad);

          quadrature::add((ValueType)1.0,
                          *d_nuclearCorrectionPotAtRhoQuad,
                          (ValueType)1.0,
                          *d_totalAuxChargePotAtbSmearQuad,
                          *d_totalAuxChargePotAtbSmearQuad,
                          *d_linAlgOpContext);
        }

      quadrature::add((ValueType)-1.0,
                      *d_totalAuxChargePotAtbSmearQuad,
                      (ValueType)1.0,
                      *d_numericalCancelScratch1,
                      *d_nuclearCorrectionPotAtRhoQuad,
                      *d_linAlgOpContext);

      d_totalAuxChargePotAtbSmearQuad->setValue(0);

      d_linearSolverFunction = std::make_shared<
        electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpace,
                                                      dim>>(
        d_feBMTotalCharge,
        d_feBDTotalChargeStiffnessMatrix,
        d_feBDTotalChargeRhs,
        *d_totalChargeDensity,
        ksdft::PoissonProblemDefaults::PC_TYPE,
        d_linAlgOpContext,
        ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        d_numComponents);
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
          feBDTotalChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasisData,
                                          memorySpace>> feBDHamiltonian)
    {
      d_isNumerical                    = false;
      d_atomCoordinates                = atomCoordinates;
      d_feBDTotalChargeRhs             = feBDTotalChargeRhs;
      d_feBDNuclearChargeRhs           = nullptr;
      d_feBMTotalCharge                = feBMTotalCharge;
      d_feBDTotalChargeStiffnessMatrix = feBDTotalChargeStiffnessMatrix;
      d_feBasisOp =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeBasisData,
                                                  memorySpace,
                                                  dim>>(d_feBDTotalChargeRhs,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      d_feBasisOpHamiltonian =
        std::make_shared<basis::FEBasisOperations<ValueTypeBasisCoeff,
                                                  ValueTypeWaveFnBasisData,
                                                  memorySpace,
                                                  dim>>(feBDHamiltonian,
                                                        d_maxCellBlock,
                                                        d_numComponents);

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = d_feBDTotalChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainerHam = feBDHamiltonian->getQuadratureRuleContainer();

      // create nuclear and electron charge densities and total charge potential
      // with correction
      d_totalChargeDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainer, d_numComponents);

      d_totalAuxChargePotentialQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_totalChargePotentialQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      d_nuclearCorrectionPotQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainerHam, d_numComponents);

      // get the input quadraturevaluescontainer for poisson solve
      d_nuclearChargesDensity =
        new quadrature::QuadratureValuesContainer<RealType, memorySpace>(
          quadRuleContainer, d_numComponents);

      d_totalAuxChargePotAtbSmearQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainer, d_numComponents);

      d_nuclearCorrectionPotAtRhoQuad =
        new quadrature::QuadratureValuesContainer<ValueType, memorySpace>(
          quadRuleContainer, d_numComponents);

      // Init the phi_el multivector
      d_totalChargePotential =
        new linearAlgebra::MultiVector<ValueType, memorySpace>(
          d_feBMTotalCharge->getMPIPatternP2P(),
          d_linAlgOpContext,
          d_numComponents);

      const utils::SmearChargeDensityFunction smfunc(d_atomCoordinates,
                                                     d_atomCharges,
                                                     d_smearedChargeRadius);

      RealType d_totNuclearChargeQuad = 0;
      for (size_type iCell = 0; iCell < quadRuleContainer->nCells(); iCell++)
        {
          size_type           quadId = 0;
          std::vector<double> jxw    = quadRuleContainer->getCellJxW(iCell);
          for (auto j : quadRuleContainer->getCellRealPoints(iCell))
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
                        *d_linAlgOpContext);

      int rank;
      utils::mpi::MPICommRank(
        d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator(), &rank);

      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);
      rootCout << "Integral of nuclear charges over domain: "
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
                  a[iComp] = (ValueType)(d_externalPotentialFunction)(j) -
                             (ValueType)(smfuncPot)(j);
                }
              RealType *b = a.data();
              d_nuclearCorrectionPotQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
        }

      for (size_type iCell = 0; iCell < quadRuleContainer->nCells(); iCell++)
        {
          size_type quadId = 0;
          for (auto j : quadRuleContainer->getCellRealPoints(iCell))
            {
              std::vector<RealType> a(d_numComponents);
              for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                {
                  a[iComp] = (ValueType)(d_externalPotentialFunction)(j) -
                             (ValueType)(smfuncPot)(j);
                }
              RealType *b = a.data();
              d_nuclearCorrectionPotAtRhoQuad
                ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
                                                                       quadId,
                                                                       b);
              quadId = quadId + 1;
            }
          // for (size_type iComp = 0; iComp < d_numComponents; iComp++)
          //   {
          //     std::vector<ValueType> a(
          //       quadRuleContainer->nCellQuadraturePoints(iCell), 0);
          //     size_type quadId = 0;

          //     for (auto j : quadRuleContainer->getCellRealPoints(iCell))
          //       {
          //         a[quadId] = (ValueType)(d_externalPotentialFunction)(j) -
          //                     (ValueType)(smfuncPot)(j);
          //         quadId = quadId + 1;
          //       }
          //     ValueType *b = a.data();
          //     d_nuclearCorrectionPotAtRhoQuad
          //       ->template setCellQuadValues<utils::MemorySpace::HOST>(iCell,
          //                                                              iComp,
          //                                                              b);
          //   }
        }

      d_linearSolverFunction = std::make_shared<
        electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpace,
                                                      dim>>(
        d_feBMTotalCharge,
        d_feBDTotalChargeStiffnessMatrix,
        d_feBDTotalChargeRhs,
        *d_totalChargeDensity,
        ksdft::PoissonProblemDefaults::PC_TYPE,
        d_linAlgOpContext,
        ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
        d_numComponents);
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
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &electronChargeDensity)
    {
      d_electronChargeDensity = &electronChargeDensity;
      utils::throwException(
        electronChargeDensity.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .isCartesianTensorStructured() ?
          electronChargeDensity.getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() ==
            d_feBDTotalChargeRhs->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes() :
          electronChargeDensity.getQuadratureRuleContainer() ==
            d_feBDTotalChargeRhs->getQuadratureRuleContainer(),
        "The electronic and total charge rhs for poisson solve quadrature rule"
        " should be same, otherwise they cannot be added in ElectrostaticFE reinitField()");

      /*---- solve poisson problem for b+rho system ---*/

      // add nuclear and electron charge densities

      quadrature::add((RealType)1.0,
                      *d_nuclearChargesDensity,
                      (RealType)1.0,
                      electronChargeDensity,
                      *d_totalChargeDensity,
                      *d_linAlgOpContext);

      // Scale by 4\pi
      quadrature::scale((RealType)(4 * utils::mathConstants::pi),
                        *d_totalChargeDensity,
                        *d_linAlgOpContext);

      d_linearSolverFunction->reinit(d_feBMTotalCharge,
                                     d_feBDTotalChargeRhs,
                                     *d_totalChargeDensity);

      linearAlgebra::LinearAlgebraProfiler profiler;

      std::shared_ptr<linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpace>>
        CGSolve =
          std::make_shared<linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                         ValueTypeBasisCoeff,
                                                         memorySpace>>(
            ksdft::PoissonProblemDefaults::MAX_ITER,
            ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
            ksdft::PoissonProblemDefaults::RELATIVE_TOL,
            ksdft::PoissonProblemDefaults::DIVERGENCE_TOL,
            profiler);

      CGSolve->solve(*d_linearSolverFunction);

      d_linearSolverFunction->getSolution(*d_totalChargePotential);

      /* Change this to feBasisOperations for electrostaic basis with same
       * quadrulecontainer as hamiltonian*/
      d_feBasisOp->interpolate(*d_totalChargePotential,
                               *d_feBMTotalCharge,
                               *d_totalAuxChargePotentialQuad);
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
      // d_totalAuxChargePotentialQuad

      quadrature::add((ValueType)1.0,
                      *d_totalAuxChargePotentialQuad,
                      (ValueType)1.0,
                      *d_nuclearCorrectionPotQuad,
                      *d_totalChargePotentialQuad,
                      *d_linAlgOpContext);

      d_feBasisOpHamiltonian->computeFEMatrices(
        basis::realspace::LinearLocalOp::IDENTITY,
        basis::realspace::VectorMathOp::MULT,
        basis::realspace::VectorMathOp::MULT,
        basis::realspace::LinearLocalOp::IDENTITY,
        *d_totalChargePotentialQuad,
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
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs)
    {
      // Solve poisson problem for individual atoms
      std::shared_ptr<
        const basis::FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>
        feBDHNuclearCharge = std::dynamic_pointer_cast<
          const basis::
            FEBasisDofHandler<ValueTypeBasisCoeff, memorySpace, dim>>(
          feBDNuclearChargeRhs->getBasisDofHandler());
      utils::throwException(
        feBDHNuclearCharge != nullptr,
        "Could not cast BasisDofHandler of the input Field to FEBasisDofHandler "
        "in ElectrostaticLocalFE");

      std::shared_ptr<const quadrature::QuadratureRuleContainer>
        quadRuleContainer = feBDNuclearChargeRhs->getQuadratureRuleContainer();

      std::shared_ptr<
        electrostatics::PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                      ValueTypeBasisCoeff,
                                                      memorySpace,
                                                      dim>>
        linearSolverFunctionNuclear = nullptr;

      d_nuclearChargeQuad.resize(d_numAtoms, 0);
      for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
        {
          std::shared_ptr<const utils::ScalarSpatialFunctionReal> smfunc =
            std::make_shared<const utils::SmearChargePotentialFunction>(
              d_atomCoordinates[iAtom],
              d_atomCharges[iAtom],
              d_smearedChargeRadius[iAtom]);

          d_feBMNuclearCharge[iAtom] =
            std::make_shared<basis::FEBasisManager<ValueTypeBasisCoeff,
                                                   ValueTypeBasisData,
                                                   memorySpace,
                                                   dim>>(feBDHNuclearCharge,
                                                         smfunc);

          smfunc = std::make_shared<const utils::SmearChargeDensityFunction>(
            d_atomCoordinates[iAtom],
            d_atomCharges[iAtom],
            d_smearedChargeRadius[iAtom]);

          d_nuclearChargeQuad[iAtom] = 0;
          for (size_type iCell = 0; iCell < quadRuleContainer->nCells();
               iCell++)
            {
              size_type           quadId = 0;
              std::vector<double> jxw    = quadRuleContainer->getCellJxW(iCell);
              for (auto j : quadRuleContainer->getCellRealPoints(iCell))
                {
                  std::vector<RealType> a(d_numComponents);
                  for (size_type iComp = 0; iComp < d_numComponents; iComp++)
                    {
                      a[iComp] = (RealType)(*smfunc)(j);
                      d_nuclearChargeQuad[iAtom] +=
                        (RealType)(*smfunc)(j)*jxw[quadId];
                    }
                  RealType *b = a.data();
                  d_totalChargeDensity
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
                            *d_totalChargeDensity,
                            *d_linAlgOpContext);

          if (iAtom == 0)
            {
              linearSolverFunctionNuclear =
                std::make_shared<electrostatics::PoissonLinearSolverFunctionFE<
                  ValueTypeBasisData,
                  ValueTypeBasisCoeff,
                  memorySpace,
                  dim>>(d_feBMNuclearCharge[iAtom],
                        feBDNuclearChargeStiffnessMatrix,
                        feBDNuclearChargeRhs,
                        *d_totalChargeDensity,
                        ksdft::PoissonProblemDefaults::PC_TYPE,
                        d_linAlgOpContext,
                        ksdft::KSDFTDefaults::CELL_BATCH_SIZE_GRAD_EVAL,
                        d_numComponents);
            }
          else
            {
              linearSolverFunctionNuclear->reinit(d_feBMNuclearCharge[iAtom],
                                                  feBDNuclearChargeRhs,
                                                  *d_totalChargeDensity);
            }

          linearAlgebra::LinearAlgebraProfiler profiler;

          std::shared_ptr<linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                          ValueTypeBasisCoeff,
                                                          memorySpace>>
            CGSolve = std::make_shared<
              linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                            ValueTypeBasisCoeff,
                                            memorySpace>>(
              ksdft::PoissonProblemDefaults::MAX_ITER,
              ksdft::PoissonProblemDefaults::ABSOLUTE_TOL,
              ksdft::PoissonProblemDefaults::RELATIVE_TOL,
              ksdft::PoissonProblemDefaults::DIVERGENCE_TOL,
              profiler);

          CGSolve->solve(*linearSolverFunctionNuclear);

          d_nuclearChargesPotential[iAtom] =
            new linearAlgebra::MultiVector<ValueType, memorySpace>(
              d_feBMNuclearCharge[iAtom]->getMPIPatternP2P(),
              d_linAlgOpContext,
              d_numComponents);
          linearSolverFunctionNuclear->getSolution(
            *d_nuclearChargesPotential[iAtom]);
        }
      d_totalChargeDensity->setValue(0);
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
      auto jxwStorage = d_feBDTotalChargeRhs->getJxWInAllCells();

      d_feBasisOp->interpolate(*d_totalChargePotential,
                               *d_feBMTotalCharge,
                               *d_totalAuxChargePotAtbSmearQuad);

      quadrature::add((RealType)1.0,
                      *d_nuclearChargesDensity,
                      (RealType)1.0,
                      *d_electronChargeDensity,
                      *d_totalChargeDensity,
                      *d_linAlgOpContext);

      RealType totalEnergy =
        ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          ValueTypeWaveFnBasisData,
          memorySpace,
          dim>(*d_totalAuxChargePotAtbSmearQuad,
               *d_totalChargeDensity,
               jxwStorage,
               d_linAlgOpContext,
               d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      totalEnergy = totalEnergy * 0.5;

      // self energy computation // TODO : Do this only once in reinitBasis
      RealType selfEnergy = 0;

      if (d_isNumerical)
        {
          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainer =
              d_feBDNuclearChargeRhs->getQuadratureRuleContainer();

          auto jxwStorageNucl = d_feBDNuclearChargeRhs->getJxWInAllCells();

          for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
            {
              const utils::SmearChargeDensityFunction smfunc(
                d_atomCoordinates[iAtom],
                d_atomCharges[iAtom],
                d_smearedChargeRadius[iAtom]);

              for (size_type iCell = 0; iCell < quadRuleContainer->nCells();
                   iCell++)
                {
                  size_type           quadId = 0;
                  std::vector<double> jxw =
                    quadRuleContainer->getCellJxW(iCell);
                  for (auto j : quadRuleContainer->getCellRealPoints(iCell))
                    {
                      std::vector<RealType> a(d_numComponents);
                      for (size_type iComp = 0; iComp < d_numComponents;
                           iComp++)
                        {
                          a[iComp] = (RealType)(smfunc)(j);
                        }
                      RealType *b = a.data();
                      d_numericalCancelScratch2
                        ->template setCellQuadValues<utils::MemorySpace::HOST>(
                          iCell, quadId, b);
                      quadId = quadId + 1;
                    }
                }

              // Scale by d_atomCharges[iAtom]/d_nuclearChargeQuad[iAtom]
              quadrature::scale((RealType)std::abs(d_atomCharges[iAtom] /
                                                   d_nuclearChargeQuad[iAtom]),
                                *d_numericalCancelScratch2,
                                *d_linAlgOpContext);

              basis::FEBasisOperations<ValueTypeBasisCoeff,
                                       ValueTypeBasisData,
                                       memorySpace,
                                       dim>
                feBasisOp(d_feBDNuclearChargeRhs,
                          d_maxCellBlock,
                          d_numComponents);

              feBasisOp.interpolate(*d_nuclearChargesPotential[iAtom],
                                    *d_feBMNuclearCharge[iAtom],
                                    *d_numericalCancelScratch1);

              RealType selfEnergyAtom =
                ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
                  ValueTypeBasisData,
                  ValueTypeBasisCoeff,
                  ValueTypeWaveFnBasisData,
                  memorySpace,
                  dim>(*d_numericalCancelScratch1,
                       *d_numericalCancelScratch2,
                       jxwStorageNucl,
                       d_linAlgOpContext,
                       d_feBMNuclearCharge[iAtom]
                         ->getMPIPatternP2P()
                         ->mpiCommunicator());

              selfEnergy += selfEnergyAtom;
            }
          selfEnergy *= 0.5;
        }
      else
        {
          /**
          for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
            {
              const utils::SmearChargePotentialFunction smfunc(
                d_atomCoordinates[iAtom],
                d_atomCharges[iAtom],
                d_smearedChargeRadius[iAtom]);

              double Ig = 10976. / (17875 * d_smearedChargeRadius[iAtom]);
              selfEnergy += (RealType)(0.5 * std::pow(d_atomCharges[iAtom], 2) *
                                       (Ig - (smfunc(d_atomCoordinates[iAtom]) /
                                              d_atomCharges[iAtom])));
            }
          selfEnergy *= -1;
          **/

          std::shared_ptr<const quadrature::QuadratureRuleContainer>
            quadRuleContainer =
              d_feBDTotalChargeRhs->getQuadratureRuleContainer();

          auto jxwStorageNucl = d_feBDTotalChargeRhs->getJxWInAllCells();

          for (unsigned int iAtom = 0; iAtom < d_numAtoms; iAtom++)
            {
              RealType                                atomNuclearChargeQuad = 0;
              const utils::SmearChargeDensityFunction smfunc(
                d_atomCoordinates[iAtom],
                d_atomCharges[iAtom],
                d_smearedChargeRadius[iAtom]);

              const utils::SmearChargePotentialFunction smfunc1(
                d_atomCoordinates[iAtom],
                d_atomCharges[iAtom],
                d_smearedChargeRadius[iAtom]);

              for (size_type iCell = 0; iCell < quadRuleContainer->nCells();
                   iCell++)
                {
                  size_type           quadId = 0;
                  std::vector<double> jxw =
                    quadRuleContainer->getCellJxW(iCell);
                  for (auto j : quadRuleContainer->getCellRealPoints(iCell))
                    {
                      std::vector<RealType> a(d_numComponents);
                      for (size_type iComp = 0; iComp < d_numComponents;
                           iComp++)
                        {
                          a[iComp] = (RealType)smfunc(j);
                          atomNuclearChargeQuad +=
                            (RealType)smfunc(j) * jxw[quadId];
                        }
                      RealType *b = a.data();
                      d_totalChargeDensity
                        ->template setCellQuadValues<utils::MemorySpace::HOST>(
                          iCell, quadId, b);

                      for (size_type iComp = 0; iComp < d_numComponents;
                           iComp++)
                        {
                          a[iComp] = smfunc1(j);
                        }
                      b = a.data();
                      d_totalAuxChargePotAtbSmearQuad
                        ->template setCellQuadValues<utils::MemorySpace::HOST>(
                          iCell, quadId, b);
                      quadId = quadId + 1;
                    }
                }

              utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                utils::mpi::MPIInPlace,
                &atomNuclearChargeQuad,
                1,
                utils::mpi::Types<RealType>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

              // Scale by d_atomCharges[iAtom]/totalNuclearChargeAtom
              quadrature::scale((RealType)std::abs(d_atomCharges[iAtom] /
                                                   atomNuclearChargeQuad),
                                *d_totalChargeDensity,
                                *d_linAlgOpContext);

              RealType selfEnergyAtom =
                ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
                  ValueTypeBasisData,
                  ValueTypeBasisCoeff,
                  ValueTypeWaveFnBasisData,
                  memorySpace,
                  dim>(
                  *d_totalAuxChargePotAtbSmearQuad,
                  *d_totalChargeDensity,
                  jxwStorageNucl,
                  d_linAlgOpContext,
                  d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

              selfEnergy += selfEnergyAtom;
            }
          selfEnergy *= 0.5;
        }

      // correction energy evaluation

      RealType correctionEnergy =
        ElectrostaticLocalFEInternal::getIntegralFieldTimesRho<
          ValueTypeBasisData,
          ValueTypeBasisCoeff,
          ValueTypeWaveFnBasisData,
          memorySpace,
          dim>(*d_nuclearCorrectionPotAtRhoQuad,
               *d_electronChargeDensity,
               jxwStorage,
               d_linAlgOpContext,
               d_feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator());

      d_energy = totalEnergy - selfEnergy + correctionEnergy;
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

  } // end of namespace ksdft
} // end of namespace dftefe
