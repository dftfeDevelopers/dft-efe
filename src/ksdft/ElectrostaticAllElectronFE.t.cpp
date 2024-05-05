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

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::
      ElectrostaticAllElectronFE(
        const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>
          &             feBasisDataStorage,
        const size_type maxCellTimesNumVecs)
      : d_feBasisDataStorage(&feBasisDataStorage)
    {
      reinit();
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::
      reinit(std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                         ValueTypeBasisData,
                                                         memorySpace,
                                                         dim>> feBasisManager,
             std::shared_ptr<
               const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
               feBasisDataStorageStiffnessMatrix,
             std::shared_ptr<
               const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
               feBasisDataStorageRhs,
             const quadrature::QuadratureValuesContainer<ValueType, memorySpace>
               &                                     chargeDensity,
             const linearAlgebra::PreconditionerType pcType,
             std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
               linAlgOpContext)
    {
      std::shared_ptr<
        dftefe::linearAlgebra::MultiVector<ValueTypeBasisData, memorySpace>>
        solution;

      std::shared_ptr<
        dftefe::linearAlgebra::LinearSolverFunction<ValueTypeBasisData,
                                                    ValueTypeBasisCoeff,
                                                    memorySpace>>
        linearSolverFunction =
          std::make_shared<dftefe::electrostatics::
                             PoissonLinearSolverFunctionFE<ValueTypeBasisData,
                                                           ValueTypeBasisCoeff,
                                                           memorySpace,
                                                           dim>>(
            feBasisManager,
            feBasisDataStorageStiffnessMatrix,
            feBasisDataStorageRhs,
            chargeDensity,
            pcType,
            linAlgOpContext,
            50);

      dftefe::linearAlgebra::LinearAlgebraProfiler profiler;

      std::shared_ptr<
        dftefe::linearAlgebra::LinearSolverImpl<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpace>>
        CGSolve = std::make_shared<
          dftefe::linearAlgebra::CGLinearSolver<ValueTypeBasisData,
                                                ValueTypeBasisCoeff,
                                                memorySpace>>(
          maxIter, absoluteTol, relativeTol, divergenceTol, profiler);

      CGSolve->solve(*linearSolverFunction);

      linearSolverFunction->getSolution(*solution);
    }

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    Storage
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::getHamiltonian() const
    {}

    template <typename ValueTypeBasisData,
              typename ValueTypeBasisCoeff,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticAllElectronFE<ValueTypeBasisData,
                               ValueTypeBasisCoeff,
                               memorySpace,
                               dim>::evalEnergy() const
    {
      feBasisOp.interpolate(*solution,
                            *basisManager,
                            quadValuesContainerNumerical);

      auto                      iter1 = quadValuesContainer.begin();
      auto                      iter2 = quadValuesContainerNumerical.begin();
      const std::vector<double> JxW   = quadRuleContainer->getJxW();
      double                    e     = 0;
      for (unsigned int i = 0; i < numQuadraturePoints; i++)
        {
          for (unsigned int j = 0; j < numComponents; j++)
            {
              e += *(i * numComponents + j + iter1) *
                   *(i * numComponents + j + iter2) * JxW[i] * 0.5;
            }
        }
      energy[iProb] = e;
    }

    dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
      energy.data(),
      mpiReducedEnergy.data(),
      energy.size(),
      dftefe::utils::mpi::MPIDouble,
      dftefe::utils::mpi::MPISum,
      comm);

    double Ig                   = 10976. / (17875 * rc);
    double analyticalSelfEnergy = 0, numericalSelfEnergy = 0;
    for (unsigned int i = 0; i < nAtoms; i++)
      {
        std::vector<dftefe::utils::Point> coord{atomCoordinatesVec[i]};
        analyticalSelfEnergy +=
          0.5 * (Ig - vSmear(atomCoordinatesVec[i], coord, rc));
        numericalSelfEnergy += mpiReducedEnergy[i];
      }

  }

  template <typename ValueTypeBasisData,
            typename ValueTypeBasisCoeff,
            utils::MemorySpace memorySpace,
            size_type          dim>
  RealType
  ElectrostaticAllElectronFE<ValueTypeBasisData,
                             ValueTypeBasisCoeff,
                             memorySpace,
                             dim>::getEnergy() const
  {
    return d_energy;
  }

} // end of namespace ksdft
} // end of namespace dftefe
