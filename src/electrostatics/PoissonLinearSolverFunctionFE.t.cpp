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

#include <utils/Defaults.h>
namespace dftefe
{
  namespace electrostatics
  {
    namespace PoissonLinearSolverFunctionFEInternal
    {
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      getDiagonal(
        linearAlgebra::Vector<ValueTypeOperator, memorySpace> &diagonal,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>>      feBasisManager,
        std::shared_ptr<const LaplaceOperatorContextFE<ValueTypeOperator,
                                                       ValueTypeOperand,
                                                       memorySpace,
                                                       dim>>   AXContext,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type cellBlockSize)
      {
        const size_type numLocallyOwnedCells =
          feBasisManager->nLocallyOwnedCells();
        std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          numCellDofs[iCell] = feBasisManager->nLocallyOwnedCellDofs(iCell);

        auto itCellLocalIdsBegin =
          feBasisManager->locallyOwnedCellLocalDofIdsBegin();

        std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                           0);
        std::copy(numCellDofs.begin(),
                  numCellDofs.begin() + numLocallyOwnedCells,
                  locallyOwnedCellsNumDoFsSTL.begin());

        utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
          numLocallyOwnedCells);
        locallyOwnedCellsNumDoFs.copyFrom(locallyOwnedCellsNumDoFsSTL);

        basis::FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
          addCellWiseBasisDataToDiagonalData(
            AXContext->getBasisGradNiGradNjInAllCells()->data(),
            itCellLocalIdsBegin,
            locallyOwnedCellsNumDoFs,
            diagonal.data());

        // function to do a static condensation to send the constraint nodes to
        // its parent nodes
        feBasisManager->getConstraints().distributeChildToParent(diagonal, 1);

        // Function to add the values to the local node from its corresponding
        // ghost nodes from other processors.
        diagonal.accumulateAddLocallyOwned();

        diagonal.updateGhostValues();
      }

    } // end of namespace PoissonLinearSolverFunctionFEInternal


    //
    // Constructor
    //

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      PoissonLinearSolverFunctionFE(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          feBasisDataStorageStiffnessMatrix,
        const std::map<
          std::string,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>>
          &feBasisDataStorageRhs,
        const std::map<
          std::string,
          const quadrature::QuadratureValuesContainer<ValueType, memorySpace> &>
          &                                     inpRhs,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxFieldBlock)
      : d_feBasisManagerField(feBasisManagerField)
      , d_feBasisDataStorageStiffnessMatrix(feBasisDataStorageStiffnessMatrix)
      , d_numComponents(
          !inpRhs.empty() ? inpRhs.begin()->second.getNumberComponents() : 0)
      , d_pcType(pcType)
      , d_fieldInHomoDBCVec(feBasisManagerField->getMPIPatternP2P(),
                            linAlgOpContext,
                            d_numComponents,
                            ValueTypeOperand())
      , d_linAlgOpContext(linAlgOpContext)
      , d_maxCellBlock(maxCellBlock)
      , d_maxFieldBlock(maxFieldBlock)
      , d_p(feBasisManagerField->getMPIPatternP2P()->mpiCommunicator(),
            "Poisson Solver")
      , d_rootCout(std::cout)
    {
      int rank;
      utils::mpi::MPICommRank(
        feBasisManagerField->getMPIPatternP2P()->mpiCommunicator(), &rank);
      d_rootCout.setCondition(rank == 0);

      utils::throwException(
        !inpRhs.empty(),
        "The input QuadValuesContainer Map in PoissonSolver cannot be empty.");

      auto iter = feBasisDataStorageRhs.begin();
      while (iter != feBasisDataStorageRhs.end())
        {
          d_feBasisDataStorageRhs[iter->first] = iter->second;
          utils::throwException(
            ((d_feBasisDataStorageStiffnessMatrix->getBasisDofHandler())
               .get() == (iter->second->getBasisDofHandler()).get()),
            "The BasisDofHandler of the datastorages does not match in PoissonLinearSolverFunctionFE.");
          iter++;
        }

      utils::throwException(
        (feBasisDataStorageStiffnessMatrix->getBasisDofHandler().get() ==
         &feBasisManagerField->getBasisDofHandler()),
        "The BasisDofHandler of the dataStorages and basisManager should be same in PoissonLinearSolverFunctionFE.");

      std::shared_ptr<
        const basis::FEBasisDofHandler<ValueTypeOperator, memorySpace, dim>>
        basisDofHandler = std::dynamic_pointer_cast<
          const basis::FEBasisDofHandler<ValueTypeOperator, memorySpace, dim>>(
          feBasisDataStorageStiffnessMatrix->getBasisDofHandler());
      utils::throwException(
        basisDofHandler != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler "
        "in PoissonLinearSolverFunctionFE");

      // Set up BasisManager
      std::shared_ptr<const utils::ScalarSpatialFunctionReal> zeroFunction =
        std::make_shared<utils::ScalarZeroFunctionReal>();

      d_p.registerStart("Initilization");
      d_feBasisManagerHomo =
        std::make_shared<basis::FEBasisManager<ValueTypeOperand,
                                               ValueTypeOperator,
                                               memorySpace,
                                               dim>>(basisDofHandler,
                                                     zeroFunction);

      d_fieldInHomoDBCVec.updateGhostValues();
      feBasisManagerField->getConstraints().distributeParentToChild(
        d_fieldInHomoDBCVec, d_numComponents);

      auto mpiPatternP2PHomo = d_feBasisManagerHomo->getMPIPatternP2P();

      linearAlgebra::MultiVector<ValueType, memorySpace> b1(mpiPatternP2PHomo,
                                                            linAlgOpContext,
                                                            d_numComponents,
                                                            ValueType());
      d_b = b1;
      linearAlgebra::MultiVector<ValueType, memorySpace> x(mpiPatternP2PHomo,
                                                           linAlgOpContext,
                                                           d_numComponents,
                                                           ValueType());
      d_x = x;
      linearAlgebra::MultiVector<ValueType, memorySpace> initial(
        mpiPatternP2PHomo, linAlgOpContext, d_numComponents, ValueType());
      d_initial = initial;

      d_AxContext = std::make_shared<LaplaceOperatorContextFE<ValueTypeOperator,
                                                              ValueTypeOperand,
                                                              memorySpace,
                                                              dim>>(
        *d_feBasisManagerHomo,
        *d_feBasisManagerHomo,
        feBasisDataStorageStiffnessMatrix,
        linAlgOpContext,
        d_maxCellBlock,
        d_maxFieldBlock); // solving the AX = b

      d_AxContextNHDB =
        std::make_shared<LaplaceOperatorContextFE<ValueTypeOperator,
                                                  ValueTypeOperand,
                                                  memorySpace,
                                                  dim>>(
          *d_feBasisManagerField,
          *d_feBasisManagerHomo,
          d_AxContext->getBasisGradNiGradNjInAllCells(),
          d_maxCellBlock,
          d_maxFieldBlock); // handling the inhomogeneous DBC in RHS

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        mpiPatternP2PHomo, linAlgOpContext);

      if (d_pcType == linearAlgebra::PreconditionerType::JACOBI)
        {
          PoissonLinearSolverFunctionFEInternal::
            getDiagonal<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>(
              diagonal,
              d_feBasisManagerHomo,
              d_AxContext,
              linAlgOpContext,
              d_maxCellBlock);


          d_feBasisManagerHomo->getConstraints().setConstrainedNodes(diagonal,
                                                                     1,
                                                                     1.0);

          d_PCContext = std::make_shared<
            linearAlgebra::PreconditionerJacobi<ValueTypeOperator,
                                                ValueTypeOperand,
                                                memorySpace>>(diagonal);

          diagonal.updateGhostValues();
        }

      else if (d_pcType == linearAlgebra::PreconditionerType::NONE)
        d_PCContext =
          std::make_shared<linearAlgebra::PreconditionerNone<ValueTypeOperator,
                                                             ValueTypeOperand,
                                                             memorySpace>>();
      else
        utils::throwException(false, "Unknown PreConditionerType");
      d_p.registerEnd("Initilization");

      d_rhsQuadValComponent.clear();
      reinit(d_feBasisManagerField, inpRhs);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      PoissonLinearSolverFunctionFE(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          feBasisDataStorageStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          feBasisDataStorageRhs,
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpace> &                        inpRhs,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxFieldBlock)
      : PoissonLinearSolverFunctionFE(
          feBasisManagerField,
          feBasisDataStorageStiffnessMatrix,
          std::map<
            std::string,
            std::shared_ptr<
              const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>>(
            {{"Field", feBasisDataStorageRhs}}),
          std::map<std::string,
                   const quadrature::QuadratureValuesContainer<
                     linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                            ValueTypeOperand>,
                     memorySpace> &>({{"Field", inpRhs}}),
          pcType,
          linAlgOpContext,
          maxCellBlock,
          maxFieldBlock)
    {}

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      reinit(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        const std::map<std::string,
                       const quadrature::QuadratureValuesContainer<
                         linearAlgebra::blasLapack::
                           scalar_type<ValueTypeOperator, ValueTypeOperand>,
                         memorySpace> &> &                inpRhs)
    {
      auto iter = d_feBasisDataStorageRhs.begin();
      while (iter != d_feBasisDataStorageRhs.end())
        {
          auto iter1 = inpRhs.find(iter->first);
          if (iter1 != inpRhs.end())
            utils::throwException(
              (iter1->second.getQuadratureRuleContainer()
                   ->getQuadratureRuleAttributes()
                   .isCartesianTensorStructured() ?
                 iter1->second.getQuadratureRuleContainer()
                     ->getQuadratureRuleAttributes() ==
                   iter->second->getQuadratureRuleContainer()
                     ->getQuadratureRuleAttributes() :
                 iter1->second.getQuadratureRuleContainer() ==
                   iter->second->getQuadratureRuleContainer()) &&
                d_numComponents == iter1->second.getNumberComponents(),
              "Either the input field and feBasisDataStorageRhs quadrature rule"
              " are not same same,  for PoissonLinearSolverFunctionFE reinit or input"
              "field has different components than that when constructed.");
          else
            utils::throwException(
              false,
              "The inpRhs corresponding to a feBasisDataStorageRhs couldn't be found in PoissonLinearSolver.");
          iter++;
        }

      d_p.reset();
      if (d_feBasisManagerField != feBasisManagerField)
        {
          d_p.registerStart("Initilization");
          utils::throwException(
            (&(d_feBasisManagerField->getBasisDofHandler()) ==
             &(feBasisManagerField->getBasisDofHandler())),
            "The BasisDofHandler of the feBasisManagerField in reinit does not match with that in constructor in PoissonLinearSolverFunctionFE.");

          d_feBasisManagerField = feBasisManagerField;
          d_fieldInHomoDBCVec.updateGhostValues();
          d_feBasisManagerField->getConstraints().distributeParentToChild(
            d_fieldInHomoDBCVec, d_numComponents);

          d_AxContextNHDB->reinit(*d_feBasisManagerField,
                                  *d_feBasisManagerHomo);
          d_p.registerEnd("Initilization");
        }

      // Compute RHS

      std::vector<ValueType> ones(0);
      ones.resize(d_numComponents, (ValueType)1.0);
      std::vector<ValueType> nOnes(0);
      nOnes.resize(d_numComponents, (ValueType)-1.0);

      d_b.setValue(0.0);
      linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> b1(d_b, 0.0),
        b(d_b, 0.0);

      d_p.registerStart("Rhs Computation");
      iter = d_feBasisDataStorageRhs.begin();
      while (iter != d_feBasisDataStorageRhs.end())
        {
          ValueType max(1e6);
          if (d_rhsQuadValComponent.find(iter->first) !=
              d_rhsQuadValComponent.end())
            {
              quadrature::add<ValueType, memorySpace>(
                (ValueType)1,
                inpRhs.find(iter->first)->second,
                (ValueType)(-1),
                d_rhsQuadValComponent.find(iter->first)->second,
                *d_linAlgOpContext);

              max = linearAlgebra::blasLapack::amax(
                d_rhsQuadValComponent.find(iter->first)->second.nEntries(),
                d_rhsQuadValComponent.find(iter->first)->second.data(),
                1,
                *d_linAlgOpContext);

              utils::mpi::MPIAllreduce<memorySpace>(
                utils::mpi::MPIInPlace,
                &max,
                1,
                utils::mpi::Types<ValueType>::getMPIDatatype(),
                utils::mpi::MPIMax,
                feBasisManagerField->getMPIPatternP2P()->mpiCommunicator());

              d_rhsQuadValComponent.find(iter->first)->second =
                inpRhs.find(iter->first)->second;
            }

          if (std::abs(max) > 1e-12)
            {
              // Set up basis Operations for RHS
              basis::FEBasisOperations<ValueTypeOperand,
                                       ValueTypeOperator,
                                       memorySpace,
                                       dim>
                feBasisOperations(iter->second,
                                  d_maxCellBlock,
                                  d_maxFieldBlock);

              feBasisOperations.integrateWithBasisValues(
                inpRhs.find(iter->first)->second, *d_feBasisManagerHomo, b1);

              if (d_rhsQuadValComponent.find(iter->first) !=
                  d_rhsQuadValComponent.end())
                {
                  d_rhsMultiVecComponent.find(iter->first)->second = b1;
                }
              else
                {
                  d_rhsQuadValComponent[iter->first] =
                    inpRhs.find(iter->first)->second;
                  d_rhsMultiVecComponent[iter->first] = b1;
                }

              linearAlgebra::add(ones, b1, ones, b, b);
            }
          else
            {
              d_rootCout << "Skipped " << iter->first << " RHS evaluation\n";
              linearAlgebra::add(
                ones,
                d_rhsMultiVecComponent.find(iter->first)->second,
                ones,
                b,
                b);
            }
          iter++;
        }

      linearAlgebra::MultiVector<ValueType, memorySpace> rhsNHDB(d_b, 0.0);

      d_AxContextNHDB->apply(d_fieldInHomoDBCVec, rhsNHDB, true, true);

      linearAlgebra::add(ones, b, nOnes, rhsNHDB, d_b);
      d_p.registerEnd("Rhs Computation");
      d_p.print();

      // for (unsigned int i = 0 ; i < d_b.locallyOwnedSize() ; i++)
      //   {
      //     std::cout << i  << " " << *(rhsNHDB.data()+i) << " \t ";
      //   }

      for (int i = 0; i < d_numComponents; i++)
        std::cout << "rhs-norm: " << rhsNHDB.l2Norms()[i]
                  << " d_b-norm: " << d_b.l2Norms()[i]
                  << " b-norm: " << b.l2Norms()[i] << "\t";
      std::cout << "\n";
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      reinit(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpace> &inpRhs)
    {
      std::map<std::string,
               const quadrature::QuadratureValuesContainer<
                 linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                        ValueTypeOperand>,
                 memorySpace> &>
        inpRhsMap = {{"Field", inpRhs}};

      reinit(feBasisManagerField, inpRhsMap);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::
      OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
      PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::getAxContext() const
    {
      return *d_AxContext;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::
      OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
      PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::getPCContext() const
    {
      return *d_PCContext;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      setSolution(const linearAlgebra::MultiVector<
                  linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                         ValueTypeOperand>,
                  memorySpace> &x)
    {
      d_x = x;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      getSolution(linearAlgebra::MultiVector<
                  linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                         ValueTypeOperand>,
                  memorySpace> &solution)
    {
      size_type numComponents = solution.getNumberComponents();
      solution.setValue(0.0);

      for (size_type i = 0; i < solution.locallyOwnedSize(); i++)
        {
          for (size_type j = 0; j < numComponents; j++)
            {
              solution.data()[i * numComponents + j] =
                d_x.data()[i * numComponents + j] +
                d_fieldInHomoDBCVec.data()[i * numComponents + j];
            }
        }

      solution.updateGhostValues();

      d_feBasisManagerField->getConstraints().distributeParentToChild(
        solution, numComponents);

      // this is done for a particular case for poisson solve each
      // scf guess but have to be modified with a reinit parameter
      d_initial = solution;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getRhs() const
    {
      return d_b;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::MultiVector<
      linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                             ValueTypeOperand>,
      memorySpace> &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getInitialGuess() const
    {
      return d_initial;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const utils::mpi::MPIComm &
    PoissonLinearSolverFunctionFE<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::getMPIComm() const
    {
      return d_feBasisManagerHomo->getMPIPatternP2P()->mpiCommunicator();
    }

  } // end of namespace electrostatics
} // end of namespace dftefe
