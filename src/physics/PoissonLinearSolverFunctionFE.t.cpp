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
  namespace physics
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
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          feBasisDataStorage)
      {
        const size_type numLocallyOwnedCells =
          feBasisManager->nLocallyOwnedCells();
        std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          numCellDofs[iCell] = feBasisManager->nLocallyOwnedCellDofs(iCell);

        auto itCellLocalIdsBegin =
          feBasisManager->locallyOwnedCellLocalDofIdsBegin();

        // access cell-wise discrete Laplace operator
        auto gradNiGradNjInAllCells =
          feBasisDataStorage->getBasisGradNiGradNjInAllCells();

        std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                           0);
        std::copy(numCellDofs.begin(),
                  numCellDofs.begin() + numLocallyOwnedCells,
                  locallyOwnedCellsNumDoFsSTL.begin());

        utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
          numLocallyOwnedCells);
        locallyOwnedCellsNumDoFs.copyFrom(locallyOwnedCellsNumDoFsSTL);

        basis::FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
          addCellWiseBasisDataToDiagonalData(gradNiGradNjInAllCells.data(),
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
        const size_type maxCellTimesNumVecs)
      : d_feBasisManagerField(feBasisManagerField)
      , d_numComponents(inpRhs.getNumberComponents())
      , d_pcType(pcType)
    {
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

      utils::throwException(
        ((feBasisDataStorageRhs->getBasisDofHandler()).get() ==
         (feBasisDataStorageStiffnessMatrix->getBasisDofHandler()).get()) &&
          ((feBasisDataStorageRhs->getBasisDofHandler()).get() ==
           &(feBasisManagerField->getBasisDofHandler())),
        "The BasisDofHandler of the dataStorages and basisManager should be same in PoissonLinearSolverFunctionFE.");

      std::shared_ptr<
        const basis::FEBasisDofHandler<ValueTypeOperator, memorySpace, dim>>
        basisDofHandler = std::dynamic_pointer_cast<
          const basis::FEBasisDofHandler<ValueTypeOperator, memorySpace, dim>>(
          feBasisDataStorageRhs->getBasisDofHandler());
      utils::throwException(
        basisDofHandler != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler "
        "in PoissonLinearSolverFunctionFE");

      // set up MPIPatternP2P for the constraints
      auto mpiPatternP2PField = feBasisManagerField->getMPIPatternP2P();

      // set up vh with inhomogeneous BC

      linearAlgebra::MultiVector<ValueTypeOperand, memorySpace>
        fieldInHomoDBCVec(mpiPatternP2PField,
                          linAlgOpContext,
                          d_numComponents,
                          ValueTypeOperand());
      d_fieldInHomoDBCVec = fieldInHomoDBCVec;

      // Set up BasisManager
      std::shared_ptr<const utils::ScalarSpatialFunctionReal> zeroFunction =
        std::make_shared<utils::ScalarZeroFunctionReal>();

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
        *feBasisDataStorageStiffnessMatrix,
        maxCellTimesNumVecs); // solving the AX = b

      std::shared_ptr<const linearAlgebra::OperatorContext<ValueTypeOperator,
                                                           ValueTypeOperand,
                                                           memorySpace>>
        AxContextNHDB =
          std::make_shared<LaplaceOperatorContextFE<ValueTypeOperator,
                                                    ValueTypeOperand,
                                                    memorySpace,
                                                    dim>>(
            *d_feBasisManagerField,
            *d_feBasisManagerHomo,
            *feBasisDataStorageStiffnessMatrix,
            maxCellTimesNumVecs); // handling the inhomogeneous DBC in RHS

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        mpiPatternP2PHomo, linAlgOpContext);

      if (d_pcType == linearAlgebra::PreconditionerType::JACOBI)
        {
          PoissonLinearSolverFunctionFEInternal::
            getDiagonal<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>(
              diagonal,
              d_feBasisManagerHomo,
              feBasisDataStorageStiffnessMatrix);


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

      // Compute RHS

      std::vector<ValueType> ones(0);
      ones.resize(d_numComponents, (ValueType)1.0);
      std::vector<ValueType> nOnes(0);
      nOnes.resize(d_numComponents, (ValueType)-1.0);

      d_b.setValue(0.0);
      linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> b(d_b, 0.0);

      // Set up basis Operations for RHS
      basis::
        FEBasisOperations<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
          feBasisOperations(feBasisDataStorageRhs, maxCellTimesNumVecs);

      feBasisOperations.integrateWithBasisValues(inpRhs,
                                                 *d_feBasisManagerHomo,
                                                 b);

      linearAlgebra::MultiVector<ValueType, memorySpace> rhsNHDB(d_b, 0.0);

      AxContextNHDB->apply(d_fieldInHomoDBCVec, rhsNHDB);

      linearAlgebra::add(ones, b, nOnes, rhsNHDB, d_b);

      // for (unsigned int i = 0 ; i < d_b.locallyOwnedSize() ; i++)
      //   {
      //     std::cout << i  << " " << *(rhsNHDB.data()+i) << " \t ";
      //   }

      // for(int i = 0 ; i < d_numComponents ; i++)
      // std::cout << "rhs-norm: " << rhsNHDB.l2Norms()[i] << " d_b-norm: " <<
      // d_b.l2Norms()[i] << " b-norm: " << b.l2Norms()[i] << "\t";
      // std::cout << "\n";
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

  } // end of namespace physics
} // end of namespace dftefe
