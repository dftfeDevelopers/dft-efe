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
  namespace basis
  {
    namespace L2ProjectionLinearSolverFunctionInternal
    {
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      getDiagonal(
        linearAlgebra::Vector<ValueTypeOperator, memorySpace> &diagonal,
        std::shared_ptr<const FEBasisManager<ValueTypeOperand,
                                             ValueTypeOperator,
                                             memorySpace,
                                             dim>>             feBasisManager,
        std::shared_ptr<const CFEOverlapOperatorContext<ValueTypeOperator,
                                                        ValueTypeOperand,
                                                        memorySpace,
                                                        dim>>
          cfeOverlapOperatorContext)
      {
        const size_type numLocallyOwnedCells =
          feBasisManager->nLocallyOwnedCells();
        std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          numCellDofs[iCell] = feBasisManager->nLocallyOwnedCellDofs(iCell);

        auto itCellLocalIdsBegin =
          feBasisManager->locallyOwnedCellLocalDofIdsBegin();

        // access cell-wise discrete Laplace operator
        auto NiNjInAllCells =
          cfeOverlapOperatorContext->getBasisOverlapInAllCells();

        std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                           0);
        std::copy(numCellDofs.begin(),
                  numCellDofs.begin() + numLocallyOwnedCells,
                  locallyOwnedCellsNumDoFsSTL.begin());

        utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
          numLocallyOwnedCells);
        locallyOwnedCellsNumDoFs.copyFrom(locallyOwnedCellsNumDoFsSTL);

        basis::FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
          addCellWiseBasisDataToDiagonalData(NiNjInAllCells.data(),
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
    } // end of namespace L2ProjectionLinearSolverFunctionInternal


    //
    // Constructor
    //
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    L2ProjectionLinearSolverFunction<ValueTypeOperator,
                                     ValueTypeOperand,
                                     memorySpace,
                                     dim>::
      L2ProjectionLinearSolverFunction(
        std::shared_ptr<const FEBasisManager<ValueTypeOperand,
                                             ValueTypeOperator,
                                             memorySpace,
                                             dim>> cfeBasisManager,
        std::shared_ptr<const CFEOverlapOperatorContext<ValueTypeOperator,
                                                        ValueTypeOperand,
                                                        memorySpace,
                                                        dim>>
          cfeBasisDataStorageOverlapMatrix,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          cfeBasisDataStorageRhs,
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpace> &                        inp,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxFieldBlock)
      : d_feBasisManager(cfeBasisManager)
      , d_b(cfeBasisManager->getMPIPatternP2P(),
            linAlgOpContext,
            inp.getNumberComponents())
      , d_pcType(pcType)
      , d_x(cfeBasisManager->getMPIPatternP2P(),
            linAlgOpContext,
            inp.getNumberComponents())
      , d_initial(cfeBasisManager->getMPIPatternP2P(),
                  linAlgOpContext,
                  inp.getNumberComponents())
      , d_AxContext(cfeBasisDataStorageOverlapMatrix)
      , d_maxCellBlock(maxCellBlock)
      , d_maxFieldBlock(maxFieldBlock)
    {
      d_mpiPatternP2P = cfeBasisManager->getMPIPatternP2P();

      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

      // Get M marix for Md = \integral N_e. N_c
      // d_AxContext =
      //   std::make_shared<CFEOverlapOperatorContext<ValueTypeOperator,
      //                                                      ValueTypeOperand,
      //                                                      memorySpace,
      //                                                      dim>>(
      //     *d_feBasisManager,
      //     *cfeBasisDataStorageOverlapMatrix,
      //     maxCellTimesNumVecs);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        d_mpiPatternP2P, linAlgOpContext);

      if (d_pcType == linearAlgebra::PreconditionerType::JACOBI)
        {
          L2ProjectionLinearSolverFunctionInternal::
            getDiagonal<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>(
              diagonal, d_feBasisManager, d_AxContext);


          d_feBasisManager->getConstraints().setConstrainedNodes(diagonal,
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

      d_b.setValue(0.0);

      // Set up basis Operations for RHS
      FEBasisOperations<ValueTypeOperator, ValueTypeOperator, memorySpace, dim>
        cfeBasisOperations(cfeBasisDataStorageRhs,
                           d_maxCellBlock,
                           d_maxFieldBlock);

      // Integrate this with different quarature rule. (i.e. adaptive for the
      // enrichment functions) , inp will be in adaptive grid
      cfeBasisOperations.integrateWithBasisValues(inp, *d_feBasisManager, d_b);

      // for (unsigned int i = 0 ; i < d_b.locallyOwnedSize() ; i++)
      //   {
      //     std::cout << i  << " " << *(rhsNHDB.data()+i) << " \t ";
      //   }

      // for(int i = 0 ; i < inp.getNumberComponents() ; i++)
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
      L2ProjectionLinearSolverFunction<ValueTypeOperator,
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
      L2ProjectionLinearSolverFunction<ValueTypeOperator,
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
    L2ProjectionLinearSolverFunction<ValueTypeOperator,
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
    L2ProjectionLinearSolverFunction<ValueTypeOperator,
                                     ValueTypeOperand,
                                     memorySpace,
                                     dim>::
      getSolution(linearAlgebra::MultiVector<
                  linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                         ValueTypeOperand>,
                  memorySpace> &solution)
    {
      size_type numComponents = solution.getNumberComponents();

      for (size_type i = 0; i < solution.locallyOwnedSize(); i++)
        {
          for (size_type j = 0; j < numComponents; j++)
            {
              solution.data()[i * numComponents + j] =
                d_x.data()[i * numComponents + j];
            }
        }

      solution.updateGhostValues();

      d_feBasisManager->getConstraints().distributeParentToChild(solution,
                                                                 numComponents);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &
    L2ProjectionLinearSolverFunction<ValueTypeOperator,
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
    L2ProjectionLinearSolverFunction<ValueTypeOperator,
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
    L2ProjectionLinearSolverFunction<ValueTypeOperator,
                                     ValueTypeOperand,
                                     memorySpace,
                                     dim>::getMPIComm() const
    {
      return d_mpiPatternP2P->mpiCommunicator();
    }

  } // namespace basis
} // end of namespace dftefe
