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
                utils::MemorySpace memorySpace>
      void
      computeDiagonalCellWiseLocal(
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &NiNjInAllCells,
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand> *y,
        const size_type               numLocallyOwnedCells,
        const std::vector<size_type> &numCellDofs,
        const size_type *             cellLocalIdsStartPtr,
        const size_type               cellBlockSize)
      {
        //
        // Perform ye = Diagonal(Ae), where
        // Ae is the discrete Overlap operator for the e-th cell.
        // That is, \f$Ae_ij=\int_{\Omega_e} N_i \cdot N_j
        // d\textbf{r} $\f,
        // (\f$Ae_ij$\f is the integral of the dot product of the gradient of
        // i-th and j-th basis function in the e-th cell.
        //
        // ye is part of output vector (y),
        // respectively, belonging to e-th cell.
        //

        size_type BStartOffset       = 0;
        size_type cellLocalIdsOffset = 0;
        for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
             cellStartId += cellBlockSize)
          {
            const size_type cellEndId =
              std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
            const size_type        numCellsInBlock = cellEndId - cellStartId;
            std::vector<size_type> cellsInBlockNumDoFsSTL(numCellsInBlock, 0);
            std::copy(numCellDofs.begin() + cellStartId,
                      numCellDofs.begin() + cellEndId,
                      cellsInBlockNumDoFsSTL.begin());

            const size_type cellsInBlockNumCumulativeDoFs =
              std::accumulate(cellsInBlockNumDoFsSTL.begin(),
                              cellsInBlockNumDoFsSTL.end(),
                              0);

            utils::MemoryStorage<size_type, memorySpace> cellsInBlockNumDoFs(
              numCellsInBlock);
            cellsInBlockNumDoFs.copyFrom(cellsInBlockNumDoFsSTL);

            // allocate memory for cell-wise data for y
            utils::MemoryStorage<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                     ValueTypeOperand>,
              memorySpace>
              yCellValues(cellsInBlockNumCumulativeDoFs,
                          utils::Types<linearAlgebra::blasLapack::scalar_type<
                            ValueTypeOperator,
                            ValueTypeOperand>>::zero);

            const ValueTypeOperator *BBlock =
              NiNjInAllCells.data() + BStartOffset;
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                   ValueTypeOperand> *C =
              yCellValues.begin();

            size_type BBlockOffset = 0;
            size_type COffset      = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
              {
                size_type nDoFs = cellsInBlockNumDoFsSTL[iCell];
                for (size_type j = 0; j < nDoFs; j++)
                  {
                    *(C + COffset + j) =
                      *(BBlock + BBlockOffset + j * nDoFs + j);
                  }
                COffset += nDoFs;
                BBlockOffset += nDoFs * nDoFs;
              }

            FECellWiseDataOperations<
              linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                     ValueTypeOperand>,
              memorySpace>::addCellWiseDataToFieldData(yCellValues,
                                                       1,
                                                       cellLocalIdsStartPtr +
                                                         cellLocalIdsOffset,
                                                       cellsInBlockNumDoFs,
                                                       y);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                BStartOffset +=
                  cellsInBlockNumDoFsSTL[iCell] * cellsInBlockNumDoFsSTL[iCell];
                cellLocalIdsOffset += cellsInBlockNumDoFsSTL[iCell];
              }
          }
      }

      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      getDiagonal(
        linearAlgebra::Vector<ValueTypeOperator, memorySpace> &diagonal,
        std::shared_ptr<
          const FEBasisHandler<ValueTypeOperator, memorySpace, dim>>
          feBasisHandler,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeOperator, memorySpace>>
                          feBasisDataStorage,
        const std::string basisInterfaceCoeffConstraint,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const size_type                             maxCellTimesNumVecs)
      {
        const size_type numLocallyOwnedCells =
          feBasisHandler->nLocallyOwnedCells();
        std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          numCellDofs[iCell] = feBasisHandler->nLocallyOwnedCellDofs(iCell);

        auto itCellLocalIdsBegin =
          feBasisHandler->locallyOwnedCellLocalDofIdsBegin(
            basisInterfaceCoeffConstraint);

        // access cell-wise discrete Laplace operator
        auto NiNjInAllCells =
          feBasisDataStorage->getBasisOverlapInAllCells(
            quadratureRuleAttributes);

        const size_type cellBlockSize = maxCellTimesNumVecs;

        //
        // get processor local part of the diagonal
        //
        L2ProjectionLinearSolverFunctionInternal::computeDiagonalCellWiseLocal<
          ValueTypeOperator,
          ValueTypeOperand,
          memorySpace>(NiNjInAllCells,
                       diagonal.begin(),
                       numLocallyOwnedCells,
                       numCellDofs,
                       itCellLocalIdsBegin,
                       cellBlockSize);

        // function to do a static condensation to send the constraint nodes to
        // its parent nodes
        feBasisHandler->getConstraints(basisInterfaceCoeffConstraint)
          .distributeChildToParent(diagonal, 1);

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
        std::shared_ptr<
          const FEBasisHandler<ValueTypeOperator, memorySpace, dim>>
                                             cfeBasisHandler,
        std::shared_ptr<
          const FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          cfeBasisDataStorage,
        FEBasisOperations<ValueTypeOperand, ValueTypeOperator,memorySpace,
         dim> cfeBasisOperations,
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpace> & inp,
        const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
        const std::string basisInterfaceCoeffConstraint,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellTimesNumVecs)
      : d_feBasisHandler(cfeBasisHandler)
      , d_feBasisDataStorage(cfeBasisDataStorage)
      , d_b(cfeBasisHandler->getMPIPatternP2P(basisInterfaceCoeffConstraint),
            linAlgOpContext,
            inp.getNumberComponents())
      , d_pcType(pcType)
      , d_x(cfeBasisHandler->getMPIPatternP2P(basisInterfaceCoeffConstraint),
            linAlgOpContext,
            inp.getNumberComponents())
      , d_initial(cfeBasisHandler->getMPIPatternP2P(
                    basisInterfaceCoeffConstraint),
                  linAlgOpContext,
                  inp.getNumberComponents())
      , d_basisInterfaceCoeffConstraint(basisInterfaceCoeffConstraint)
    {
      d_mpiPatternP2P =
        cfeBasisHandler->getMPIPatternP2P(basisInterfaceCoeffConstraint);

      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

      d_AxContext =
        std::make_shared<BasisOverlapOperatorContext<ValueTypeOperator,
                                                           ValueTypeOperand,
                                                           memorySpace,
                                                           dim>>(
          *d_feBasisHandler,
          *d_feBasisDataStorage,
          basisInterfaceCoeffConstraint,
          basisInterfaceCoeffConstraint,
          quadratureRuleAttributes,
          maxCellTimesNumVecs);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        d_mpiPatternP2P, linAlgOpContext);

      if (d_pcType == linearAlgebra::PreconditionerType::JACOBI)
        {
          L2ProjectionLinearSolverFunctionInternal::
            getDiagonal<ValueTypeOperator, ValueTypeOperand, memorySpace, dim>(
              diagonal,
              d_feBasisHandler,
              d_feBasisDataStorage,
              basisInterfaceCoeffConstraint,
              quadratureRuleAttributes,
              maxCellTimesNumVecs);


          d_feBasisHandler->getConstraints(basisInterfaceCoeffConstraint)
            .setConstrainedNodes(diagonal, 1, 1.0);

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
      ones.resize(inp.getNumberComponents(), (ValueType)1.0);
      std::vector<ValueType> nOnes(0);
      nOnes.resize(inp.getNumberComponents(), (ValueType)-1.0);

      d_b.setValue(0.0);

      cfeBasisOperations.integrateWithBasisValues(inp,
                                                 quadratureRuleAttributes,
                                                 *d_feBasisHandler,
                                                 basisInterfaceCoeffConstraint,
                                                 d_b);

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

      d_feBasisHandler->getConstraints(d_basisInterfaceCoeffConstraint)
        .distributeParentToChild(solution, numComponents);
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

  } // end of namespace physics
} // end of namespace dftefe
