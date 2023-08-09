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
                utils::MemorySpace memorySpace>
      void
      computeDiagonalCellWiseLocal(
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &                     gradNiGradNjInAllCells,
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand> *y,
        const size_type                              numLocallyOwnedCells,
        const std::vector<size_type> &               numCellDofs,
        const size_type *                            cellLocalIdsStartPtr,
        const size_type                              cellBlockSize)
      {
        //
        // Perform ye = Diagonal(Ae), where
        // Ae is the discrete Laplace operator for the e-th cell.
        // That is, \f$Ae_ij=\int_{\Omega_e} \nabla N_i \cdot \nabla N_j
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
            utils::MemoryStorage<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand>, memorySpace> yCellValues(
              cellsInBlockNumCumulativeDoFs,
              utils::Types<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand>>::zero);

            const ValueTypeOperator *BBlock = gradNiGradNjInAllCells.data() + BStartOffset;
            linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand> *C = yCellValues.begin();

            size_type BBlockOffset = 0;
            size_type COffset = 0;
            for ( size_type iCell = 0 ; iCell < numCellsInBlock ; iCell++ )
            {
              size_type nDoFs = cellsInBlockNumDoFsSTL[iCell];
              for ( size_type j = 0 ; j < nDoFs ; j++ )
              {
                *(C + COffset + j) = *(BBlock + BBlockOffset + j*nDoFs + j);
              }
              COffset += nDoFs;
              BBlockOffset += nDoFs * nDoFs;
            }

            basis::FECellWiseDataOperations<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,ValueTypeOperand>, memorySpace>::
              addCellWiseDataToFieldData(yCellValues,
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
        getDiagonal(linearAlgebra::Vector<ValueTypeOperator, memorySpace> &diagonal,
          std::shared_ptr<const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>>
            feBasisHandler,
          std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
            feBasisDataStorage,
          const std::string                                    constraintsHangingwHomogeneous,
          const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
          const size_type                 maxCellTimesNumVecs)
      {

        const size_type numLocallyOwnedCells =
          feBasisHandler->nLocallyOwnedCells();
        std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
        for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
          numCellDofs[iCell] = feBasisHandler->nLocallyOwnedCellDofs(iCell);
        
        auto itCellLocalIdsBegin =
          feBasisHandler->locallyOwnedCellLocalDofIdsBegin(constraintsHangingwHomogeneous);

        // access cell-wise discrete Laplace operator
        auto gradNiGradNjInAllCells =
          feBasisDataStorage->getBasisGradNiGradNjInAllCells(
            quadratureRuleAttributes);

        const size_type cellBlockSize = maxCellTimesNumVecs;

          //
          // get processor local part of the diagonal
          //
        PoissonLinearSolverFunctionFEInternal::computeDiagonalCellWiseLocal
                <ValueTypeOperator,
                ValueTypeOperand,
                memorySpace>(
          gradNiGradNjInAllCells,
          diagonal.begin(),
          numLocallyOwnedCells,
          numCellDofs,
          itCellLocalIdsBegin,
          cellBlockSize);

        // function to do a static condensation to send the constraint nodes to
        // its parent nodes
        feBasisHandler->getConstraints(constraintsHangingwHomogeneous).distributeChildToParent(diagonal, 1);

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
                                  dim>::PoissonLinearSolverFunctionFE(
      std::shared_ptr<const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>>
        feBasisHandler,
      const basis::FEBasisOperations<ValueTypeOperator,ValueTypeOperand,memorySpace,dim>
        & feBasisOperations,
      std::shared_ptr<const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
        feBasisDataStorage,
      const quadrature::QuadratureValuesContainer<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
        ValueTypeOperand>, memorySpace> 
        & inp,
      const quadrature::QuadratureRuleAttributes &quadratureRuleAttributes,
      const std::string                                    constraintsHanging,
      const std::string                                    constraintsHangingwHomogeneous,
      const linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> & inhomogeneousDirichletBCVector,
      const linearAlgebra::PreconditionerType              pcType,
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContext,
      const size_type                 maxCellTimesNumVecs)
      : d_feBasisHandler(feBasisHandler)
      , d_feBasisDataStorage(feBasisDataStorage)
      , d_b(feBasisHandler->getMPIPatternP2P(constraintsHangingwHomogeneous),linAlgOpContext,inp.getNumberComponents())
      , d_pcType(pcType)
      , d_constraintsHanging(constraintsHanging)
      , d_x(feBasisHandler->getMPIPatternP2P(constraintsHangingwHomogeneous),linAlgOpContext,inp.getNumberComponents())
      , d_initial(feBasisHandler->getMPIPatternP2P(constraintsHangingwHomogeneous),linAlgOpContext,inp.getNumberComponents())
      , d_inhomogeneousDirichletBCVector(feBasisHandler->getMPIPatternP2P(constraintsHanging),linAlgOpContext,inp.getNumberComponents())
    {
      d_inhomogeneousDirichletBCVector = inhomogeneousDirichletBCVector;
      d_mpiPatternP2PHangingwHomogeneous = feBasisHandler->getMPIPatternP2P(constraintsHangingwHomogeneous);

      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;

      d_AxContext = std::make_shared<physics::LaplaceOperatorContextFE
        <ValueTypeOperator, ValueTypeOperand, memorySpace, dim>>( *d_feBasisHandler,
        *d_feBasisDataStorage,
        constraintsHangingwHomogeneous,
        constraintsHangingwHomogeneous,
        quadratureRuleAttributes,
        maxCellTimesNumVecs); // solving the AX = b

      auto AxContextNHDB = std::make_shared<physics::LaplaceOperatorContextFE
        <ValueTypeOperator, ValueTypeOperand, memorySpace, dim>>( *d_feBasisHandler,
        *d_feBasisDataStorage,
        constraintsHanging,
        constraintsHangingwHomogeneous,
        quadratureRuleAttributes,
        maxCellTimesNumVecs); // handling the inhomogeneous DBC in RHS

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(d_mpiPatternP2PHangingwHomogeneous,linAlgOpContext);   

      if (d_pcType == linearAlgebra::PreconditionerType::JACOBI)
      {
        PoissonLinearSolverFunctionFEInternal::getDiagonal
                <ValueTypeOperator,
                ValueTypeOperand,
                memorySpace,
                dim>(
            diagonal,
            feBasisHandler,
            feBasisDataStorage,
            constraintsHangingwHomogeneous,
            quadratureRuleAttributes,
            maxCellTimesNumVecs);

        // for (unsigned int i = 0 ; i < diagonal.locallyOwnedSize() ; i++)
        //   {
        //     std::cout << "diagonal[" <<i<<"] : "<< *(diagonal.data()+i) << " ";
        //   }        

        feBasisHandler->getConstraints(constraintsHangingwHomogeneous).setConstrainedNodes(diagonal, 1, 1.0);

        d_PCContext = std::make_shared<linearAlgebra::PreconditionerJacobi
          <ValueTypeOperator, ValueTypeOperand, memorySpace>>(diagonal);

        diagonal.updateGhostValues();

      }
      else if (d_pcType == linearAlgebra::PreconditionerType::NONE)
        d_PCContext = std::make_shared<linearAlgebra::PreconditionerNone
          <ValueTypeOperator, ValueTypeOperand, memorySpace>>();
      else
      utils::throwException(
        false,
        "Unknown PreConditionerType");

      // Compute RHS

      std::vector<ValueType> ones(0);
      ones.resize(inp.getNumberComponents(), (ValueType)1.0);
      std::vector<ValueType> nOnes(0);
      nOnes.resize(inp.getNumberComponents(), (ValueType)-1.0);

      d_b.setValue(0.0);
      linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> b(d_b, 0.0);

      // feBasisOperations.integrateWithBasisValues(
      //   inp,
      //   quadratureRuleAttributes,
      //   *d_feBasisHandler,
      //   constraintsHangingwHomogeneous,
      //   b);

      linearAlgebra::MultiVector<ValueType, memorySpace> rhsNHDB(d_b, 0.0);

      AxContextNHDB->apply(d_inhomogeneousDirichletBCVector, rhsNHDB);

      linearAlgebra::add(ones, b, nOnes, rhsNHDB, d_b);

    // for (unsigned int i = 0 ; i < d_b.locallyOwnedSize() ; i++)
    //   {
    //     std::cout << i  << " " << *(rhsNHDB.data()+i) << " \t ";
    //   }

    //std::cout << "rhs-norm: " << rhsNHDB.l2Norms()[0] << " d_b-norm: " << d_b.l2Norms()[0] << " b-norm: " << b.l2Norms()[0] << "\n";

    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
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
    const linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
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
                                  dim>::setSolution(const linearAlgebra::MultiVector<linearAlgebra::blasLapack::scalar_type
                                    <ValueTypeOperator,ValueTypeOperand>, memorySpace>
                                   &x)
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
                                  dim>::getSolution(linearAlgebra::MultiVector<linearAlgebra::blasLapack::scalar_type
                                      <ValueTypeOperator,ValueTypeOperand>, memorySpace> &solution) 
    {
      std::vector<linearAlgebra::blasLapack::scalar_type
                                      <ValueTypeOperator,ValueTypeOperand>> ones(0);
      ones.resize(solution.getNumberComponents(), (linearAlgebra::blasLapack::scalar_type
                                      <ValueTypeOperator,ValueTypeOperand>)1.0);

      dftefe::linearAlgebra::add(ones, d_x, ones, d_inhomogeneousDirichletBCVector, solution);

      solution.updateGhostValues();

      d_feBasisHandler->getConstraints(d_constraintsHanging).distributeParentToChild(solution, solution.getNumberComponents());
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
    const linearAlgebra::MultiVector<linearAlgebra::blasLapack::scalar_type
                                    <ValueTypeOperator,ValueTypeOperand>, memorySpace> &
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
      return d_mpiPatternP2PHangingwHomogeneous->mpiCommunicator();
    }

  } // end of namespace physics
} // end of namespace dftefe