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
    namespace CFEOverlapInverseOpContextGLLInternal
    {
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      class OverlapMatrixInverseLinearSolverFunctionFE
        : public linearAlgebra::LinearSolverFunction<ValueTypeOperator,
                                                     ValueTypeOperand,
                                                     memorySpace>
      {
      public:
        /**
         * @brief define ValueType as the superior (bigger set) of the
         * ValueTypeOperator and ValueTypeOperand
         * (e.g., between double and complex<double>, complex<double>
         * is the bigger set)
         */
        using ValueType =
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>;

      public:
        /**
         * @brief This constructor creates an instance of a base LinearSolverFunction called OverlapMatrixInverseLinearSolverFunctionFE
         */
        OverlapMatrixInverseLinearSolverFunctionFE(
          const basis::FEBasisManager<ValueTypeOperand,
                                      ValueTypeOperator,
                                      memorySpace,
                                      dim> &    feBasisManager,
          const CFEOverlapOperatorContext<ValueTypeOperator,
                                          ValueTypeOperand,
                                          memorySpace,
                                          dim> &MContext,
          std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
            linAlgOpContext)
          : d_feBasisManager(&feBasisManager)
          , d_linAlgOpContext(linAlgOpContext)
          , d_AxContext(&MContext)
        {
          d_PCContext = std::make_shared<
            linearAlgebra::PreconditionerNone<ValueTypeOperator,
                                              ValueTypeOperand,
                                              memorySpace>>();
        }

        void
        reinit(linearAlgebra::MultiVector<ValueType, memorySpace> &X)
        {
          d_numComponents = X.getNumberComponents();

          // set up MPIPatternP2P for the constraints
          auto mpiPatternP2P = d_feBasisManager->getMPIPatternP2P();

          linearAlgebra::MultiVector<ValueType, memorySpace> x(
            mpiPatternP2P, d_linAlgOpContext, d_numComponents, ValueType());
          d_x = x;
          linearAlgebra::MultiVector<ValueType, memorySpace> initial(
            mpiPatternP2P, d_linAlgOpContext, d_numComponents, ValueType());
          d_initial = initial;

          // Compute RHS
          d_feBasisManager->getConstraints().distributeChildToParent(
            X, d_numComponents);

          d_b = X;
        }

        ~OverlapMatrixInverseLinearSolverFunctionFE() = default;

        const linearAlgebra::
          OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
          getAxContext() const
        {
          return *d_AxContext;
        }

        const linearAlgebra::
          OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace> &
          getPCContext() const
        {
          return *d_PCContext;
        }

        void
        setSolution(const linearAlgebra::MultiVector<
                    linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                           ValueTypeOperand>,
                    memorySpace> &x)
        {
          d_x = x;
        }


        void
        getSolution(linearAlgebra::MultiVector<
                    linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                           ValueTypeOperand>,
                    memorySpace> &solution)
        {
          size_type numComponents = solution.getNumberComponents();
          solution.setValue(0.0);

          solution = d_x;
          solution.updateGhostValues();

          d_feBasisManager->getConstraints().distributeParentToChild(
            solution, numComponents);
        }

        const linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &
        getRhs() const
        {
          return d_b;
        }

        const linearAlgebra::MultiVector<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpace> &
        getInitialGuess() const
        {
          return d_initial;
        }

        const utils::mpi::MPIComm &
        getMPIComm() const
        {
          return d_feBasisManager->getMPIPatternP2P()->mpiCommunicator();
        }

      private:
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                  d_linAlgOpContext;
        size_type d_numComponents;
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
            *                                              d_feBasisManager;
        const linearAlgebra::OperatorContext<ValueTypeOperator,
                                             ValueTypeOperand,
                                             memorySpace> *d_AxContext;
        std::shared_ptr<const linearAlgebra::OperatorContext<ValueTypeOperator,
                                                             ValueTypeOperand,
                                                             memorySpace>>
                                                           d_PCContext;
        linearAlgebra::MultiVector<ValueType, memorySpace> d_x;
        linearAlgebra::MultiVector<ValueType, memorySpace> d_b;
        linearAlgebra::MultiVector<ValueType, memorySpace> d_initial;

      }; // end of class

      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeBasisOverlapMatrix(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalGLLBasisDataStorage,
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>> cfeBDH,
        utils::MemoryStorage<ValueTypeOperator, memorySpace> &NiNjInAllCells,
        linearAlgebra::LinAlgOpContext<memorySpace> &         linAlgOpContext)
      {
        // Set up the overlap matrix quadrature storages.

        const size_type numLocallyOwnedCells = cfeBDH->nLocallyOwnedCells();
        std::vector<size_type> dofsInCellVec(0);
        dofsInCellVec.resize(numLocallyOwnedCells, 0);
        size_type cumulativeBasisOverlapId = 0;

        size_type basisOverlapSize = 0;
        size_type cellId           = 0;

        size_type       dofsPerCell;
        const size_type dofsPerCellCFE = cfeBDH->nCellDofs(cellId);

        auto locallyOwnedCellIter = cfeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != cfeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = cfeBDH->nCellDofs(cellId);
            basisOverlapSize += dofsInCellVec[cellId] * dofsInCellVec[cellId];
            cellId++;
          }

        std::vector<ValueTypeOperator> basisOverlapTmp(0);

        NiNjInAllCells.resize(basisOverlapSize, ValueTypeOperator(0));
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeOperator(0));

        // auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex = 0;

        // const utils::MemoryStorage<ValueTypeOperator, memorySpace>
        //   &basisDataInAllCellsClassicalBlock =
        //     classicalGLLBasisDataStorage.getBasisDataInAllCells();

        locallyOwnedCellIter = cfeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != cfeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = dofsInCellVec[cellIndex];
            size_type nQuadPointInCellClassicalBlock =
              classicalGLLBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesClassicalBlock =
              classicalGLLBasisDataStorage.getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);

            const utils::MemoryStorage<ValueTypeOperator, memorySpace> &
              basisDataInCell = classicalGLLBasisDataStorage.getBasisDataInCell(
                cellIndex); /*GLL Quad rule*/

            std::vector<ValueTypeOperator> JxWxNCellConj(
              dofsPerCell * nQuadPointInCellClassicalBlock);

            size_type stride = 0;
            size_type m = 1, n = dofsPerCell,
                      k = nQuadPointInCellClassicalBlock;

            linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                              ValueTypeOperator,
                                                              memorySpace>(
              1,
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Conj,
              &stride,
              &stride,
              &stride,
              &m,
              &n,
              &k,
              cellJxWValuesClassicalBlock.data(),
              basisDataInCell.data(),
              JxWxNCellConj.data(),
              linAlgOpContext);

            linearAlgebra::blasLapack::
              gemm<ValueTypeOperand, ValueTypeOperand, memorySpace>(
                'N',
                'T',
                dofsPerCell,
                dofsPerCell,
                nQuadPointInCellClassicalBlock,
                (ValueTypeOperand)1.0,
                JxWxNCellConj.data(),
                dofsPerCell,
                basisDataInCell.data(),
                dofsPerCell,
                (ValueTypeOperand)0.0,
                basisOverlapTmp.data() + cumulativeBasisOverlapId,
                dofsPerCell,
                linAlgOpContext);

            // const ValueTypeOperator *cumulativeClassicalBlockDofQuadPoints =
            //   basisDataInAllCellsClassicalBlock.data(); /*GLL Quad rule*/

            // for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
            //   {
            //     for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
            //       {
            //         *basisOverlapTmpIter = 0.0;
            //         // Ni_classical* Ni_classical of the
            //         classicalBlockBasisData for (unsigned int qPoint = 0;
            //              qPoint < nQuadPointInCellClassicalBlock;
            //              qPoint++)
            //           {
            //             *basisOverlapTmpIter +=
            //               *(cumulativeClassicalBlockDofQuadPoints +
            //                 dofsPerCell * qPoint + iNode
            //                 /*nQuadPointInCellClassicalBlock * iNode +
            //                 qPoint*/) *
            //               *(cumulativeClassicalBlockDofQuadPoints +
            //                 dofsPerCell * qPoint + jNode
            //                 /*nQuadPointInCellClassicalBlock * jNode +
            //                 qPoint*/) *
            //               cellJxWValuesClassicalBlock[qPoint];
            //           }
            //         basisOverlapTmpIter++;
            //       }
            //   }

            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(),
          NiNjInAllCells.data(),
          basisOverlapTmp.data());
      }

    } // namespace CFEOverlapInverseOpContextGLLInternal

    // Write M^-1 apply on a matrix for GLL with spectral finite element
    // M^-1 does not have a cell structure.

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    CFEOverlapInverseOpContextGLL<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      CFEOverlapInverseOpContextGLL(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
            &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalGLLBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : d_diagonalInv(d_feBasisManager->getMPIPatternP2P(), linAlgOpContext)
      , d_feBasisManager(&feBasisManager)
      , d_isCGSolved(false)
    {
      const size_type numLocallyOwnedCells =
        d_feBasisManager->nLocallyOwnedCells();

      std::shared_ptr<
        const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
        cfeBDH = std::dynamic_pointer_cast<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
          classicalGLLBasisDataStorage.getBasisDofHandler());
      utils::throwException(
        cfeBDH != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler "
        "in CFEOverlapInverseOpContext");

      utils::throwException(cfeBDH->totalRanges() == 1,
                            "This function is only for classical FE basis.");

      utils::throwException(
        classicalGLLBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily() == quadrature::QuadratureFamily::GLL,
        "The quadrature rule for integration of Classical FE dofs has to be GLL."
        "Contact developers if extra options are needed.");

      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBegin =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      // access cell-wise discrete Laplace operator

      utils::MemoryStorage<ValueTypeOperator, memorySpace> NiNjInAllCells(0);

      CFEOverlapInverseOpContextGLLInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalGLLBasisDataStorage,
             cfeBDH,
             NiNjInAllCells,
             *linAlgOpContext);

      // auto NiNjInAllCells =
      //   classicalGLLBasisDataStorage.getBasisOverlapInAllCells();

      std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                         0);
      std::copy(numCellDofs.begin(),
                numCellDofs.begin() + numLocallyOwnedCells,
                locallyOwnedCellsNumDoFsSTL.begin());

      utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
        numLocallyOwnedCells);
      locallyOwnedCellsNumDoFs.copyFrom(locallyOwnedCellsNumDoFsSTL);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(NiNjInAllCells.data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           diagonal.data());

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      // NOTE ::: In a global matrix sense this step can be thought as doing
      // a kind of mass lumping. It is seen that doing such mass lumping in
      // overlap inverse made the scfs converge faster . Without this step the
      // HX residual was not dropping below 1e-3 for non-conforming mesh.
      d_feBasisManager->getConstraints().distributeChildToParent(diagonal, 1);

      d_feBasisManager->getConstraints().setConstrainedNodes(diagonal, 1, 1.0);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      diagonal.accumulateAddLocallyOwned();

      diagonal.updateGhostValues();

      linearAlgebra::blasLapack::reciprocalX(diagonal.localSize(),
                                             1.0,
                                             diagonal.data(),
                                             d_diagonalInv.data(),
                                             *(diagonal.getLinAlgOpContext()));

      d_feBasisManager->getConstraints().setConstrainedNodesToZero(
        d_diagonalInv, 1);
    }


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    CFEOverlapInverseOpContextGLL<ValueTypeOperator,
                                  ValueTypeOperand,
                                  memorySpace,
                                  dim>::
      CFEOverlapInverseOpContextGLL(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
            &                                 feBasisManager,
        const CFEOverlapOperatorContext<ValueTypeOperator,
                                        ValueTypeOperand,
                                        memorySpace,
                                        dim> &MContext,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
             linAlgOpContext,
        bool isCGSolved)
      : d_feBasisManager(&feBasisManager)
      , d_linAlgOpContext(linAlgOpContext)
      , d_isCGSolved(isCGSolved)
    {
      if (d_isCGSolved)
        {
          d_overlapInvPoisson = std::make_shared<
            CFEOverlapInverseOpContextGLLInternal::
              OverlapMatrixInverseLinearSolverFunctionFE<ValueTypeOperator,
                                                         ValueTypeOperand,
                                                         memorySpace,
                                                         dim>>(
            *d_feBasisManager, MContext, linAlgOpContext);

          linearAlgebra::LinearAlgebraProfiler profiler;

          d_CGSolve =
            std::make_shared<linearAlgebra::CGLinearSolver<ValueTypeOperator,
                                                           ValueTypeOperand,
                                                           memorySpace>>(
              100000, 1e-10, 1e-12, 1e10, profiler);
        }
      else
        {
          utils::throwException(false,
                                "Could not have other options than cgsolve.");
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    CFEOverlapInverseOpContextGLL<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<ValueType, memorySpace> &       Y,
                  bool updateGhostX,
                  bool updateGhostY) const
    {
      if (d_isCGSolved)
        {
          if (updateGhostX)
            X.updateGhostValues();
          std::shared_ptr<
            CFEOverlapInverseOpContextGLLInternal::
              OverlapMatrixInverseLinearSolverFunctionFE<ValueTypeOperator,
                                                         ValueTypeOperand,
                                                         memorySpace,
                                                         dim>>
            overlapInvPoisson = std::dynamic_pointer_cast<
              CFEOverlapInverseOpContextGLLInternal::
                OverlapMatrixInverseLinearSolverFunctionFE<ValueTypeOperator,
                                                           ValueTypeOperand,
                                                           memorySpace,
                                                           dim>>(
              d_overlapInvPoisson);
          Y.setValue(0.0);
          overlapInvPoisson->reinit(X);
          d_CGSolve->solve(*overlapInvPoisson);
          overlapInvPoisson->getSolution(Y);
          if (updateGhostY)
            Y.updateGhostValues();
        }
      else
        {
          if (updateGhostX)
            X.updateGhostValues();
          // update the child nodes based on the parent nodes
          d_feBasisManager->getConstraints().distributeParentToChild(
            X, X.getNumberComponents());

          Y.setValue(0.0);

          linearAlgebra::blasLapack::khatriRaoProduct(
            linearAlgebra::blasLapack::Layout::ColMajor,
            1,
            X.getNumberComponents(),
            d_diagonalInv.localSize(),
            d_diagonalInv.data(),
            X.data(),
            Y.data(),
            *(d_diagonalInv.getLinAlgOpContext()));

          // function to do a static condensation to send the constraint nodes
          // to its parent nodes
          // TODO : The distributeChildToParent of the result of M_inv*X is
          // processor local and for adaptive quadrature the M_inv is not
          // diagonal. Implement that case here.
          d_feBasisManager->getConstraints().distributeChildToParent(
            Y, Y.getNumberComponents());

          // Function to update the ghost values of the result
          if (updateGhostY)
            Y.updateGhostValues();
        }
    }
  } // namespace basis
} // namespace dftefe
