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
    // Write M^-1 apply on a matrix for GLL with spectral finite element
    // M^-1 does not have a cell structure.

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    FEOverlapInverseOperatorContext<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
      FEOverlapInverseOperatorContext(
        const basis::FEBasisHandler<ValueTypeOperator, memorySpace, dim>
          &                                         feBasisHandler,
        const basis::FEOverlapOperatorContext<ValueTypeOperator,
                                              ValueTypeOperand,
                                              memorySpace,
                                              dim> &feOverlapOperatorContext,
        const std::string                           constraints,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : d_diagonalInv(d_feBasisHandler->getMPIPatternP2P(constraints),
                      linAlgOpContext)
      , d_feBasisHandler(&feBasisHandler)
      , d_constraints(constraints)
    {
      const size_type numLocallyOwnedCells =
        d_feBasisHandler->nLocallyOwnedCells();

      const FEBasisManager &basisManager =
        dynamic_cast<const FEBasisManager &>(feBasisHandler.getBasisManager());
      utils::throwException(
        &basisManager != nullptr,
        "Could not cast BasisManager of the input vector to FEBasisManager ");

      utils::throwException(basisManager.totalRanges() == 1,
                            "This function is only for classical FE basis.");

      utils::throwException(
        feOverlapOperatorContext.getFEBasisDataStorage()
            .getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily() == dftefe::quadrature::QuadratureFamily::GLL,
        "The quadrature rule for integration of Classical FE dofs has to be GLL."
        "Contact developers if extra options are needed.");

      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisHandler->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBegin =
        d_feBasisHandler->locallyOwnedCellLocalDofIdsBegin(constraints);

      // access cell-wise discrete Laplace operator
      auto NiNjInAllCells =
        feOverlapOperatorContext.getBasisOverlapInAllCells();

      std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                         0);
      std::copy(numCellDofs.begin(),
                numCellDofs.begin() + numLocallyOwnedCells,
                locallyOwnedCellsNumDoFsSTL.begin());

      utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
        numLocallyOwnedCells);
      locallyOwnedCellsNumDoFs.copyFrom(locallyOwnedCellsNumDoFsSTL);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        d_feBasisHandler->getMPIPatternP2P(d_constraints), linAlgOpContext);

      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(NiNjInAllCells.data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           diagonal.data());

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      d_feBasisHandler->getConstraints(d_constraints)
        .distributeChildToParent(diagonal, 1);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      diagonal.accumulateAddLocallyOwned();

      diagonal.updateGhostValues();

      linearAlgebra::blasLapack::reciprocalX(diagonal.localSize(),
                                             1.0,
                                             diagonal.data(),
                                             d_diagonalInv.data(),
                                             *(diagonal.getLinAlgOpContext()));
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    FEOverlapInverseOperatorContext<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const
    {
      X.updateGhostValues();
      // update the child nodes based on the parent nodes
      d_feBasisHandler->getConstraints(d_constraints)
        .distributeParentToChild(X, X.getNumberComponents());

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

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      // TODO : The distributeChildToParent of the result of M_inv*X is
      // processor local and for adaptive quadrature the M_inv is not diagonal.
      // Implement that case here.
      d_feBasisHandler->getConstraints(d_constraints)
        .distributeChildToParent(Y, Y.getNumberComponents());

      // Function to update the ghost values of the result
      Y.updateGhostValues();
    }
  } // namespace basis
} // namespace dftefe
