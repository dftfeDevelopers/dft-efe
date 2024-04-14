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
#include <utils/MathFunctions.h>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <algorithm>
// #include <mkl.h>
namespace dftefe
{
  namespace basis
  {
    namespace OrthoEFEOverlapInverseOpContextGLLInternal
    {
      /*
      // Functions from intel MKL library
      //   // LU decomoposition of a general matrix
      //   void dgetrf(int* M, int *N, double* A, int* lda, int* IPIV, int*
      //   INFO);

      //   // generate inverse of a matrix given its LU decomposition
      //   void dgetri(int* N, double* A, int* lda, int* IPIV, double* WORK,
      //   int* lwork, int* INFO);

      void
      inverse(double *A, int N)
      {
        int *   IPIV  = new int[N];
        int     LWORK = N * N;
        double *WORK  = new double[LWORK];
        int     INFO;

        dgetrf(&N, &N, A, &N, IPIV, &INFO);
        dgetri(&N, A, &N, IPIV, WORK, &LWORK, &INFO);

        delete[] IPIV;
        delete[] WORK;
      }
      */
      // Use this for data storage of orthogonalized EFE only
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeBasisOverlapMatrix(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockBasisDataStorage,
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>> cfeBDH,
        std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                                 ValueTypeOperator,
                                                 memorySpace,
                                                 dim>>                 efeBDH,
        utils::MemoryStorage<ValueTypeOperator, memorySpace> &NiNjInAllCells)
      {
        std::shared_ptr<
          const EnrichmentClassicalInterfaceSpherical<ValueTypeOperator,
                                                      memorySpace,
                                                      dim>>
          eci = efeBDH->getEnrichmentClassicalInterface();

        size_type nTotalEnrichmentIds =
          eci->getEnrichmentIdsPartition()->nTotalEnrichmentIds();

        // Set up the overlap matrix quadrature storages.

        const size_type numLocallyOwnedCells = efeBDH->nLocallyOwnedCells();
        std::vector<size_type> dofsInCellVec(0);
        dofsInCellVec.resize(numLocallyOwnedCells, 0);
        size_type cumulativeBasisOverlapId = 0;

        size_type       basisOverlapSize = 0;
        size_type       cellId           = 0;
        const size_type feOrder          = efeBDH->getFEOrder(cellId);

        size_type       dofsPerCell;
        const size_type dofsPerCellCFE = cfeBDH->nCellDofs(cellId);

        auto locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();

        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = efeBDH->nCellDofs(cellId);
            basisOverlapSize += dofsInCellVec[cellId] * dofsInCellVec[cellId];
            cellId++;
          }

        std::vector<ValueTypeOperator> basisOverlapTmp(0);

        NiNjInAllCells.resize(basisOverlapSize, ValueTypeOperator(0));
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeOperator(0));

        auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex           = 0;

        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsClassicalBlock =
            classicalBlockGLLBasisDataStorage.getBasisDataInAllCells();
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsEnrichmentBlock =
            enrichmentBlockBasisDataStorage.getBasisDataInAllCells();

        size_type cumulativeEnrichmentBlockDofQuadPointsOffset = 0;

        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != efeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsPerCell = dofsInCellVec[cellIndex];
            size_type nQuadPointInCellClassicalBlock =
              classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesClassicalBlock =
              classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);

            size_type nQuadPointInCellEnrichmentBlock =
              enrichmentBlockBasisDataStorage.getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEnrichmentBlock =
              enrichmentBlockBasisDataStorage.getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);

            const ValueTypeOperator *cumulativeClassicalBlockDofQuadPoints =
              basisDataInAllCellsClassicalBlock.data(); /*GLL Quad rule*/

            const ValueTypeOperator *cumulativeEnrichmentBlockDofQuadPoints =
              basisDataInAllCellsEnrichmentBlock.data() +
              cumulativeEnrichmentBlockDofQuadPointsOffset;

            for (unsigned int iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (unsigned int jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisOverlapTmpIter = 0.0;
                    // Ni_classical* Ni_classical of the classicalBlockBasisData
                    if (iNode < dofsPerCellCFE && jNode < dofsPerCellCFE)
                      {
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellClassicalBlock;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeClassicalBlockDofQuadPoints +
                                nQuadPointInCellClassicalBlock * iNode +
                                qPoint) *
                              *(cumulativeClassicalBlockDofQuadPoints +
                                nQuadPointInCellClassicalBlock * jNode +
                                qPoint) *
                              cellJxWValuesClassicalBlock[qPoint];
                          }
                      }

                    else if (iNode >= dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
                        // Ni_pristine*Ni_pristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlock;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeEnrichmentBlockDofQuadPoints +
                                nQuadPointInCellEnrichmentBlock * iNode +
                                qPoint) *
                              *(cumulativeEnrichmentBlockDofQuadPoints +
                                nQuadPointInCellEnrichmentBlock * jNode +
                                qPoint) *
                              cellJxWValuesEnrichmentBlock[qPoint];
                          }
                      }
                    basisOverlapTmpIter++;
                  }
              }

            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
            cumulativeEnrichmentBlockDofQuadPointsOffset +=
              nQuadPointInCellEnrichmentBlock * dofsPerCell;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(),
          NiNjInAllCells.data(),
          basisOverlapTmp.data());
      }

    } // namespace OrthoEFEOverlapInverseOpContextGLLInternal

    // Write M^-1 apply on a matrix for GLL with spectral finite element
    // M^-1 does not have a cell structure.

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    OrthoEFEOverlapInverseOpContextGLL<ValueTypeOperator,
                                       ValueTypeOperand,
                                       memorySpace,
                                       dim>::
      OrthoEFEOverlapInverseOpContextGLL(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
            &feBasisManager,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockGLLBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : d_feBasisManager(&feBasisManager)
      , d_linAlgOpContext(linAlgOpContext)
      , d_diagonalInv(d_feBasisManager->getMPIPatternP2P(), linAlgOpContext)
    {
      const size_type numLocallyOwnedCells =
        d_feBasisManager->nLocallyOwnedCells();

      const BasisDofHandler &basisDofHandler =
        feBasisManager.getBasisDofHandler();

      const EFEBasisDofHandler<ValueTypeOperand,
                               ValueTypeOperator,
                               memorySpace,
                               dim> &efebasisDofHandler =
        dynamic_cast<const EFEBasisDofHandler<ValueTypeOperand,
                                              ValueTypeOperator,
                                              memorySpace,
                                              dim> &>(basisDofHandler);
      utils::throwException(
        &efebasisDofHandler != nullptr,
        "Could not cast BasisDofHandler of the input to EFEBasisDofHandler.");

      d_efebasisDofHandler = &efebasisDofHandler;

      utils::throwException(
        efebasisDofHandler.isOrthogonalized(),
        "The Enrichment functions have to be orthogonalized for this class to do the application of overlap inverse.");

      utils::throwException(
        classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily() == dftefe::quadrature::QuadratureFamily::GLL,
        "The quadrature rule for integration of Classical FE dofs has to be GLL."
        "Contact developers if extra options are needed.");

      std::shared_ptr<
        const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
        cfeBDH = std::dynamic_pointer_cast<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
          classicalBlockGLLBasisDataStorage.getBasisDofHandler());
      utils::throwException(
        cfeBDH != nullptr,
        "Could not cast BasisDofHandler to FEBasisDofHandler "
        "in OrthoEFEOverlapInverseOperatorContext for the Classical data storage of classical dof block.");

      const EFEBasisDataStorage<ValueTypeOperator, memorySpace>
        &enrichmentBlockBasisDataStorageEFE = dynamic_cast<
          const EFEBasisDataStorage<ValueTypeOperator, memorySpace> &>(
          enrichmentBlockBasisDataStorage);
      utils::throwException(
        &enrichmentBlockBasisDataStorageEFE != nullptr,
        "Could not cast FEBasisDataStorage to EFEBasisDataStorage "
        "in EFEOverlapOperatorContext for enrichmentBlockBasisDataStorage.");

      std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                               ValueTypeOperator,
                                               memorySpace,
                                               dim>>
        efeBDH =
          std::dynamic_pointer_cast<const EFEBasisDofHandler<ValueTypeOperand,
                                                             ValueTypeOperator,
                                                             memorySpace,
                                                             dim>>(
            enrichmentBlockBasisDataStorage.getBasisDofHandler());
      utils::throwException(
        efeBDH != nullptr,
        "Could not cast BasisDofHandler to EFEBasisDofHandler "
        "in OrthoEFEOverlapInverseOperatorContext for the Enrichment data storage of enrichment dof blocks.");

      utils::throwException(
        cfeBDH->getTriangulation() == efeBDH->getTriangulation() &&
          cfeBDH->getFEOrder(0) == efeBDH->getFEOrder(0),
        "The EFEBasisDataStorage and and Classical FEBasisDataStorage have different triangulation or FEOrder"
        "in OrthoEFEOverlapInverseOperatorContext.");

      utils::throwException(
        &efebasisDofHandler == efeBDH.get(),
        "In OrthoEFEOverlapInverseOperatorContext the feBasisManager and enrichmentBlockBasisDataStorage should"
        "come from same basisDofHandler.");


      const size_type numCellClassicalDofs = utils::mathFunctions::sizeTypePow(
        (efebasisDofHandler.getFEOrder(0) + 1), dim);
      d_nglobalEnrichmentIds = efebasisDofHandler.nGlobalEnrichmentNodes();

      std::vector<size_type> numCellDofs(numLocallyOwnedCells, 0);
      for (size_type iCell = 0; iCell < numLocallyOwnedCells; ++iCell)
        numCellDofs[iCell] = d_feBasisManager->nLocallyOwnedCellDofs(iCell);

      auto itCellLocalIdsBegin =
        d_feBasisManager->locallyOwnedCellLocalDofIdsBegin();

      utils::MemoryStorage<ValueTypeOperator, memorySpace> NiNjInAllCells(0);

      OrthoEFEOverlapInverseOpContextGLLInternal::computeBasisOverlapMatrix<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(classicalBlockGLLBasisDataStorage,
             enrichmentBlockBasisDataStorage,
             cfeBDH,
             efeBDH,
             NiNjInAllCells);

      // // access cell-wise discrete Laplace operator
      // auto NiNjInAllCells =
      //   efeOverlapOperatorContext.getBasisOverlapInAllCells();

      std::vector<size_type> locallyOwnedCellsNumDoFsSTL(numLocallyOwnedCells,
                                                         0);
      std::copy(numCellDofs.begin(),
                numCellDofs.begin() + numLocallyOwnedCells,
                locallyOwnedCellsNumDoFsSTL.begin());

      utils::MemoryStorage<size_type, memorySpace> locallyOwnedCellsNumDoFs(
        numLocallyOwnedCells);
      locallyOwnedCellsNumDoFs.template copyFrom(locallyOwnedCellsNumDoFsSTL);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      // Create the diagonal of the classical block matrix which is diagonal for
      // GLL with spectral quadrature
      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(NiNjInAllCells.data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           diagonal.data());

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      d_feBasisManager->getConstraints().distributeChildToParent(diagonal, 1);

      // Function to add the values to the local node from its corresponding
      // ghost nodes from other processors.
      diagonal.accumulateAddLocallyOwned();

      diagonal.updateGhostValues();

      linearAlgebra::blasLapack::reciprocalX(diagonal.localSize(),
                                             1.0,
                                             diagonal.data(),
                                             d_diagonalInv.data(),
                                             *(diagonal.getLinAlgOpContext()));

      // Now form the enrichment block matrix.
      d_basisOverlapEnrichmentBlock =
        std::make_shared<utils::MemoryStorage<ValueTypeOperator, memorySpace>>(
          d_nglobalEnrichmentIds * d_nglobalEnrichmentIds);

      std::vector<ValueTypeOperator> basisOverlapEnrichmentBlockSTL(
        d_nglobalEnrichmentIds * d_nglobalEnrichmentIds, 0),
        basisOverlapEnrichmentBlockSTLTmp(d_nglobalEnrichmentIds *
                                            d_nglobalEnrichmentIds,
                                          0);

      size_type cellId                     = 0;
      size_type cumulativeBasisDataInCells = 0;
      for (auto enrichmentVecInCell :
           efebasisDofHandler.getEnrichmentIdsPartition()
             ->overlappingEnrichmentIdsInCells())
        {
          size_type nCellEnrichmentDofs = enrichmentVecInCell.size();
          for (unsigned int j = 0; j < nCellEnrichmentDofs; j++)
            {
              for (unsigned int k = 0; k < nCellEnrichmentDofs; k++)
                {
                  *(basisOverlapEnrichmentBlockSTLTmp.data() +
                    enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                    enrichmentVecInCell[k]) +=
                    *(NiNjInAllCells.data() + cumulativeBasisDataInCells +
                      (numCellClassicalDofs + nCellEnrichmentDofs) *
                        (numCellClassicalDofs + j) +
                      numCellClassicalDofs + k);
                }
            }
          cumulativeBasisDataInCells += utils::mathFunctions::sizeTypePow(
            (nCellEnrichmentDofs + numCellClassicalDofs), 2);
          cellId += 1;
        }

      int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        basisOverlapEnrichmentBlockSTLTmp.data(),
        basisOverlapEnrichmentBlockSTL.data(),
        basisOverlapEnrichmentBlockSTLTmp.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      // do inversion of enrichment block using slate lapackpp
      // utils::MemoryStorage<size_type, memorySpace>
      // ipiv(d_nglobalEnrichmentIds);

      linearAlgebra::blasLapack::inverse<ValueTypeOperator, memorySpace>(
        d_nglobalEnrichmentIds,
        basisOverlapEnrichmentBlockSTL.data(),
        *(d_diagonalInv.getLinAlgOpContext()));

      /* //do inversion of enrichment block using intel mkl
      EFEBlockInverse::inverse(basisOverlapEnrichmentBlockSTL.data(),
                                d_nglobalEnrichmentIds);*/

      d_basisOverlapEnrichmentBlock
        ->template copyFrom<utils::MemorySpace::HOST>(
          basisOverlapEnrichmentBlockSTL.data());
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    OrthoEFEOverlapInverseOpContextGLL<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const
    {
      const size_type numComponents = X.getNumberComponents();
      const size_type nlocallyOwnedEnrichmentIds =
        d_feBasisManager->getLocallyOwnedRanges()[1].second -
        d_feBasisManager->getLocallyOwnedRanges()[1].first;
      const size_type nlocallyOwnedClassicalIds =
        d_feBasisManager->getLocallyOwnedRanges()[0].second -
        d_feBasisManager->getLocallyOwnedRanges()[0].first;

      X.updateGhostValues();
      // update the child nodes based on the parent nodes
      d_feBasisManager->getConstraints().distributeParentToChild(
        X, X.getNumberComponents());

      Y.setValue(0.0);

      linearAlgebra::blasLapack::khatriRaoProduct(
        linearAlgebra::blasLapack::Layout::ColMajor,
        1,
        numComponents,
        d_diagonalInv.localSize(),
        d_diagonalInv.data(),
        X.begin(),
        Y.begin(),
        *(d_diagonalInv.getLinAlgOpContext()));

      utils::MemoryStorage<ValueTypeOperand, memorySpace> XenrichedGlobalVec(
        d_nglobalEnrichmentIds * numComponents),
        XenrichedGlobalVecTmp(d_nglobalEnrichmentIds * numComponents),
        YenrichedGlobalVec(d_nglobalEnrichmentIds * numComponents);

      XenrichedGlobalVecTmp.template copyFrom<memorySpace>(
        X.begin(),
        nlocallyOwnedEnrichmentIds * numComponents,
        nlocallyOwnedClassicalIds * numComponents,
        ((d_feBasisManager->getLocallyOwnedRanges()[1].first) -
         (d_efebasisDofHandler->getGlobalRanges()[0].second)) *
          numComponents);

      int err = utils::mpi::MPIAllreduce<memorySpace>(
        XenrichedGlobalVecTmp.data(),
        XenrichedGlobalVec.data(),
        XenrichedGlobalVecTmp.size(),
        utils::mpi::MPIDouble,
        utils::mpi::MPISum,
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(err);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      // Do dgemm

      ValueType alpha = 1.0;
      ValueType beta  = 0.0;

      linearAlgebra::blasLapack::
        gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
          linearAlgebra::blasLapack::Layout::ColMajor,
          linearAlgebra::blasLapack::Op::NoTrans,
          linearAlgebra::blasLapack::Op::Trans,
          numComponents,
          d_nglobalEnrichmentIds,
          d_nglobalEnrichmentIds,
          alpha,
          XenrichedGlobalVec.data(),
          numComponents,
          d_basisOverlapEnrichmentBlock->data(),
          d_nglobalEnrichmentIds,
          beta,
          YenrichedGlobalVec.begin(),
          numComponents,
          *d_linAlgOpContext);

      YenrichedGlobalVec.template copyTo<memorySpace>(
        Y.begin(),
        nlocallyOwnedEnrichmentIds * numComponents,
        ((d_feBasisManager->getLocallyOwnedRanges()[1].first) -
         (d_efebasisDofHandler->getGlobalRanges()[0].second)) *
          numComponents,
        nlocallyOwnedClassicalIds * numComponents);

      Y.updateGhostValues();

      // function to do a static condensation to send the constraint nodes to
      // its parent nodes
      d_feBasisManager->getConstraints().distributeChildToParent(
        Y, Y.getNumberComponents());

      // Function to update the ghost values of the Y
      Y.updateGhostValues();
    }
  } // namespace basis
} // namespace dftefe
