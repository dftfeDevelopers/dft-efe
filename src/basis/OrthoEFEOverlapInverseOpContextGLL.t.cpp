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
#include <utils/ConditionalOStream.h>
namespace dftefe
{
  namespace basis
  {
    namespace OrthoEFEOverlapInverseOpContextGLLInternal
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
                                      dim> &         feBasisManager,
          const OrthoEFEOverlapOperatorContext<ValueTypeOperator,
                                               ValueTypeOperand,
                                               memorySpace,
                                               dim> &MContext,
          std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                          linAlgOpContext,
          const size_type maxCellTimesNumVecs)
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

      }; // end of class PoissonLinearSolverFunctionFE

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
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
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
          &basisDataInAllCellsEnrichmentBlockEnrichment =
            enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInAllCells();

        size_type cumulativeEnrichmentBlockEnrichmentDofQuadPointsOffset = 0;

        locallyOwnedCellIter = efeBDH->beginLocallyOwnedCells();


        const std::unordered_map<global_size_type,
                                 utils::OptimizedIndexSet<size_type>>
          *enrichmentIdToClassicalLocalIdMap =
            &eci->getClassicalComponentLocalIdsMap();
        const std::unordered_map<global_size_type,
                                 std::vector<ValueTypeOperator>>
          *enrichmentIdToInterfaceCoeffMap =
            &eci->getClassicalComponentCoeffMap();

        std::shared_ptr<const FEBasisManager<ValueTypeOperator,
                                             ValueTypeOperator,
                                             memorySpace,
                                             dim>>
          cfeBasisManager =
            std::dynamic_pointer_cast<const FEBasisManager<ValueTypeOperator,
                                                           ValueTypeOperator,
                                                           memorySpace,
                                                           dim>>(
              eci->getCFEBasisManager());


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

            size_type nQuadPointInCellEnrichmentBlockEnrichment =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEnrichmentBlockEnrichment =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);


            size_type nQuadPointInCellEnrichmentBlockClassical =
              enrichmentBlockClassicalBasisDataStorage
                .getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellIndex);
            std::vector<double> cellJxWValuesEnrichmentBlockClassical =
              enrichmentBlockClassicalBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellJxW(cellIndex);


            const ValueTypeOperator *cumulativeClassicalBlockDofQuadPoints =
              basisDataInAllCellsClassicalBlock.data(); /*GLL Quad rule*/

            const ValueTypeOperator
              *cumulativeEnrichmentBlockEnrichmentDofQuadPoints =
                basisDataInAllCellsEnrichmentBlockEnrichment.data() +
                cumulativeEnrichmentBlockEnrichmentDofQuadPointsOffset;


            std::vector<utils::Point> quadRealPointsVec =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellRealPoints(cellIndex);

            std::vector<size_type> vecClassicalLocalNodeId(0);

            size_type numEnrichmentIdsInCell = dofsPerCell - dofsPerCellCFE;

            std::vector<ValueTypeOperator> classicalComponentInQuadValuesEC(0);

            classicalComponentInQuadValuesEC.resize(
              nQuadPointInCellEnrichmentBlockClassical * numEnrichmentIdsInCell,
              (ValueTypeOperator)0);

            std::vector<ValueTypeOperator> classicalComponentInQuadValuesEE(0);

            classicalComponentInQuadValuesEE.resize(
              nQuadPointInCellEnrichmentBlockEnrichment *
                numEnrichmentIdsInCell,
              (ValueTypeOperator)0);

            if (numEnrichmentIdsInCell > 0)
              {
                cfeBasisManager->getCellDofsLocalIds(cellIndex,
                                                     vecClassicalLocalNodeId);

                std::vector<ValueTypeOperator> coeffsInCell(
                  dofsPerCellCFE * numEnrichmentIdsInCell, 0);

                for (size_type cellEnrichId = 0;
                     cellEnrichId < numEnrichmentIdsInCell;
                     cellEnrichId++)
                  {
                    // get the enrichmentIds
                    global_size_type enrichmentId =
                      eci->getEnrichmentId(cellIndex, cellEnrichId);

                    // get the vectors of non-zero localIds and coeffs
                    auto iter =
                      enrichmentIdToInterfaceCoeffMap->find(enrichmentId);
                    auto it =
                      enrichmentIdToClassicalLocalIdMap->find(enrichmentId);
                    if (iter != enrichmentIdToInterfaceCoeffMap->end() &&
                        it != enrichmentIdToClassicalLocalIdMap->end())
                      {
                        const std::vector<ValueTypeOperator>
                          &coeffsInLocalIdsMap = iter->second;

                        for (size_type i = 0; i < dofsPerCellCFE; i++)
                          {
                            size_type pos   = 0;
                            bool      found = false;
                            it->second.getPosition(vecClassicalLocalNodeId[i],
                                                   pos,
                                                   found);
                            if (found)
                              {
                                coeffsInCell[numEnrichmentIdsInCell * i +
                                             cellEnrichId] =
                                  coeffsInLocalIdsMap[pos];
                              }
                          }
                      }
                  }

                utils::MemoryStorage<ValueTypeOperator,
                                     utils::MemorySpace::HOST>
                  basisValInCellEC =
                    enrichmentBlockClassicalBasisDataStorage.getBasisDataInCell(
                      cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::Trans,
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockClassical,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  basisValInCellEC.data(),
                  nQuadPointInCellEnrichmentBlockClassical,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEC.data(),
                  numEnrichmentIdsInCell,
                  *eci->getLinAlgOpContext());

                utils::MemoryStorage<ValueTypeOperator,
                                     utils::MemorySpace::HOST>
                  basisValInCellEE = enrichmentBlockEnrichmentBasisDataStorage
                                       .getBasisDataInCell(cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector

                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::Op::NoTrans,
                  linearAlgebra::blasLapack::Op::Trans,
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockEnrichment,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  basisValInCellEE.data(),
                  nQuadPointInCellEnrichmentBlockEnrichment,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEE.data(),
                  numEnrichmentIdsInCell,
                  *eci->getLinAlgOpContext());
              }


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
                        /**
                        // Ni_pristine*Ni_pristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            *basisOverlapTmpIter +=
                              *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints
                        + nQuadPointInCellEnrichmentBlockEnrichment * iNode +
                                qPoint) *
                              *(cumulativeEnrichmentBlockEnrichmentDofQuadPoints
                        + nQuadPointInCellEnrichmentBlockEnrichment * jNode +
                                qPoint) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                          **/

                        ValueTypeOperator NpiNpj     = (ValueTypeOperator)0,
                                          ciNciNpj   = (ValueTypeOperator)0,
                                          NpicjNcj   = (ValueTypeOperator)0,
                                          ciNcicjNcj = (ValueTypeOperator)0;
                        // Ni_pristine*Ni_pristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NpiNpj +=
                              efeBDH->getEnrichmentValue(
                                cellIndex,
                                iNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              efeBDH->getEnrichmentValue(
                                cellIndex,
                                jNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // Ni_pristine* interpolated ci's in
                        // Ni_classicalQuadratureOfPristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            ciNciNpj +=
                              classicalComponentInQuadValuesEE
                                [numEnrichmentIdsInCell * qPoint +
                                 (iNode - dofsPerCellCFE)] *
                              efeBDH->getEnrichmentValue(
                                cellIndex,
                                jNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // Ni_pristine* interpolated ci's in
                        // Ni_classicalQuadratureOfPristine at quadpoints
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockEnrichment;
                             qPoint++)
                          {
                            NpicjNcj +=
                              efeBDH->getEnrichmentValue(
                                cellIndex,
                                iNode - dofsPerCellCFE,
                                quadRealPointsVec[qPoint]) *
                              classicalComponentInQuadValuesEE
                                [numEnrichmentIdsInCell * qPoint +
                                 (jNode - dofsPerCellCFE)] *
                              cellJxWValuesEnrichmentBlockEnrichment[qPoint];
                          }
                        // interpolated ci's in Ni_classicalQuadrature of Mc = d
                        // * interpolated ci's in Ni_classicalQuadrature of Mc =
                        // d
                        for (unsigned int qPoint = 0;
                             qPoint < nQuadPointInCellEnrichmentBlockClassical;
                             qPoint++)
                          {
                            ciNcicjNcj +=
                              classicalComponentInQuadValuesEC
                                [numEnrichmentIdsInCell * qPoint +
                                 (iNode - dofsPerCellCFE)] *
                              classicalComponentInQuadValuesEC
                                [numEnrichmentIdsInCell * qPoint +
                                 (jNode - dofsPerCellCFE)] *
                              cellJxWValuesEnrichmentBlockClassical[qPoint];
                          }
                        *basisOverlapTmpIter +=
                          NpiNpj - NpicjNcj - ciNciNpj + ciNcicjNcj;
                      }
                    basisOverlapTmpIter++;
                  }
              }

            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            cellIndex++;
            cumulativeEnrichmentBlockEnrichmentDofQuadPointsOffset +=
              nQuadPointInCellEnrichmentBlockEnrichment * dofsPerCell;
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
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : d_feBasisManager(&feBasisManager)
      , d_linAlgOpContext(linAlgOpContext)
      , d_diagonalInv(d_feBasisManager->getMPIPatternP2P(), linAlgOpContext)
      , d_isCGSolved(false)
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
            .getQuadratureFamily() == quadrature::QuadratureFamily::GLL,
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
          enrichmentBlockEnrichmentBasisDataStorage);
      utils::throwException(
        &enrichmentBlockBasisDataStorageEFE != nullptr,
        "Could not cast FEBasisDataStorage to EFEBasisDataStorage "
        "in EFEOverlapOperatorContext for enrichmentBlockEnrichmentBasisDataStorage.");

      std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                               ValueTypeOperator,
                                               memorySpace,
                                               dim>>
        efeBDH =
          std::dynamic_pointer_cast<const EFEBasisDofHandler<ValueTypeOperand,
                                                             ValueTypeOperator,
                                                             memorySpace,
                                                             dim>>(
            enrichmentBlockEnrichmentBasisDataStorage.getBasisDofHandler());
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
        "In OrthoEFEOverlapInverseOperatorContext the feBasisManager and enrichmentBlockEnrichmentBasisDataStorage should"
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
             enrichmentBlockEnrichmentBasisDataStorage,
             enrichmentBlockClassicalBasisDataStorage,
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
      /**
      int rank;
      utils::mpi::MPICommRank(
        d_feBasisManager->getMPIPatternP2P()->mpiCommunicator(), &rank);

      utils::ConditionalOStream rootCout(std::cout);
      rootCout.setCondition(rank == 0);

      rootCout << "Enrichment Block Matrix: " << std::endl;
      for (size_type i = 0; i < d_nglobalEnrichmentIds; i++)
        {
          rootCout << "[";
          for (size_type j = 0; j < d_nglobalEnrichmentIds; j++)
            {
              rootCout << *(basisOverlapEnrichmentBlockSTL.data() +
                            i * d_nglobalEnrichmentIds + j)
                       << "\t";
            }
          rootCout << "]" << std::endl;
        }

      linearAlgebra::blasLapack::inverse<ValueTypeOperator, memorySpace>(
        d_nglobalEnrichmentIds,
        basisOverlapEnrichmentBlockSTL.data(),
        *(d_diagonalInv.getLinAlgOpContext()));

      rootCout << "Enrichment Block Inverse Matrix: " << std::endl;
      for (size_type i = 0; i < d_nglobalEnrichmentIds; i++)
        {
          rootCout << "[";
          for (size_type j = 0; j < d_nglobalEnrichmentIds; j++)
            {
              rootCout << *(basisOverlapEnrichmentBlockSTL.data() +
                            i * d_nglobalEnrichmentIds + j)
                       << "\t";
            }
          rootCout << "]" << std::endl;
        }
      **/
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
    OrthoEFEOverlapInverseOpContextGLL<ValueTypeOperator,
                                       ValueTypeOperand,
                                       memorySpace,
                                       dim>::
      OrthoEFEOverlapInverseOpContextGLL(
        const basis::
          FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
            &                                      feBasisManager,
        const OrthoEFEOverlapOperatorContext<ValueTypeOperator,
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
            OrthoEFEOverlapInverseOpContextGLLInternal::
              OverlapMatrixInverseLinearSolverFunctionFE<ValueTypeOperator,
                                                         ValueTypeOperand,
                                                         memorySpace,
                                                         dim>>(
            *d_feBasisManager, MContext, linAlgOpContext, 50);

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
    OrthoEFEOverlapInverseOpContextGLL<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<ValueType, memorySpace> &Y) const
    {
      if (!d_isCGSolved)
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

          utils::MemoryStorage<ValueTypeOperand, memorySpace>
            XenrichedGlobalVec(d_nglobalEnrichmentIds * numComponents),
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

          // function to do a static condensation to send the constraint nodes
          // to its parent nodes
          d_feBasisManager->getConstraints().distributeChildToParent(
            Y, Y.getNumberComponents());

          // Function to update the ghost values of the Y
          Y.updateGhostValues();
        }
      else
        {
          std::shared_ptr<
            OrthoEFEOverlapInverseOpContextGLLInternal::
              OverlapMatrixInverseLinearSolverFunctionFE<ValueTypeOperator,
                                                         ValueTypeOperand,
                                                         memorySpace,
                                                         dim>>
            overlapInvPoisson = std::dynamic_pointer_cast<
              OrthoEFEOverlapInverseOpContextGLLInternal::
                OverlapMatrixInverseLinearSolverFunctionFE<ValueTypeOperator,
                                                           ValueTypeOperand,
                                                           memorySpace,
                                                           dim>>(
              d_overlapInvPoisson);
          Y.setValue(0.0);
          overlapInvPoisson->reinit(X);
          d_CGSolve->solve(*overlapInvPoisson);
          overlapInvPoisson->getSolution(Y);
        }
    }
  } // namespace basis
} // namespace dftefe
