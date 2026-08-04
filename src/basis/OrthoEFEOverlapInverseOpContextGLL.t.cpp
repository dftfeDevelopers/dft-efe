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
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>> ccfeBDH,
        std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                                 ValueTypeOperator,
                                                 memorySpace,
                                                 dim>>                 eefeBDH,
        utils::MemoryStorage<ValueTypeOperator, memorySpace> &basisOverlap)
      {
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>
          &linAlgOpContext =
            *linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST;

        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          ecfeBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            enrichmentBlockClassicalBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          ecfeBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in OrthoEFEOverlapOperatorContext for the Classical data storage of enrichment dof blocks.");

        std::shared_ptr<
          const EnrichmentClassicalInterfaceSpherical<ValueTypeOperator,
                                                      memorySpace,
                                                      dim>>
          eci = eefeBDH->getEnrichmentClassicalInterface();

        size_type nTotalEnrichmentIds =
          eci->getEnrichmentIdsPartition()->nTotalEnrichmentIds();

        // Set up the overlap matrix quadrature storages.

        const size_type numLocallyOwnedCells = eefeBDH->nLocallyOwnedCells();
        std::vector<size_type> dofsInCellVec(0);
        dofsInCellVec.resize(numLocallyOwnedCells, 0);
        size_type cumulativeBasisOverlapId = 0;

        size_type       basisOverlapSize = 0;
        size_type       cellId           = 0;
        const size_type feOrder          = eefeBDH->getFEOrder(cellId);

        size_type       dofsPerCell;
        const size_type dofsPerCellCFE = ccfeBDH->nCellDofs(cellId);

        auto      locallyOwnedCellIter = eefeBDH->beginLocallyOwnedCells();
        size_type numCumulativeDofsxQuadEFEInAllCells       = 0;
        size_type numCumulativeEnrichDofsxQuadEFEInAllCells = 0;

        for (; locallyOwnedCellIter != eefeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = eefeBDH->nCellDofs(cellId);
            numCumulativeDofsxQuadEFEInAllCells +=
              dofsInCellVec[cellId] * enrichmentBlockEnrichmentBasisDataStorage
                                        .getQuadratureRuleContainer()
                                        ->nCellQuadraturePoints(cellId);
            numCumulativeEnrichDofsxQuadEFEInAllCells +=
              (dofsInCellVec[cellId] - dofsPerCellCFE) *
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellId);
            basisOverlapSize += dofsInCellVec[cellId] * dofsInCellVec[cellId];
            cellId++;
          }

        std::vector<ValueTypeOperator> basisOverlapTmp(0);

        basisOverlap.resize(basisOverlapSize, ValueTypeOperator(0));
        basisOverlapTmp.resize(basisOverlapSize, ValueTypeOperator(0));

        auto      basisOverlapTmpIter = basisOverlapTmp.begin();
        size_type cellIndex           = 0;

        locallyOwnedCellIter = eefeBDH->beginLocallyOwnedCells();

        size_type cumulativeDofQuadPointsOffsetCFE            = 0,
                  cumulativeDofQuadPointsOffsetEnrichBlockCFE = 0,
                  cumulativeDofQuadPointsOffsetEnrichBlockEFE = 0;

        bool isConstantDofsAndQuadPointsInCellCFE = false;
        quadrature::QuadratureFamily quadFamily =
          classicalBlockGLLBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        if ((quadFamily == quadrature::QuadratureFamily::GAUSS ||
             quadFamily == quadrature::QuadratureFamily::GLL ||
             quadFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED) &&
            !ccfeBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCellCFE = true;

        bool isConstantDofsAndQuadPointsInCellEnrichBlockCFE = false;
        quadrature::QuadratureFamily quadFamily1 =
          enrichmentBlockClassicalBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes()
            .getQuadratureFamily();
        if ((quadFamily1 == quadrature::QuadratureFamily::GAUSS ||
             quadFamily1 == quadrature::QuadratureFamily::GLL ||
             quadFamily1 == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED) &&
            !ecfeBDH->isVariableDofsPerCell())
          isConstantDofsAndQuadPointsInCellEnrichBlockCFE = true;

        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsClassicalBlock =
            classicalBlockGLLBasisDataStorage.getBasisDataInAllCells();
        const utils::MemoryStorage<ValueTypeOperator, memorySpace>
          &basisDataInAllCellsEnrichmentBlockClassical =
            enrichmentBlockClassicalBasisDataStorage.getBasisDataInAllCells();

        utils::MemoryStorage<ValueTypeOperator, memorySpace>
          basisDataInAllCellsEnrichmentBlockEnrichment(
            numCumulativeDofsxQuadEFEInAllCells);
        std::pair<size_type, size_type> cellPair(0, numLocallyOwnedCells);
        enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInCellRange(
          cellPair, basisDataInAllCellsEnrichmentBlockEnrichment);

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsClassicalBlockHost(
            basisDataInAllCellsClassicalBlock.size());

        basisDataInAllCellsClassicalBlockHost.copyFrom(
          basisDataInAllCellsClassicalBlock);

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsEnrichmentBlockClassicalHost(
            basisDataInAllCellsEnrichmentBlockClassical.size());

        basisDataInAllCellsEnrichmentBlockClassicalHost.copyFrom(
          basisDataInAllCellsEnrichmentBlockClassical);

        utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
          basisDataInAllCellsEnrichmentBlockEnrichmentHost(
            basisDataInAllCellsEnrichmentBlockEnrichment.size());

        basisDataInAllCellsEnrichmentBlockEnrichmentHost.copyFrom(
          basisDataInAllCellsEnrichmentBlockEnrichment);

        utils::MemoryStorage<double, memorySpace>
          quadValuesInAllCellsEnrichmentMemSpace(
            numCumulativeEnrichDofsxQuadEFEInAllCells);
        eefeBDH->getEnrichmentClassicalInterface()
          ->getEnrichmentValuesInCellRangeAtQuadPts(
            *enrichmentBlockEnrichmentBasisDataStorage
               .getQuadratureRuleContainer(),
            quadValuesInAllCellsEnrichmentMemSpace.data(),
            *eefeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext(),
            std::make_pair((size_type)0, numLocallyOwnedCells));
        std::vector<double> quadValuesInAllCellsEnrichment(
          numCumulativeEnrichDofsxQuadEFEInAllCells);
        utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
          numCumulativeEnrichDofsxQuadEFEInAllCells,
          quadValuesInAllCellsEnrichment.data(),
          quadValuesInAllCellsEnrichmentMemSpace.data());

        size_type cumulativeQuadEnrichBlockEnrichxenrichInCell = 0;

        for (; locallyOwnedCellIter != eefeBDH->endLocallyOwnedCells();
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
              basisDataInAllCellsClassicalBlockHost.data() +
              cumulativeDofQuadPointsOffsetCFE; /*GLL Quad rule*/

            // const ValueTypeOperator
            //   *cumulativeEnrichmentBlockEnrichmentDofQuadPoints =
            //     basisDataInCellEnrichmentBlockEnrichment.data();


            std::vector<utils::Point> quadRealPointsVec =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellRealPoints(cellIndex);

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
                std::vector<ValueTypeOperator> coeffsInCell(
                  dofsPerCellCFE * numEnrichmentIdsInCell, 0);

                coeffsInCell =
                  eefeBDH->getEnrichmentClassicalInterface()
                    ->getClassicalComponentCoeffsInCellOEFE(cellIndex);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector
                ValueTypeOperator *B =
                  basisDataInAllCellsEnrichmentBlockClassicalHost.data() +
                  cumulativeDofQuadPointsOffsetEnrichBlockCFE;
                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'N',
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockClassical,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  B,
                  dofsPerCellCFE,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEC.data(),
                  numEnrichmentIdsInCell,
                  linAlgOpContext);

                // Do a gemm (\Sigma c_i N_i^classical)
                // and get the quad values in std::vector
                B = basisDataInAllCellsEnrichmentBlockEnrichmentHost.data() +
                    cumulativeDofQuadPointsOffsetEnrichBlockEFE;
                linearAlgebra::blasLapack::gemm<ValueTypeOperator,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'N',
                  numEnrichmentIdsInCell,
                  nQuadPointInCellEnrichmentBlockEnrichment,
                  dofsPerCellCFE,
                  (ValueTypeOperator)1.0,
                  coeffsInCell.data(),
                  numEnrichmentIdsInCell,
                  B,
                  dofsPerCell,
                  (ValueTypeOperator)0.0,
                  classicalComponentInQuadValuesEE.data(),
                  numEnrichmentIdsInCell,
                  linAlgOpContext);
              }

            std::vector<ValueTypeOperator> basisOverlapClassicalBlock(
              dofsPerCellCFE * dofsPerCellCFE);
            std::vector<ValueTypeOperator> JxWxNCell(
              dofsPerCellCFE * nQuadPointInCellClassicalBlock, 0);

            size_type stride = 0;
            size_type m = 1, n = dofsPerCellCFE,
                      k = nQuadPointInCellClassicalBlock;

            linearAlgebra::blasLapack::scaleStridedVarBatched<
              ValueTypeOperator,
              ValueTypeOperator,
              utils::MemorySpace::HOST>(
              1,
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              &stride,
              &stride,
              &stride,
              &m,
              &n,
              &k,
              cellJxWValuesClassicalBlock.data(),
              cumulativeClassicalBlockDofQuadPoints,
              JxWxNCell.data(),
              linAlgOpContext);

            linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                            ValueTypeOperand,
                                            utils::MemorySpace::HOST>(
              'N',
              'C',
              n,
              n,
              k,
              (ValueTypeOperand)1.0,
              JxWxNCell.data(),
              n,
              cumulativeClassicalBlockDofQuadPoints,
              n,
              (ValueTypeOperand)0.0,
              basisOverlapClassicalBlock.data(),
              n,
              linAlgOpContext);


            std::vector<ValueTypeOperator> basisOverlapEEBlock1(
              numEnrichmentIdsInCell * numEnrichmentIdsInCell, 0),
              basisOverlapEEBlock2(numEnrichmentIdsInCell *
                                     numEnrichmentIdsInCell,
                                   0),
              basisOverlapEEBlock3(numEnrichmentIdsInCell *
                                     numEnrichmentIdsInCell,
                                   0);

            if (numEnrichmentIdsInCell > 0)
              {
                // Ni_pristine*Ni_pristine at quadpoints
                JxWxNCell.resize(numEnrichmentIdsInCell *
                                   nQuadPointInCellEnrichmentBlockEnrichment,
                                 0);

                m = 1, n = numEnrichmentIdsInCell,
                k = nQuadPointInCellEnrichmentBlockEnrichment;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  utils::MemorySpace::HOST>(
                  1,
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::ScalarOp::Identity,
                  linearAlgebra::blasLapack::ScalarOp::Identity,
                  &stride,
                  &stride,
                  &stride,
                  &m,
                  &n,
                  &k,
                  cellJxWValuesEnrichmentBlockEnrichment.data(),
                  quadValuesInAllCellsEnrichment.data() +
                    cumulativeQuadEnrichBlockEnrichxenrichInCell,
                  JxWxNCell.data(),
                  linAlgOpContext);

                linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                                ValueTypeOperand,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'C',
                  n,
                  n,
                  k,
                  (ValueTypeOperand)1.0,
                  JxWxNCell.data(),
                  n,
                  quadValuesInAllCellsEnrichment.data() +
                    cumulativeQuadEnrichBlockEnrichxenrichInCell,
                  n,
                  (ValueTypeOperand)0.0,
                  basisOverlapEEBlock1.data(),
                  n,
                  linAlgOpContext);

                // interpolated ci's in Ni_classicalQuadrature of Mc = d
                // * interpolated ci's in Ni_classicalQuadrature of Mc =
                // d
                JxWxNCell.resize(numEnrichmentIdsInCell *
                                   nQuadPointInCellEnrichmentBlockClassical,
                                 0);

                m = 1, n = numEnrichmentIdsInCell,
                k = nQuadPointInCellEnrichmentBlockClassical;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  utils::MemorySpace::HOST>(
                  1,
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::ScalarOp::Identity,
                  linearAlgebra::blasLapack::ScalarOp::Identity,
                  &stride,
                  &stride,
                  &stride,
                  &m,
                  &n,
                  &k,
                  cellJxWValuesEnrichmentBlockClassical.data(),
                  classicalComponentInQuadValuesEC.data(),
                  JxWxNCell.data(),
                  linAlgOpContext);

                linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'C',
                  n,
                  n,
                  k,
                  (ValueTypeOperand)1.0,
                  JxWxNCell.data(),
                  n,
                  classicalComponentInQuadValuesEC.data(),
                  n,
                  (ValueTypeOperator)0.0,
                  basisOverlapEEBlock2.data(),
                  n,
                  linAlgOpContext);

                // Ni_pristine* interpolated ci's in
                // Ni_classicalQuadratureOfPristine at quadpoints

                JxWxNCell.resize(numEnrichmentIdsInCell *
                                   nQuadPointInCellEnrichmentBlockEnrichment,
                                 0);

                m = 1, n = numEnrichmentIdsInCell,
                k = nQuadPointInCellEnrichmentBlockEnrichment;

                linearAlgebra::blasLapack::scaleStridedVarBatched<
                  ValueTypeOperator,
                  ValueTypeOperator,
                  utils::MemorySpace::HOST>(
                  1,
                  linearAlgebra::blasLapack::Layout::ColMajor,
                  linearAlgebra::blasLapack::ScalarOp::Identity,
                  linearAlgebra::blasLapack::ScalarOp::Identity,
                  &stride,
                  &stride,
                  &stride,
                  &m,
                  &n,
                  &k,
                  cellJxWValuesEnrichmentBlockEnrichment.data(),
                  quadValuesInAllCellsEnrichment.data() +
                    cumulativeQuadEnrichBlockEnrichxenrichInCell,
                  JxWxNCell.data(),
                  linAlgOpContext);

                linearAlgebra::blasLapack::gemm<ValueTypeOperand,
                                                ValueTypeOperator,
                                                utils::MemorySpace::HOST>(
                  'N',
                  'C',
                  n,
                  n,
                  k,
                  (ValueTypeOperand)1.0,
                  classicalComponentInQuadValuesEE.data(),
                  n,
                  JxWxNCell.data(),
                  n,
                  (ValueTypeOperator)0.0,
                  basisOverlapEEBlock3.data(),
                  n,
                  linAlgOpContext);
              }

            for (size_type iNode = 0; iNode < dofsPerCell; iNode++)
              {
                for (size_type jNode = 0; jNode < dofsPerCell; jNode++)
                  {
                    *basisOverlapTmpIter = 0.0;
                    // Ni_classical* Ni_classical of the classicalBlockBasisData
                    if (iNode < dofsPerCellCFE && jNode < dofsPerCellCFE)
                      {
                        *basisOverlapTmpIter =
                          *(basisOverlapClassicalBlock.data() +
                            iNode * dofsPerCellCFE + jNode);
                      }

                    else if (iNode >= dofsPerCellCFE && jNode >= dofsPerCellCFE)
                      {
                        *basisOverlapTmpIter =
                          *(basisOverlapEEBlock1.data() +
                            (iNode - dofsPerCellCFE) * numEnrichmentIdsInCell +
                            (jNode - dofsPerCellCFE)) +
                          *(basisOverlapEEBlock2.data() +
                            (iNode - dofsPerCellCFE) * numEnrichmentIdsInCell +
                            (jNode - dofsPerCellCFE)) -
                          *(basisOverlapEEBlock3.data() +
                            (iNode - dofsPerCellCFE) * numEnrichmentIdsInCell +
                            (jNode - dofsPerCellCFE)) -
                          *(basisOverlapEEBlock3.data() +
                            (jNode - dofsPerCellCFE) * numEnrichmentIdsInCell +
                            (iNode - dofsPerCellCFE));
                      }
                    basisOverlapTmpIter++;
                  }
              }

            cumulativeBasisOverlapId += dofsPerCell * dofsPerCell;
            if (!isConstantDofsAndQuadPointsInCellCFE)
              cumulativeDofQuadPointsOffsetCFE +=
                nQuadPointInCellClassicalBlock * dofsPerCellCFE;
            if (!isConstantDofsAndQuadPointsInCellEnrichBlockCFE)
              cumulativeDofQuadPointsOffsetEnrichBlockCFE +=
                nQuadPointInCellEnrichmentBlockClassical * dofsPerCellCFE;
            cumulativeDofQuadPointsOffsetEnrichBlockEFE +=
              nQuadPointInCellEnrichmentBlockEnrichment * dofsPerCell;
            cumulativeQuadEnrichBlockEnrichxenrichInCell +=
              numEnrichmentIdsInCell *
              nQuadPointInCellEnrichmentBlockEnrichment;
            cellIndex++;
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          basisOverlapTmp.size(), basisOverlap.data(), basisOverlapTmp.data());
      }

      // Use this for data storage of orthogonalized EFE only
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      computeBasisOverlapMatrixBlocked(
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &classicalBlockBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorage,
        const FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockClassicalBasisDataStorage,
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>> ccfeBDH,
        std::shared_ptr<const EFEBasisDofHandler<ValueTypeOperand,
                                                 ValueTypeOperator,
                                                 memorySpace,
                                                 dim>>                 eefeBDH,
        utils::MemoryStorage<ValueTypeOperator, memorySpace> &basisOverlap,
        const size_type                                       cellBlockSize)
      {
        std::shared_ptr<
          const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>
          ecfeBDH = std::dynamic_pointer_cast<
            const FEBasisDofHandler<ValueTypeOperand, memorySpace, dim>>(
            enrichmentBlockClassicalBasisDataStorage.getBasisDofHandler());
        utils::throwException(
          ecfeBDH != nullptr,
          "Could not cast BasisDofHandler to FEBasisDofHandler "
          "in OrthoEFEOverlapOperatorContext for the Classical data storage of enrichment dof blocks.");

        linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext =
          *eefeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext();

        utils::throwException(
          ccfeBDH->getTriangulation() == ecfeBDH->getTriangulation() &&
            ccfeBDH->getFEOrder(0) == ecfeBDH->getFEOrder(0),
          "The EFEBasisDataStorage and and Classical FEBasisDataStorage have different triangulation or FEOrder"
          "in OrthoEFEOverlapOperatorContext.");

        // interpolate the ci 's to the enrichment quadRuleAttr quadpoints

        const EFEBasisDataStorage<ValueTypeOperator, memorySpace>
          &enrichmentBlockEnrichmentBasisDataStorageEFE = dynamic_cast<
            const EFEBasisDataStorage<ValueTypeOperator, memorySpace> &>(
            enrichmentBlockEnrichmentBasisDataStorage);
        utils::throwException(
          &enrichmentBlockEnrichmentBasisDataStorageEFE != nullptr,
          "Could not cast FEBasisDataStorage to EFEBasisDataStorage "
          "in OrthoEFEOverlapOperatorContext for enrichmentBlockEnrichmentBasisDataStorage.");

        // Set up the overlap matrix quadrature storages.

        const size_type numLocallyOwnedCells = eefeBDH->nLocallyOwnedCells();
        size_type       numCumulativeEnrichDofsxQuadEFEInAllCells = 0;
        std::vector<size_type> dofsInCellVec(0);
        dofsInCellVec.resize(numLocallyOwnedCells, 0);

        size_type basisOverlapSize = 0;
        size_type cellId           = 0;

        const size_type dofsPerCellCFE = ccfeBDH->nCellDofs(cellId);

        auto locallyOwnedCellIter = eefeBDH->beginLocallyOwnedCells();
        for (; locallyOwnedCellIter != eefeBDH->endLocallyOwnedCells();
             ++locallyOwnedCellIter)
          {
            dofsInCellVec[cellId] = eefeBDH->nCellDofs(cellId);
            basisOverlapSize += dofsInCellVec[cellId] * dofsInCellVec[cellId];
            numCumulativeEnrichDofsxQuadEFEInAllCells +=
              (dofsInCellVec[cellId] - dofsPerCellCFE) *
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->nCellQuadraturePoints(cellId);
            cellId++;
          }

        basisOverlap.resize(basisOverlapSize, ValueTypeOperator(0));

        size_type cellIndex = 0;

        utils::MemoryStorage<double, memorySpace>
          quadValuesInAllCellsEnrichment(
            numCumulativeEnrichDofsxQuadEFEInAllCells);
        eefeBDH->getEnrichmentClassicalInterface()
          ->getEnrichmentValuesInCellRangeAtQuadPts(
            *enrichmentBlockEnrichmentBasisDataStorage
               .getQuadratureRuleContainer(),
            quadValuesInAllCellsEnrichment.data(),
            *eefeBDH->getEnrichmentClassicalInterface()->getLinAlgOpContext(),
            std::make_pair((size_type)0, numLocallyOwnedCells));

        auto coeffsInAllCellsHost =
          eefeBDH->getEnrichmentClassicalInterface()
            ->getClassicalComponentCoeffsInAllCellsOEFE();
        utils::MemoryStorage<ValueTypeOperator, memorySpace> coeffsInAllCells(
          coeffsInAllCellsHost.size());
        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          coeffsInAllCellsHost.size(),
          coeffsInAllCells.data(),
          coeffsInAllCellsHost.data());

        const double *cellJxWValuesClassicalBlockPtr =
          classicalBlockBasisDataStorage.getJxWInAllCells().data();

        const double *cellJxWValuesEnrichmentBlockClassicalPtr =
          enrichmentBlockClassicalBasisDataStorage.getJxWInAllCells().data();
        ;

        const double *cellJxWValuesEnrichmentBlockEnrichmentPtr =
          enrichmentBlockEnrichmentBasisDataStorage.getJxWInAllCells().data();
        ;

        size_type numCumulativeDofsxDofsCellsInBlock           = 0;
        size_type cumulativeQuadEnrichBlockEnrichxenrichInCell = 0;
        size_type cumulativeCoeffsInCellRange                  = 0;

        for (size_type cellStartId = 0; cellStartId < numLocallyOwnedCells;
             cellStartId += cellBlockSize)
          {
            ValueTypeOperator *basisOverlapStartPtrInCellBlock =
              basisOverlap.data() + numCumulativeDofsxDofsCellsInBlock;

            size_type cellQuadStartIdsClassicalBlock =
              classicalBlockBasisDataStorage.getQuadratureRuleContainer()
                ->getCellQuadStartId(cellStartId);
            size_type cellQuadStartIdsEnrichmentBlockClassical =
              enrichmentBlockClassicalBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellQuadStartId(cellStartId);
            size_type cellQuadStartIdsEnrichmentBlockEnrichment =
              enrichmentBlockEnrichmentBasisDataStorage
                .getQuadratureRuleContainer()
                ->getCellQuadStartId(cellStartId);

            const size_type cellEndId =
              std::min(cellStartId + cellBlockSize, numLocallyOwnedCells);
            const size_type numCellsInBlock = cellEndId - cellStartId;

            std::vector<size_type> dofsPerCellInCellBlock(numCellsInBlock, 0),
              numEnrichmentIdsInCellBlock(numCellsInBlock, 0),
              nQuadPointInCellBlockClassicalBlock(numCellsInBlock, 0),
              nQuadPointInCellBlockEnrichmentBlockClassical(numCellsInBlock, 0),
              nQuadPointInCellBlockEnrichmentBlockEnrichment(numCellsInBlock,
                                                             0);

            std::vector<size_type> mSizes(numCellsInBlock, 0);
            std::vector<size_type> nSizes(numCellsInBlock, 0);
            std::vector<size_type> kSizes(numCellsInBlock, 0);
            std::vector<size_type> ldaSizes(numCellsInBlock, 0);
            std::vector<size_type> ldbSizes(numCellsInBlock, 0);
            std::vector<size_type> ldcSizes(numCellsInBlock, 0);
            std::vector<size_type> strideA(numCellsInBlock, 0);
            std::vector<size_type> strideB(numCellsInBlock, 0);
            std::vector<size_type> strideC(numCellsInBlock, 0);

            std::copy(dofsInCellVec.begin() + cellStartId,
                      dofsInCellVec.begin() + cellEndId,
                      dofsPerCellInCellBlock.begin());

            size_type cumulativeDofsCFExQuadClassicalBlock           = 0;
            size_type cumulativeDofsCFExQuadEnrichmentBlockClassical = 0;
            size_type cumulativeDofsxQuadEnrichmentBlockEnrichment   = 0;
            size_type cumulativeEnrichxQuadEnrichmentBlockClassical  = 0;
            size_type cumulativeEnrichxQuadEnrichmentBlockEnrichment = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
              {
                cellIndex = iCell + cellStartId;

                numEnrichmentIdsInCellBlock[iCell] =
                  dofsPerCellInCellBlock[iCell] - dofsPerCellCFE;

                nQuadPointInCellBlockClassicalBlock[iCell] =
                  classicalBlockBasisDataStorage.getQuadratureRuleContainer()
                    ->nCellQuadraturePoints(cellIndex);
                nQuadPointInCellBlockEnrichmentBlockClassical[iCell] =
                  enrichmentBlockClassicalBasisDataStorage
                    .getQuadratureRuleContainer()
                    ->nCellQuadraturePoints(cellIndex);
                nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell] =
                  enrichmentBlockEnrichmentBasisDataStorage
                    .getQuadratureRuleContainer()
                    ->nCellQuadraturePoints(cellIndex);

                cumulativeDofsCFExQuadClassicalBlock +=
                  dofsPerCellCFE * nQuadPointInCellBlockClassicalBlock[iCell];
                cumulativeDofsCFExQuadEnrichmentBlockClassical +=
                  dofsPerCellCFE *
                  nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                cumulativeDofsxQuadEnrichmentBlockEnrichment +=
                  dofsPerCellInCellBlock[iCell] *
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];

                cumulativeEnrichxQuadEnrichmentBlockClassical +=
                  numEnrichmentIdsInCellBlock[iCell] *
                  nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                cumulativeEnrichxQuadEnrichmentBlockEnrichment +=
                  numEnrichmentIdsInCellBlock[iCell] *
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
              }

            //------- basis data in cell range ------
            std::pair<size_type, size_type> cellPair(cellStartId, cellEndId);

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              basisDataInCellRangeClassicalBlock(
                cumulativeDofsCFExQuadClassicalBlock);
            classicalBlockBasisDataStorage.getBasisDataInCellRange(
              cellPair, basisDataInCellRangeClassicalBlock);

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              basisDataInCellRangeEnrichmentBlockClassical(
                cumulativeDofsCFExQuadEnrichmentBlockClassical);
            enrichmentBlockClassicalBasisDataStorage.getBasisDataInCellRange(
              cellPair, basisDataInCellRangeEnrichmentBlockClassical);

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              basisDataInCellRangeEnrichmentBlockEnrichment(
                cumulativeDofsxQuadEnrichmentBlockEnrichment);
            enrichmentBlockEnrichmentBasisDataStorage.getBasisDataInCellRange(
              cellPair, basisDataInCellRangeEnrichmentBlockEnrichment);
            //------ basis data in cell range -------

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              classicalComponentInQuadValuesEC(
                cumulativeEnrichxQuadEnrichmentBlockClassical);

            utils::MemoryStorage<ValueTypeOperator, memorySpace>
              classicalComponentInQuadValuesEE(
                cumulativeEnrichxQuadEnrichmentBlockEnrichment);

            // Do a gemm (\Sigma c_i N_i^classical)
            std::vector<char> transA(numCellsInBlock, 'N');
            std::vector<char> transB(numCellsInBlock, 'N');
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                kSizes[iCell]   = dofsPerCellCFE;
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = kSizes[iCell];
                ldcSizes[iCell] = mSizes[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = mSizes[iCell] * nSizes[iCell];
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperator,
                                                             ValueTypeOperator,
                                                             memorySpace>(
              numCellsInBlock,
              transA.data(),
              transB.data(),
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              (ValueTypeOperator)1.0,
              coeffsInAllCells.data() + cumulativeCoeffsInCellRange,
              ldaSizes.data(),
              basisDataInCellRangeEnrichmentBlockClassical.data(),
              ldbSizes.data(),
              (ValueTypeOperator)0.0,
              classicalComponentInQuadValuesEC.data(),
              ldcSizes.data(),
              linAlgOpContext);

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                kSizes[iCell]   = dofsPerCellCFE;
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = dofsPerCellInCellBlock[iCell];
                ldcSizes[iCell] = mSizes[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = dofsPerCellInCellBlock[iCell] * nSizes[iCell];
                strideC[iCell]  = mSizes[iCell] * nSizes[iCell];
              }

            // Do a gemm (\Sigma c_i N_i^classical) for enrichment quad rule

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperator,
                                                             ValueTypeOperator,
                                                             memorySpace>(
              numCellsInBlock,
              transA.data(),
              transB.data(),
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              (ValueTypeOperator)1.0,
              coeffsInAllCells.data() + cumulativeCoeffsInCellRange,
              ldaSizes.data(),
              basisDataInCellRangeEnrichmentBlockEnrichment.data(),
              ldbSizes.data(),
              (ValueTypeOperator)0.0,
              classicalComponentInQuadValuesEE.data(),
              ldcSizes.data(),
              linAlgOpContext);

            // strideC for sub-blocks written directly into basisOverlap:
            //   strideC_full[i]       = dofsPerCell[i]^2  (CC, EC)
            //   strideC_colOffset[i]  = dofsPerCell[i]^2 +
            //     dofsPerCFE*(dofsPerCell[i+1]-dofsPerCell[i])  (CE, EE)
            std::vector<size_type> strideC_full(numCellsInBlock, 0);
            std::vector<size_type> strideC_colOffset(numCellsInBlock, 0);
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                strideC_full[iCell] =
                  dofsPerCellInCellBlock[iCell] * dofsPerCellInCellBlock[iCell];
                strideC_colOffset[iCell] =
                  strideC_full[iCell] +
                  ((iCell + 1 < numCellsInBlock) ?
                     dofsPerCellCFE * (dofsPerCellInCellBlock[iCell + 1] -
                                       dofsPerCellInCellBlock[iCell]) :
                     0);
              }

            // ------------------- Classical - Classical Block
            // --------------------
            size_type JxWxNCellSize = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]  = 1;
                nSizes[iCell]  = dofsPerCellCFE;
                kSizes[iCell]  = nQuadPointInCellBlockClassicalBlock[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
              }
            utils::MemoryStorage<ValueTypeOperator, memorySpace> JxWxNCell(
              JxWxNCellSize);

            linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                              ValueTypeOperator,
                                                              memorySpace>(
              numCellsInBlock,
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              cellJxWValuesClassicalBlockPtr + cellQuadStartIdsClassicalBlock,
              basisDataInCellRangeClassicalBlock.data(),
              JxWxNCell.data(),
              linAlgOpContext);

            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'C');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell]   = dofsPerCellCFE;
                nSizes[iCell]   = dofsPerCellCFE;
                kSizes[iCell]   = nQuadPointInCellBlockClassicalBlock[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = nSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_full[iCell];
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                                             ValueTypeOperand,
                                                             memorySpace>(
              numCellsInBlock,
              transA.data(),
              transB.data(),
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              (ValueTypeOperand)1.0,
              JxWxNCell.data(),
              ldaSizes.data(),
              basisDataInCellRangeClassicalBlock.data(),
              ldbSizes.data(),
              (ValueTypeOperand)0.0,
              basisOverlapStartPtrInCellBlock,
              ldcSizes.data(),
              linAlgOpContext);
            // ------------------- Classical - Classical Block
            // --------------------

            // ------------------- Enrichment - Enrichment Block
            // -------------------- pristine with pristine
            JxWxNCellSize = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = 1;
                nSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
              }
            JxWxNCell.resize(JxWxNCellSize, 0);

            linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                              ValueTypeOperator,
                                                              memorySpace>(
              numCellsInBlock,
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              cellJxWValuesEnrichmentBlockEnrichmentPtr +
                cellQuadStartIdsEnrichmentBlockEnrichment,
              quadValuesInAllCellsEnrichment.data() +
                cumulativeQuadEnrichBlockEnrichxenrichInCell,
              JxWxNCell.data(),
              linAlgOpContext);

            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'C');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = mSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_colOffset[iCell];
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                                             ValueTypeOperand,
                                                             memorySpace>(
              numCellsInBlock,
              transA.data(),
              transB.data(),
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              (ValueTypeOperand)1.0,
              JxWxNCell.data(),
              ldaSizes.data(),
              quadValuesInAllCellsEnrichment.data() +
                cumulativeQuadEnrichBlockEnrichxenrichInCell,
              ldbSizes.data(),
              (ValueTypeOperand)0.0,
              basisOverlapStartPtrInCellBlock +
                dofsPerCellInCellBlock[0] * dofsPerCellCFE + dofsPerCellCFE,
              ldcSizes.data(),
              linAlgOpContext);

            // interpolated ci's in Ni_classicalQuadrature of Mc = d
            // * interpolated ci's in Ni_classicalQuadrature of Mc =
            // d
            JxWxNCellSize = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = 1;
                nSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
              }
            JxWxNCell.resize(JxWxNCellSize, 0);

            linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                              ValueTypeOperator,
                                                              memorySpace>(
              numCellsInBlock,
              linearAlgebra::blasLapack::Layout::ColMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              cellJxWValuesEnrichmentBlockClassicalPtr +
                cellQuadStartIdsEnrichmentBlockClassical,
              classicalComponentInQuadValuesEC.data(),
              JxWxNCell.data(),
              linAlgOpContext);


            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'C');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockClassical[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = nSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_colOffset[iCell];
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                                             ValueTypeOperator,
                                                             memorySpace>(
              numCellsInBlock,
              transA.data(),
              transB.data(),
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              (ValueTypeOperand)1.0,
              JxWxNCell.data(),
              ldaSizes.data(),
              classicalComponentInQuadValuesEC.data(),
              ldbSizes.data(),
              (ValueTypeOperator)1.0,
              basisOverlapStartPtrInCellBlock +
                dofsPerCellInCellBlock[0] * dofsPerCellCFE + dofsPerCellCFE,
              ldcSizes.data(),
              linAlgOpContext);

            // Ni_pristine* interpolated ci's in
            // Ni_classicalQuadratureOfPristine at quadpoints

            JxWxNCellSize = 0;
            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = 1;
                nSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                strideA[iCell] = mSizes[iCell] * kSizes[iCell];
                strideB[iCell] = kSizes[iCell] * nSizes[iCell];
                strideC[iCell] = mSizes[iCell] * nSizes[iCell] * kSizes[iCell];
                JxWxNCellSize += nSizes[iCell] * kSizes[iCell] * mSizes[iCell];
              }
            JxWxNCell.resize(JxWxNCellSize, 0);

            linearAlgebra::blasLapack::scaleStridedVarBatched<ValueTypeOperator,
                                                              ValueTypeOperator,
                                                              memorySpace>(
              numCellsInBlock,
              linearAlgebra::blasLapack::Layout::RowMajor,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              linearAlgebra::blasLapack::ScalarOp::Identity,
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              cellJxWValuesEnrichmentBlockEnrichmentPtr +
                cellQuadStartIdsEnrichmentBlockEnrichment,
              quadValuesInAllCellsEnrichment.data() +
                cumulativeQuadEnrichBlockEnrichxenrichInCell,
              JxWxNCell.data(),
              linAlgOpContext);

            std::fill(transA.begin(), transA.end(), 'N');
            std::fill(transB.begin(), transB.end(), 'N');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                ldaSizes[iCell] = mSizes[iCell];
                ldbSizes[iCell] = kSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_colOffset[iCell];
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                                             ValueTypeOperator,
                                                             memorySpace>(
              numCellsInBlock,
              transA.data(),
              transB.data(),
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              (ValueTypeOperand)-1.0,
              classicalComponentInQuadValuesEE.data(),
              ldaSizes.data(),
              JxWxNCell.data(),
              ldbSizes.data(),
              (ValueTypeOperator)1.0,
              basisOverlapStartPtrInCellBlock +
                dofsPerCellInCellBlock[0] * dofsPerCellCFE + dofsPerCellCFE,
              ldcSizes.data(),
              linAlgOpContext);

            std::fill(transA.begin(), transA.end(), 'T');
            std::fill(transB.begin(), transB.end(), 'T');

            for (size_type iCell = 0; iCell < numCellsInBlock; ++iCell)
              {
                mSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                nSizes[iCell] = numEnrichmentIdsInCellBlock[iCell];
                kSizes[iCell] =
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
                ldaSizes[iCell] = kSizes[iCell];
                ldbSizes[iCell] = nSizes[iCell];
                ldcSizes[iCell] = dofsPerCellInCellBlock[iCell];
                strideA[iCell]  = mSizes[iCell] * kSizes[iCell];
                strideB[iCell]  = kSizes[iCell] * nSizes[iCell];
                strideC[iCell]  = strideC_colOffset[iCell];
              }

            linearAlgebra::blasLapack::gemmStridedVarBatched<ValueTypeOperand,
                                                             ValueTypeOperator,
                                                             memorySpace>(
              numCellsInBlock,
              transA.data(),
              transB.data(),
              strideA.data(),
              strideB.data(),
              strideC.data(),
              mSizes.data(),
              nSizes.data(),
              kSizes.data(),
              (ValueTypeOperand)-1.0,
              JxWxNCell.data(),
              ldaSizes.data(),
              classicalComponentInQuadValuesEE.data(),
              ldbSizes.data(),
              (ValueTypeOperator)1.0,
              basisOverlapStartPtrInCellBlock +
                dofsPerCellInCellBlock[0] * dofsPerCellCFE + dofsPerCellCFE,
              ldcSizes.data(),
              linAlgOpContext);

            for (size_type iCell = 0; iCell < numCellsInBlock; iCell++)
              {
                numCumulativeDofsxDofsCellsInBlock +=
                  dofsPerCellInCellBlock[iCell] * dofsPerCellInCellBlock[iCell];
                cumulativeCoeffsInCellRange +=
                  dofsPerCellCFE * numEnrichmentIdsInCellBlock[iCell];
                cumulativeQuadEnrichBlockEnrichxenrichInCell +=
                  numEnrichmentIdsInCellBlock[iCell] *
                  nQuadPointInCellBlockEnrichmentBlockEnrichment[iCell];
              }
          }
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

      OrthoEFEOverlapInverseOpContextGLLInternal::
        computeBasisOverlapMatrixBlocked<ValueTypeOperator,
                                         ValueTypeOperand,
                                         memorySpace,
                                         dim>(
          classicalBlockGLLBasisDataStorage,
          enrichmentBlockEnrichmentBasisDataStorage,
          enrichmentBlockClassicalBasisDataStorage,
          cfeBDH,
          efeBDH,
          NiNjInAllCells,
          BasisDataStorageDefaults<memorySpace>::CELL_BATCH_SIZE);

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
      locallyOwnedCellsNumDoFs.copyFrom(locallyOwnedCellsNumDoFsSTL);

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> diagonal(
        d_feBasisManager->getMPIPatternP2P(), linAlgOpContext);

      const size_type numCumulativeDofsCells =
        std::accumulate(locallyOwnedCellsNumDoFsSTL.begin(),
                        locallyOwnedCellsNumDoFsSTL.end(),
                        0);

      // Create the diagonal of the classical block matrix which is diagonal for
      // GLL with spectral quadrature
      FECellWiseDataOperations<ValueTypeOperator, memorySpace>::
        addCellWiseBasisDataToDiagonalData(NiNjInAllCells.data(),
                                           itCellLocalIdsBegin,
                                           locallyOwnedCellsNumDoFs,
                                           numCumulativeDofsCells,
                                           diagonal.data(),
                                           *linAlgOpContext);

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

      utils::MemoryStorage<ValueTypeOperator, utils::MemorySpace::HOST>
        NiNjInAllCellsHost(NiNjInAllCells.size());
      NiNjInAllCellsHost.template copyFrom<memorySpace>(NiNjInAllCells.data());

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
          for (size_type j = 0; j < nCellEnrichmentDofs; j++)
            {
              for (size_type k = 0; k < nCellEnrichmentDofs; k++)
                {
                  *(basisOverlapEnrichmentBlockSTLTmp.data() +
                    enrichmentVecInCell[j] * d_nglobalEnrichmentIds +
                    enrichmentVecInCell[k]) +=
                    *(NiNjInAllCellsHost.data() + cumulativeBasisDataInCells +
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
        utils::mpi::Types<ValueTypeOperator>::getMPIDatatype(),
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
      **/

      for (size_type i = 0; i < d_nglobalEnrichmentIds; i++)
        {
          if (std::abs(*(basisOverlapEnrichmentBlockSTL.data() +
                         i * d_nglobalEnrichmentIds + i)) < 1e-10)
            {
              utils::throwException(
                false,
                "One of diagonal elements of M is very small : " +
                  std::to_string(i));
            }
        }

      linearAlgebra::blasLapack::inverse<ValueTypeOperator,
                                         utils::MemorySpace::HOST>(
        d_nglobalEnrichmentIds,
        basisOverlapEnrichmentBlockSTL.data(),
        *(linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST));

      for (size_type i = 0; i < d_nglobalEnrichmentIds; i++)
        {
          if (std::abs(*(basisOverlapEnrichmentBlockSTL.data() +
                         i * d_nglobalEnrichmentIds + i)) > 1e10)
            {
              utils::throwException(
                false, "One of diagonal elements of MInv is very large.");
            }
        }

      /**
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
    OrthoEFEOverlapInverseOpContextGLL<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
                  linearAlgebra::MultiVector<ValueType, memorySpace> &       Y,
                  bool updateGhostX,
                  bool updateGhostY) const
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

          if (updateGhostX)
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
            YenrichedGlobalVec(d_nglobalEnrichmentIds * numComponents);

          utils::MemoryStorage<ValueTypeOperand, utils::MemorySpace::HOST>
            XenrichedGlobalVecTmp(d_nglobalEnrichmentIds * numComponents);

          XenrichedGlobalVecTmp.template copyFrom<memorySpace>(
            X.begin(),
            nlocallyOwnedEnrichmentIds * numComponents,
            nlocallyOwnedClassicalIds * numComponents,
            ((d_feBasisManager->getLocallyOwnedRanges()[1].first) -
             (d_efebasisDofHandler->getGlobalRanges()[0].second)) *
              numComponents);

          int err = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
            utils::mpi::MPIInPlace,
            XenrichedGlobalVecTmp.data(),
            XenrichedGlobalVecTmp.size(),
            utils::mpi::Types<ValueTypeOperand>::getMPIDatatype(),
            utils::mpi::MPISum,
            d_feBasisManager->getMPIPatternP2P()->mpiCommunicator());
          std::pair<bool, std::string> mpiIsSuccessAndMsg =
            utils::mpi::MPIErrIsSuccessAndMsg(err);
          utils::throwException(mpiIsSuccessAndMsg.first,
                                "MPI Error:" + mpiIsSuccessAndMsg.second);

          XenrichedGlobalVec.template copyFrom<utils::MemorySpace::HOST>(
            XenrichedGlobalVecTmp);

          // Do dgemm

          ValueType alpha = 1.0;
          ValueType beta  = 0.0;

          linearAlgebra::blasLapack::
            gemm<ValueTypeOperator, ValueTypeOperand, memorySpace>(
              'N',
              'T',
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
          if (updateGhostY)
            Y.updateGhostValues();
        }
      else
        {
          if (updateGhostX)
            X.updateGhostValues();
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
          if (updateGhostY)
            Y.updateGhostValues();
        }
    }
  } // namespace basis
} // namespace dftefe
