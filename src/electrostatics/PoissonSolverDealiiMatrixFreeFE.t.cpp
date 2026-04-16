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
  namespace electrostatics
  {
    namespace PoissonSolverDealiiMatrixFreeFEInternal
    {
      template <typename ValueTypeOperator,
                typename ValueTypeOperand,
                utils::MemorySpace memorySpace,
                size_type          dim>
      void
      getDealiiQuadRule(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>
                                 feBasisDataStorage,
        dealii::Quadrature<dim> &quadRuleDealii,
        unsigned int &           num1DQuadPoints)
      {
        const quadrature::QuadratureRuleAttributes quadAttr =
          feBasisDataStorage->getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes();
        const quadrature::QuadratureFamily quadratureFamily =
          quadAttr.getQuadratureFamily();

        utils::throwException(
          quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
            quadratureFamily == quadrature::QuadratureFamily::GLL ||
            quadratureFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED,
          "The quadrature rule has to be uniform quadrature like GAUSS , GLL or GAUSS_SUBDIVIDED for Dealii Matrix Free.");

        num1DQuadPoints = (unsigned int)(std::cbrt(
          feBasisDataStorage->getQuadratureRuleContainer()
            ->nCellQuadraturePoints(0)));

        if (auto cfeBDSDealii = std::dynamic_pointer_cast<
              const basis::CFEBDSOnTheFlyComputeDealii<ValueTypeOperand,
                                                       ValueTypeOperator,
                                                       memorySpaceHost,
                                                       dim>>(
              feBasisDataStorage))
          quadRuleDealii = cfeBDSDealii->getDealiiQuadratureRule();
        else if (auto cfeBDSDealii = std::dynamic_pointer_cast<
                   const basis::CFEBasisDataStorageDealii<ValueTypeOperand,
                                                          ValueTypeOperator,
                                                          memorySpaceHost,
                                                          dim>>(
                   feBasisDataStorage))
          quadRuleDealii = cfeBDSDealii->getDealiiQuadratureRule();
        else
          utils::throwException(
            false,
            "Could not cast FEBasisDataStorage to CFEBasisDataStorageDealii or CFEBDSOnTheFlyComputeDealii "
            "in PoissonSolverDealiiMatrixFreeFE.");
      }
    } // end of namespace PoissonSolverDealiiMatrixFreeFEInternal


    //
    // Constructor
    //

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
      PoissonSolverDealiiMatrixFreeFE(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpaceHost,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>
          feBasisDataStorageStiffnessMatrix,
        const std::map<
          std::string,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>>
          &feBasisDataStorageRhs,
        const std::map<
          std::string,
          const quadrature::QuadratureValuesContainer<ValueType, memorySpaceHost> &>
          &                                     inpRhs,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : d_feBasisManagerField(feBasisManagerField)
      , d_numComponents(
          !inpRhs.empty() ? inpRhs.begin()->second.getNumberComponents() : 0)
      , d_pcType(pcType)
      , d_p(feBasisManagerField->getMPIPatternP2P()->mpiCommunicator(),
            "Poisson Solver")
      , pcout(std::cout)
      , d_dealiiMatrixFree(
          std::make_shared<dealii::MatrixFree<dim, ValueTypeOperator>>())
      , d_dofHandlerIndex(0)
      , d_matrixFreeQuadCompStiffnessMatrix(0)
      , d_dealiiQuadratureRuleVec(1, dealii::Quadrature<dim>())
      , d_scratchMultiVecHost(feBasisManagerField->getMPIPatternP2P(),
                          linearAlgebra::LinAlgOpContextDefaults::LINALG_OP_CONTXT_HOST ,
                          d_numComponents,
                          ValueType())
    {
      int rank;
      utils::mpi::MPICommRank(this->getMPIComm(), &rank);
      pcout.setCondition(rank == 0);

      // Store linAlgOpContext for later device vector creation and device BLAS.
      d_linAlgOpContext = linAlgOpContext;

      utils::throwException(
        d_numComponents == 1,
        "Number Components of QuadratureValuesContainer has to be 1 for Dealii Matrix Free Poisson Solve.");

      utils::throwException(
        !inpRhs.empty(),
        "The input QuadValuesContainer Map in PoissonSolver cannot be empty.");

      auto iter = feBasisDataStorageRhs.begin();
      while (iter != feBasisDataStorageRhs.end())
        {
          d_feBasisDataStorageRhs[iter->first] = iter->second;
          utils::throwException(
            ((feBasisDataStorageStiffnessMatrix->getBasisDofHandler()).get() ==
             (iter->second->getBasisDofHandler()).get()),
            "The BasisDofHandler of the datastorages does not match in PoissonLinearSolverFunctionFE.");
          iter++;
        }

      utils::throwException(
        (feBasisDataStorageStiffnessMatrix->getBasisDofHandler().get() ==
         &feBasisManagerField->getBasisDofHandler()),
        "The BasisDofHandler of the dataStorages and basisManager should be same in PoissonLinearSolverFunctionFE.");

      // Check wether the dofhandler and constrints come from classical basis or
      // not
      std::shared_ptr<const basis::CFEBasisDofHandlerDealii<ValueTypeOperator,
                                                            memorySpaceHost,
                                                            dim>>
        cfeBasisDofHandlerDealii = std::dynamic_pointer_cast<
          const basis::
            CFEBasisDofHandlerDealii<ValueTypeOperator, memorySpaceHost, dim>>(
          feBasisDataStorageStiffnessMatrix->getBasisDofHandler());
      utils::throwException(
        cfeBasisDofHandlerDealii.get() != nullptr,
        "Could not cast BasisDofHandler to CFEBasisDofHandlerDealii "
        "in PoissonSolverDealiiMatrixFreeFE.");

      // Set up Homogoneous Constraints BasisManager
      std::shared_ptr<const utils::ScalarSpatialFunctionReal> zeroFunction =
        std::make_shared<utils::ScalarZeroFunctionReal>();

      d_feBasisManagerHomo =
        std::make_shared<basis::FEBasisManager<ValueTypeOperand,
                                               ValueTypeOperator,
                                               memorySpaceHost,
                                               dim>>(cfeBasisDofHandlerDealii,
                                                     zeroFunction);

      const basis::CFEConstraintsLocalDealii<ValueTypeOperator,
                                             memorySpaceHost,
                                             dim> &cfeConstraintsLocalDealii =
        dynamic_cast<const basis::CFEConstraintsLocalDealii<ValueTypeOperator,
                                                            memorySpaceHost,
                                                            dim> &>(
          d_feBasisManagerHomo->getConstraints());
      utils::throwException(
        &cfeConstraintsLocalDealii != nullptr,
        "Could not cast ConstraintsLocal to CFEConstraintsLocalDealii "
        "in PoissonSolverDealiiMatrixFreeFE.");

      d_feOrder          = cfeBasisDofHandlerDealii->getFEOrder(0);
      d_dealiiDofHandler = cfeBasisDofHandlerDealii->getDoFHandler();
      d_dealiiAffineConstraintMatrix =
        &cfeConstraintsLocalDealii.getAffineConstraints();

      PoissonSolverDealiiMatrixFreeFEInternal::getDealiiQuadRule<
        ValueTypeOperator,
        ValueTypeOperand,
        memorySpace,
        dim>(feBasisDataStorageStiffnessMatrix,
             d_dealiiQuadratureRuleVec[0],
             d_num1DQuadPointsStiffnessMatrix);
      unsigned int count = 1;
      auto         iter1 = feBasisDataStorageRhs.begin();
      d_num1DQuadPointsRhs.clear();
      d_nonTensorSructuredQuadeRhs.clear();
      while (iter1 != feBasisDataStorageRhs.end())
        {
          const quadrature::QuadratureRuleAttributes quadAttr =
            iter1->second->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes();
          const quadrature::QuadratureFamily quadratureFamily =
            quadAttr.getQuadratureFamily();

          if (!(quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
                quadratureFamily == quadrature::QuadratureFamily::GLL ||
                quadratureFamily ==
                  quadrature::QuadratureFamily::GAUSS_SUBDIVIDED))
            {
              d_nonTensorSructuredQuadeRhs.push_back(
                distributedCPUVec<ValueTypeOperator>());
            }
          else
            {
              d_dealiiQuadratureRuleVec.push_back(dealii::Quadrature<dim>());
              PoissonSolverDealiiMatrixFreeFEInternal::getDealiiQuadRule<
                ValueTypeOperator,
                ValueTypeOperand,
                memorySpace,
                dim>(iter1->second,
                     d_dealiiQuadratureRuleVec[count],
                     d_num1DQuadPointsRhs[iter1->first]);
              count += 1;
            }
          iter1++;
        }

      typename dealii::MatrixFree<dim>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        dealii::MatrixFree<dim>::AdditionalData::partition_partition;
      additional_data.mapping_update_flags = dealii::update_values |
                                             dealii::update_gradients |
                                             dealii::update_JxW_values;

      // create dealiiMatrixFree
      d_dealiiMatrixFree->reinit(
        d_mappingDealii,
        std::vector<const dealii::DoFHandler<dim> *>{d_dealiiDofHandler.get()},
        std::vector<const dealii::AffineConstraints<ValueTypeOperand> *>{
          d_dealiiAffineConstraintMatrix},
        d_dealiiQuadratureRuleVec,
        additional_data);

      d_cellIdToCellIndexMap.clear();
      auto cellPtr =
        d_dealiiMatrixFree->get_dof_handler(d_dofHandlerIndex).begin_active();
      auto endcPtr =
        d_dealiiMatrixFree->get_dof_handler(d_dofHandlerIndex).end();

      unsigned int iCell = 0;
      for (; cellPtr != endcPtr; ++cellPtr)
        if (cellPtr->is_locally_owned())
          {
            d_cellIdToCellIndexMap[cellPtr->id()] = iCell;
            ++iCell;
          }

      // std::vector<global_size_type> ghostIndices(0);
      // ghostIndices.resize(feBasisManagerHomo->nGhost(), 0);
      // for(int i = 0 ; i < ghostIndices.size() ; i++)
      // {
      //   ghostIndices[i] = *(feBasisManagerHomo->getGhostIndices().data()+i);
      // }

      // //const dealii::IndexSet ghostIndexSet, locallyOwnedIndexSet;
      // d_ghostIndexSet.fill_index_vector(ghostIndices);
      // d_locallyOwnedIndexSet.add_range(feBasisManagerHomo->getLocallyOwnedRanges()[0].first,
      //   feBasisManagerHomo->getLocallyOwnedRanges()[0].second);

      d_x.reinit(d_dealiiMatrixFree->get_vector_partitioner(d_dofHandlerIndex));
      // d_x.reinit(d_locallyOwnedIndexSet, d_ghostIndexSet,
      // feBasisManagerField->getMPIPatternP2P()->mpiCommunicator());
      d_initial.reinit(d_x);

      utils::throwException(
        d_pcType == dftefe::linearAlgebra::PreconditionerType::JACOBI,
        "Only JACOBI preconditioner avaliable for Dealii Matrix Free Poisson Solve. Contact developers for other options.");

      for (auto &i : d_nonTensorSructuredQuadeRhs)
        {
          i.reinit(d_x);
        }
      // ---------------------------------------------------------------
      // Device-specific initialisation (only when memorySpace == DEVICE)
      // ---------------------------------------------------------------
      if constexpr (memorySpace == utils::MemorySpace::DEVICE)
        {
#ifndef DFTEFE_WITH_DEVICE
          utils::throwException(
            false,
            "PoissonSolverDealiiMatrixFreeFE: memorySpace == DEVICE requires "
            "compilation with DFTEFE_WITH_DEVICE.");
#else

          auto partitioner = d_x.get_partitioner();
          const std::pair<dealii::types::global_dof_index,
                            dealii::types::global_dof_index> &locallyOwnedRange =
              partitioner->local_range();
          std::vector<dealii::types::global_dof_index> ghostIndices =
            (partitioner->ghost_indices()).get_index_vector();

          d_mpiPatternP2PDevice =
            std::make_shared<utils::mpi::MPIPatternP2P<memorySpace>>(
              std::pair<uInt, uInt>(locallyOwnedRange.first,
                                                  locallyOwnedRange.second),
              std::vector<uInt>(ghostIndices.begin(), ghostIndices.end()),
              partitioner->get_mpi_communicator());

          // Create device-resident vectors sharing the same MPI layout as d_x.
          //auto mpiPattern = d_feBasisManagerHomo->getMPIPatternP2P();

          // d_mpiPatternP2PDevice = std::make_shared<
          //   const utils::mpi::MPIPatternP2P<dftefe::utils::MemorySpace::DEVICE>>(
          //     mpiPattern->getLocallyOwnedRange(0),
          //     mpiPattern->getGhostIndices(),
          //     getMPIComm());
              
          d_diagonalADevice = std::make_unique<
            linearAlgebra::Vector<ValueTypeOperator, memorySpace>>(
            d_mpiPatternP2PDevice, d_linAlgOpContext, ValueTypeOperator(0));

          d_rhsDevice = std::make_unique<
            linearAlgebra::Vector<ValueTypeOperator, memorySpace>>(
            d_mpiPatternP2PDevice, d_linAlgOpContext, ValueTypeOperator(0));

          d_initialDevice = std::make_unique<
            linearAlgebra::Vector<ValueTypeOperator, memorySpace>>(
            d_mpiPatternP2PDevice, d_linAlgOpContext, ValueTypeOperator(0));

          utils::throwException(
            d_num1DQuadPointsStiffnessMatrix == d_feOrder + 1,
            "The quadrature point for the stiffness matrix in Laplacian has to be 1 more than the feOrder for using dftefe::electrostatics::matrixFreeWrapperDevice class.");

          // Device matrix-free Laplace operator.
          // nDofsPerDim = feOrder + 1 (number of 1-D quadrature / dof points).
          d_matrixFreeWrapperDevice = std::make_unique<
            dftefe::MatrixFreeWrapperClass<ValueTypeOperator,
                                          dftefe::operatorList::Laplace,
                                          dftefe::utils::MemorySpace::DEVICE,
                                          false>>(
            static_cast<std::uint32_t>(d_feOrder + 1),
            getMPIComm(),
            d_dealiiMatrixFree.get(),
            *d_dealiiAffineConstraintMatrix,
            static_cast<std::uint32_t>(d_dofHandlerIndex),
            static_cast<std::uint32_t>(d_matrixFreeQuadCompStiffnessMatrix),
            static_cast<uInt>(1));

          d_matrixFreeWrapperDevice->init();
#endif // DFTEFE_WITH_DEVICE
        }

      computeDiagonalA();
      reinit(feBasisManagerField, inpRhs);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
      PoissonSolverDealiiMatrixFreeFE(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpaceHost,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>
          feBasisDataStorageStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>
          feBasisDataStorageRhs,
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpaceHost> &                        inpRhs,
        const linearAlgebra::PreconditionerType pcType,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
          linAlgOpContext)
      : PoissonSolverDealiiMatrixFreeFE(
          feBasisManagerField,
          feBasisDataStorageStiffnessMatrix,
          std::map<
            std::string,
            std::shared_ptr<
              const basis::FEBasisDataStorage<ValueTypeOperator, memorySpaceHost>>>(
            {{"Field", feBasisDataStorageRhs}}),
          std::map<std::string,
                   const quadrature::QuadratureValuesContainer<
                     linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                            ValueTypeOperand>,
                     memorySpaceHost> &>({{"Field", inpRhs}}),
          pcType,
          linAlgOpContext)
    {}

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
      reinit(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpaceHost,
                                                    dim>> feBasisManagerField,
        const std::map<std::string,
                       const quadrature::QuadratureValuesContainer<
                         linearAlgebra::blasLapack::
                           scalar_type<ValueTypeOperator, ValueTypeOperand>,
                         memorySpaceHost> &> &                inpRhs)
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
              " are not same same,  for PoissonSolverDealiiMatrixFreeFE reinit or input"
              "field has different components than that when constructed.");
          else
            utils::throwException(
              false,
              "The inpRhs corresponding to a feBasisDataStorageRhs couldn't be found in PoissonLinearSolver.");
          iter++;
        }

      if (d_feBasisManagerField != feBasisManagerField)
        {
          utils::throwException(
            (&(d_feBasisManagerField->getBasisDofHandler()) ==
             &(feBasisManagerField->getBasisDofHandler())),
            "The BasisDofHandler of the feBasisManagerField in reinit does not match with that in constructor in PoissonSolverDealiiMatrixFreeFE.");

          d_feBasisManagerField = feBasisManagerField;
        }

      const basis::CFEConstraintsLocalDealii<ValueTypeOperator,
                                             memorySpaceHost,
                                             dim> &cfeConstraintsLocalDealii =
        dynamic_cast<const basis::CFEConstraintsLocalDealii<ValueTypeOperator,
                                                            memorySpaceHost,
                                                            dim> &>(
          d_feBasisManagerField->getConstraints());
      utils::throwException(
        &cfeConstraintsLocalDealii != nullptr,
        "Could not cast ConstraintsLocal to CFEConstraintsLocalDealii "
        "in PoissonSolverDealiiMatrixFreeFE.");

      d_constraintsInfo = &cfeConstraintsLocalDealii.getAffineConstraints();

      utils::mpi::MPIBarrier(this->getMPIComm());
      double start_time = utils::mpi::MPIWtime();
      double time;
      // Compute RHS
      computeRhs(d_rhs, inpRhs);
      utils::mpi::MPIBarrier(this->getMPIComm());
      time = utils::mpi::MPIWtime();

      pcout << "Time for compute rhs: " << time - start_time << std::endl;

      // Upload RHS to device (d_rhsDevice already sized in constructor).
      if constexpr (memorySpace == utils::MemorySpace::DEVICE)
        {
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            d_rhsDevice->locallyOwnedSize(),
            d_rhsDevice->data(),
            d_rhs.begin());
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
      reinit(
        std::shared_ptr<const basis::FEBasisManager<ValueTypeOperand,
                                                    ValueTypeOperator,
                                                    memorySpaceHost,
                                                    dim>> feBasisManagerField,
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpaceHost> &inpRhs)
    {
      std::map<std::string,
               const quadrature::QuadratureValuesContainer<
                 linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                        ValueTypeOperand>,
                 memorySpaceHost> &>
        inpRhsMap = {{"Field", inpRhs}};

      reinit(feBasisManagerField, inpRhsMap);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::setSolution(const distributedCPUVec<ValueType> &x)
    {
      d_x = x;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
      getSolution(linearAlgebra::MultiVector<
                  linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                         ValueTypeOperand>,
                  memorySpaceHost> &solution)
    {
      solution.setValue(0.0);

      for (size_type i = 0; i < solution.locallyOwnedSize(); i++)
        {
          solution.data()[i] = *(d_x.begin() + i);
        }

      solution.updateGhostValues();

      d_feBasisManagerField->getConstraints().distributeParentToChild(solution,
                                                                      1);

      // this is done for a particular case for poisson solve each
      // scf guess but have to be modified with a reinit parameter
      d_initial = d_x;

      if constexpr (memorySpace == utils::MemorySpace::DEVICE)
        {
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            d_initialDevice->locallyOwnedSize(),
            d_initialDevice->data(),
            d_initial.begin());
        }
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const basis::FEEvaluationWrapperBase::distributedCPUVec<ValueTypeOperand> &
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::getRhs() const
    {
      return d_rhs;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const basis::FEEvaluationWrapperBase::distributedCPUVec<
      linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                             ValueTypeOperand>> &
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
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
    const linearAlgebra::Vector<ValueTypeOperator, utils::MemorySpace::DEVICE> &
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::getRhsDevice() const
    {
      return *d_rhsDevice;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const linearAlgebra::Vector<ValueTypeOperator, utils::MemorySpace::DEVICE> &
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::getInitialGuessDevice() const
    {
      return *d_initialDevice;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::solve(const double absTolerance,
                                                const unsigned int
                                                  maxNumberIterations)
    {
      this->CGsolve(absTolerance, maxNumberIterations, true);
    }

    // Ax

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::AX(const dealii::MatrixFree<dim, double> &      matrixFreeData,
               distributedCPUVec<double> &                  dst,
               const distributedCPUVec<double> &            src,
               const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      dealii::VectorizedArray<double> quarter =
        dealii::make_vectorized_array(1.0 /* / (4.0 * M_PI)*/);

      //  dealii::FEEvaluation<dim, FEOrderElectro, FEOrderElectro + 1> fe_eval(
      //    matrixFreeData,
      //    d_dofHandlerIndex,
      //    d_matrixFreeQuadCompStiffnessMatrix);

      basis::DealiiFEEvaluationWrapper<1> fe_eval_wrap(
        d_feOrder,
        d_num1DQuadPointsStiffnessMatrix,
        *d_dealiiMatrixFree,
        d_dofHandlerIndex,
        d_matrixFreeQuadCompStiffnessMatrix);

      basis::FEEvaluationWrapperBase &fe_eval =
        fe_eval_wrap.getFEEvaluationWrapperBase();

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          fe_eval.reinit(cell);
          // fe_eval.gather_evaluate(src,dealii::EvaluationFlags::gradients);
          fe_eval.readDoFValues(src);
          fe_eval.evaluate(dealii::EvaluationFlags::gradients);
          //  for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          //    {
          //      fe_eval.submit_gradient(fe_eval.get_gradient(q) * quarter, q);
          //    }
          fe_eval.submitInterpolatedGradientsAndMultiply(quarter);
          fe_eval.integrate(dealii::EvaluationFlags::gradients);
          fe_eval.distributeLocalToGlobal(dst);
          // fe_eval.integrate_scatter(dealii::EvaluationFlags::gradients,dst);
        }
    }


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::computeDiagonalA()
    {
      d_diagonalA.reinit(d_x);
      d_diagonalA = 0;

      const dealii::DoFHandler<dim> &dofHandler =
        d_dealiiMatrixFree->get_dof_handler(d_dofHandlerIndex);

      const dealii::Quadrature<dim> &quadrature =
        d_dealiiMatrixFree->get_quadrature(d_matrixFreeQuadCompStiffnessMatrix);
      dealii::FEValues<dim>  fe_values(dofHandler.get_fe(),
                                      quadrature,
                                      dealii::update_gradients |
                                        dealii::update_JxW_values);
      const unsigned int     dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
      const unsigned int     num_quad_points = quadrature.size();
      dealii::Vector<double> elementalDiagonalA(dofs_per_cell);
      std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);

      // parallel loop over all elements
      typename dealii::DoFHandler<dim>::active_cell_iterator
        cell = dofHandler.begin_active(),
        endc = dofHandler.end();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            cell->get_dof_indices(local_dof_indices);

            elementalDiagonalA = 0.0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int q_point = 0; q_point < num_quad_points;
                   ++q_point)
                elementalDiagonalA(i) += /*(1.0 / (4.0 * M_PI)) **/
                  (fe_values.shape_grad(i, q_point) *
                   fe_values.shape_grad(i, q_point)) *
                  fe_values.JxW(q_point);

            d_dealiiAffineConstraintMatrix->distribute_local_to_global(
              elementalDiagonalA, local_dof_indices, d_diagonalA);
          }

      // MPI operation to sync data
      d_diagonalA.compress(dealii::VectorOperation::add);

      for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
        if (d_diagonalA.in_local_range(i))
          if (!d_dealiiAffineConstraintMatrix->is_constrained(i))
            d_diagonalA(i) = 1.0 / d_diagonalA(i);

      d_diagonalA.compress(dealii::VectorOperation::insert);

      // Upload Jacobi diagonal to device.
      if constexpr (memorySpace == utils::MemorySpace::DEVICE)
        {
          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            d_diagonalADevice->locallyOwnedSize(),
            d_diagonalADevice->data(),
            d_diagonalA.begin());
        }
    }

    // Matrix-Free Jacobi preconditioner application

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace,
      dim>::precondition_Jacobi(distributedCPUVec<double> &      dst,
                                const distributedCPUVec<double> &src) const
    {
      // dst = src;
      // dst.scale(d_diagonalA);

      for (unsigned int i = 0; i < dst.locally_owned_size(); i++)
        dst.local_element(i) =
          d_diagonalA.local_element(i) * src.local_element(i);
    }


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::vmult(distributedCPUVec<double> &Ax,
                                                distributedCPUVec<double> &x)
    {
      Ax = 0.0;
      x.update_ghost_values();
      AX(*d_dealiiMatrixFree,
         Ax,
         x,
         std::make_pair(0, d_dealiiMatrixFree->n_cell_batches()));
      Ax.compress(dealii::VectorOperation::add);
    }


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
      computeRhs(distributedCPUVec<double> &rhs,
                 const std::map<
                   std::string,
                   const quadrature::QuadratureValuesContainer<
                     linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                            ValueTypeOperand>,
                     memorySpaceHost> &> &inpRhs)
    {
      typename dealii::DoFHandler<dim>::active_cell_iterator subCellPtr;
      rhs.reinit(d_x);
      rhs = 0;
      //  if (d_isStoreSmearedChargeRhs)
      //    {
      //      d_rhsSmearedCharge.reinit(d_x);
      //      d_rhsSmearedCharge = 0;
      //    }

      // A * boundary for static condensation
      const dealii::DoFHandler<dim> &dofHandler =
        d_dealiiMatrixFree->get_dof_handler(d_dofHandlerIndex);

      const unsigned int dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
      typename dealii::DoFHandler<dim>::active_cell_iterator
        cell = dofHandler.begin_active(),
        endc = dofHandler.end();


      distributedCPUVec<double> tempvec;
      tempvec.reinit(rhs);
      tempvec = 0.0;
      tempvec.update_ghost_values();
      d_constraintsInfo->distribute(tempvec);
      tempvec.update_ghost_values();

      //  dealii::FEEvaluation<dim, FEOrderElectro, FEOrderElectro + 1> fe_eval(
      //    *d_dealiiMatrixFree,
      //    d_dofHandlerIndex,
      //    d_matrixFreeQuadCompStiffnessMatrix);

      basis::DealiiFEEvaluationWrapper<1> fe_eval_wrap(
        d_feOrder,
        d_num1DQuadPointsStiffnessMatrix,
        *d_dealiiMatrixFree,
        d_dofHandlerIndex,
        d_matrixFreeQuadCompStiffnessMatrix);

      basis::FEEvaluationWrapperBase &fe_eval =
        fe_eval_wrap.getFEEvaluationWrapperBase();

      const dealii::Quadrature<dim> &quadratureRuleAxTemp =
        d_dealiiMatrixFree->get_quadrature(d_matrixFreeQuadCompStiffnessMatrix);

      int isPerformStaticCondensation = (tempvec.linfty_norm() > 1e-10) ? 1 : 0;

      utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
        &isPerformStaticCondensation, 1, utils::mpi::MPIInt, 0, getMPIComm());

      if (isPerformStaticCondensation == 1)
        {
          dealii::VectorizedArray<double> quarter =
            dealii::make_vectorized_array(-1.0 /* / (4.0 * M_PI)*/);
          for (unsigned int macrocell = 0;
               macrocell < d_dealiiMatrixFree->n_cell_batches();
               ++macrocell)
            {
              fe_eval.reinit(macrocell);
              fe_eval.readDoFValuesPlain(tempvec);
              fe_eval.evaluate(dealii::EvaluationFlags::gradients);
              //  for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              //    {
              //      fe_eval.submit_gradient(-quarter *
              //      fe_eval.get_gradient(q), q);
              //    }
              fe_eval.submitInterpolatedGradientsAndMultiply(quarter);
              fe_eval.integrate(dealii::EvaluationFlags::gradients);
              fe_eval.distributeLocalToGlobal(rhs);
            }
        }

      unsigned int matrixFreeQuadratureComponentRhs = 1;
      unsigned int nonTensorStructQuadInRhsCount    = 0;
      auto         iter = d_feBasisDataStorageRhs.begin();
      while (iter != d_feBasisDataStorageRhs.end())
        {
          const quadrature::QuadratureRuleAttributes quadAttr =
            iter->second->getQuadratureRuleContainer()
              ->getQuadratureRuleAttributes();
          const quadrature::QuadratureFamily quadratureFamily =
            quadAttr.getQuadratureFamily();

          if (!(quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
                quadratureFamily == quadrature::QuadratureFamily::GLL ||
                quadratureFamily ==
                  quadrature::QuadratureFamily::GAUSS_SUBDIVIDED))
            {
              // Set up basis Operations for RHS
              basis::FEBasisOperations<ValueTypeOperand,
                                       ValueTypeOperator,
                                       memorySpaceHost,
                                       dim>
                feBasisOperations(iter->second, 1, d_numComponents);

              feBasisOperations.integrateWithBasisValues(
                inpRhs.find(iter->first)->second,
                *d_feBasisManagerHomo,
                d_scratchMultiVecHost);

              for (size_type i = 0; i < d_scratchMultiVecHost.locallyOwnedSize();
                   i++)
                {
                  *(d_nonTensorSructuredQuadeRhs[nonTensorStructQuadInRhsCount]
                      .begin() +
                    i) = d_scratchMultiVecHost.data()[i];
                }
              nonTensorStructQuadInRhsCount += 1;
            }
          else
            {
              // dealii::FEEvaluation<
              // 3,
              // FEOrderElectro,
              // C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>()>
              // fe_eval_density(*d_dealiiMatrixFree,
              //                 d_dofHandlerIndex,
              //                 matrixFreeQuadratureComponentRhs);

              basis::DealiiFEEvaluationWrapper<1> fe_eval_density_wrap(
                d_feOrder,
                d_num1DQuadPointsRhs[iter->first],
                *d_dealiiMatrixFree,
                d_dofHandlerIndex,
                matrixFreeQuadratureComponentRhs);

              basis::FEEvaluationWrapperBase &fe_eval_density =
                fe_eval_density_wrap.getFEEvaluationWrapperBase();

              dealii::AlignedVector<dealii::VectorizedArray<double>> rhoQuads(
                fe_eval_density.totalNumberofQuadraturePoints(),
                dealii::make_vectorized_array(0.0));
              for (unsigned int macrocell = 0;
                   macrocell < d_dealiiMatrixFree->n_cell_batches();
                   ++macrocell)
                {
                  fe_eval_density.reinit(macrocell);

                  std::fill(rhoQuads.begin(),
                            rhoQuads.end(),
                            dealii::make_vectorized_array(0.0));
                  const unsigned int numSubCells =
                    d_dealiiMatrixFree->n_active_entries_per_cell_batch(
                      macrocell);
                  for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                       ++iSubCell)
                    {
                      subCellPtr = d_dealiiMatrixFree->get_cell_iterator(
                        macrocell, iSubCell, d_dofHandlerIndex);
                      dealii::CellId subCellId = subCellPtr->id();
                      unsigned int   cellIndex =
                        d_cellIdToCellIndexMap[subCellId];
                      const double *tempVec =
                        inpRhs.find(iter->first)->second.data() +
                        cellIndex *
                          fe_eval_density.totalNumberofQuadraturePoints();

                      for (unsigned int q = 0;
                           q < fe_eval_density.totalNumberofQuadraturePoints();
                           ++q)
                        rhoQuads[q][iSubCell] = tempVec[q];
                    }


                  // for (unsigned int q = 0; q < fe_eval_density.n_q_points;
                  // ++q)
                  //   {
                  //     fe_eval_density.submit_value(rhoQuads[q], q);
                  //   }
                  fe_eval_density.submitValues(rhoQuads);
                  fe_eval_density.integrate(dealii::EvaluationFlags::values);
                  fe_eval_density.distributeLocalToGlobal(rhs);
                }
              matrixFreeQuadratureComponentRhs++;
            }
          iter++;
        }

      // MPI operation to sync data
      rhs.compress(dealii::VectorOperation::add);

      //  if (d_isReuseSmearedChargeRhs)
      //    rhs += d_rhsSmearedCharge;

      for (auto &a : d_nonTensorSructuredQuadeRhs)
        {
          rhs += a;
        }

      //  if (d_isStoreSmearedChargeRhs)
      //    d_rhsSmearedCharge.compress(dealii::VectorOperation::add);

      // FIXME: check if this is really required
      d_dealiiAffineConstraintMatrix->set_zero(rhs);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const utils::mpi::MPIComm &
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::getMPIComm() const
    {
      return d_feBasisManagerField->getMPIPatternP2P()->mpiCommunicator();
    }


    // solve
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::CGsolve(const double absTolerance,
                                                  const unsigned int
                                                       maxNumberIterations,
                                                  bool distributeFlag)
    {
      // Dispatch to the GPU path when the memory space is DEVICE.
      if constexpr (memorySpace == utils::MemorySpace::DEVICE)
        {
          CGsolveDevice(absTolerance, maxNumberIterations, distributeFlag);
          return;
        }

      // get RHS
      basis::FEEvaluationWrapperBase::distributedCPUVec<double> rhs, gvec, dvec,
        hvec;
      rhs = this->getRhs();

      MPI_Barrier(this->getMPIComm());
      double time = utils::mpi::MPIWtime();

      int rank;
      utils::mpi::MPICommRank(this->getMPIComm(), &rank);
      utils::ConditionalOStream pcout(std::cout, rank == 0);

      bool conv = false; // false : converged; true : converged

      basis::FEEvaluationWrapperBase::distributedCPUVec<double> x =
        this->getInitialGuess();

      double res = 0.0, initial_res = 0.0;
      int    it = 0;

      try
        {
          x.update_ghost_values();

          // resize the vectors, but do not set the values since they'd be
          // overwritten soon anyway.
          gvec.reinit(x, true);
          dvec.reinit(x, true);
          hvec.reinit(x, true);

          gvec.zero_out_ghost_values();
          dvec.zero_out_ghost_values();
          hvec.zero_out_ghost_values();

          double gh    = 0.0;
          double beta  = 0.0;
          double alpha = 0.0;

          // compute residual. if vector is zero, then short-circuit the full
          // computation
          if (!x.all_zero())
            {
              this->vmult(gvec, x);
              gvec.add(-1., rhs);
            }
          else
            {
              // gvec.equ(-1., rhs);
              for (unsigned int i = 0; i < gvec.locally_owned_size(); i++)
                gvec.local_element(i) = -rhs.local_element(i);
            }

          res         = gvec.l2_norm();
          initial_res = res;
          if (res < absTolerance)
            conv = true;
          if (conv)
            {
              pcout << std::endl;
              pcout << "initial abs. residual: " << initial_res
                    << " , current abs. residual: " << res
                    << " , nsteps: " << it
                    << " , abs. tolerance criterion:  " << absTolerance
                    << "\n\n";
              return;
            }
          while ((!conv) && (it < maxNumberIterations))
            {
              it++;

              if (it > 1)
                {
                  this->precondition_Jacobi(hvec, gvec);
                  beta = gh;
                  DFTEFE_AssertWithMsg(std::abs(beta) != 0.,
                                       "Division by zero\n");
                  gh   = gvec * hvec;
                  beta = gh / beta;

                  dvec.sadd(beta, -1., hvec);
                }
              else
                {
                  this->precondition_Jacobi(hvec, gvec);
                  dvec.equ(-1., hvec);
                  gh = gvec * hvec;
                }

              this->vmult(hvec, dvec);
              alpha = dvec * hvec;

              DFTEFE_AssertWithMsg(std::abs(alpha) != 0., "Division by zero\n");
              alpha = gh / alpha;

              for (unsigned int i = 0; i < x.locally_owned_size(); i++)
                x.local_element(i) += alpha * dvec.local_element(i);
              // x.add(alpha, dvec);

              res = std::sqrt(std::abs(gvec.add_and_dot(alpha, hvec, gvec)));

              if (res < absTolerance)
                conv = true;
            }
          if (!conv)
            {
              DFTEFE_AssertWithMsg(false,
                                   "DFT-EFE Error: Solver did not converge\n");
            }

          x.update_ghost_values();

          if (distributeFlag)
            d_constraintsInfo->distribute(x);

          this->setSolution(x);
        }
      catch (...)
        {
          DFTEFE_AssertWithMsg(
            false,
            "DFT-EFE Error: Poisson solver did not converge as per set tolerances."
            "consider increasing MAXIMUM ITERATIONS in Poisson problem parameters."
            "In rare cases for all-electron problems this can also occur due to a known parallel constraints"
            "issue in dealii library.");
          pcout
            << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
          pcout << "Current abs. residual: " << res << std::endl;
        }

      pcout << std::endl;
      pcout << "initial abs. residual: " << initial_res
            << " , current abs. residual: " << res << " , nsteps: " << it
            << " , abs. tolerance criterion:  " << absTolerance << "\n\n";

      utils::mpi::MPIBarrier(this->getMPIComm());
      time = utils::mpi::MPIWtime() - time;

      pcout << "Time for Poisson/Helmholtz problem CG iterations: " << time
            << std::endl;
    }

    // =========================================================================
    // computeAXDevice
    //
    // Computes Ax = K * x on the GPU using dftfe's MatrixFreeWrapperClass.
    // The homogeneous constraints are handled by the wrapper: it distributes
    // constraints before the AX kernel and accumulates contributions back.
    // Ghost values of x must be communicated before entry (handled internally
    // via x.updateGhostValues()).
    // =========================================================================
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
      computeAXDevice(
        linearAlgebra::Vector<ValueTypeOperator, utils::MemorySpace::DEVICE> &Ax,
        linearAlgebra::Vector<ValueTypeOperator, utils::MemorySpace::DEVICE> &x)
    {
      if constexpr (memorySpace != utils::MemorySpace::DEVICE)
        {
          utils::throwException(false,
                                "computeAXDevice called for non-DEVICE memory space.");
          return;
        }

#ifndef DFTEFE_WITH_DEVICE
      utils::throwException(
        false,
        "computeAXDevice requires compilation with DFTEFE_WITH_DEVICE.");
#else
      //pcout << "Enter AX; x l2Norm: " << x.l2Norm()<< "\n"<<std::flush;

      // Zero the output vector (locally owned + ghost).
      Ax.setValue(ValueTypeOperator(0));

      // Update ghost values on device via MPI (uses dft-efe MPIPatternP2P).
      x.updateGhostValues();

      //pcout << "Update Ghost AX; x l2Norm: " << x.l2Norm()<< "\n"<<std::flush;

      // Apply homogeneous constraints on device (sets constrained dofs to 0,
      // distributes slave→master using the constraint matrix supplied to the
      // MatrixFreeWrapperClass constructor).
      d_matrixFreeWrapperDevice->constraintsDistribute(x.data());

      //pcout << "After ConstraintP2C and update AX; x l2Norm: " << x.l2Norm()<< "\n"<<std::flush;
      //pcout << "After ConstraintP2C and update AX; Ax l2Norm: " << Ax.l2Norm()<< "\n"<<std::flush;

      // Execute the matrix-free Laplace AX kernel on device.
      // NOTE: MatrixFree::init() bakes coeff = 1/(4*pi) into d_jacobianFactor
      // for the Laplace operator (DFT-FE convention), but the host AX() uses
      // quarter = 1.0.  Scale Ax by 4*pi to match the host convention.
      d_matrixFreeWrapperDevice->computeAX(Ax.data(), x.data());

      // Scale the full local storage (owned + ghost) by 4*pi so that ghost
      // contributions passed to accumulateAddLocallyOwned() are also corrected.
      linearAlgebra::blasLapack::ascale(
        Ax.locallyOwnedSize() + Ax.ghostSize(),
        ValueTypeOperator(4.0 * M_PI),
        Ax.begin(),
        Ax.begin(),
        *d_linAlgOpContext);

      //pcout << "After Compute AX; x l2Norm: " << x.l2Norm()<< "\n"<<std::flush;
      //pcout << "After Compute AX; Ax l2Norm: " << Ax.l2Norm()<< "\n"<<std::flush;

      // Transpose-distribute: scatter master contributions to slave dofs and
      // accumulate elemental results from ghost dofs back to locally-owned dofs.
      d_matrixFreeWrapperDevice->constraintsDistributeTranspose(Ax.data(),
                                                                x.data());

      //pcout << "ConstraintC2P AX; x l2Norm: " << x.l2Norm()<< "\n"<<std::flush;
      //pcout << "ConstraintC2P AX; Ax l2Norm: " << Ax.l2Norm()<< "\n"<<std::flush;
                                                            
      // MPI reduction: add ghost contributions to locally-owned dofs.
      Ax.accumulateAddLocallyOwned();
      
      //pcout << "Leave AX; x l2Norm: " << x.l2Norm()<< "\n"<<std::flush;
      //  pcout << "Leave AX; Ax l2Norm: " << Ax.l2Norm()<< "\n"<<std::flush;

#endif // DFTEFE_WITH_DEVICE
    }


    // =========================================================================
    // CGsolveDevice
    //
    // Preconditioned conjugate gradient solver running entirely on the GPU.
    // All CG vector operations use dft-efe's blasLapack / linearAlgebra APIs
    // with memorySpace == DEVICE.  The matrix-free AX is performed via
    // computeAXDevice().
    //
    //  * The RHS and Jacobi diagonal are already on device (uploaded by reinit
    //    and computeDiagonalA respectively).
    //  * We start from x = initial on device.  Because the RHS already contains the
    //    static-condensation correction (-K_fc * u_bc), the CG solution gives
    //    the free-dof values directly.  
    //  * Full inhomogeneous constraints are
    //    applied on the CPU after copying the solution back.
    //  * Ghost communication is handled by MultiVector::updateGhostValues() /
    //    accumulateAddLocallyOwned() (dft-efe) and by the MatrixFreeWrapper
    //    constraint routines (dftfe).
    // =========================================================================
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::CGsolveDevice(
      const double       absTolerance,
      const unsigned int maxNumberIterations,
      bool               distributeFlag)
    {
      if constexpr (memorySpace != utils::MemorySpace::DEVICE)
        {
          utils::throwException(false,
                                "CGsolveDevice called for non-DEVICE memory space.");
          return;
        }

#ifndef DFTEFE_WITH_DEVICE
      utils::throwException(
        false,
        "CGsolveDevice requires compilation with DFTEFE_WITH_DEVICE.");
#else

      utils::mpi::MPIBarrier(getMPIComm());
      double time = utils::mpi::MPIWtime();

      // Allocate device CG work vectors from the same MPI layout as d_x.
      linearAlgebra::Vector<ValueTypeOperator, memorySpace> x;
      x = getInitialGuessDevice();
      
      linearAlgebra::Vector<ValueTypeOperator, memorySpace> rhsDevice;
      rhsDevice = getRhsDevice();

      MPI_Barrier(getMPIComm());
      time = MPI_Wtime();

      linearAlgebra::Vector<ValueTypeOperator, memorySpace> &d_Jacobi = *d_diagonalADevice;

      d_xLocalDof = x.locallyOwnedSize() * x.numVectors();

      d_devSum.resize(1);
      d_devSumPtr = d_devSum.data();

      double     res = 0.0, initial_res = 0.0;
      bool       conv = false;
      size_type it   = 0;

      try
        {
          x.updateGhostValues();

          /// reinit temporary vectors for cgsolver device
          d_qvec = x;
          d_rvec = x;
          d_dvec = x;

          d_qvec.setValue(0.);
          d_rvec.setValue(0.);
          d_dvec.setValue(0.);

          double alpha = 0.0;
          double beta  = 0.0;
          double delta = 0.0;

          // r = Ax
          computeAXDevice(d_rvec, x);

          // r = Ax - rhs
          double mOne = -1.0;
         linearAlgebra::blasLapack::axpy(d_xLocalDof, 
                                  mOne, 
                                  rhsDevice.begin(), 
                                  1, 
                                  d_rvec.begin(), 
                                  1,
                                  *d_linAlgOpContext);

          // res = r.r
          res = d_rvec.l2Norm();
          initial_res = res;
          if (res < absTolerance)
            conv = true;
          if (conv)
            return;

          while ((!conv) && (it < maxNumberIterations))
            {
              it++;

              if (it > 1)
                {
                  beta = delta;
                  DFTEFE_AssertWithMsg(std::abs(beta) != 0., "Division by zero\n");

                  // d = M^(-1) * r
                  // delta = d.r
                  delta =
                    applyPreconditionAndComputeDotProduct(d_Jacobi.begin());

                  beta = delta / beta;

                  // q = beta * q - d
                  saddDevice(d_qvec.begin(), d_dvec.begin(), beta, d_xLocalDof);
                }
              else
                {
                  // delta = r.(M^(-1) * r)
                  // q = -M^(-1) * r
                  delta = applyPreconditionComputeDotProductAndSadd(
                    d_Jacobi.begin());
                }

              // d = Aq
              computeAXDevice(d_dvec, d_qvec);

              // alpha = q.d
              dotDevice(d_xLocalDof, d_qvec.begin(), d_dvec.begin(), alpha, *d_linAlgOpContext);

              DFTEFE_AssertWithMsg(std::abs(alpha) != 0.,"Division by zero\n");
              alpha = delta / alpha;

              // res = r.r
              // r += alpha * d
              // x += alpha * q
              res = scaleXRandComputeNorm(x.begin(), alpha);

              if (res < absTolerance)
                conv = true;
            }

          if (!conv)
            {
              DFTEFE_AssertWithMsg(false,
                            "DFT-FE Error: Solver did not converge\n");
            }

          // ------------------------------------------------------------------
          // Copy device solution back to CPU, apply full inhomogeneous
          // constraints, then store via setSolution.
          // ------------------------------------------------------------------
          utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
            d_xLocalDof, d_x.begin(), x.data());

          d_x.update_ghost_values();

          if (distributeFlag)
            d_constraintsInfo->distribute(d_x);

          setSolution(d_x);
        }

      catch (...)
        {
          DFTEFE_AssertWithMsg(
            false,
            "DFT-EFE Error: Poisson solver did not converge as per set tolerances."
            "consider increasing MAXIMUM ITERATIONS in Poisson problem parameters."
            "In rare cases for all-electron problems this can also occur due to a known parallel constraints"
            "issue in dealii library.");
          pcout
            << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
          pcout << "Current abs. residual in Device: " << res << std::endl;
        }

      pcout << std::endl;
      pcout << "initial abs. residual in Device: " << initial_res
            << " , current abs. residual in Device: " << res << " , nsteps: " << it
            << " , abs. tolerance criterion in Device:  " << absTolerance << "\n\n";

      MPI_Barrier(getMPIComm());
      time = MPI_Wtime() - time;

      pcout << "Time for Device Poisson/Helmholtz problem CG iterations: "
            << time << std::endl;
#endif // DFTEFE_WITH_DEVICE
    }


    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    double
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::applyPreconditionAndComputeDotProduct(
    const double *d_jacobi)
  {
    double local_sum = 0.0, sum = 0.0;
    dftefe::utils::deviceMemset(d_devSumPtr, 0, sizeof(double));

    applyPreconditionAndComputeDotProductDevice(
      d_dvec.begin(), d_devSumPtr, d_rvec.begin(), d_jacobi, d_xLocalDof);

    dftefe::utils::MemoryTransfer<
      dftefe::utils::MemorySpace::HOST,
      dftefe::utils::MemorySpace::DEVICE>::copy(1, &local_sum, d_devSum.begin());

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, getMPIComm());

    return sum;
  }


      template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    double
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
  applyPreconditionComputeDotProductAndSadd(
    const double *d_jacobi)
  {
    double local_sum = 0.0, sum = 0.0;
    dftefe::utils::deviceMemset(d_devSumPtr, 0, sizeof(double));

    applyPreconditionComputeDotProductAndSaddDevice(
      d_qvec.begin(), d_devSumPtr, d_rvec.begin(), d_jacobi, d_xLocalDof);

    dftefe::utils::MemoryTransfer<
      dftefe::utils::MemorySpace::HOST,
      dftefe::utils::MemorySpace::DEVICE>::copy(1, &local_sum, d_devSum.begin());

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, getMPIComm());

    return sum;
  }


      template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    double
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
  scaleXRandComputeNorm(double *x, const double &alpha)
  {
    double local_sum = 0.0, sum = 0.0;
    dftefe::utils::deviceMemset(d_devSumPtr, 0, sizeof(double));

    scaleXRandComputeNormDevice(x,
                                d_rvec.begin(),
                                d_devSumPtr,
                                d_qvec.begin(),
                                d_dvec.begin(),
                                alpha,
                                d_xLocalDof);

    dftefe::utils::MemoryTransfer<
      dftefe::utils::MemorySpace::HOST,
      dftefe::utils::MemorySpace::DEVICE>::copy(1, &local_sum, d_devSum.begin());

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, getMPIComm());

    return std::sqrt(sum);
  }


      template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::
  dotDevice(const size_type size, double *x, double *y, double &alpha,
           linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
  {
    double result = linearAlgebra::blasLapack::dot(size,
                                 x,
                                1,
                                y,
                                1,
                                linAlgOpContext);
      MPI_Allreduce(&result,
                    &alpha,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    getMPIComm());
  }

  } // end of namespace electrostatics
} // end of namespace dftefe
