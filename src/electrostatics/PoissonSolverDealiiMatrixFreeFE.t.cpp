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
  namespace electrostatics
  {
    namespace PoissonSolverDealiiMatrixFreeFEInternal
    {
      // solve
      void
      CGsolve(dealiiLinearSolverProblem &problem,
              const double               absTolerance,
              const unsigned int         maxNumberIterations,
              bool                       distributeFlag)
      {
        int this_process;
        MPI_Comm_rank(mpi_communicator, &this_process);
        MPI_Barrier(mpi_communicator);
        double start_time = MPI_Wtime();
        double time;

        // compute RHS
        distributedCPUVec<double> rhs;
        problem.getRhs(rhs);

        MPI_Barrier(mpi_communicator);
        time = MPI_Wtime();

        pcout << "Time for compute rhs: " << time - start_time << std::endl;

        bool conv = false; // false : converged; true : converged

        distributedCPUVec<double> x = problem.getInitialGuess();

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
                problem.vmult(gvec, x);
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
              return;

            while ((!conv) && (it < maxNumberIterations))
              {
                it++;

                if (it > 1)
                  {
                    problem.precondition_Jacobi(hvec, gvec);
                    beta = gh;
                    AssertThrow(std::abs(beta) != 0.,
                                dealii::ExcMessage("Division by zero\n"));
                    gh   = gvec * hvec;
                    beta = gh / beta;
                    dvec.sadd(beta, -1., hvec);
                  }
                else
                  {
                    problem.precondition_Jacobi(hvec, gvec);
                    dvec.equ(-1., hvec);
                    gh = gvec * hvec;
                  }

                problem.vmult(hvec, dvec);
                alpha = dvec * hvec;
                AssertThrow(std::abs(alpha) != 0.,
                            dealii::ExcMessage("Division by zero\n"));
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
                AssertThrow(false,
                            dealii::ExcMessage(
                              "DFT-FE Error: Solver did not converge\n"));
              }

            x.update_ghost_values();

            if (distributeFlag)
              problem.distributeX();

            problem.setSolution(x);
          }
        catch (...)
          {
            AssertThrow(
              false,
              dealii::ExcMessage(
                "DFT-FE Error: Poisson solver did not converge as per set tolerances. consider increasing MAXIMUM ITERATIONS in Poisson problem parameters. In rare cases for all-electron problems this can also occur due to a known parallel constraints issue in dealii library. Try using set CONSTRAINTS FROM SERIAL DOFHANDLER=true under the Boundary conditions subsection."));
            pcout
              << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
            pcout << "Current abs. residual: " << res << std::endl;
          }

        pcout << std::endl;
        pcout << "initial abs. residual: " << initial_res
              << " , current abs. residual: " << res << " , nsteps: " << it
              << " , abs. tolerance criterion:  " << absTolerance << "\n\n";

        MPI_Barrier(mpi_communicator);
        time = MPI_Wtime() - time;

        pcout << "Time for Poisson/Helmholtz problem CG iterations: " << time
              << std::endl;
      }


      getDealiiQuadratureRule(
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
                                 feBasisDataStorage,
        dealii::Quadrature<dim> &quadRuleDealii,
        unsigned int &           num1DQuadPoints)
      {
        const quadrature::QuadratureRuleAttributes quadAttr =
          feBasisDataStorage.getQuadratureRuleContainer()
            ->getQuadratureRuleAttributes();
        const quadrature::QuadratureFamily quadratureFamily =
          quadAttr.getQuadratureFamily();

        utils::throwException(
          quadratureFamily == quadrature::QuadratureFamily::GAUSS ||
            quadratureFamily == quadrature::QuadratureFamily::GLL ||
            quadratureFamily == quadrature::QuadratureFamily::GAUSS_SUBDIVIDED,
          "The quadrature rule has to be uniform quadrature like GAUSS , GLL or GAUSS_SUBDIVIDED for Dealii Matrix Free.");

        num1DQuadPoints = quadAttr.getNum1DPoints();

        if (auto cfeBDSDealii = std::dynamic_pointer_cast<
              const basis::CFEBDSOnTheFlyComputeDealii<ValueTypeOperand,
                                                       ValueTypeOperator,
                                                       memorySpace,
                                                       dim>>(
              feBasisDataStorage))
          quadRuleDealii = cfeBDSDealii->getDealiiQuadratureRule();
        else if (auto cfeBDSDealii = std::dynamic_pointer_cast<
                   const basis::CFEBasisDataStorageDealii<ValueTypeOperand,
                                                          ValueTypeOperator,
                                                          memorySpace,
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
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>
          feBasisDataStorageStiffnessMatrix,
        const std::map<
          std::string,
          std::shared_ptr<
            const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>>
          &feBasisDataStorageRhs,
        const std::map<
          std::string,
          const quadrature::QuadratureValuesContainer<ValueType, memorySpace> &>
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
      , d_rootCout(std::cout)
      , d_dealiiMatrixFree(
          std::make_shared<dealii::MatrixFree<dim, ValueTypeOperator>>())
      , d_dofHandlerIndex(0)
      , d_feBasisDataStorageRhs(feBasisDataStorageRhs)
      , d_matrixFreeQuadCompStiffnessMatrix(0)
    {
      utils::throwException(
        d_numComponents == 1,
        "Number Components of QuadratureValuesContainer has to be 1 for Dealii Matrix Free Poisson Solve.");

      // Check wether the dofhandler and constrints come from classical basis or
      // not
      const basis::FEBasisDofHandler<ValueTypeOperator, memorySpace, dim>
        &cfeBasisDofHandlerDealii = std::dynamic_cast<
          const basis::
            CFEBasisDofHandlerDealii<ValueTypeOperator, memorySpace, dim> &>(
          feBasisManagerField->getBasisDofHandler());
      utils::throwException(
        cfeBasisDofHandlerDealii != nullptr,
        "Could not cast BasisDofHandler to CFEBasisDofHandlerDealii "
        "in PoissonSolverDealiiMatrixFreeFE.");

      // Set up Homogoneous Constraints BasisManager
      std::shared_ptr<const utils::ScalarSpatialFunctionReal> zeroFunction =
        std::make_shared<utils::ScalarZeroFunctionReal>();

      basis::
        FEBasisManager<ValueTypeOperand, ValueTypeOperator, memorySpace, dim>
          feBasisManagerHomo =
            std::make_shared<basis::FEBasisManager<ValueTypeOperand,
                                                   ValueTypeOperator,
                                                   memorySpace,
                                                   dim>>(
              cfeBasisDofHandlerDealii, zeroFunction);

      const basis::CFEConstraintsLocalDealii<ValueTypeOperator,
                                             memorySpace,
                                             dim> &cfeConstraintsLocalDealii =
        std::dynamic_cast<
          const basis::
            CFEConstraintsLocalDealii<ValueTypeOperator, memorySpace, dim> &>(
          feBasisManagerHomo->getConstraints());
      utils::throwException(
        cfeConstraintsLocalDealii != nullptr,
        "Could not cast ConstraintsLocal to CFEConstraintsLocalDealii "
        "in PoissonSolverDealiiMatrixFreeFE.");

      // will this work or it has to be data member ?
      std::vector<dealii::Quadrature<dim>> dealiiQuadratureRuleVec(
        1 + inpRhs.size(), dealii::Quadrature<dim>());

      d_feOrder          = cfeBasisDofHandlerDealii.getFEOrder(0);
      d_dealiiDofHandler = cfeBasisDofHandlerDealii.getDoFHandler();
      d_dealiiAffineConstraintMatrix =
        &cfeConstraintsLocalDealii.getAffineConstraints();

      PoissonSolverDealiiMatrixFreeFEInternal::getDealiiQuadratureRule(
        feBasisDataStorageStiffnessMatrix,
        dealiiQuadratureRuleVec[0],
        d_num1DQuadPointsStiffnessMatrix);

      int  count = 1;
      auto iter  = feBasisDataStorageRhs.begin();
      d_num1DQuadPointsRhs.clear();
      while (iter != feBasisDataStorageRhs.end())
        {
          PoissonSolverDealiiMatrixFreeFEInternal::getDealiiQuadratureRule(
            iter->second,
            dealiiQuadratureRuleVec[count],
            d_num1DQuadPointsRhs[iter->first]);
          count += 1;
        }

      // create dealiiMatrixFree
      dealii::MappingQ1<dim> mappingDealii;
      d_dealiiMatrixFree->reinit(
        mappingDealii,
        std::vector<const dealii::DoFHandler<dim> *>{d_dealiiDofHandler.get()},
        std::vector<const dealii::AffineConstraints<ValueTypeOperand> *>{
          d_dealiiAffineConstraintMatrix},
        dealiiQuadratureRuleVec);

      d_x(d_dealiiMatrixFree->get_vector_partitioner(d_dofHandlerIndex));
      d_initial(d_x);

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
          linAlgOpContext)
      : PoissonSolverDealiiMatrixFreeFE(
          feBasisManagerField,
          feBasisDataStorageStiffnessMatrix,
          std::map<
            std::string,
            std::shared_ptr<
              const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>>>(
            {{"Field", feBasisDataStorageRhs}}),
          std::map<std::string,
                   const quadrature::QuadratureValuesContainer<
                     linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                            ValueTypeOperand>,
                     memorySpace> &>({{"Field", inpRhs}}),
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
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        const std::map<std::string,
                       const quadrature::QuadratureValuesContainer<
                         linearAlgebra::blasLapack::
                           scalar_type<ValueTypeOperator, ValueTypeOperand>,
                         memorySpace> &> &                inpRhs)
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
                                             memorySpace,
                                             dim> &cfeConstraintsLocalDealii =
        std::dynamic_cast<
          const basis::
            CFEConstraintsLocalDealii<ValueTypeOperator, memorySpace, dim> &>(
          d_feBasisManagerField->getConstraints());
      utils::throwException(
        cfeConstraintsLocalDealii != nullptr,
        "Could not cast ConstraintsLocal to CFEConstraintsLocalDealii "
        "in PoissonSolverDealiiMatrixFreeFE.");

      *d_constraintsInfo = &cfeConstraintsLocalDealii.getAffineConstraints();

      // Compute RHS
      computeRhs(d_rhs, inpRhs);
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
                                                    memorySpace,
                                                    dim>> feBasisManagerField,
        const quadrature::QuadratureValuesContainer<
          linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                 ValueTypeOperand>,
          memorySpace> &inpRhs)
    {
      std::map<std::string,
               const quadrature::QuadratureValuesContainer<
                 linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                                        ValueTypeOperand>,
                 memorySpace> &>
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
      dim>::setSolution(distributedCPUVec<ValueType> &x)
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
                  memorySpace> &solution)
    {
      solution.setValue(0.0);

      for (size_type i = 0; i < solution.locallyOwnedSize(); i++)
        {
          solution.data()[j] = d_x.data()[j];
        }

      solution.updateGhostValues();

      d_feBasisManagerField->getConstraints().distributeParentToChild(solution,
                                                                      1);

      // this is done for a particular case for poisson solve each
      // scf guess but have to be modified with a reinit parameter
      d_initial = d_x;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    const distributedCPUVec<ValueTypeOperand> &
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
    const distributedCPUVec<ValueType> &
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
    void
    PoissonSolverDealiiMatrixFreeFE<ValueTypeOperator,
                                    ValueTypeOperand,
                                    memorySpace,
                                    dim>::solve(const double absTolerance,
                                                const unsigned int
                                                  maxNumberIterations)
    {
      PoissonSolverDealiiMatrixFreeFEInternal::CGsolve(*this,
                                                       absTolerance,
                                                       maxNumberIterations,
                                                       true);
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
        dealii::make_vectorized_array(1.0 / (4.0 * M_PI));

      //  dealii::FEEvaluation<dim, FEOrderElectro, FEOrderElectro + 1> fe_eval(
      //    matrixFreeData,
      //    d_dofHandlerIndex,
      //    d_matrixFreeQuadCompStiffnessMatrix);

      basis::DealiiFEEvaluationWrapper<1> fe_eval(
        d_feOrder,
        d_num1DQuadPointsStiffnessMatrix,
        *d_dealiiMatrixFree,
        d_dofHandlerIndex,
        d_matrixFreeQuadCompStiffnessMatrix);

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
                                      dealii::update_values |
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
                elementalDiagonalA(i) += (1.0 / (4.0 * M_PI)) *
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
                     memorySpace> &> &inpRhs)
    {
      dealii::DoFHandler<dim>::active_cell_iterator subCellPtr;
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

      //  dealii::FEEvaluation<dim, FEOrderElectro, FEOrderElectro + 1> fe_eval(
      //    *d_dealiiMatrixFree,
      //    d_dofHandlerIndex,
      //    d_matrixFreeQuadCompStiffnessMatrix);

      basis::DealiiFEEvaluationWrapper<1> fe_eval(
        d_feOrder,
        d_num1DQuadPointsStiffnessMatrix,
        *d_dealiiMatrixFree,
        d_dofHandlerIndex,
        d_matrixFreeQuadCompStiffnessMatrix);

      const dealii::Quadrature<dim> &quadratureRuleAxTemp =
        d_dealiiMatrixFree->get_quadrature(d_matrixFreeQuadCompStiffnessMatrix);

      int isPerformStaticCondensation = (tempvec.linfty_norm() > 1e-10) ? 1 : 0;

      MPI_Bcast(&isPerformStaticCondensation, 1, MPI_INT, 0, mpi_communicator);

      if (isPerformStaticCondensation == 1)
        {
          dealii::VectorizedArray<double> quarter =
            dealii::make_vectorized_array(1.0 / (4.0 * M_PI));
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

      unsigned int matrixFreeQuadratureComponentRhs = 0;
      auto         iter = d_feBasisDataStorageRhs.begin();
      while (iter != d_feBasisDataStorageRhs.end())
        {
          // dealii::FEEvaluation<
          // 3,
          // FEOrderElectro,
          // C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>()>
          // fe_eval_density(*d_dealiiMatrixFree,
          //                 d_dofHandlerIndex,
          //                 matrixFreeQuadratureComponentRhs);

          basis::DealiiFEEvaluationWrapper<1> fe_eval_density(
            d_feOrder,
            d_num1DPointsRhs[iter->first],
            *d_dealiiMatrixFree,
            d_dofHandlerIndex,
            matrixFreeQuadratureComponentRhs);

          dealii::AlignedVector<dealii::VectorizedArray<double>> rhoQuads(
            fe_eval_density.n_q_points, dealii::make_vectorized_array(0.0));
          for (unsigned int macrocell = 0;
               macrocell < d_dealiiMatrixFree->n_cell_batches();
               ++macrocell)
            {
              fe_eval_density.reinit(macrocell);

              std::fill(rhoQuads.begin(),
                        rhoQuads.end(),
                        dealii::make_vectorized_array(0.0));
              const unsigned int numSubCells =
                d_dealiiMatrixFree->n_active_entries_per_cell_batch(macrocell);
              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                {
                  subCellPtr =
                    d_dealiiMatrixFree->get_cell_iterator(macrocell,
                                                          iSubCell,
                                                          d_dofHandlerIndex);
                  dealii::CellId subCellId = subCellPtr->id();
                  unsigned int   cellIndex =
                    d_basisOperationsPtr->cellIndex(subCellId);
                  const double *tempVec =
                    inpRhs[iter->first].data() +
                    cellIndex * fe_eval_density.n_q_points;

                  for (unsigned int q = 0; q < fe_eval_density.n_q_points; ++q)
                    rhoQuads[q][iSubCell] = tempVec[q];
                }


              // for (unsigned int q = 0; q < fe_eval_density.n_q_points; ++q)
              //   {
              //     fe_eval_density.submit_value(rhoQuads[q], q);
              //   }
              fe_eval_density.submitValues(rhoQuads);
              fe_eval_density.integrate(dealii::EvaluationFlags::values);
              fe_eval_density.distributeLocalToGlobal(rhs);
            }
          matrixFreeQuadratureComponentRhs++;
        }

      // MPI operation to sync data
      rhs.compress(dealii::VectorOperation::add);

      //  if (d_isReuseSmearedChargeRhs)
      //    rhs += d_rhsSmearedCharge;

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

  } // end of namespace electrostatics
} // end of namespace dftefe
