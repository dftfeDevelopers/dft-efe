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
 * @author Bikash Kanungo
 */

#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace CGLinearSolverInternal
    {
      template <typename T>
      std::string
      toStringWithPrecision(const T a_value, const int n)
      {
        std::ostringstream out;
        out.precision(n);
        out << std::fixed << a_value;
        return out.str();
      }
    } // namespace CGLinearSolverInternal

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::
      CGLinearSolver(const size_type       maxIter,
                     const double          absoluteTol,
                     const double          relativeTol,
                     const double          divergenceTol,
                     LinearAlgebraProfiler profiler)
      : d_maxIter(maxIter)
      , d_absoluteTol(absoluteTol)
      , d_relativeTol(relativeTol)
      , d_divergenceTol(divergenceTol)
      , d_profiler(profiler)
    {}

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    LinearSolverError
    CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::solve(
      LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &linearSolverFunction)
    {
      LinearSolverError retunValue;
      auto              mpiComm = linearSolverFunction.getMPIComm();

      // register the start of the algorithm
      d_profiler.registerStart(mpiComm);

      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

      const MultiVector<ValueType, memorySpace> &b =
        linearSolverFunction.getRhs();

      const utils::mpi::MPIComm &comm = b.getMPIPatternP2P()->mpiCommunicator();
      int                        rank;
      utils::mpi::MPICommRank(comm, &rank);

      std::vector<double> bNorm(0);
      bNorm = b.l2Norms();

      size_type numComponents = b.getNumberComponents();

      MultiVector<ValueTypeOperand, memorySpace> x;
      x = linearSolverFunction.getInitialGuess();
      MultiVector<ValueTypeOperand, memorySpace> xConverged;
      xConverged = linearSolverFunction.getInitialGuess();

      // get handle to Ax
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &AxContext = linearSolverFunction.getAxContext();

      // get handle to the preconditioner
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &pcContext = linearSolverFunction.getPCContext();

      //
      // Notations used
      // x = approx. solution
      // r = residual = Ax - b
      // z = preconditioned residual = PC(r)
      // p = search direction
      // w = Ap
      // alpha = step length
      // beta = improvement relative to previous step
      //

      MultiVector<ValueType, memorySpace> r(b, 0.0);
      MultiVector<ValueType, memorySpace> w(b, 0.0);
      MultiVector<ValueType, memorySpace> z(b, 0.0);
      MultiVector<ValueType, memorySpace> p(b, 0.0);

      utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

      //
      // CG loop
      //
      std::vector<double>   rNorm(numComponents, 0.0);
      size_type             precision = d_profiler.getPrecision();
      LinearSolverErrorCode err       = LinearSolverErrorCode::OTHER_ERROR;
      size_type             iter      = 0;
      std::vector<bool>     convergeFlag(0);
      convergeFlag.resize(numComponents, false);
      bool                   divergeFlag  = false;
      bool                   allConverged = false;
      std::vector<ValueType> ones(0);
      ones.resize(numComponents, (ValueType)1.0);
      std::vector<ValueType> nOnes(0);
      nOnes.resize(numComponents, (ValueType)-1.0);
      std::vector<ValueType> alpha(numComponents, (ValueType)0),
        nAlpha(numComponents, (ValueType)0), beta(numComponents, (ValueType)0);

      //
      // @note: w is meant for storing Ap (p = search direction).
      // However, for the 0-th iteration we use it to store Ax
      // to avoid allocating memory for another Vector
      AxContext.apply(x, w, true, true);

      add(ones, b, nOnes, w, r);

      //
      // z = preconditioned r
      //
      pcContext.apply(r, z, false, false);

      //
      // p = z
      //
      p = z;

      // for (unsigned int h = 0 ; h < z.locallyOwnedSize() ; h++)
      // {
      //   for (unsigned int i = 0 ; i < z.getNumberComponents() ; i++)
      //     std::cout << "z[" <<iter << "," << h <<"] : "<< *(hostData.data()+
      //     h*z.getNumberComponents() + i)<< ",";
      //   std::cout << std::endl;
      // }

      for (iter = 1; iter <= d_maxIter; ++iter)
        {
          std::vector<double> rNorm(0);
          rNorm = r.l2Norms();

          // if (rank == 0)
          //   {
          //     if (iter % 100 == 0)
          //       {
          //         for (unsigned int i = 0; i < numComponents; i++)
          //           std::cout << rNorm[i] << ",";
          //         std::cout << "\n";
          //       }
          //   }

          std::string msg;
          for (size_type i = 0; i < numComponents; i++)
            {
              if (rNorm[i] <
                    std::max(d_absoluteTol, bNorm[i] * d_relativeTol) &&
                  convergeFlag[i] == false)
                {
                  err             = LinearSolverErrorCode::SUCCESS;
                  convergeFlag[i] = true;
                  // copy x to xConverged of the ith comp vector
                  // for (size_type j = 0; j < x.locallyOwnedSize(); j++)
                  //   *(xConverged.data() + j * numComponents + i) =
                  //     *(x.data() + j * numComponents + i);

                  // for (size_type iSize = 0; iSize < x.locallyOwnedSize();
                  //      iSize++)
                  //   memoryTransfer.copy(1,
                  //                       xConverged.data() +
                  //                         numComponents * iSize + i,
                  //                       x.data() + iSize * numComponents +
                  //                       i);

                  blasLapack::stridedBlockCopy(x.locallyOwnedSize(),
                                               1,
                                               numComponents, // srcLeadingDim
                                               i,             // srcBlockStartId
                                               numComponents, // dstLeadingDim
                                               i,             // dstBlockStartId
                                               x.data(),
                                               xConverged.data(),
                                               *x.getLinAlgOpContext());
                }

              if (rNorm[i] > d_divergenceTol && divergeFlag == false)
                {
                  err         = LinearSolverErrorCode::RESIDUAL_DIVERGENCE;
                  divergeFlag = true;
                }

              msg = "CGLinearSolver[" + std::to_string(iter) + "]";
              msg += " Abs. residual: " +
                     CGLinearSolverInternal::toStringWithPrecision(rNorm[i],
                                                                   precision);
              msg += " Rel. residual: " +
                     CGLinearSolverInternal::toStringWithPrecision(rNorm[i] /
                                                                     bNorm[i],
                                                                   precision);
              msg += "For Vector Component No: " + i;
            }

          if (std::all_of(convergeFlag.begin(),
                          convergeFlag.end(),
                          [](bool boolVal) { return boolVal; }))
            allConverged = true;

          if (divergeFlag || allConverged)
            break;

          // register start of the iteration
          // w = Ap
          AxContext.apply(p, w, true, true);

          // z^Hr (dot product of z-conjugate and r)
          std::vector<ValueType> zDotr(0);
          dot(z,
              r,
              zDotr,
              blasLapack::ScalarOp::Conj,
              blasLapack::ScalarOp::Identity);

          // p^Hw (dot product of p-conjugate and w)
          std::vector<ValueType> pDotw(0);
          dot(p,
              w,
              pDotw,
              blasLapack::ScalarOp::Conj,
              blasLapack::ScalarOp::Identity);

          // ValueType alpha = zDotr / pDotw;
          for (size_type i = 0; i < numComponents; i++)
            {
              alpha[i]  = (zDotr[i] / pDotw[i]);
              nAlpha[i] = (-zDotr[i] / pDotw[i]);
            }

          // x = x + alpha*p
          add(ones, x, alpha, p, x);

          // r = r - alpha*w
          add(ones, r, nAlpha, w, r);

          // z = preconditioned r
          pcContext.apply(r, z, false, false);

          // updated z^Hr (dot product of new z-conjugate and r)
          std::vector<ValueType> zDotrNew(0);
          dot(z,
              r,
              zDotrNew,
              blasLapack::ScalarOp::Conj,
              blasLapack::ScalarOp::Identity);

          // ValueType beta = zDotrNew / zDotr;
          for (size_type i = 0; i < numComponents; i++)
            {
              beta[i] = (zDotrNew[i] / zDotr[i]);
            }
          // p = z + beta*p
          add(ones, z, beta, p, p);

          // register end of the iteration
          // d_profiler.registerIterEnd(msg);
        }

      linearSolverFunction.setSolution(xConverged);

      if (iter > d_maxIter)
        {
          err = LinearSolverErrorCode::FAILED_TO_CONVERGE;
        }

      std::string msg = "";
      retunValue      = LinearSolverErrorMsg::isSuccessAndMsg(err);

      if (retunValue.isSuccess)
        {
          msg = "CGLinear solve converged in maximum " + std::to_string(iter) +
                " iterations";
        }
      else
        msg = retunValue.msg;

      // register end of CG
      // --------- TODO : fix the profiler class------------
      if (rank == 0)
        d_profiler.registerEnd(msg);

      return retunValue;
    }

    // template <typename ValueTypeOperator,
    //           typename ValueTypeOperand,
    //           utils::MemorySpace memorySpace>
    // LinearSolverError
    // CGLinearSolver<ValueTypeOperator, ValueTypeOperand, memorySpace>::solve(
    //   LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, memorySpace>
    //     &linearSolverFunction)
    // {
    //   LinearSolverError retunValue;
    //   auto              mpiComm = linearSolverFunction.getMPIComm();

    //   // register the start of the algorithm
    //   d_profiler.registerStart(mpiComm);

    //   using ValueType =
    //     blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

    //   const MultiVector<ValueType, memorySpace> &b =
    //     linearSolverFunction.getRhs();

    //   const utils::mpi::MPIComm &comm =
    //   b.getMPIPatternP2P()->mpiCommunicator(); int rank;
    //   utils::mpi::MPICommRank(comm, &rank);

    //   std::vector<double> bNorm(0);
    //   bNorm = b.l2Norms();

    //   size_type numComponents = b.getNumberComponents();

    //   MultiVector<ValueTypeOperand, memorySpace> x;
    //   x = linearSolverFunction.getInitialGuess();
    //   MultiVector<ValueTypeOperand, memorySpace> xConverged;
    //   xConverged = linearSolverFunction.getInitialGuess();

    //   // get handle to Ax
    //   const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
    //     &AxContext = linearSolverFunction.getAxContext();

    //   // get handle to the preconditioner
    //   const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
    //     &pcContext = linearSolverFunction.getPCContext();

    //   //
    //   // Notations used
    //   // x = approx. solution
    //   // r = residual = Ax - b
    //   // z = preconditioned residual = PC(r)
    //   // p = search direction
    //   // w = Ap
    //   // alpha = step length
    //   // beta = improvement relative to previous step
    //   //

    //   MultiVector<ValueType, memorySpace> r(b, 0.0);
    //   MultiVector<ValueType, memorySpace> w(b, 0.0);
    //   MultiVector<ValueType, memorySpace> z(b, 0.0);
    //   MultiVector<ValueType, memorySpace> p(b, 0.0);

    //   utils::MemoryTransfer<memorySpace, memorySpace> memoryTransfer;

    //   //
    //   // CG loop
    //   //
    //   std::vector<double>   rNorm(numComponents, 0.0);
    //   size_type             precision = d_profiler.getPrecision();
    //   LinearSolverErrorCode err       = LinearSolverErrorCode::OTHER_ERROR;
    //   size_type             iter      = 0;
    //   std::vector<bool>     convergeFlag(0);
    //   convergeFlag.resize(numComponents, false);
    //   bool                   divergeFlag  = false;
    //   bool                   allConverged = false;
    //   std::vector<ValueType> ones(0);
    //   ones.resize(numComponents, (ValueType)1.0);
    //   std::vector<ValueType> nOnes(0);
    //   nOnes.resize(numComponents, (ValueType)-1.0);
    //         // std::cout << std::endl;
    //         //           utils::MemoryStorage<ValueType,
    //         utils::MemorySpace::HOST>
    //         //                  hostData(p.locallyOwnedSize() *
    //         p.getNumberComponents(), 0.0);

    //   for (; iter <= d_maxIter; ++iter)
    //     {
    //       // register start of the iteration
    //       // d_profiler.registerIterStart(iter);
    //       if (iter == 0)
    //         {
    //           //
    //           // @note: w is meant for storing Ap (p = search direction).
    //           // However, for the 0-th iteration we use it to store Ax
    //           // to avoid allocating memory for another Vector
    //           AxContext.apply(x, w, true, true);
    //           //  std::cout << std::endl;
    //           //         hostData.template copyFrom<memorySpace>(x.data());
    //           //         ValueType sumx = 0;
    //           //         for(int i = 0 ; i< hostData.size() ; i++ )
    //           //         {
    //           //           sumx += *(hostData.data() + i);
    //           //         }
    //           //         utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //           //           utils::mpi::MPIInPlace,
    //           //           &sumx,
    //           //           1,
    //           //           utils::mpi::Types<ValueType>::getMPIDatatype(),
    //           //           utils::mpi::MPISum,
    //           //           z.getMPIPatternP2P()->mpiCommunicator());
    //           //         std::cout << "sumx: "<< sumx << std::endl;
    //           // std::cout << std::endl;
    //           //         hostData.template copyFrom<memorySpace>(w.data());
    //           //         ValueType sumw = 0;
    //           //         for(int i = 0 ; i< hostData.size() ; i++ )
    //           //         {
    //           //           sumw += *(hostData.data() + i);
    //           //         }
    //           //         utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //           //           utils::mpi::MPIInPlace,
    //           //           &sumw,
    //           //           1,
    //           //           utils::mpi::Types<ValueType>::getMPIDatatype(),
    //           //           utils::mpi::MPISum,
    //           //           z.getMPIPatternP2P()->mpiCommunicator());
    //           //         std::cout << "sumw: "<< sumw << std::endl;

    //           add(ones, b, nOnes, w, r);

    //           //
    //           // z = preconditioned r
    //           //
    //           pcContext.apply(r, z, false, false);

    //           //
    //           // p = z
    //           //
    //           p = z;

    //           // for (unsigned int h = 0 ; h < z.locallyOwnedSize() ; h++)
    //           // {
    //           //   for (unsigned int i = 0 ; i < z.getNumberComponents() ;
    //           i++)
    //           //     std::cout << "z[" <<iter << "," << h <<"] : "<<
    //           *(hostData.data()+ h*z.getNumberComponents() + i)<< ",";
    //           //   std::cout << std::endl;
    //           // }
    //         }
    //       else
    //         {
    //           // w = Ap
    //           AxContext.apply(p, w, true, true);
    //           //  std::cout << std::endl;
    //           //         hostData.template copyFrom<memorySpace>(p.data());
    //           //         ValueType sump = 0;
    //           //         for(int i = 0 ; i< hostData.size() ; i++ )
    //           //         {
    //           //           sump += *(hostData.data() + i);
    //           //         }
    //           //         utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //           //           utils::mpi::MPIInPlace,
    //           //           &sump,
    //           //           1,
    //           //           utils::mpi::Types<ValueType>::getMPIDatatype(),
    //           //           utils::mpi::MPISum,
    //           //           z.getMPIPatternP2P()->mpiCommunicator());
    //           //         std::cout << "sump: "<< sump << std::endl;
    //           // std::cout << std::endl;
    //           //         hostData.template copyFrom<memorySpace>(w.data());
    //           //         ValueType sumw = 0;
    //           //         for(int i = 0 ; i< hostData.size() ; i++ )
    //           //         {
    //           //           sumw += *(hostData.data() + i);
    //           //         }
    //           //         utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //           //           utils::mpi::MPIInPlace,
    //           //           &sumw,
    //           //           1,
    //           //           utils::mpi::Types<ValueType>::getMPIDatatype(),
    //           //           utils::mpi::MPISum,
    //           //           z.getMPIPatternP2P()->mpiCommunicator());
    //           //         std::cout << "sumw: "<< sumw << std::endl;
    //           // z^Hr (dot product of z-conjugate and r)
    //           std::vector<ValueType> zDotr(0);
    //           dot(z,
    //               r,
    //               zDotr,
    //               blasLapack::ScalarOp::Conj,
    //               blasLapack::ScalarOp::Identity);
    //         // std::cout << std::endl;
    //         //           hostData.template copyFrom<memorySpace>(z.data());
    //         //           ValueType sumz = 0;
    //         //           for(int i = 0 ; i< hostData.size() ; i++ )
    //         //           {
    //         //             sumz += *(hostData.data() + i);
    //         //           }
    //         //           utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //         //             utils::mpi::MPIInPlace,
    //         //             &sumz,
    //         //             1,
    //         //             utils::mpi::Types<ValueType>::getMPIDatatype(),
    //         //             utils::mpi::MPISum,
    //         //             z.getMPIPatternP2P()->mpiCommunicator());
    //         //           std::cout << "sumz: "<< sumz << std::endl;

    //         // std::cout << std::endl;
    //         //           hostData.template copyFrom<memorySpace>(r.data());
    //         //           ValueType sumr = 0;
    //         //           for(int i = 0 ; i< hostData.size() ; i++ )
    //         //           {
    //         //             sumr += *(hostData.data() + i);
    //         //           }
    //         //           utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //         //             utils::mpi::MPIInPlace,
    //         //             &sumr,
    //         //             1,
    //         //             utils::mpi::Types<ValueType>::getMPIDatatype(),
    //         //             utils::mpi::MPISum,
    //         //             z.getMPIPatternP2P()->mpiCommunicator());
    //         //           std::cout << "sumr: "<< sumr << std::endl;

    //           // p^Hw (dot product of p-conjugate and w)
    //           std::vector<ValueType> pDotw(0);
    //           dot(p,
    //               w,
    //               pDotw,
    //               blasLapack::ScalarOp::Conj,
    //               blasLapack::ScalarOp::Identity);

    //           // ValueType alpha = zDotr / pDotw;
    //           std::vector<ValueType> alpha(0), nAlpha(0);
    //           for (size_type i = 0; i < numComponents; i++)
    //             {
    //               alpha.push_back(zDotr[i] / pDotw[i]);
    //               nAlpha.push_back(-zDotr[i] / pDotw[i]);
    //               //std::cout << "alpha: "<<alpha[i] <<  " zDotr[i]:
    //               "<<zDotr[i] << " pDotw[i]: "<<pDotw[i]<< std::endl;
    //             }

    //           // x = x + alpha*p
    //           add(ones, x, alpha, p, x);

    //           // r = r - alpha*w
    //           add(ones, r, nAlpha, w, r);

    //           // z = preconditioned r
    //           pcContext.apply(r, z, false, false);

    //           // updated z^Hr (dot product of new z-conjugate and r)
    //           std::vector<ValueType> zDotrNew(0);
    //           dot(z,
    //               r,
    //               zDotrNew,
    //               blasLapack::ScalarOp::Conj,
    //               blasLapack::ScalarOp::Identity);

    //           // ValueType beta = zDotrNew / zDotr;
    //           std::vector<ValueType> beta(0);
    //           for (size_type i = 0; i < numComponents; i++)
    //           {
    //             beta.push_back(zDotrNew[i] / zDotr[i]);
    //               //std::cout << "beta: "<<beta[i] << " zDotrNew[i]:
    //               "<<zDotrNew[i]<< " zDotr[i]: "<<zDotr[i]<<std::endl;
    //           }
    //         //    std::cout << std::endl;
    //         //           hostData.template copyFrom<memorySpace>(p.data());
    //         //           sump = 0;
    //         //           for(int i = 0 ; i< hostData.size() ; i++ )
    //         //           {
    //         //             sump += *(hostData.data() + i);
    //         //           }
    //         //           utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //         //             utils::mpi::MPIInPlace,
    //         //             &sump,
    //         //             1,
    //         //             utils::mpi::Types<ValueType>::getMPIDatatype(),
    //         //             utils::mpi::MPISum,
    //         //             z.getMPIPatternP2P()->mpiCommunicator());
    //         //           std::cout << "sump prev: "<< sump << std::endl;

    //         // std::cout << std::endl;
    //         //           hostData.template copyFrom<memorySpace>(z.data());
    //         //           sumz = 0;
    //         //           for(int i = 0 ; i< hostData.size() ; i++ )
    //         //           {
    //         //             sumz += *(hostData.data() + i);
    //         //           }
    //         //           utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //         //             utils::mpi::MPIInPlace,
    //         //             &sumz,
    //         //             1,
    //         //             utils::mpi::Types<ValueType>::getMPIDatatype(),
    //         //             utils::mpi::MPISum,
    //         //             z.getMPIPatternP2P()->mpiCommunicator());
    //                   // std::cout << "sumz prev: "<< sumz << std::endl;
    //           // p = z + beta*p
    //           add(ones, z, beta, p, p);
    //           //  std::cout << std::endl;
    //           //         hostData.template copyFrom<memorySpace>(p.data());
    //           //         sump = 0;
    //           //         for(int i = 0 ; i< hostData.size() ; i++ )
    //           //         {
    //           //           sump += *(hostData.data() + i);
    //           //         }
    //           //         utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
    //           //           utils::mpi::MPIInPlace,
    //           //           &sump,
    //           //           1,
    //           //           utils::mpi::Types<ValueType>::getMPIDatatype(),
    //           //           utils::mpi::MPISum,
    //           //           z.getMPIPatternP2P()->mpiCommunicator());
    //           //         std::cout << "sump later: "<< sump << std::endl;

    //           // if constexpr (memorySpace == utils::MemorySpace::DEVICE)
    //           //   dftefe::utils::deviceSynchronize();
    //         }

    //       std::vector<double> rNorm(0);
    //       rNorm = r.l2Norms();

    //       // if (rank == 0)
    //       //   {
    //       //     if (iter % 100 == 0)
    //       //       {
    //       //         for (unsigned int i = 0; i < numComponents; i++)
    //       //           std::cout << rNorm[i] << ",";
    //       //         std::cout << "\n";
    //       //       }
    //       //   }

    //       std::string msg;
    //       for (size_type i = 0; i < numComponents; i++)
    //         {
    //           if (rNorm[i] <
    //                 std::max(d_absoluteTol, bNorm[i] * d_relativeTol) &&
    //               convergeFlag[i] == false)
    //             {
    //               err             = LinearSolverErrorCode::SUCCESS;
    //               convergeFlag[i] = true;
    //               // copy x to xConverged of the ith comp vector
    //               // for (size_type j = 0; j < x.locallyOwnedSize(); j++)
    //               //   *(xConverged.data() + j * numComponents + i) =
    //               //     *(x.data() + j * numComponents + i);

    //               for (size_type iSize = 0; iSize < x.locallyOwnedSize();
    //                    iSize++)
    //                 memoryTransfer.copy(1,
    //                                     xConverged.data() +
    //                                       numComponents * iSize + i,
    //                                     x.data() + iSize * numComponents +
    //                                     i);
    //             }

    //           if (rNorm[i] > d_divergenceTol && divergeFlag == false)
    //             {
    //               err         = LinearSolverErrorCode::RESIDUAL_DIVERGENCE;
    //               divergeFlag = true;
    //             }

    //           msg = "CGLinearSolver[" + std::to_string(iter) + "]";
    //           msg += " Abs. residual: " +
    //                  CGLinearSolverInternal::toStringWithPrecision(rNorm[i],
    //                                                                precision);
    //           msg += " Rel. residual: " +
    //                  CGLinearSolverInternal::toStringWithPrecision(rNorm[i] /
    //                                                                  bNorm[i],
    //                                                                precision);
    //           msg += "For Vector Component No: " + i;
    //         }

    //       if (std::all_of(convergeFlag.begin(),
    //                       convergeFlag.end(),
    //                       [](bool boolVal) { return boolVal; }))
    //         allConverged = true;

    //       if (divergeFlag || allConverged)
    //         break;

    //       // register end of the iteration
    //       // d_profiler.registerIterEnd(msg);
    //     }

    //   linearSolverFunction.setSolution(xConverged);

    //   if (iter > d_maxIter)
    //     {
    //       err = LinearSolverErrorCode::FAILED_TO_CONVERGE;
    //     }

    //   std::string msg = "";
    //   retunValue      = LinearSolverErrorMsg::isSuccessAndMsg(err);

    //   if (retunValue.isSuccess)
    //     {
    //       msg = "CGLinear solve converged in maximum " + std::to_string(iter)
    //       +
    //             " iterations";
    //     }
    //   else
    //     msg = retunValue.msg;

    //   // register end of CG
    //   // --------- TODO : fix the profiler class------------
    //   if (rank == 0)
    //     d_profiler.registerEnd(msg);

    //   return retunValue;
    // }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
