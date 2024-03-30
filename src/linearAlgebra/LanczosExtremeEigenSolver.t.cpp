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

#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <utils/DataTypeOverloads.h>
#include <cstdlib>
#include <ctime>
#include <type_traits>
#include <string>
#include <complex>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace LanczosExtremeEigenSolverInternal
    {
      template <typename T>
      class generate
      {
      public:
        inline static T
        randomNumber()
        {
          T retVal;
          retVal = static_cast<T>(std::rand()) / RAND_MAX;
          return retVal;
        }
      };

      template <>
      class generate<std::complex<double>>
      {
      public:
        inline static std::complex<double>
        randomNumber()
        {
          std::complex<double> retVal;
          retVal.real(static_cast<double>(std::rand()) / RAND_MAX);
          retVal.imag(static_cast<double>(std::rand()) / RAND_MAX);
          return retVal;
        }
      };

      template <>
      class generate<std::complex<float>>
      {
      public:
        inline static std::complex<float>
        randomNumber()
        {
          std::complex<float> retVal;
          retVal.real(static_cast<float>(std::rand()) / RAND_MAX);
          retVal.imag(static_cast<float>(std::rand()) / RAND_MAX);
          return retVal;
        }
      };

    } // namespace LanczosExtremeEigenSolverInternal

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
      LanczosExtremeEigenSolver(
        const size_type                              maxKrylovSubspaceSize,
        const size_type                              numLowerExtermeEigenValues,
        const size_type                              numUpperExtermeEigenValues,
        std::vector<double> &                        tolerance,
        double                                       lanczosBetaTolerance,
        const Vector<ValueTypeOperand, memorySpace> &initialGuess)
    {
      reinit(maxKrylovSubspaceSize,
             numLowerExtermeEigenValues,
             numUpperExtermeEigenValues,
             tolerance,
             lanczosBetaTolerance,
             initialGuess);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
      LanczosExtremeEigenSolver(
        const size_type      maxKrylovSubspaceSize,
        const size_type      numLowerExtermeEigenValues,
        const size_type      numUpperExtermeEigenValues,
        std::vector<double> &tolerance,
        double               lanczosBetaTolerance,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                      mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext)
    {
      reinit(maxKrylovSubspaceSize,
             numLowerExtermeEigenValues,
             numUpperExtermeEigenValues,
             tolerance,
             lanczosBetaTolerance,
             mpiPatternP2P,
             linAlgOpContext);
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
      reinit(const size_type      maxKrylovSubspaceSize,
             const size_type      numLowerExtermeEigenValues,
             const size_type      numUpperExtermeEigenValues,
             std::vector<double> &tolerance,
             double               lanczosBetaTolerance,
             const Vector<ValueTypeOperand, memorySpace> &initialGuess)
    {
      d_maxKrylovSubspaceSize      = ((global_size_type)maxKrylovSubspaceSize <=
                                 d_initialGuess.globalSize()) ?
                                       maxKrylovSubspaceSize :
                                       d_initialGuess.globalSize();
      d_initialGuess               = initialGuess;
      d_numLowerExtermeEigenValues = numLowerExtermeEigenValues;
      d_numUpperExtermeEigenValues = numUpperExtermeEigenValues;
      d_tolerance                  = tolerance;
      d_lanczosBetaTolerance       = lanczosBetaTolerance;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    LanczosExtremeEigenSolver<ValueTypeOperator,
                              ValueTypeOperand,
                              memorySpace>::
      reinit(const size_type      maxKrylovSubspaceSize,
             const size_type      numLowerExtermeEigenValues,
             const size_type      numUpperExtermeEigenValues,
             std::vector<double> &tolerance,
             double               lanczosBetaTolerance,
             std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                                                           mpiPatternP2P,
             std::shared_ptr<LinAlgOpContext<memorySpace>> linAlgOpContext)
    {
      // Get the rank of the process
      int rank;
      utils::mpi::MPICommRank(mpiPatternP2P->mpiCommunicator(), &rank);
      std::srand(std::time(nullptr) * (rank + 1));

      Vector<ValueTypeOperand, memorySpace> initialGuess(mpiPatternP2P,
                                                         linAlgOpContext);

      d_initialGuess = initialGuess;

      std::vector<ValueTypeOperand> initialGuessSTL(
        d_initialGuess.locallyOwnedSize());

      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        d_initialGuess.locallyOwnedSize(),
        initialGuessSTL.data(),
        d_initialGuess.data());

      LanczosExtremeEigenSolverInternal::generate<ValueTypeOperand>
        generateNumber;
      // todo - implement random class in utils and modify this
      for (size_type i = 0; i < d_initialGuess.locallyOwnedSize(); i++)
        {
          *(initialGuessSTL.data() + i) = generateNumber.randomNumber();
        }

      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        initialGuessSTL.size(), d_initialGuess.data(), initialGuessSTL.data());

      d_maxKrylovSubspaceSize = ((global_size_type)maxKrylovSubspaceSize <=
                                 d_initialGuess.globalSize()) ?
                                  maxKrylovSubspaceSize :
                                  d_initialGuess.globalSize();

      d_numLowerExtermeEigenValues = numLowerExtermeEigenValues;
      d_numUpperExtermeEigenValues = numUpperExtermeEigenValues;
      d_tolerance                  = tolerance;
      d_lanczosBetaTolerance       = lanczosBetaTolerance;
    }

    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    EigenSolverError
    LanczosExtremeEigenSolver<
      ValueTypeOperator,
      ValueTypeOperand,
      memorySpace>::solve(const OpContext &                    A,
                          std::vector<RealType> &              eigenValues,
                          MultiVector<ValueType, memorySpace> &eigenVectors,
                          bool             computeEigenVectors,
                          const OpContext &B,
                          const OpContext &BInv)
    {
      EigenSolverError retunValue;

      size_type numWantedEigenValues =
        d_numLowerExtermeEigenValues + d_numUpperExtermeEigenValues;
      // solve the Lanczos until the eigenValues are calculated till tolerance

      utils::throwException(
        d_maxKrylovSubspaceSize >= numWantedEigenValues,
        "Maximum Krylov subspace size should be more than number of required eigenPairs.");

      eigenValues.clear();
      std::vector<RealType> eigenValuesPrev(0);
      eigenValuesPrev.resize(numWantedEigenValues, (RealType)0);
      std::vector<bool> isToleranceReached(numWantedEigenValues, false);
      bool              isSuccess = false;
      size_type         krylovSubspaceSize;

      std::vector<RealType>  alphaVec(0), betaVec(0);
      std::vector<ValueType> alpha(1, (ValueType)0), beta(1, (ValueType)0);

      ValueType ones = (ValueType)1.0, nBeta, nAlpha;

      Vector<ValueType, memorySpace> temp(d_initialGuess, (ValueType)0.0);
      Vector<ValueType, memorySpace> v(d_initialGuess, (ValueType)0.0);
      Vector<ValueType, memorySpace> q(d_initialGuess, (ValueType)0.0);
      Vector<ValueType, memorySpace> qPrev(d_initialGuess, (ValueType)0.0);

      std::vector<Vector<ValueType, memorySpace>> krylovSubspOrthoVec(0);
      utils::MemoryStorage<ValueType, memorySpace>
        krylovSubspOrthoVecMemStorage(0);

      // memory for the eigenVectors
      utils::MemoryStorage<ValueType, memorySpace> eigenVectorsKrylovSubspace(
        0);
      utils::MemoryStorage<ValueType, memorySpace>
        wantedEigenVectorsKrylovSubspace(0);

      EigenSolverErrorCode err = EigenSolverErrorCode::OTHER_ERROR;
      retunValue               = EigenSolverErrorMsg::isSuccessAndMsg(err);

      // normalize the initialGuess with B norm set q = b/norm
      // compute B-norm = (initGuess)^TB(initGuess)

      B.apply(d_initialGuess, temp);

      dot<ValueTypeOperand, ValueType, memorySpace>(
        d_initialGuess,
        temp,
        alpha,
        blasLapack::ScalarOp::Conj,
        blasLapack::ScalarOp::Identity);

      alpha[0] = std::sqrt(alpha[0]);

      blasLapack::ascale<ValueType, ValueTypeOperand, memorySpace>(
        d_initialGuess.locallyOwnedSize(),
        (ValueType)(1.0 / alpha[0]),
        d_initialGuess.data(),
        q.data(),
        *d_initialGuess.getLinAlgOpContext());

      for (size_type iter = 1; iter <= d_maxKrylovSubspaceSize; iter++)
        {
          // push the q orthogonal krylov subspace vectors if needed later
          if (computeEigenVectors)
            krylovSubspOrthoVec.push_back(q);

          // v = BInv A q_i

          temp.setValue((ValueType)0.0);
          A.apply(q, temp);
          BInv.apply(temp, v);

          // get \alpha = q_i^TAq_i

          dot<ValueType, ValueType, memorySpace>(
            q,
            temp,
            alpha,
            blasLapack::ScalarOp::Conj,
            blasLapack::ScalarOp::Identity);

          alphaVec.push_back((RealType)alpha[0]);

          // std::cout << "alphaVec: ";
          // for(auto i : alphaVec)
          // {
          //   std::cout << i << ",";
          // }
          // std::cout << "\n";

          // std::cout << "betaVec: ";
          // for(auto i : betaVec)
          // {
          //   std::cout << i << ",";
          // }
          // std::cout << "\n";

          if (iter >= numWantedEigenValues)
            {
              krylovSubspaceSize = iter;

              utils::MemoryStorage<ValueType, memorySpace> eigenValuesIter(
                alphaVec.size());
              eigenValuesIter.template copyFrom<utils::MemorySpace::HOST>(
                alphaVec.data());
              utils::MemoryStorage<ValueType, memorySpace> betaVecTemp(
                betaVec.size());
              betaVecTemp.template copyFrom<utils::MemorySpace::HOST>(
                betaVec.data());

              if (computeEigenVectors)
                {
                  eigenVectorsKrylovSubspace.resize(
                    krylovSubspaceSize * krylovSubspaceSize,
                    utils::Types<ValueType>::zero);
                  LapackError lapackReturn =
                    blasLapack::steqr<ValueType, memorySpace>(
                      blasLapack::Job::Vec,
                      krylovSubspaceSize,
                      eigenValuesIter.data(),
                      betaVecTemp.data(),
                      eigenVectorsKrylovSubspace.data(),
                      krylovSubspaceSize,
                      *d_initialGuess.getLinAlgOpContext());

                  if (lapackReturn.err ==
                      LapackErrorCode::FAILED_REAL_TRIDIAGONAL_EIGENPROBLEM)
                    {
                      err        = EigenSolverErrorCode::LAPACK_ERROR;
                      retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
                      retunValue.msg += lapackReturn.msg;
                      break;
                    }
                }
              else
                {
                  LapackError lapackReturn =
                    blasLapack::steqr<ValueType, memorySpace>(
                      blasLapack::Job::NoVec,
                      krylovSubspaceSize,
                      eigenValuesIter.data(),
                      betaVecTemp.data(),
                      eigenVectorsKrylovSubspace.data(),
                      krylovSubspaceSize,
                      *d_initialGuess.getLinAlgOpContext());

                  if (lapackReturn.err ==
                      LapackErrorCode::FAILED_REAL_TRIDIAGONAL_EIGENPROBLEM)
                    {
                      err        = EigenSolverErrorCode::LAPACK_ERROR;
                      retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
                      retunValue.msg += lapackReturn.msg;
                      break;
                    }
                }

              eigenValues.clear();

              // To store the sliced vector
              eigenValues.resize(numWantedEigenValues);

              // std::cout << "iter: "<<iter << "\n";
              // std::cout << "eigenValuesIter: ";
              // for(auto i : eigenValuesIter)
              // {
              //   std::cout << i << ",";
              // }
              // std::cout << "\n";

              eigenValuesIter.template copyTo<utils::MemorySpace::HOST>(
                eigenValues.data(), d_numLowerExtermeEigenValues, 0, 0);

              eigenValuesIter.template copyTo<utils::MemorySpace::HOST>(
                eigenValues.data(),
                d_numUpperExtermeEigenValues,
                eigenValuesIter.size() - d_numUpperExtermeEigenValues,
                d_numLowerExtermeEigenValues);

              for (size_type eigId = 0; eigId < numWantedEigenValues; eigId++)
                {
                  isToleranceReached[eigId] =
                    (std::abs(eigenValuesPrev[eigId] - eigenValues[eigId]) <=
                     d_tolerance[eigId]) ?
                      true :
                      false;
                }
              if (std::all_of(isToleranceReached.begin(),
                              isToleranceReached.end(),
                              [](bool v) { return v; }))
                {
                  err        = EigenSolverErrorCode::SUCCESS;
                  retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
                  isSuccess  = true;
                  break;
                }
              eigenValuesPrev = eigenValues;
            }
          nAlpha = (ValueType)(-1.0) * (ValueType)alpha[0];
          nBeta  = (ValueType)(-1.0) * (ValueType)beta[0];

          // get v = v - \alpha_i * q_i - \beta_i-1 * q_i-1

          add(ones, v, nAlpha, q, v);
          add(ones, v, nBeta, qPrev, v);

          // compute \beta_i = bnorm v
          temp.setValue((ValueType)0.0);
          B.apply(v, temp);

          dot<ValueType, ValueType, memorySpace>(
            v,
            temp,
            beta,
            blasLapack::ScalarOp::Conj,
            blasLapack::ScalarOp::Identity);

          beta[0] = std::sqrt(beta[0]);

          if (beta[0] < d_lanczosBetaTolerance)
            {
              if (krylovSubspaceSize >= numWantedEigenValues)
                isSuccess = true;
              err        = EigenSolverErrorCode::LANCZOS_BETA_ZERO;
              retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
              break;
            }

          betaVec.push_back((RealType)beta[0]);

          qPrev = q;

          // get q_i+1 = v/\beta_i
          blasLapack::ascale<ValueType, ValueType, memorySpace>(
            v.locallyOwnedSize(),
            (ValueType)(1.0 / beta[0]),
            v.data(),
            q.data(),
            *d_initialGuess.getLinAlgOpContext());
        }

      if (krylovSubspaceSize > d_maxKrylovSubspaceSize)
        {
          isSuccess  = true;
          err        = EigenSolverErrorCode::LANCZOS_SUBSPACE_INSUFFICIENT;
          retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
        }

      if (computeEigenVectors && isSuccess)
        {
          // copy to wantedEigenVectorsKrylovSubspace and rotate
          wantedEigenVectorsKrylovSubspace.resize(
            krylovSubspaceSize * numWantedEigenValues,
            utils::Types<ValueType>::zero);
          krylovSubspOrthoVecMemStorage.resize(krylovSubspaceSize *
                                                 q.locallyOwnedSize(),
                                               utils::Types<ValueType>::zero);

          // std::cout << "krylovSubspOrthoVec: \n";
          for (size_type vecId = 0; vecId < krylovSubspaceSize; vecId++)
            {
              utils::MemoryTransfer<memorySpace, memorySpace>::copy(
                q.locallyOwnedSize(),
                krylovSubspOrthoVecMemStorage.data() +
                  vecId * q.locallyOwnedSize(),
                krylovSubspOrthoVec[vecId].data());

              // for(size_type j = 0 ; j <
              // krylovSubspOrthoVec[vecId].locallyOwnedSize() ; j++)
              // {
              //   std::cout << *(krylovSubspOrthoVec[vecId].data()+j) << ",";
              // }
              // std::cout << "\n";
            }

          // std::cout << "krylovSubspOrthoVecMemStorage: \n" ;
          // for(size_type j = 0 ; j < krylovSubspaceSize ; j++)
          // {
          //   std::cout << "[";
          //   for(size_type i = 0 ; i < q.locallyOwnedSize() ; i++)
          //   {
          //     std::cout <<
          //     *(krylovSubspOrthoVecMemStorage.data()+j*q.locallyOwnedSize()+i)
          //     << ",";
          //   }
          //   std::cout << "]\n";
          // }

          utils::MemoryTransfer<memorySpace, memorySpace>::copy(
            d_numLowerExtermeEigenValues * krylovSubspaceSize,
            wantedEigenVectorsKrylovSubspace.data(),
            eigenVectorsKrylovSubspace.data());

          utils::MemoryTransfer<memorySpace, memorySpace>::copy(
            d_numUpperExtermeEigenValues * krylovSubspaceSize,
            wantedEigenVectorsKrylovSubspace.data() +
              d_numLowerExtermeEigenValues * krylovSubspaceSize,
            eigenVectorsKrylovSubspace.data() +
              krylovSubspaceSize * krylovSubspaceSize -
              d_numUpperExtermeEigenValues * krylovSubspaceSize);

          // std::cout << "eigenVectorsKrylovSubspace: \n" ;
          // for(size_type j = 0 ; j < krylovSubspaceSize ; j++)
          // {
          //   std::cout << "[";
          //   for(size_type i = 0 ; i < krylovSubspaceSize ; i++)
          //   {
          //     std::cout <<
          //     *(eigenVectorsKrylovSubspace.data()+i*krylovSubspaceSize+j) <<
          //     ",";
          //   }
          //   std::cout << "]\n";
          // }

          // std::cout << "wantedEigenVectorsKrylovSubspace: \n" ;
          // for(size_type j = 0 ; j < numWantedEigenValues ; j++)
          // {
          //   std::cout << "[";
          //   for(size_type i = 0 ; i < krylovSubspaceSize ; i++)
          //   {
          //     std::cout <<
          //     *(wantedEigenVectorsKrylovSubspace.data()+j*krylovSubspaceSize+i)
          //     << ",";
          //   }
          //   std::cout << "]\n";
          // }

          blasLapack::gemm<ValueType, ValueType, memorySpace>(
            blasLapack::Layout::ColMajor,
            blasLapack::Op::Trans,
            blasLapack::Op::Trans,
            numWantedEigenValues,
            q.locallyOwnedSize(),
            krylovSubspaceSize,
            (ValueType)1.0,
            wantedEigenVectorsKrylovSubspace.data(),
            krylovSubspaceSize,
            krylovSubspOrthoVecMemStorage.data(),
            q.locallyOwnedSize(),
            (ValueType)0.0,
            eigenVectors.data(),
            numWantedEigenValues,
            *d_initialGuess.getLinAlgOpContext());
        }

      return retunValue;
    }
  } // end of namespace linearAlgebra
} // end of namespace dftefe
