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
          retVal = static_cast<T>((std::rand()) / RAND_MAX);
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
        retVal.real(static_cast<double>((std::rand()) / RAND_MAX));
        retVal.imag(static_cast<double>((std::rand()) / RAND_MAX));
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
        retVal.real(static_cast<float>((std::rand()) / RAND_MAX));
        retVal.imag(static_cast<float>((std::rand()) / RAND_MAX));
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
        std::vector<double>                          &tolerance,
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
        const size_type     maxKrylovSubspaceSize,
        const size_type     numLowerExtermeEigenValues,
        const size_type     numUpperExtermeEigenValues,
        std::vector<double> &tolerance,
        double              lanczosBetaTolerance,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>> mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>>           linAlgOpContext)
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
      reinit(const size_type     maxKrylovSubspaceSize,
             const size_type     numLowerExtermeEigenValues,
             const size_type     numUpperExtermeEigenValues,
             std::vector<double> &tolerance,
             double              lanczosBetaTolerance,
             const Vector<ValueTypeOperand, memorySpace> &initialGuess)
    {
      d_maxKrylovSubspaceSize      = maxKrylovSubspaceSize;
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
      reinit(
        const size_type     maxKrylovSubspaceSize,
        const size_type     numLowerExtermeEigenValues,
        const size_type     numUpperExtermeEigenValues,
        std::vector<double> &tolerance,
        double              lanczosBetaTolerance,
        std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>> mpiPatternP2P,
        std::shared_ptr<LinAlgOpContext<memorySpace>>           linAlgOpContext)
    {
      // Get the rank of the process
      int rank;
      utils::mpi::MPICommRank(mpiPatternP2P->mpiCommunicator(), &rank);

      Vector<ValueType, memorySpace> initialGuess(mpiPatternP2P, linAlgOpContext);

      d_initialGuess = initialGuess;

      std::vector<ValueType> initialGuessSTL(d_initialGuess.locallyOwnedSize());

      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        d_initialGuess.locallyOwnedSize(), initialGuessSTL.data(), d_initialGuess.data());

      LanczosExtremeEigenSolverInternal::generate<ValueType> rand;
      // todo - implement random class in utils and modify this
      for (size_type i = 0; i < d_initialGuess.locallyOwnedSize(); i++)
        {
          *(initialGuessSTL.data() + i) = rand.randomNumber();
        }

      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        initialGuessSTL.size(), d_initialGuess.data(), initialGuessSTL.data());

      d_maxKrylovSubspaceSize      = maxKrylovSubspaceSize;
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
      std::vector<bool>     isToleranceReached;
      bool                  isSuccess = false;
      size_type             krylovSubspaceSize;

      std::vector<RealType> alphaVec(0), betaVec(0), eigenValuesIter(0),
        betaVecTemp(0);
      std::vector<ValueType> alpha(1), beta(1);

      ValueType ones = (ValueType)1.0;

      Vector<ValueType, memorySpace> temp(d_initialGuess, 0.0);
      Vector<ValueType, memorySpace> v(d_initialGuess, 0.0);
      Vector<ValueType, memorySpace> q(d_initialGuess, 0.0);
      Vector<ValueType, memorySpace> qPrev(d_initialGuess, 0.0);

      std::vector<ValueType> qSTL(q.locallyOwnedSize());
      std::vector<ValueType> vSTL(q.locallyOwnedSize());

      std::vector<Vector<ValueType, memorySpace>> krylovSubspOrthoVec(0);
      utils::MemoryStorage<ValueType, memorySpace>
        krylovSubspOrthoVecMemStorage(0);

      // memory for the eigenVectors
      utils::MemoryStorage<ValueType, utils::MemorySpace::HOST>
        eigenVectorsKrylovSubspace(0);
      utils::MemoryStorage<ValueType, memorySpace>
        wantedEigenVectorsKrylovSubspace(0);

      EigenSolverErrorCode err = EigenSolverErrorCode::OTHER_ERROR;
      retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);

      // normalize the initialGuess with B norm set q = b/norm
      // compute B-norm = (initGuess)^TB(initGuess)

      B.apply(d_initialGuess, temp);

      dot<ValueType, ValueType, memorySpace>(d_initialGuess,
                                             temp,
                                             alpha,
                                             blasLapack::ScalarOp::Conj, 
                                             blasLapack::ScalarOp::Identity);

      std::cout << "alpha[0]: "<<alpha[0]<<"\n";

      std::vector<ValueType> initialGuessSTL(d_initialGuess.locallyOwnedSize());

      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        d_initialGuess.locallyOwnedSize(), initialGuessSTL.data(), d_initialGuess.data());

      for (size_type h = 0; h < d_initialGuess.locallyOwnedSize(); h++)
        {
          *(qSTL.data() + h) = *(initialGuessSTL.data() + h) / alpha[0];
        }

      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        qSTL.size(), q.data(), qSTL.data());


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
            q, temp, alpha, blasLapack::ScalarOp::Conj, blasLapack::ScalarOp::Identity);

          alphaVec.push_back((RealType)alpha[0]);

          if (iter >= numWantedEigenValues)
            {
              krylovSubspaceSize = iter;
              eigenValuesIter    = alphaVec;
              betaVecTemp        = betaVec;
                std::cout << "alphavec: ";
                for(auto i : alphaVec)
                {
                  std::cout << i << ",";
                }
                std::cout << "\n";
                std::cout << "betavec: ";
                for(auto i : betaVec)
                {
                  std::cout << i << ",";
                }
                std::cout << "\n";
              if (computeEigenVectors)
                {
                  eigenVectorsKrylovSubspace.resize(
                    krylovSubspaceSize * krylovSubspaceSize,
                    utils::Types<ValueType>::zero);
                  size_type lapackReturn =
                    lapack::steqr(lapack::Job::Vec,
                                  krylovSubspaceSize,
                                  eigenValuesIter.data(),
                                  betaVecTemp.data(),
                                  eigenVectorsKrylovSubspace.data(),
                                  krylovSubspaceSize);

                  if (lapackReturn != 0)
                    {
                      err = EigenSolverErrorCode::LAPACK_STEQR_ERROR;
                      retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
                      retunValue.msg += std::to_string(lapackReturn);                    
                      break;
                    }
                }
              else
                {
                  size_type lapackReturn =
                    lapack::steqr(lapack::Job::NoVec,
                                  krylovSubspaceSize,
                                  eigenValuesIter.data(),
                                  betaVecTemp.data(),
                                  eigenVectorsKrylovSubspace.data(),
                                  krylovSubspaceSize);

                  if (lapackReturn != 0)
                    {
                      err = EigenSolverErrorCode::LAPACK_STEQR_ERROR;
                      retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
                      retunValue.msg += std::to_string(lapackReturn);
                      break;
                    }
                }

              eigenValues.clear();
              eigenValuesPrev.clear();

              // To store the sliced vector
              eigenValues.resize(numWantedEigenValues);
              eigenValuesPrev.resize(numWantedEigenValues, (RealType)0);

              auto start = eigenValuesIter.begin();
              auto end =
                eigenValuesIter.end() + d_numLowerExtermeEigenValues + 1;

              std::copy(start, end, eigenValues.begin());

              start = eigenValuesIter.end() - d_numUpperExtermeEigenValues;
              end   = eigenValuesIter.end();

              std::copy(start,
                        end,
                        eigenValues.begin() + d_numLowerExtermeEigenValues);

              isToleranceReached.clear();
              for (size_type eigId = 0; eigId <= numWantedEigenValues; eigId++)
                {
                  isToleranceReached.push_back(
                    (eigenValuesPrev[eigId] - eigenValues[eigId] <=
                     d_tolerance[eigId]));
                }
              if (std::all_of(isToleranceReached.begin(),
                              isToleranceReached.end(),
                              [](bool v) { return v; }))
                {
                  err       = EigenSolverErrorCode::SUCCESS;
                  retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
                  isSuccess = true;
                  break;
                }
              eigenValuesPrev = eigenValues;
            }

          ValueType nAlpha = (ValueType)(-1.0) * (ValueType)alpha[0],
                    nBeta  = (ValueType)(-1.0) * (ValueType)beta[0];

          // get v = v - \alpha_iq_i - \beta_i-1q_i-1

          add(ones, v, nAlpha, q, v);
          add(ones, v, nBeta, qPrev, v);

          // compute \beta_i = bnorm v
          temp.setValue((ValueType)0.0);
          B.apply(v, temp);

          dot<ValueType, ValueType, memorySpace>(
            v, temp, beta, blasLapack::ScalarOp::Conj, blasLapack::ScalarOp::Identity);

          if (beta[0] < d_lanczosBetaTolerance)
            {
              if(krylovSubspaceSize >= numWantedEigenValues)
                isSuccess = true;
              err = EigenSolverErrorCode::LANCZOS_BETA_ZERO;
              retunValue = EigenSolverErrorMsg::isSuccessAndMsg(err);
              break;
            }

          utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
            v.locallyOwnedSize(), vSTL.data(), v.data());

          betaVec.push_back((RealType)beta[0]);

          // get q_i+1 = v/\beta_i

          for (size_type h = 0; h < v.locallyOwnedSize(); h++)
            {
              *(qSTL.data() + h) = *(vSTL.data() + h) / beta[0];
            }

          qPrev = q;

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            qSTL.size(), q.data(), qSTL.data());
        }

      if (computeEigenVectors && isSuccess)
        {
          // copy to wantedEigenVectorsKrylovSubspace and rotate
          wantedEigenVectorsKrylovSubspace.resize(
            krylovSubspaceSize * numWantedEigenValues,
            utils::Types<ValueType>::zero);
          krylovSubspOrthoVecMemStorage.resize(krylovSubspaceSize * q.locallyOwnedSize(),
                                               utils::Types<ValueType>::zero);

          for (size_type vecId = 0; vecId < krylovSubspaceSize; vecId++)
            {
              utils::MemoryTransfer<memorySpace, memorySpace>::copy(
                q.locallyOwnedSize(),
                krylovSubspOrthoVecMemStorage.data() + vecId * q.locallyOwnedSize(),
                krylovSubspOrthoVec[vecId].data());
            }

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            d_numLowerExtermeEigenValues * krylovSubspaceSize,
            wantedEigenVectorsKrylovSubspace.data(),
            eigenVectorsKrylovSubspace.data());

          utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
            d_numUpperExtermeEigenValues * krylovSubspaceSize,
            wantedEigenVectorsKrylovSubspace.data() +
              d_numLowerExtermeEigenValues * krylovSubspaceSize,
            eigenVectorsKrylovSubspace.data() +
              krylovSubspaceSize * krylovSubspaceSize -
              d_numUpperExtermeEigenValues * krylovSubspaceSize);

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
