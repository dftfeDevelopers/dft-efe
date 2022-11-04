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

/*
 * @brief This example tests the linear Conjugate Gradient (CG) algorithm for 
 * a complex Hermitian positive definite matrix. We create a random Hermitian
 * positive definite matrix and constrain its condition number to a pre-defined
 * value. The size of the matrix and its pre-defined condition number are
 * hard-coded at the beginning of the main() function. Further, the various 
 * tolerances for the CG solver are also hard-coded at the beginning of the 
 * main() function.
 */

#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <complex>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/CGLinearSolver.h>
#include <linearAlgebra/LinearSolverFunction.h>
#include <utils/MPITypes.h>
#include <utils/RandNumGen.h>

using namespace dftefe;

extern "C" {
  // LU decomoposition of a general matrix
  void zgetrf(int* M, int *N, std::complex<double> * A, int* lda, int* IPIV, int* INFO);

  // generate inverse of a matrix given its LU decomposition
  void zgetri(int* N, std::complex<double> * A, int* lda, int* IPIV, std::complex<double> * WORK, int* lwork, int* INFO);
  void zheev(char * JOBZ, char * UPLO, int * N, std::complex<double> * A, int * LDA, double * W,
	     std::complex<double> * WORK, int * LWORK, double * RWORK, int * INFO);
}

namespace 
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

  void inverse(std::complex<double> * A, int N)
  {
    int *IPIV = new int[N];
    int LWORK = N*N;
    std::complex<double> * WORK = new std::complex<double>[LWORK];
    int INFO;

    zgetrf(&N,&N,A,&N,IPIV,&INFO);
    zgetri(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    delete[] IPIV;
    delete[] WORK;
  }
  
  void getEigenvalues(std::complex<double> * A, double * eigs, int N)
  {
    char jobz = 'N';
    char uplo = 'U';
    int lwork = 2*N-1;
    std::complex<double> * work = new std::complex<double>[lwork];
    double * rwork = new double[3*N-2];
    int info;

    zheev(&jobz, &uplo, &N, A, &N, eigs, work, &lwork, rwork, &info); 

    delete work;
    delete rwork;
  }
  
  template<typename T>
    void
    linearTransformMatrix(std::vector<T> & A, const T a, const T b, int N)
    {
      for(unsigned int i = 0; i < N; ++i)
      {
        for(unsigned int j = 0; j < N; ++j)
	{
	  A[i*N+j] *= a;
	}
	A[i*N+i] += b;
      }
    }

  template <typename T>
    inline T
    conjugate(const T &x)
    {
      return std::conj(x);
    }

  template <>
    inline double
    conjugate(const double &x)
    {
      return x;
    }

  template <>
    inline float
    conjugate(const float &x)
    {
      return x;
    }

  template <typename T>
    class RandMatHermitianPositiveDefiniteMatGen
    {
      public:
	RandMatHermitianPositiveDefiniteMatGen(const unsigned int N):
	  d_A(N*N, T(0.0))
      {
	std::vector<T> BMat(N*N);
        std::vector<T> BTMat(N*N);
        // random symmetrix matrix
        utils::RandNumGen<T> rng(T(0.0), T(1.0));
        for(unsigned int i = 0; i < N; ++i)
        {
          for(unsigned int j = 0; j < N; ++j)
          {
            const T x = rng.generate(); 
            BMat[i*N + j] = x;
            BTMat[j*N+i] =  conjugate(x);
          }
	}
	
	// Create a positive semi-definite matrix 
        // A = B^* B (i.e., B-adjoint times B)
        // Use dgemms to accelerate this
        for(unsigned int i = 0; i < N; ++i)
        {
          for(unsigned int j = 0; j < N; ++j)
          {
            for (unsigned int k = 0 ; k < N ; ++k)
            {
              d_A[i*N + j] += BTMat[i*N+k] * BMat[k*N+j];
            }
          }
	}


      }
	std::vector<T> getA() const 
	{
	  return d_A;
	}

      private:
	std::vector<T> d_A;
    };

  //
  // A test OperatorContext for Ax
  //
  template <typename ValueTypeOperator,
	   typename ValueTypeOperand>
	     class OperatorContextA: public linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand ,utils::MemorySpace::HOST>
	   {
	     public:
	       OperatorContextA(const std::vector<ValueTypeOperator> & A, const unsigned int N):
		 d_N(N),
		 d_A(A)
	     {
	     }

	       void
		 apply(linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST> &x,
		     linearAlgebra::Vector<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
		     utils::MemorySpace::HOST> &                        y) const override 
		 {
		   using ValueType = linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
		   y.setValue(ValueType());
		   for(unsigned int i = 0; i < d_N; ++i)
		   {
		     for(unsigned int j = 0; j < d_N; ++j)
		       *(y.data() + i) += d_A[i*d_N+j]*(*(x.data()+j));
		   }
		 }

	       void
		 apply(linearAlgebra::MultiVector<ValueTypeOperand, utils::MemorySpace::HOST> &X,
		     linearAlgebra::MultiVector<
		     linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
		     utils::MemorySpace::HOST> &Y) const override
		 {
		   utils::throwException(false, "In TestCGLinearSolverSerialHostDouble "
		       " the OperatorContextA apply() for MultiVector is not implemented");
		 }

	     private:
	       unsigned int d_N;
	       std::vector<ValueTypeOperator> d_A;
	   };// end of clas OperatorContextA

  //
  // A test OperatorContext for Jacobi preconditioner
  //
  template <typename ValueTypeOperator,
	   typename ValueTypeOperand>
	     class OperatorContextJacobiPC: public linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand, utils::MemorySpace::HOST>
	   {
	     public:
	       OperatorContextJacobiPC(const std::vector<ValueTypeOperator> & diag):
		 d_N(diag.size()),
		 d_diag(diag)
	     {}

	       void
		 apply(linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST> &x,
		     linearAlgebra::Vector<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
		     utils::MemorySpace::HOST> &                        y) const override 
		 {
		   using ValueType = linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
		   y.setValue(ValueType(0.0));
		   for(unsigned int i = 0; i < d_N; ++i)
		   {
		     *(y.data() + i)= (1.0/d_diag[i])*(*(x.data()+i));
		   }
		 }

	       void
		 apply(linearAlgebra::MultiVector<ValueTypeOperand, utils::MemorySpace::HOST> &X,
		     linearAlgebra::MultiVector<
		     linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
		     utils::MemorySpace::HOST> &Y) const override
		 {
		   utils::throwException(false, "In TestCGLinearSolverSerialHostDouble "
		       " the OperatorContextJacobiPC apply() for MultiVector is not implemented");
		 }

	     private:
	       unsigned int d_N;
	       std::vector<ValueTypeOperator> d_diag;
	   };// end of clas OperatorContextA

  //
  // A test LinearSovlerFunction
  //
  template <typename ValueTypeOperator,
	   typename ValueTypeOperand>
	     class LinearSolverFunctionTest: public linearAlgebra::LinearSolverFunction<ValueTypeOperator, ValueTypeOperand, utils::MemorySpace::HOST>
	   {
	     public:
	       using ValueType = linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

	     public:
	       LinearSolverFunctionTest(const std::vector<ValueTypeOperator> & A, const std::vector<ValueType> & b, const unsigned int N, 
		   std::shared_ptr<linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>> laoc ):
		 d_N(N),
		 d_x(N, laoc, 0.0),
		 d_b(N, laoc, 0.0)
	     {
	       d_AxContext = std::make_shared<OperatorContextA<ValueTypeOperator, ValueTypeOperand>>(A,N);
	       std::vector<ValueTypeOperator> diag(N,ValueTypeOperator(0.0));
	       for(unsigned int i = 0; i < N; ++i)
		 diag[i] = A[i*N+i];
	       d_PCContext = std::make_shared<OperatorContextJacobiPC<ValueTypeOperator, ValueTypeOperand>>(diag);
	       for(unsigned int i = 0; i < N; ++i)
	       {
		 *(d_b.data() + i) = b[i];
	       }
	     }

	       ~LinearSolverFunctionTest() = default;

	       const linearAlgebra::OperatorContext<ValueTypeOperator,
		     ValueTypeOperand,
		     utils::MemorySpace::HOST> &
		       getAxContext() const override 
		       {
			 return *d_AxContext;

		       }

	       const linearAlgebra::OperatorContext<ValueTypeOperator,
		     ValueTypeOperand,
		     utils::MemorySpace::HOST> &
		       getPCContext() const override 
		       {
			 return *d_PCContext;
		       }

	       void
		 setSolution(const linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST> &x) override 
		 {
		   d_x = x;
		 }

	       const linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST> &
		 getSolution() const override 
		 {
		   return d_x;
		 }


	       linearAlgebra::Vector<ValueType, utils::MemorySpace::HOST>
		 getRhs() const override 
		 {
		   return d_b;
		 }

	       linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST>
		 getInitialGuess() const override   
		 {
		   linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST> z(d_x, 0.0);
		   return z;
		 }

	       const utils::mpi::MPIComm &
		 getMPIComm() const override 
		 {
		   return utils::mpi::MPICommSelf;
		 }

	     private:
	       unsigned int d_N;
	       std::shared_ptr<linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand, utils::MemorySpace::HOST>> d_AxContext;
	       std::shared_ptr<linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand, utils::MemorySpace::HOST>> d_PCContext;
	       linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST> d_x;
	       linearAlgebra::Vector<ValueType, utils::MemorySpace::HOST> d_b;


	   }; // end of clas LinearSolverFunctionTest


}// end of local namespace  

int main()
{
  //
  // define various parameters for the test
  //

  //
  // matrix parameters
  //

  // size of the matrix
  const unsigned int N = 100;
  // pre-defined condition number of the matrix
  const double conditionNumber = 1e6;
  // set the lowest eigenvalue for the matrix
  const double eigLow = 1e-3;
  // set the highest eigenvalue for the matrix in accordance with the lowest
  // eigenvalue and the condition number
  const double eigHigh= eigLow*conditionNumber;
  
 
  //
  // CG parameters
  //

  // absolute tolerance for the residual (r = Ax - b) in the linear solve
  const double absTol = 1e-12;
  // relative tolerance for the residual (r = Ax - b) in the linear solve
  const double relTol = 1e-15;
  // tolerance for divergence of the residual 
  // (i.e., exit CG solver if the residual exceeds this value)
  const double resDivTol = 1e10;
  // tolerance to check accuracy of exact x (x = A^{-1}b) and the x from the 
  // CG linear solver
  const double diffRelTol = 1e-14*conditionNumber;
  // Max. iterations for the CG linear solver
  const unsigned int maxIter = 3*N;

  //
  // initialize MPI
  //
  int mpiInitFlag = 0;
  utils::mpi::MPIInitialized(&mpiInitFlag);
  if(!mpiInitFlag)
  {
    utils::mpi::MPIInit(NULL, NULL);
  }

  const utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;

  linearAlgebra::blasLapack::BlasQueue<Host> queue;
  std::shared_ptr<linearAlgebra::LinAlgOpContext<Host>> laoc = 
    std::make_shared<linearAlgebra::LinAlgOpContext<Host>>(&queue);

  //
  // create random symmetric positive definite matrix
  //
  RandMatHermitianPositiveDefiniteMatGen<std::complex<double>> rMatGen(N);
  std::vector<std::complex<double>> A = rMatGen.getA();
  std::cout << "Generated Hermitian positive definite matrix" << std::endl; 
  
  // evaluate the eigenvalues of A
  std::vector<std::complex<double>> ACopy(A);
  std::vector<double> eigs(N);
  getEigenvalues(&ACopy[0], &eigs[0], N);
  std::sort(eigs.begin(), eigs.end());
  double eig1 = eigs[0];
  double eig2 = eigs[N-1];
  std::cout << "Min. eigenvalue: " << eig1<< " Max. eigenvalue: " << 
    eig2 << " Condition Number: " << eig2/eig1<< std::endl;

  // linearly transform A to map the eigenvalues to eigLow and eigHigh
  const double u = (eigLow-eigHigh)/(eig1-eig2);
  const double v = (eigHigh*eig1 - eigLow*eig2)/(eig1-eig2);
  std::complex<double> alpha(u,0.0);
  std::complex<double> beta(v,0.0);
  linearTransformMatrix(A, alpha, beta, N);
 
  ACopy = A;
  getEigenvalues(&ACopy[0], &eigs[0], N);
  std::sort(eigs.begin(), eigs.end());
  eig1 = eigs[0];
  eig2 = eigs[N-1];
  std::cout << "Min. eigenvalue: " << eig1 << " Max. eigenvalue: " << 
    eig2 << " Condition Number: " << eig2/eig1 << std::endl;
  
  std::vector<std::complex<double>> b(N,0.0);
  utils::RandNumGen<std::complex<double>> rng(0.0, 1.0);
  for(unsigned int i = 0; i < N; ++i)
    b[i] = rng.generate();

  std::vector<std::complex<double>> AInv(A);
  inverse(AInv.data(), N);
  std::vector<std::complex<double>> x(N,0.0);
  for(unsigned int i = 0; i < N; ++i)
  {
    for(unsigned int j = 0; j < N; ++j)
      x[i] += AInv[i*N + j]*b[j];
  }

  LinearSolverFunctionTest<std::complex<double>, std::complex<double>> lsf(A, b, N, laoc);
  linearAlgebra::CGLinearSolver<std::complex<double>, std::complex<double>, Host> cgls(maxIter, absTol, relTol, resDivTol, linearAlgebra::LinearAlgebraProfiler()); 
  linearAlgebra::Error err = cgls.solve(lsf);
  const linearAlgebra::Vector<std::complex<double>, Host> & xcg = lsf.getSolution();
  linearAlgebra::Vector<std::complex<double>, utils::MemorySpace::HOST> bVec = lsf.getRhs();
  const double bNorm = bVec.l2Norm();
  double diffRelL2 = 0.0;
  for(unsigned int i = 0; i < N; ++i)
    diffRelL2 += pow(std::abs(x[i] - *(xcg.data()+i)), 2.0);

  diffRelL2 = sqrt(diffRelL2)/bNorm;
  utils::throwException(diffRelL2 < diffRelTol, 
      "TestCGLinearSolverSerialHostComplexDouble.cpp failed. "
      "L2 norm tolerance required: " + 
      toStringWithPrecision(diffRelTol, 16) + 
      ", L2 norm attained: " + 
      toStringWithPrecision(diffRelL2, 16));

  //
  // gracefully end MPI
  //
  int mpiFinalFlag = 0;
  utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    utils::mpi::MPIFinalize();
  }
}
