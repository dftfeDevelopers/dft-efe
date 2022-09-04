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
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/CGLinearSolver.h>
#include <linearAlgebra/LinearSolverFunction.h>
#include <utils/MPITypes.h>
#include <utils/RandNumGen.h>

using namespace dftefe;

extern "C" {
  // C = alphaA.B + betaC    
  void dgemm(char* TRANSA, char* TRANSB, const int* M,
               const int* N, const int* K, double* alpha, double* A,
               const int* LDA, double* B, const int* LDB, double* beta,
               double* C, const int* LDC);
   
   // Y= alphaA.X + betaY                                               
   void dgemv(char* TRANS, const int* M, const int* N,
               double* alpha, double* A, const int* LDA, double* X,
               const int* INCX, double* beta, double* C, const int* INCY);
   
   // LU decomoposition of a general matrix
    void dgetrf(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
}

namespace 
{

void inverse(double* A, int N)
{
    int *IPIV = new int[N];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf(&N,&N,A,&N,IPIV,&INFO);
    dgetri(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    delete[] IPIV;
    delete[] WORK;
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
    class RandMatHermitianGen
    {
      public:
	RandMatHermitianGen(const unsigned int N):
	  d_A(N*N, T(0.0))
      {
	// random symmetrix matrix
	utils::RandNumGen<T> rng(T(0.0), T(1.0));
	for(unsigned int i = 0; i < N; ++i)
	{
	  for(unsigned int j = i; j < N; ++j)
	  {
	    const T x = rng.generate(); 
	    d_A[i*N + j] = x;
	    d_A[j*N + i] = conjugate(x);
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
	     {}

	     void
	       apply(const linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST> &x,
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
	       apply(const linearAlgebra::MultiVector<ValueTypeOperand, utils::MemorySpace::HOST> &X,
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
	       apply(const linearAlgebra::Vector<ValueTypeOperand, utils::MemorySpace::HOST> &x,
		   linearAlgebra::Vector<linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
		   utils::MemorySpace::HOST> &                        y) const override 
	       {
		 using ValueType = linearAlgebra::blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
		 y.setValue(ValueType(0.0));
		 for(unsigned int i = 0; i < d_N; ++i)
		 {
		     *(y.data() + i)= d_diag[i]*(*(x.data()+i));
		 }
	       }

	     void
	       apply(const linearAlgebra::MultiVector<ValueTypeOperand, utils::MemorySpace::HOST> &X,
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
		   d_x =x;
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
  
  const utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
  const unsigned int N= 4;
  
  linearAlgebra::blasLapack::BlasQueue<Host> queue;
  std::shared_ptr<linearAlgebra::LinAlgOpContext<Host>> laoc = 
    std::make_shared<linearAlgebra::LinAlgOpContext<Host>>(&queue);
  
  RandMatHermitianGen<double> rMatGen(N);
  std::vector<double> A = rMatGen.getA();
  std::vector<double> b(N,0.0);
  utils::RandNumGen<double> rng(0.0, 1.0);
  for(unsigned int i = 0; i < N; ++i)
    b[i] = rng.generate();

  std::vector<double> AInv(A);
  inverse(AInv.data(), N);
  std::vector<double> x(N,0.0);
  for(unsigned int i = 0; i < N; ++i)
  {
    for(unsigned int j = 0; j < N; ++j)
      x[i] = AInv[i*N + j]*b[j];
  }

  LinearSolverFunctionTest<double, double> lsf(A,b,N, laoc);
  linearAlgebra::CGLinearSolver<double, double, Host> cgls(1000, 1e-12, 1e-12, 1e6, linearAlgebra::LinearAlgebraProfiler()); 
  linearAlgebra::Error err = cgls.solve(lsf);
  const linearAlgebra::Vector<double, Host> & xcg = lsf.getSolution();
  for(unsigned int i = 0; i < N; ++i)
    std::cout << "[" << i << "] diff:" <<  x[i] - *(xcg.data()+i) << std::endl;
}
