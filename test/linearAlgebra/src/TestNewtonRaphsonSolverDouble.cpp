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

/*
 * @brief This example tests the linear Conjugate Gradient (CG) algorithm for 
 * a real symmetric positive definite matrix on the HOST (CPU). 
 * We create a random symmetric positive definite matrix and constrain its 
 * condition number to a pre-defined value. The size of the matrix and its 
 * pre-defined condition number are hard-coded at the beginning of the main() 
 * function. Further, the various tolerances for the CG solver are also 
 * hard-coded at the beginning of the main() function.
 * The test case is hardcoded for single vector case and NOT FOR MULTIVECTOR.
 */


#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <algorithm>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/NewtonRaphsonSolver.h>
#include <linearAlgebra/NewtonRaphsonSolverFunction.h>
#include <utils/RandNumGen.h>

using namespace dftefe;

namespace test
{
  //
  // A test NewtonRaphsonSovlerFunction
  //
  template <typename ValueType>
	     class FunctionTest: public linearAlgebra::NewtonRaphsonSolverFunction<ValueType>
	   {
	     public:
	       FunctionTest():
          d_x((ValueType)(0.1))
	        {}

	       ~FunctionTest() = default;

	       const ValueType
		       getValue(ValueType &x) const override 
		       {
			      return x*x + 2*x + 1;
		       }

	       const ValueType
		       getForce(ValueType &x) const override 
		       {
			        return 2*x + 2;
		       }

	       void
          setSolution(const ValueType &x) override 
          {
            d_x =x;
          }

          void
          getSolution(ValueType &solution) override 
          {
            solution = d_x;
          }

          const ValueType &
          getInitialGuess() const override   
          {
            return d_x;
          }
          private:
            ValueType d_x;
	   }; 

    double
    fermiDirac(const double eigenValue,
               const double fermiEnergy,
               const double kb,
               const double T)
    {
      const double factor = (eigenValue - fermiEnergy) / (kb * T);

      double retValue;

      if(factor >= 0)
        retValue = std::exp(-factor) / (1.0 + std::exp(-factor));
      else 
        retValue = 1.0 / (1.0 + std::exp(factor));

      std::cout << "Value: " << factor << ", " << retValue << "\n";

      return  retValue;                            
    }

    double
    fermiDiracDer(const double eigenValue,
                  const double fermiEnergy,
                  const double kb,
                  const double T)
    {
      const double factor = (eigenValue - fermiEnergy) / (kb * T);
      const double beta   = 1.0 / (kb * T);

      double retValue;

      if(factor >= 0)
        retValue = (beta * std::exp(-factor) / (1.0 + std::exp(-factor)) /
                (1.0 + std::exp(-factor)));
      else 
        retValue = (beta * std::exp(factor) / (1.0 + std::exp(factor)) /
                (1.0 + std::exp(factor)));

      std::cout << "Der: " << factor << ", " << retValue << "\n";

      return  retValue;
    }

  //
  // A test NewtonRaphsonSovlerFunction
  //
	     class FractionalOccupancyFunction: public linearAlgebra::NewtonRaphsonSolverFunction<double>
	   {
	     public:
	       FractionalOccupancyFunction(const std::vector<double> &eigenValues,
          const size_type      numElectrons,
          const double         kb,
          const double         T,
          const double         initialGuess)
          : d_x(initialGuess)
          , d_initialGuess(initialGuess)
          , d_eigenValues(eigenValues)
          , d_kb(kb)
          , d_T(T)
          , d_numElectrons(numElectrons)
	        {}

	       ~FractionalOccupancyFunction() = default;

	       const double
		       getValue(double &x) const override 
		       {
              double retValue = 0;

              for (auto &i : d_eigenValues)
                {
                  retValue += 2 * fermiDirac(i, x, d_kb, d_T);
                }
              retValue -= (double)d_numElectrons;
              return retValue;
		       }

	       const double
		       getForce(double &x) const override 
		       {
              double retValue = 0;

              for (auto &i : d_eigenValues)
                {
                  retValue += 2 * fermiDiracDer(i, x, d_kb, d_T);
                }
              return retValue;
		       }

	       void
          setSolution(const double &x) override 
          {
            d_x =x;
          }

          void
          getSolution(double &solution) override 
          {
            solution = d_x;
          }

          const double &
          getInitialGuess() const override   
          {
            return d_x;
          }

          private:
            double              d_x;
            double              d_initialGuess;
            std::vector<double> d_eigenValues;
            size_type           d_numElectrons;
            double              d_kb;
            double              d_T;
	   }; 

}// end of local namespace  

int main()
{

  test::FunctionTest<double> *lsf = new test::FunctionTest<double>();
  linearAlgebra::NewtonRaphsonSolver<double> nrs(5e1, 1e-10, 1e-14); 
  linearAlgebra::NewtonRaphsonError err = nrs.solve(*lsf);
  double solution;
  lsf->getSolution(solution);
  std::cout << solution << "\n";
  std::cout << err.msg <<"\n";
  delete lsf;

  test::FractionalOccupancyFunction *lsf1 = new test::FractionalOccupancyFunction
    ({-0.3871218951, 109.8516437, 110.1915159, 110.5144201, 110.6296845, 110.7654463, 111.0254295, 111.1454389, 111.2021994, 111.4086254, 111.5410011, 111.6166645, 111.9046748, 112.0428015, 112.2300994},
      1,
      3.166811429e-06,
      10,
      -0.3871218951);

  err = nrs.solve(*lsf1);
  lsf1->getSolution(solution);
  std::cout << solution << "\n";
  std::cout << err.msg <<"\n";
  delete lsf1;

}
