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
			      return x*x - 2;
		       }

	       const ValueType
		       getForce(ValueType &x) const override 
		       {
			        return 2*x;
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

          void
          setInitialGuess(ValueType &x) override   
          {
            d_x = x;
          }

          private:
            ValueType d_x;
	   }; 


}// end of local namespace  

int main()
{

  test::FunctionTest<double> *lsf = new test::FunctionTest<double>();
  linearAlgebra::NewtonRaphsonSolver<double> nrs(1e3, 1e-4, 1e-14); 
  linearAlgebra::NewtonRaphsonError err = nrs.solve(*lsf);
  double solution;
  lsf->getSolution(solution);
  std::cout << solution << "\n";
  std::cout << err.msg <<"\n";
  delete lsf;
}
