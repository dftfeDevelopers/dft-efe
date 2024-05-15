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

#ifndef dftefeFractionalOccupancyFunction_h
#define dftefeFractionalOccupancyFunction_h

#include <linearAlgebra/NewtonRaphsonSolver.h>
#include <linearAlgebra/NewtonRaphsonSolverFunction.h>

namespace dftefe
{
  namespace ksdft
  {
    double
    fermiDirac(const double eigenValue,
               const double fermiEnergy,
               const double kb,
               const double T);

    double
    fermiDiracDer(const double eigenValue,
                  const double fermiEnergy,
                  const double kb,
                  const double T);

    class FractionalOccupancyFunction
      : public linearAlgebra::NewtonRaphsonSolverFunction<double>
    {
    public:
      /**
       * @brief Constructor
       */
      FractionalOccupancyFunction(std::vector<double> &eigenValues,
                                  const size_type      numElectrons,
                                  const double         kb,
                                  const double         T);

      ~FractionalOccupancyFunction() = default;

      const double
      getValue(double &x) const override;

      const double
      getForce(double &x) const override;

      void
      setSolution(const double &x) override;

      void
      getSolution(double &solution) override;

      const double &
      getInitialGuess() const override;

      void
      setInitialGuess(double &x) override;

    private:
      double              d_x;
      std::vector<double> d_eigenValues;
      size_type           d_numElectrons;
      double              d_kb;
      double              d_T;

    }; // end of class FractionalOccupancyFunction
  }    // end of namespace ksdft
} // end of namespace dftefe
#endif // dftefeFractionalOccupancyFunction_h
