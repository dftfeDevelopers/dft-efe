#include <cmath>
#include <vector>
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
               const double T)
    {
      const double factor = (eigenValue - fermiEnergy) / (kb * T);
      return (factor >= 0) ? std::exp(-factor) / (1.0 + std::exp(-factor)) :
                             1.0 / (1.0 + std::exp(factor));
    }

    double
    fermiDiracDer(const double eigenValue,
                  const double fermiEnergy,
                  const double kb,
                  const double T)
    {
      const double factor = (eigenValue - fermiEnergy) / (kb * T);
      const double beta   = 1.0 / (kb * T);
      return (factor >= 0) ?
               (-beta * std::exp(-factor) / (1.0 + std::exp(-factor)) /
                (1.0 + std::exp(-factor))) :
               (-beta * std::exp(factor) / (1.0 + std::exp(factor)) /
                (1.0 + std::exp(factor)));
    }

    class FractionalOccupancyFunction
      : public linearAlgebra::NewtonRaphsonSolverFunction<double>
    {
    public:
      FractionalOccupancyFunction(std::vector<double> &eigenValues,
                                  const size_type      numElectrons,
                                  const double         kb,
                                  const double         T)
        : d_x((double)(0.1))
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
            retValue += fermiDirac(i, x, d_kb, d_T);
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
            retValue += fermiDiracDer(i, x, d_kb, d_T);
          }
        return retValue;
      }

      void
      setSolution(const double &x) override
      {
        d_x = x;
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

      void
      setInitialGuess(double &x) override
      {
        d_x = x;
      }

    private:
      double              d_x;
      std::vector<double> d_eigenValues;
      size_type           d_numElectrons;
      double              d_kb;
      double              d_T;
    };

  } // namespace ksdft
} // namespace dftefe
