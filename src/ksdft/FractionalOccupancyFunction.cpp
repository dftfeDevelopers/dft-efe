#include <cmath>
#include <vector>
#include <ksdft/FractionalOccupancyFunction.h>

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

    FractionalOccupancyFunction::FractionalOccupancyFunction(
      std::vector<double> &eigenValues,
      const size_type      numElectrons,
      const double         kb,
      const double         T)
      : d_x((double)(0.1))
      , d_eigenValues(eigenValues)
      , d_kb(kb)
      , d_T(T)
      , d_numElectrons(numElectrons)
    {}

    const double
    FractionalOccupancyFunction::getValue(double &x) const
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
    FractionalOccupancyFunction::getForce(double &x) const
    {
      double retValue = 0;

      for (auto &i : d_eigenValues)
        {
          retValue += 2 * fermiDiracDer(i, x, d_kb, d_T);
        }
      return retValue;
    }

    void
    FractionalOccupancyFunction::setSolution(const double &x)
    {
      d_x = x;
    }

    void
    FractionalOccupancyFunction::getSolution(double &solution)
    {
      solution = d_x;
    }

    const double &
    FractionalOccupancyFunction::getInitialGuess() const
    {
      return d_x;
    }

    void
    FractionalOccupancyFunction::setInitialGuess(double &x)
    {
      d_x = x;
    }

  } // namespace ksdft
} // namespace dftefe
