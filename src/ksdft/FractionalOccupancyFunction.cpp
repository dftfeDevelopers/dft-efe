#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <ksdft/FractionalOccupancyFunction.h>
#include <ksdft/Defaults.h>
#include <utils/Exceptions.h>

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
               (beta * std::exp(-factor) / (1.0 + std::exp(-factor)) /
                (1.0 + std::exp(-factor))) :
               (beta * std::exp(factor) / (1.0 + std::exp(factor)) /
                (1.0 + std::exp(factor)));
    }

    FractionalOccupancyFunction::FractionalOccupancyFunction(
      const std::vector<double> &eigenValues,
      const size_type            numElectrons,
      const double               kb,
      const double               T,
      const double               initialGuess)
      : d_x(initialGuess)
      , d_initialGuess(initialGuess)
      , d_eigenValues(eigenValues)
      , d_kb(kb)
      , d_T(T)
      , d_numElectrons(numElectrons)
      , d_lowerBound(0)
      , d_upperBound(0)
    {
      computeBracket();
    }

    void
    FractionalOccupancyFunction::computeBracket()
    {
      // [min(d_eigenValues), max(d_eigenValues)] provably brackets the
      // root: getValue() is a sum of Fermi-Dirac sigmoids, each strictly
      // increasing in x, so getValue() itself is strictly increasing;
      // combined with 0 < numElectrons < 2*d_eigenValues.size() (always
      // true physically), getValue(min) < 0 < getValue(max). As a
      // defensive fallback in case that doesn't hold in floating point
      // anyway, widen both ends outward by the bracket's own width and
      // recheck, up to 1000 times, before giving up - mirrors DFT-FE's
      // Fermi energy solver (src/dft/fermiEnergy.cc).
      const size_type maxBracketExpansionIter = 1000;

      double xLeft =
        *std::min_element(d_eigenValues.begin(), d_eigenValues.end());
      double xRight =
        *std::max_element(d_eigenValues.begin(), d_eigenValues.end());

      double spectrumWidth = xRight - xLeft;
      if (spectrumWidth <= 0.0)
        spectrumWidth = 1.0;

      double yLeft  = getValue(xLeft);
      double yRight = getValue(xRight);

      bool intervalFound = yLeft * yRight <= 0.0;
      for (size_type iter = 0;
           !intervalFound && iter < maxBracketExpansionIter;
           ++iter)
        {
          xLeft -= spectrumWidth;
          xRight += spectrumWidth;

          yLeft  = getValue(xLeft);
          yRight = getValue(xRight);

          intervalFound = yLeft * yRight <= 0.0;
        }

      utils::throwException(
        intervalFound,
        "FractionalOccupancyFunction: failed to find a chemical potential "
        "interval that brackets the target electron count after " +
          std::to_string(maxBracketExpansionIter) +
          " bracket expansions. xLeft=" + std::to_string(xLeft) +
          " (getValue=" + std::to_string(yLeft) +
          "), xRight=" + std::to_string(xRight) +
          " (getValue=" + std::to_string(yRight) +
          "). Check for NaN/Inf eigenvalues; the number of wanted "
          "eigenvalues may also be insufficient.");

      d_lowerBound = xLeft;
      d_upperBound = xRight;
    }

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
      // Returns d_x rather than d_initialGuess: d_x tracks the current
      // best estimate (updated by setSolution()), so if a BisectionSolver
      // has already been run on this function, a subsequent
      // NewtonRaphsonSolver seeded via getInitialGuess() picks up the
      // bisected root instead of restarting from the original guess.
      return d_x;
    }

    const double
    FractionalOccupancyFunction::getLowerBound() const
    {
      return d_lowerBound;
    }

    const double
    FractionalOccupancyFunction::getUpperBound() const
    {
      return d_upperBound;
    }

  } // namespace ksdft
} // namespace dftefe
