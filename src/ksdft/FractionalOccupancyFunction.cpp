#include <cmath>
#include <vector>
#include <FractionalOccupancyFunction.h>

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

  } // namespace ksdft
} // namespace dftefe
