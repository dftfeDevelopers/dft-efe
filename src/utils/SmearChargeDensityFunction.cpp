#include <cmath>
#include <utils/SmearChargeDensityFunction.h>

namespace dftefe
{
  namespace utils
  {
    SmearChargeDensityFunction::SmearChargeDensityFunction(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<double> &      atomCharges,
      const std::vector<double> &      smearedChargeRadius)
      : d_atomCoordinates(atomCoordinates)
      , d_rc(smearedChargeRadius)
      , d_z(atomCharges)
      , d_numAtoms(atomCoordinates.size())
      , d_dim(atomCoordinates[0].size())
    {
#ifdef DFTEFE_WITH_DEVICE
      std::vector<double> atomCoordsFlat = utils::flatten(atomCoordinates);
      d_atomCoordsFlatDevice.resize(atomCoordsFlat.size());
      d_rcDevice.resize(d_rc.size());
      d_zDevice.resize(d_z.size());
      MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST>::copy(
        atomCoordsFlat.size(),
        d_atomCoordsFlatDevice.data(),
        atomCoordsFlat.data());
      MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST>::copy(
        d_rc.size(), d_rcDevice.data(), d_rc.data());
      MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST>::copy(
        d_z.size(), d_zDevice.data(), d_z.data());
#endif
    }

    SmearChargeDensityFunction::SmearChargeDensityFunction(
      const std::vector<utils::Point> &atomCoordinates,
      const std::vector<double> &      atomCharges,
      const double &                   smearedChargeRadius)
      : SmearChargeDensityFunction(
          atomCoordinates,
          atomCharges,
          std::vector<double>(atomCoordinates.size(), smearedChargeRadius))
    {}

    SmearChargeDensityFunction::SmearChargeDensityFunction(
      const utils::Point &atomCoordinates,
      const double        atomCharges,
      const double        smearedChargeRadius)
      : SmearChargeDensityFunction(
          std::vector<utils::Point>{atomCoordinates},
          std::vector<double>{atomCharges},
          std::vector<double>{smearedChargeRadius})
    {}

    double
    SmearChargeDensityFunction::operator()(const utils::Point &point) const
    {
      double ret = 0;
      for (size_type i = 0; i < d_atomCoordinates.size(); i++)
        {
          double r = 0;
          for (size_type j = 0; j < point.size(); j++)
            {
              r += std::pow((point[j] - d_atomCoordinates[i][j]), 2);
            }
          r = std::sqrt(r);
          if (r > d_rc[i])
            ret += 0;
          else
            ret += d_z[i] * -21 * std::pow((r - d_rc[i]), 3) *
                   (6 * r * r + 3 * r * d_rc[i] + d_rc[i] * d_rc[i]) /
                   (5 * M_PI * std::pow(d_rc[i], 8));
        }
      return ret;
    }

    std::vector<double>
    SmearChargeDensityFunction::operator()(
      const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> returnValue(N, 0.0);
      for (size_type i = 0; i < d_atomCoordinates.size(); i++)
        {
          for (size_type iPoint = 0; iPoint < N; ++iPoint)
            {
              double r = 0;
              for (size_type j = 0; j < points[iPoint].size(); j++)
                {
                  r +=
                    std::pow((points[iPoint][j] - d_atomCoordinates[i][j]), 2);
                }
              r = std::sqrt(r);
              if (r < d_rc[i])
                returnValue[iPoint] +=
                  d_z[i] * -21 * std::pow((r - d_rc[i]), 3) *
                  (6 * r * r + 3 * r * d_rc[i] + d_rc[i] * d_rc[i]) /
                  (5 * M_PI * std::pow(d_rc[i], 8));
            }
        }
      return returnValue;
    }

    void
    SmearChargeDensityFunction::evalHost(size_type     numPoints,
                                         const double *t,
                                         double *      q) const
    {
      utils::Point              p(d_dim);
      std::vector<utils::Point> points(numPoints, p);
      for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
        for (size_type iDim = 0; iDim < d_dim; ++iDim)
          points[iPoint][iDim] = t[iPoint * d_dim + iDim];
      std::vector<double> retValue = (*this)(points);
      std::copy(retValue.begin(), retValue.end(), q);
    }

  } // namespace utils
} // namespace dftefe
