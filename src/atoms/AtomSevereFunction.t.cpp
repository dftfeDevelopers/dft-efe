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

namespace dftefe
{
  namespace atoms
  {
    template <utils::MemorySpace memorySpace>
    AtomSevereFunction<memorySpace>::AtomSevereFunction(
      std::shared_ptr<const AtomSphericalDataContainer>
                                       atomSphericalDataContainer,
      const std::vector<std::string> & atomSymbol,
      const std::vector<utils::Point> &atomCoordinates,
      const std::string                fieldName,
      const size_type                  derivativeType,
      const size_type                  sphericalValPower,
      const double                     constant,
      linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext)
      : AtomSuperpositionFunction<memorySpace>(atomSphericalDataContainer,
                                               atomSymbol,
                                               atomCoordinates,
                                               fieldName,
                                               linAlgOpContext)
      , d_dim(atomCoordinates[0].size())
      , d_constant(constant)
    {
      utils::throwException(
        (derivativeType == 0 || derivativeType == 1) &&
          (sphericalValPower == 1 || sphericalValPower == 2) &&
          !(derivativeType == 1 && sphericalValPower == 1),
        "AtomSevereFunction: valid combinations are (derivativeType=0, "
        "sphericalValPower=1), (derivativeType=0, sphericalValPower=2), "
        "or (derivativeType=1, sphericalValPower=2).");

      if (derivativeType == 0 && sphericalValPower == 1)
        d_atomSupType = AtomSuperpositionFuncType::Identity;
      else if (derivativeType == 0 && sphericalValPower == 2)
        d_atomSupType = AtomSuperpositionFuncType::IdentitySq;
      else
        d_atomSupType = AtomSuperpositionFuncType::GradDotGradSq;
    }

    template <utils::MemorySpace memorySpace>
    double
    AtomSevereFunction<memorySpace>::operator()(
      const utils::Point &point) const
    {
      std::vector<double> t(d_dim);
      for (size_type iDim = 0; iDim < d_dim; ++iDim)
        t[iDim] = point[iDim];
      double q = 0.0;
      AtomSuperpositionFunction<memorySpace>::evalHost(
        1, d_atomSupType, d_constant, t.data(), &q);
      return q;
    }

    template <utils::MemorySpace memorySpace>
    std::vector<double>
    AtomSevereFunction<memorySpace>::operator()(
      const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> t(N * d_dim);
      for (size_type iPoint = 0; iPoint < N; ++iPoint)
        for (size_type iDim = 0; iDim < d_dim; ++iDim)
          t[iPoint * d_dim + iDim] = points[iPoint][iDim];
      std::vector<double> q(N, 0.0);
      AtomSuperpositionFunction<memorySpace>::evalHost(
        N, d_atomSupType, d_constant, t.data(), q.data());
      return q;
    }

    template <utils::MemorySpace memorySpace>
    void
    AtomSevereFunction<memorySpace>::evalHost(size_type     numPoints,
                                              const double *t,
                                              double *      q) const
    {
      AtomSuperpositionFunction<memorySpace>::evalHost(
        numPoints, d_atomSupType, d_constant, t, q);
    }

#ifdef DFTEFE_WITH_DEVICE
    template <utils::MemorySpace memorySpace>
    void
    AtomSevereFunction<memorySpace>::evalDevice(size_type     numPoints,
                                                const double *t,
                                                double *      q) const
    {
      AtomSuperpositionFunction<memorySpace>::evalDevice(
        numPoints, d_atomSupType, d_constant, t, q);
    }
#endif

  } // namespace atoms
} // namespace dftefe
