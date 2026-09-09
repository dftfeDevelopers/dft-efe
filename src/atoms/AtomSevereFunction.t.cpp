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

#include <utils/SmearChargeDensityFunction.h>
#include <utils/PointChargePotentialFunction.h>
#include <atoms/SphericalHarmonics.h>
#include <type_traits>

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      template <typename FuncType>
      std::string
      getFieldName(FuncType type)
      {
        if constexpr (std::is_same_v<FuncType, AtomSevereFuncType::PSP>)
          {
            return "vlocal";
          }
        else
          {
            switch (type)
              {
                case AtomSevereFuncType::Atomic::density:
                  return "density";
                case AtomSevereFuncType::Atomic::vNuclearSq:
                case AtomSevereFuncType::Atomic::gradVNuclearSq:
                case AtomSevereFuncType::Atomic::bTimesVNuclear:
                  return "vnuclear";
                case AtomSevereFuncType::Atomic::vTotalSq:
                case AtomSevereFuncType::Atomic::gradVTotalSq:
                case AtomSevereFuncType::Atomic::bPlusRhoTimesVTotal:
                  return "vtotal";
                case AtomSevereFuncType::Atomic::orbitalSq:
                case AtomSevereFuncType::Atomic::gradOrbitalSq:
                case AtomSevereFuncType::Atomic::vExtTimesOrbitalSq:
                  return "orbital";
                default:
                  return "";
              }
          }
      }

      template <typename FuncType>
      AtomSuperpositionFuncType
      getAtomSupType(FuncType type)
      {
        if constexpr (std::is_same_v<FuncType, AtomSevereFuncType::PSP>)
          {
            return AtomSuperpositionFuncType::Identity;
          }
        else
          {
            switch (type)
              {
                case AtomSevereFuncType::Atomic::density:
                  return AtomSuperpositionFuncType::Identity;
                case AtomSevereFuncType::Atomic::gradVNuclearSq:
                case AtomSevereFuncType::Atomic::gradVTotalSq:
                case AtomSevereFuncType::Atomic::gradOrbitalSq:
                  return AtomSuperpositionFuncType::GradDotGradSq;
                default:
                  return AtomSuperpositionFuncType::IdentitySq;
              }
          }
      }

      template <typename FuncType>
      bool
      getIsComposite(FuncType type)
      {
        if constexpr (std::is_same_v<FuncType, AtomSevereFuncType::PSP>)
          return false;
        else
          return (type == AtomSevereFuncType::Atomic::bPlusRhoTimesVTotal ||
                  type == AtomSevereFuncType::Atomic::bTimesVNuclear ||
                  type == AtomSevereFuncType::Atomic::vExtTimesOrbitalSq);
      }

      template <typename FuncType>
      AtomSevereFuncType::Atomic
      getAtomicType(FuncType type)
      {
        if constexpr (std::is_same_v<FuncType, AtomSevereFuncType::Atomic>)
          return type;
        else
          return AtomSevereFuncType::Atomic::density;
      }
    } // namespace

    template <utils::MemorySpace memorySpace>
    template <typename FuncType>
    AtomSevereFunction<memorySpace>::AtomSevereFunction(
      std::shared_ptr<const AtomSphericalDataContainer>
                                                   atomSphericalDataContainer,
      const std::vector<std::string> &             atomSymbol,
      const std::vector<utils::Point> &            atomCoordinates,
      const std::vector<double> &                  atomCharges,
      double                                       smearedChargeRadius,
      FuncType                                     type,
      double                                       constant,
      linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext)
      : AtomSuperpositionFunction<memorySpace>(atomSphericalDataContainer,
                                               atomSymbol,
                                               atomCoordinates,
                                               getFieldName(type),
                                               linAlgOpContext)
      , d_atomSupType(getAtomSupType(type))
      , d_constant(constant)
      , d_isComposite(getIsComposite(type))
      , d_atomicType(getAtomicType(type))
    {
      if (d_isComposite)
        {
          if constexpr (std::is_same_v<FuncType, AtomSevereFuncType::Atomic>)
            {
              if (type == AtomSevereFuncType::Atomic::bPlusRhoTimesVTotal ||
                  type == AtomSevereFuncType::Atomic::bTimesVNuclear)
                {
                  d_b = std::make_shared<utils::SmearChargeDensityFunction>(
                    atomCoordinates, atomCharges, smearedChargeRadius);
                }
              if (type == AtomSevereFuncType::Atomic::bPlusRhoTimesVTotal)
                {
                  d_rho = std::make_shared<AtomSevereFunction<memorySpace>>(
                    atomSphericalDataContainer,
                    atomSymbol,
                    atomCoordinates,
                    atomCharges,
                    smearedChargeRadius,
                    AtomSevereFuncType::Atomic::density,
                    1.0 / (Clm(0, 0) * Dm(0) * Qm(0, 0)),
                    linAlgOpContext);
                }
              if (type == AtomSevereFuncType::Atomic::vExtTimesOrbitalSq)
                {
                  d_vext =
                    std::make_shared<utils::PointChargePotentialFunction>(
                      atomCoordinates, atomCharges);
                }
            }
        }
    }

    template <utils::MemorySpace memorySpace>
    double
    AtomSevereFunction<memorySpace>::operator()(const utils::Point &point) const
    {
      std::vector<double> t(this->d_dim);
      for (size_type iDim = 0; iDim < this->d_dim; ++iDim)
        t[iDim] = point[iDim];
      double q = 0.0;
      evalHost(1, t.data(), &q);
      return q;
    }

    template <utils::MemorySpace memorySpace>
    std::vector<double>
    AtomSevereFunction<memorySpace>::operator()(
      const std::vector<utils::Point> &points) const
    {
      const size_type     N = points.size();
      std::vector<double> t(N * this->d_dim);
      for (size_type iPoint = 0; iPoint < N; ++iPoint)
        for (size_type iDim = 0; iDim < this->d_dim; ++iDim)
          t[iPoint * this->d_dim + iDim] = points[iPoint][iDim];
      std::vector<double> q(N, 0.0);
      evalHost(N, t.data(), q.data());
      return q;
    }

    template <utils::MemorySpace memorySpace>
    void
    AtomSevereFunction<memorySpace>::evalHost(size_type     numPoints,
                                              const double *t,
                                              double *      q) const
    {
      if (!d_isComposite)
        {
          AtomSuperpositionFunction<memorySpace>::evalHost(
            numPoints, d_atomSupType, d_constant, t, q);
          return;
        }

      utils::Point              p(this->d_dim);
      std::vector<utils::Point> points(numPoints, p);
      for (size_type iPoint = 0; iPoint < numPoints; ++iPoint)
        for (size_type iDim = 0; iDim < this->d_dim; ++iDim)
          points[iPoint][iDim] = t[iPoint * this->d_dim + iDim];

      std::fill(q, q + numPoints, 0.0);

      if (d_atomicType == AtomSevereFuncType::Atomic::bPlusRhoTimesVTotal)
        {
          auto bVals   = (*d_b)(points);
          auto rhoVals = (*d_rho)(points);
          for (size_type e = 0; e < this->d_numEnrichmentFuncTotal; ++e)
            {
              utils::Point origin(
                this->d_atomCoordinatesVec[this->d_enrichmentToAtomId[e]]);
              for (size_type i = 0; i < numPoints; ++i)
                {
                  double val =
                    this->d_sphericalDataVecAll[e]->getValue(points[i], origin);
                  q[i] += std::fabs(val * (bVals[i] + rhoVals[i]));
                }
            }
        }
      else if (d_atomicType == AtomSevereFuncType::Atomic::bTimesVNuclear)
        {
          auto bVals = (*d_b)(points);
          for (size_type e = 0; e < this->d_numEnrichmentFuncTotal; ++e)
            {
              utils::Point origin(
                this->d_atomCoordinatesVec[this->d_enrichmentToAtomId[e]]);
              for (size_type i = 0; i < numPoints; ++i)
                {
                  double val =
                    this->d_sphericalDataVecAll[e]->getValue(points[i], origin);
                  q[i] += std::fabs(val * bVals[i]);
                }
            }
        }
      else if (d_atomicType == AtomSevereFuncType::Atomic::vExtTimesOrbitalSq)
        {
          auto vextVals = (*d_vext)(points);
          for (size_type e = 0; e < this->d_numEnrichmentFuncTotal; ++e)
            {
              utils::Point origin(
                this->d_atomCoordinatesVec[this->d_enrichmentToAtomId[e]]);
              for (size_type i = 0; i < numPoints; ++i)
                {
                  double val =
                    this->d_sphericalDataVecAll[e]->getValue(points[i], origin);
                  q[i] += std::fabs(val * val * vextVals[i]);
                }
            }
        }
    }

  } // namespace atoms
} // namespace dftefe
