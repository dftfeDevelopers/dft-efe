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
#include <utils/SmearChargePotentialFunction.h>
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
                case AtomSevereFuncType::Atomic::vSmearSq:
                  return "vSmearSq";
                case AtomSevereFuncType::Atomic::gradVSmearSq:
                  return "gradVSmearSq";
                case AtomSevereFuncType::Atomic::bTimesVSmear:
                  return "bTimesVSmear";
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
                  type == AtomSevereFuncType::Atomic::bTimesVSmear ||
                  type == AtomSevereFuncType::Atomic::vSmearSq ||
                  type == AtomSevereFuncType::Atomic::gradVSmearSq ||
                  type == AtomSevereFuncType::Atomic::vExtTimesOrbitalSq);
      }

      bool
      isNoSphericalData(AtomSevereFuncType::Atomic type)
      {
        return (type == AtomSevereFuncType::Atomic::vSmearSq ||
                type == AtomSevereFuncType::Atomic::gradVSmearSq ||
                type == AtomSevereFuncType::Atomic::bTimesVSmear);
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
              utils::throwException(
                !isNoSphericalData(type),
                "AtomSevereFunction: " + getFieldName(type) +
                  " reads no spherical data, use the constructor that takes no "
                  "AtomSphericalDataContainer.");
              if (type == AtomSevereFuncType::Atomic::bPlusRhoTimesVTotal)
                {
                  d_b = std::make_shared<utils::SmearChargeDensityFunction>(
                    atomCoordinates, atomCharges, smearedChargeRadius);
                  d_rho =
                    std::make_shared<AtomSuperpositionFunction<memorySpace>>(
                      atomSphericalDataContainer,
                      atomSymbol,
                      atomCoordinates,
                      "density",
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
    template <typename FuncType>
    AtomSevereFunction<memorySpace>::AtomSevereFunction(
      const std::vector<utils::Point> &            atomCoordinates,
      const std::vector<double> &                  atomCharges,
      double                                       smearedChargeRadius,
      FuncType                                     type,
      double                                       constant,
      linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext)
      : AtomSuperpositionFunction<memorySpace>(atomCoordinates,
                                               getFieldName(type),
                                               linAlgOpContext)
      , d_atomSupType(getAtomSupType(type))
      , d_constant(constant)
      , d_isComposite(getIsComposite(type))
      , d_atomicType(getAtomicType(type))
    {
      if constexpr (std::is_same_v<FuncType, AtomSevereFuncType::Atomic>)
        {
          utils::throwException(
            isNoSphericalData(type),
            "AtomSevereFunction: " + getFieldName(type) +
              " reads spherical data, use the constructor that takes an "
              "AtomSphericalDataContainer.");
          if (type == AtomSevereFuncType::Atomic::bTimesVSmear)
            {
              d_b = std::make_shared<utils::SmearChargeDensityFunction>(
                atomCoordinates, atomCharges, smearedChargeRadius);
            }
          if (type == AtomSevereFuncType::Atomic::bTimesVSmear ||
              type == AtomSevereFuncType::Atomic::vSmearSq)
            {
              const size_type nAtoms = atomCoordinates.size();
              d_vsmearPerAtom.resize(nAtoms);
              for (size_type iAtom = 0; iAtom < nAtoms; ++iAtom)
                d_vsmearPerAtom[iAtom] =
                  std::make_shared<utils::SmearChargePotentialFunction>(
                    atomCoordinates[iAtom],
                    atomCharges[iAtom],
                    smearedChargeRadius);
            }
        }
      else
        utils::throwException(
          false,
          "AtomSevereFunction: the PSP type reads spherical data, use the "
          "constructor that takes an AtomSphericalDataContainer.");
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
      if constexpr (memorySpace == utils::MemorySpace::DEVICE)
        utils::throwException(
          false, "AtomSevereFunction<DEVICE>::evalHost should not be called.");

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
          auto                bVals = (*d_b)(points);
          std::vector<double> rhoVals(numPoints, 0.0);
          d_rho->evaluate(numPoints,
                          AtomSuperpositionFuncType::Identity,
                          t,
                          rhoVals.data(),
                          1.0 / (Clm(0, 0) * Dm(0) * Qm(0, 0)));
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
      else if (d_atomicType == AtomSevereFuncType::Atomic::bTimesVSmear)
        {
          // the nuclear self energy integrand: no spherical factor, so this
          // branch does not loop over the enrichment functions
          auto                bVals = (*d_b)(points);
          std::vector<double> vsmVals(numPoints, 0.0);
          for (size_type iAtom = 0; iAtom < d_vsmearPerAtom.size(); ++iAtom)
            {
              auto vals = (*d_vsmearPerAtom[iAtom])(points);
              for (size_type i = 0; i < numPoints; ++i)
                vsmVals[i] += vals[i];
            }
          for (size_type i = 0; i < numPoints; ++i)
            q[i] += std::fabs(bVals[i] * vsmVals[i]);
        }
      else if (d_atomicType == AtomSevereFuncType::Atomic::vSmearSq)
        {
          // sum of the squares of the individual atomic smeared potentials,
          // matching the IdentitySq convention of the other square criteria
          for (size_type iAtom = 0; iAtom < d_vsmearPerAtom.size(); ++iAtom)
            {
              auto vals = (*d_vsmearPerAtom[iAtom])(points);
              for (size_type i = 0; i < numPoints; ++i)
                q[i] += vals[i] * vals[i];
            }
        }
      else if (d_atomicType == AtomSevereFuncType::Atomic::gradVSmearSq)
        {
          utils::throwException(
            false,
            "AtomSevereFunction: gradVSmearSq cannot be evaluated because "
            "utils::SmearChargePotentialFunction provides no gradient.");
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
