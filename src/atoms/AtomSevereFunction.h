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

#ifndef dftefeAtomSevereFunction_h
#define dftefeAtomSevereFunction_h

#include <atoms/AtomSuperpositionFunction.h>
#include <utils/ScalarSpatialFunction.h>

namespace dftefe
{
  namespace atoms
  {
    struct AtomSevereFuncType
    {
      enum class PSP
      {
        vLocal
      };

      enum class Atomic
      {
        density,
        vNuclearSq,
        gradVNuclearSq,
        vTotalSq,
        gradVTotalSq,
        orbitalSq,
        gradOrbitalSq,
        bPlusRhoTimesVTotal,
        bTimesVNuclear,
        vExtTimesOrbitalSq
      };
    };

    template <utils::MemorySpace memorySpace>
    class AtomSevereFunction : public AtomSuperpositionFunction<memorySpace>,
                               public utils::ScalarSpatialFunctionReal
    {
    public:
      template <typename FuncType>
      AtomSevereFunction(
        std::shared_ptr<const AtomSphericalDataContainer>
                                                     atomSphericalDataContainer,
        const std::vector<std::string> &             atomSymbol,
        const std::vector<utils::Point> &            atomCoordinates,
        const std::vector<double> &                  atomCharges,
        double                                       smearedChargeRadius,
        FuncType                                     type,
        double                                       constant        = 1.0,
        linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext = nullptr);

      double
      operator()(const utils::Point &point) const override;

      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;

    protected:
      void
      evalHost(size_type numPoints, const double *t, double *q) const override;

#ifdef DFTEFE_WITH_DEVICE
      void
      evalDevice(size_type     numPoints,
                 const double *t,
                 double *      q) const override;
#endif

    private:
      AtomSuperpositionFuncType                         d_atomSupType;
      double                                            d_constant;
      bool                                              d_isComposite;
      AtomSevereFuncType::Atomic                        d_atomicType;
      std::shared_ptr<utils::ScalarSpatialFunctionReal> d_b;
      std::shared_ptr<utils::ScalarSpatialFunctionReal> d_vext;
      std::shared_ptr<utils::ScalarSpatialFunctionReal> d_rho;
    };

  } // namespace atoms
} // namespace dftefe

#include <atoms/AtomSevereFunction.t.cpp>

#endif // dftefeAtomSevereFunction_h
