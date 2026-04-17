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

#include <utils/ScalarSpatialFunction.h>
#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <basis/EnrichmentDataEvalKernels.h>
#include <memory>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>

namespace dftefe
{
  namespace atoms
  {
    class AtomSevereFunction : public utils::ScalarSpatialFunctionReal
    {
    public:
      AtomSevereFunction(
        std::shared_ptr<const AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const std::vector<std::string> & atomSymbol,
        const std::vector<utils::Point> &atomCoordinates,
        const std::string                fieldName,
        const size_type                  derivativeType,
        const size_type                  sphericalValPower = 2,
        const double                     constant          = 1.0,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
          *linAlgOpContext = nullptr);

      double
      operator()(const utils::Point &point) const override;

      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;

    protected:
      void
      evalHost(size_type      numPoints,
               const double * t,
               double *       q) const override;

#ifdef DFTEFE_WITH_DEVICE
      void
      evalDevice(size_type      numPoints,
                 const double * t,
                 double *       q) const override;
#endif

    private:
      const std::shared_ptr<const AtomSphericalDataContainer>
                                     d_atomSphericalDataContainer;
      const std::vector<std::string> d_atomSymbolVec;
      std::vector<utils::Point>      d_atomCoordinatesVec;
      const std::string              d_fieldName;
      const size_type                d_derivativeType;
      const size_type                d_sphericalValPower;
      const double                   d_constant;
      size_type                      d_numAtoms;
      size_type                      d_dim;

      std::vector<std::shared_ptr<SphericalData>> d_sphericalDataVecAll;
      std::vector<double>                         d_originsFlat;
      size_type                                   d_numEnrichmentFuncTotal;

#ifdef DFTEFE_WITH_DEVICE
      linearAlgebra::LinAlgOpContext<utils::MemorySpace::DEVICE>
        *d_linAlgOpContext;
      mutable utils::MemoryStorage<double, utils::MemorySpace::DEVICE>
        d_pointsTiledDevice;
      mutable utils::MemoryStorage<double, utils::MemorySpace::DEVICE>
        d_valuesDevice;
#endif
    };

  } // namespace atoms
} // namespace dftefe

#endif // dftefeAtomSevereFunction_h
