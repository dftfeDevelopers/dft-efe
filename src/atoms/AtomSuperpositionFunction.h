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

#ifndef dftefeAtomSuperpositionFunction_h
#define dftefeAtomSuperpositionFunction_h

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
    enum class AtomSuperpositionFuncType
    {
      Identity,
      Grad,
      IdentitySq,// for the adaptive quadrature builfing
      GradDotGradSq,
    };
    
    template <utils::MemorySpace memorySpace>
    class AtomSuperpositionFunction
    {
    public:
      AtomSuperpositionFunction(
        std::shared_ptr<const AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const std::vector<std::string> & atomSymbol,
        const std::vector<utils::Point> &atomCoordinates,
        const std::string                fieldName,
        linearAlgebra::LinAlgOpContext<memorySpace> *linAlgOpContext =
          nullptr);

      void
      evaluate(const size_type                numPoints,
           const AtomSuperpositionFuncType atomSupType,
           const double *                 t,
           double *                       q,
           const double                   constant = 1.0) const
      {
#ifdef DFTEFE_WITH_DEVICE
        if (memorySpace == utils::MemorySpace::DEVICE)
          evalDevice(numPoints, atomSupType, constant, t, q);
        else
#endif
          evalHost(numPoints, atomSupType, constant, t, q);
      }

    protected:
      void
      evalHost(size_type                      numPoints,
               const AtomSuperpositionFuncType atomSupType,
               const double                   constant,
               const double *                 t,
               double *                       q) const;

#ifdef DFTEFE_WITH_DEVICE
      void
      evalDevice(size_type                      numPoints,
                 const AtomSuperpositionFuncType atomSupType,
                 const double                   constant,
                 const double *                 t,
                 double *                       q) const;
#endif

    private:
      const std::shared_ptr<const AtomSphericalDataContainer>
                                     d_atomSphericalDataContainer;
      const std::vector<std::string> d_atomSymbolVec;
      std::vector<utils::Point>      d_atomCoordinatesVec;
      const std::string              d_fieldName;
      size_type                      d_numAtoms;
      size_type                      d_dim;

      std::vector<std::shared_ptr<SphericalData>> d_sphericalDataVecAll;
      size_type                                   d_numEnrichmentFuncTotal;

      linearAlgebra::LinAlgOpContext<memorySpace> *d_linAlgOpContext;
      utils::MemoryStorage<double, memorySpace>   d_originsFlat;
      //mutable utils::MemoryStorage<double, memorySpace> d_values;
    };

  } // namespace atoms
} // namespace dftefe

#include <atoms/AtomSuperpositionFunction.t.cpp>

#endif // dftefeAtomSuperpositionFunction_h
