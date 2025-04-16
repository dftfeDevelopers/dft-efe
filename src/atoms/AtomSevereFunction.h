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
#include <memory>
#include <basis/AtomIdsPartition.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
namespace dftefe
{
  namespace atoms
  {
    template <unsigned int dim>
    class AtomSevereFunction : public utils::ScalarSpatialFunctionReal
    {
    public:
      AtomSevereFunction(std::shared_ptr<const AtomSphericalDataContainer>
                           atomSphericalDataContainer,
                         const std::vector<std::string> & atomSymbol,
                         const std::vector<utils::Point> &atomCoordinates,
                         const std::string                fieldName,
                         const size_type derivativeType,
                         const size_type sphericalValPower = 2 /* for Adaptive Quad */); // give arguments here
      double
      operator()(const utils::Point &point) const override;
      std::vector<double>
      operator()(const std::vector<utils::Point> &points) const override;

    private:
      const std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                d_atomSphericalDataContainer;
      const std::vector<std::string>  d_atomSymbolVec;
      std::vector<utils::Point> d_atomCoordinatesVec;
      const std::string               d_fieldName;
      const size_type                 d_derivativeType;
      const size_type                 d_sphericalValPower;
    };

  } // namespace atoms
} // namespace dftefe
#include <atoms/AtomSevereFunction.t.cpp>
#endif // dftefeAtomSevereFunction_h
