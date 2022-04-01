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
 * @author Bikash Kanungo
 */

#ifndef dftefeVectorAttributes_h
#define dftefeVectorAttributes_h

#include <utils/TypeConfig.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    class VectorAttributes
    {
    public:
      enum class Distribution
      {
        SERIAL,
        DISTRIBUTED
      };

      VectorAttributes(const Distribution distribution);
      VectorAttributes()  = default;
      ~VectorAttributes() = default;

      bool
      areAttributesCompatible(const VectorAttributes &vecAttributes) const;

      bool
      areDistributionCompatible(const VectorAttributes &vecAttributes) const;

      Distribution
      getDistribution() const;

    private:
      Distribution d_distribution;
    };
  } // end of namespace linearAlgebra
} // end of namespace dftefe
#endif
