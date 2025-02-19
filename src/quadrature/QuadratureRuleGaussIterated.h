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
#ifndef dftefeQuadratureRuleGaussIterated_h
#define dftefeQuadratureRuleGaussIterated_h

#include "QuadratureRule.h"

namespace dftefe
{
  namespace quadrature
  {
    class QuadratureRuleGaussIterated : public QuadratureRule
    {
    public:
      QuadratureRuleGaussIterated(const size_type dim,
                                  const size_type order1D,
                                  const size_type copies);

      size_type
      numCopies() const;

      size_type
      order1D() const;

    private:
      size_type d_numCopies, d_order1D;
    };

  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRuleGaussIterated_h
