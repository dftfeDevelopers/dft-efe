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

#ifndef dftefeOrthonormalizationFunctions_h
#define dftefeOrthonormalizationFunctions_h

#include <utils/TypeConfig.h>
#include <linearAlgebra/LinearAlgebraTypes.h>
#include <string>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/Vector.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/BlasLapackTypedef.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, utils::MemorySpace memorySpace>
    class OrthonormalizationFunctions
    {
    public:
      static void
      CholeskyGramSchmidt(const MultiVector<ValueType, memorySpace> &X,
                          MultiVector<ValueType, memorySpace> &orthogonalizedX);

    }; // end of class OrthonormalizationFunctions
  }    // end of namespace linearAlgebra
} // end of namespace dftefe
#include <linearAlgebra/OrthonormalizationFunctions.t.cpp>
#endif
