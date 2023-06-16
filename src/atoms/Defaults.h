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

#ifndef dftefeSphericalDataDefaults_h
#define dftefeSphericalDataDefaults_h

#include <utils/TypeConfig.h>

namespace dftefe
{
  namespace atoms
  {
    class SphericalDataDefaults
    {
    public:
      //
      // The polar angle tolerance
      //
      static const double POL_ANG_TOL;

      //
      // The tolerance in the derivative calcuation of the 
      // smooth cutoff function ananlytically
      //
      static const double CUTOFF_TOL;

      //
      // The spherical data are defined for a 3 dimensional case only for now.
      //
      static const size_type DEFAULT_DIM;

    }; // end of class SphericalDataDefaults
  }    // end of namespace atoms
} // end of namespace dftefe
#endif // dftefeSphericalDataDefaults_h
