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
#include <atoms/Defaults.h>
#include <limits.h>
#include <utils/TypeConfig.h>
namespace dftefe
{
  namespace atoms
  {
    /**
     * @brief Setting all the SphericalDataDefaults
     */
    const double    SphericalDataDefaults::POL_ANG_TOL = 1e-14;
    const double    SphericalDataDefaults::CUTOFF_TOL  = 1e-14;
    const double    SphericalDataDefaults::RADIUS_TOL  = 1e-14;
    const size_type SphericalDataDefaults::DEFAULT_DIM = 3;

    const std::map<std::string, std::string>
      AtomSphDataPSPDefaults::UPF_PARENT_PATH_LOOKUP{
        {"r", "PP_MESH"}, //{fieldname/metadataname, approx. path}
        {"vlocal", "PP_LOCAL"},
        {"beta", "PP_NONLOCAL"},
        {"pswfc", "PP_PSWFC"},
        {"nlcc", "PP_NLCC"},
        {"rhoatom", "PP_RHOATOM"}};

    const std::vector<std::string> AtomSphDataPSPDefaults::METADATANAMES{
      "element",
      "z_valence",
      "number_of_wfc",
      "number_of_proj",
      "core_correction",
      "dij"};

  } // end of namespace atoms
} // end of namespace dftefe
