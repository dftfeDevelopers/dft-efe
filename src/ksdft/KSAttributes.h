
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

#ifndef dftefe_KSAttributes_h
#define dftefe_KSAttributes_h

namespace dftefe
{
  namespace ksdft
  {
    enum class DensityDescrAttr
    {
      Val,
      Grad,
      Hessian,
      Laplacian,
    };

    enum class WfcDescrAttr
    {
      Tau,
    };

    enum class DensityObsAttr
    {
      Monopole,
      Dipole,
      Quadrupole,
    };

    enum class SpinMode
    {
      Unpolarized,
      Collinear,
      NonCollinear,
    };

    /**
     * Whether the nuclear potential is a pseudopotential or the bare
     * point charge (all-electron).
     */
    enum class CalculationType
    {
      PSP,
      AE,
    };
  } // namespace ksdft
} // namespace dftefe
#endif // dftefe_KSAttributes_h
