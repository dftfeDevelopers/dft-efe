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

#include <utils/PeriodicTableManager.h>
#include <algorithm>
namespace dftefe {

  namespace utils {

    namespace PeriodicTableInternal {
      PeriodicTableManager::d_atomicNumberToSymbolMap = {
       "H",
       "He",
       "Li",
       "Be",
       "B",
       "C",
       "N",
       "O",
       "F",
       "Ne",
       "Na",
       "Mg",
       "Al",
       "Si",
       "P",
       "S",
       "Cl",
       "Ar",
       "K",
       "Ca",
       "Sc",
       "Ti",
       "V",
       "Cr",
       "Mn",
       "Fe",
       "Co",
       "Ni",
       "Cu",
       "Zn",
       "Ga",
       "Ge",
       "As",
       "Se",
       "Br",
       "Kr",
       "Rb",
       "Sr",
       "Y",
       "Zr",
       "Nb",
       "Mo",
       "Tc",
       "Ru",
       "Rh",
       "Pd",
       "Ag",
       "Cd",
       "In",
       "Sn",
       "Sb",
       "Te",
       "I",
       "Xe",
       "Cs",
       "Ba",
       "La",
       "Ce",
       "Pr",
       "Nd",
       "Pm",
       "Sm",
       "Eu",
       "Gd",
       "Tb",
       "Dy",
       "Ho",
       "Er",
       "Tm",
       "Yb",
       "Lu",
       "Hf",
       "Ta",
       "W",
       "Re",
       "Os",
       "Ir",
       "Pt",
       "Au",
       "Hg",
       "Tl",
       "Pb",
       "Bi",
       "Po",
       "At",
       "Rn",
       "Fr",
       "Ra",
       "Ac",
       "Th",
       "Pa",
       "U",
       "Np",
       "Pu",
       "Am",
       "Cm",
       "Bk",
       "Cf",
       "Es",
       "Fm",
       "Md",
       "No",
       "Lr",
       "Rf",
       "Db",
       "Sg",
       "Bh",
       "Hs",
       "Mt",
       "Ds",
       "Rg",
       "Cp",
       "Uut",
       "Uuq",
       "Uup",
       "Uuh",
       "Uus",
       "Uuo"
      };
    } // end of namespace PeriodicTableInternal

  std::map<double, std::string> PeriodicTableManager::d_atomicNumberToSymbolMap;
  std::map<std::string, double> PeriodicTableManager::d_symbolToAtomicNumberMap;

  void PeriodicTableManager::PeriodicTableManager()
  {
    const int numberElements = PeriodicTableInternal::symbols.size();
    for(size_type i = 0; i < numberElements; ++i)
    {
      std::string lower = PeriodicTableInternal::symbols[i];
      std::for_each(lower.begin(), lower.end(), [](char & c){c = std::tolower(c);});
      const double atomicNumber = 1.0 + i;
      d_symbolToAtomicNumberMap[lower] = atomicNumber;
      d_atomicNumberToSymbolMap[atomicNumber] = symbols[i];
    }
  }
  } // end of namespace utils
} // end of namespace dftefe

 
