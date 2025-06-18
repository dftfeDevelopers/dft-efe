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

 #include<utils/Exceptions.h>
 #include<atoms/AtomSphericalDataPSP.h>
 #include<atoms/SphericalHarmonicFunctions.h>
 #include<string>
 #include<iostream>
 int main()
 {
  std::string atomFileName = "Cu.upf";
  const dftefe::atoms::SphericalHarmonicFunctions sphericalHarmonicFunctions(false);
  std::vector<std::string> fieldNames{}; // "vlocal", "beta", "pswfc", "nlcc", "rhoatom"
  std::vector<std::string> metadataNames = dftefe::atoms::AtomSphDataPSPDefaults::METADATANAMES;
  dftefe::atoms::AtomSphericalDataPSP atomTest(atomFileName, fieldNames, metadataNames, sphericalHarmonicFunctions);
  atomTest.addFieldName("vlocal");
  for(int i = 0 ; i < 50 ; i++)
   std::cout << i*0.5 << "\t" << atomTest.getSphericalData("vlocal",{0,0,0})->getValue(dftefe::utils::Point({0,0,0}),
     dftefe::utils::Point({i*0.5,0,0}))<<std::endl;
 }
 