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

#include<utils/Exceptions.h>
#include<atoms/AtomSphericalData.h>
#include<string>
#include<iostream>
int main()
{
  std::string atomFileName = "TestAtom.xml";
  std::vector<std::string> fieldNames{ "density", "vhartree", "vnuclear", "vtotal", "orbital" };
  std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
  std::vector<int> qNumbers{1, 0, 0};
  dftefe::atoms::AtomSphericalData atomTest(atomFileName, fieldNames, metadataNames);
  auto sphericalDataObj = atomTest.getSphericalData("vnuclear", qNumbers);
  std::vector<double> pointvec{0, 0, 2.};
  std::vector<double> originvec{0. ,0. ,0.};
  dftefe::utils::Point point(pointvec);
  dftefe::utils::Point origin(originvec);
  std::cout<<sphericalDataObj->getValue(point,origin)<<"\n";
  // std::cout<<sphericalDataObj->getGradientValue(point,origin)[1]<<"\n";
  // std::cout<<sphericalDataObj->getGradientValue(point,origin)[2]<<"\n";

      fieldNames.clear();
    fieldNames.push_back("exp");
    std::map<std::string , std::vector<std::vector<int>>> fieldToQuantumNumbersVec{{"exp" , {{1,0,0}}}};
    std::map<std::string , std::vector<std::shared_ptr<dftefe::utils::ScalarSpatialFunctionReal>>> 
      fieldToScalarSpatialFnRealVec{{"exp" , {std::make_shared<dftefe::utils::ExpModX>(0 , -1)}}};
  dftefe::atoms::AtomSphericalDataAnalytical atomTest1
    (fieldToQuantumNumbersVec, fieldToScalarSpatialFnRealVec , fieldNames, sphericalHarmonicFunctions);
  for(int i = 0 ; i < 50 ; i++)
   std::cout << i*0.5 << "\t" << atomTest1.getSphericalData("exp",{1,0,0})->getValue(dftefe::utils::Point({i*0.5,0,0}),
     dftefe::utils::Point({0,0,0}))<<std::endl;

}
