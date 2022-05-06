/******************************************************************************
* Copyright (c) 2022.                                                        *
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
* @author Ian C. Lin.
*/

#include <linearAlgebra/BlasLapack.h>
#include <vector>
#include <iostream>

int
main(int argc, char **argv)
{
 const dftefe::utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
 using namespace dftefe::linearAlgebra::blasLapack;

 double tol = 1e-12;

 dftefe::size_type Am = 10, An = 5, Bm = An, Bn = 3, Cm = Am, Cn = Bn;

 std::vector<double> colMajA = {
   -4.47950, 3.59405,  3.10196,  -6.74777, -7.62005, -0.03272, 9.19488,
   -3.19229, 1.70536,  -5.52376, 5.02534,  -4.89810, 0.11914,  3.98153,
   7.81807,  9.18583,  0.94431,  -7.22751, -7.01412, -4.84983, 6.81435,
   -4.91436, 6.28570,  -5.12950, 8.58527,  -3.00032, -6.06809, -4.97832,
   2.32089,  -0.53422, -2.96681, 6.61657,  1.70528,  0.99447,  8.34387,
   -4.28322, 5.14400,  5.07458,  -2.39108, 1.35643,  -8.48291, -8.92100,
   0.61595,  5.58334,  8.68021,  -7.40188, 1.37647,  -0.61219, -9.76196,
   -3.25755};
 std::vector<double> colMajB    = {-6.75635,
                                5.88569,
                                -3.77570,
                                0.57066,
                                -6.68703,
                                2.03964,
                                -4.74057,
                                3.08158,
                                3.78429,
                                4.96303,
                                -0.98917,
                                -8.32357,
                                -5.42046,
                                8.26675,
                                -6.95244};
 std::vector<double> colMajCRef = {
   89.14615577730,  28.87459761170,  -47.13536487310, 51.62334246520,
   11.79974547470,  112.66665505990, -39.92364529520, 4.81558075210,
   2.34609631420,   33.35030931760,  -65.28878498300, -3.82981037450,
   34.64222997030,  -16.97106108840, 48.50769588330,  -105.80323235970,
   21.87622226840,  28.57566837200,  -13.61618494700, -0.95595880270,
   -39.88364761690, 180.57289087160, -28.31665330300, -29.25840391830,
   -95.44479256550, -44.11060117130, 48.89084619830,  136.50768579580,
   92.21831704020,  82.58874393800};

 std::vector<double> C(Cm * Cn, 0.0);

 int lda = Am, ldb = Bm, ldc = Cm;

 double alpha = 1.0, beta = 0.0;

 BlasQueue<Host> queue;

 gemm(Layout::ColMajor,
      Op::NoTrans,
      Op::NoTrans,
      Am,
      Cn,
      An,
      alpha,
      colMajA.data(),
      lda,
      colMajB.data(),
      ldb,
      beta,
      C.data(),
      ldc,
      queue);

 for(dftefe::size_type i = 0; i < C.size(); ++i)
   {
     if(std::fabs(C[i]-colMajCRef[i]) > tol)
       {
         std::string msg = "At index " + std::to_string(i) +
                           " mismatch of entries after doing column major dftefe::linearAlgebra::blasWrapper::Gemm. "
                           " dftefe::linearAlgebra::blasWrapper::Gemm value: " + std::to_string(C[i]) +
                           " reference value: " + std::to_string(colMajCRef[i]);
         throw std::runtime_error(msg);
       }
   }

 std::vector<double> rowMajA = {
   -4.47950, 5.02534,  6.81435,  -2.96681, -8.48291, 3.59405,  -4.89810,
   -4.91436, 6.61657,  -8.92100, 3.10196,  0.11914,  6.28570,  1.70528,
   0.61595,  -6.74777, 3.98153,  -5.12950, 0.99447,  5.58334,  -7.62005,
   7.81807,  8.58527,  8.34387,  8.68021,  -0.03272, 9.18583,  -3.00032,
   -4.28322, -7.40188, 9.19488,  0.94431,  -6.06809, 5.14400,  1.37647,
   -3.19229, -7.22751, -4.97832, 5.07458,  -0.61219, 1.70536,  -7.01412,
   2.32089,  -2.39108, -9.76196, -5.52376, -4.84983, -0.53422, 1.35643,
   -3.25755};
 std::vector<double> rowMajB    = {-6.75635,
                                2.03964,
                                -0.98917,
                                5.88569,
                                -4.74057,
                                -8.32357,
                                -3.77570,
                                3.08158,
                                -5.42046,
                                0.57066,
                                3.78429,
                                8.26675,
                                -6.68703,
                                4.96303,
                                -6.95244};
 std::vector<double> rowMajCRef = {
   89.14615577730,   -65.28878498300, -39.88364761690, 28.87459761170,
   -3.82981037450,   180.57289087160, -47.13536487310, 34.64222997030,
   -28.31665330300,  51.62334246520,  -16.97106108840, -29.25840391830,
   11.79974547470,   48.50769588330,  -95.44479256550, 112.66665505990,
   -105.80323235970, -44.11060117130, -39.92364529520, 21.87622226840,
   48.89084619830,   4.81558075210,   28.57566837200,  136.50768579580,
   2.34609631420,    -13.61618494700, 92.21831704020,  33.35030931760,
   -0.95595880270,   82.58874393800};

 for (auto &i : C) i = 0;

 lda = An, ldb = Bn, ldc = Cn;

 gemm(Layout::RowMajor,
      Op::NoTrans,
      Op::NoTrans,
      Am,
      Cn,
      An,
      alpha,
      rowMajA.data(),
      lda,
      rowMajB.data(),
      ldb,
      beta,
      C.data(),
      ldc,
      queue);

 for(dftefe::size_type i = 0; i < C.size(); ++i)
   {
     if(std::fabs(C[i]-rowMajCRef[i]) > tol)
       {
         std::string msg = "At index " + std::to_string(i) +
                           " mismatch of entries after doing row major dftefe::linearAlgebra::blasWrapper::Gemm. "
                           " dftefe::linearAlgebra::blasWrapper::Gemm value: " + std::to_string(C[i]) +
                           " reference value: " + std::to_string(rowMajCRef[i]);
         throw std::runtime_error(msg);
       }
   }
}
