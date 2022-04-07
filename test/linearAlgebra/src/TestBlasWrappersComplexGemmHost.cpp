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

#include <linearAlgebra/BlasWrappers.h>
#include <vector>
#include <iostream>

int
main(int argc, char **argv)
{
 const dftefe::utils::MemorySpace Host = dftefe::utils::MemorySpace::HOST;
 using namespace dftefe::linearAlgebra::blasWrapper;

 double tol = 1e-12;

 dftefe::size_type Am = 10, An = 5, Bm = An, Bn = 3, Cm = Am, Cn = Bn;

 std::vector<std::complex<double>> colMajA = {
   6.5163-7.3605i, 0.7668+8.8410i, 9.9227+9.1227i, -8.4365+1.5042i, -1.1464-8.8044i, -7.8669-5.3044i, 9.2380-2.9368i, -9.9073+6.4239i, 5.4982-9.6919i, 6.3461-9.1395i, 7.3739-6.6202i, -8.3113+2.9823i, -2.0043+4.6344i, -4.8026+2.9549i, 6.0014-0.9815i, -1.3717+0.9402i, 8.2130-4.0736i, -6.3631+4.8939i, -4.7239-6.2209i, -7.0892+3.7355i, -7.2786-6.3298i, 7.3858-2.6303i, 1.5941+2.5124i, 0.9972+5.6045i, -7.1009-8.3775i, 7.0606+8.5877i, 2.4411+5.5143i, -2.9810-0.2642i, 0.2650-1.2828i, -1.9638-1.0643i, -8.4807-3.8730i, -5.2017+0.1702i, -7.5336+0.2154i, -6.3218+6.3526i, -5.2009+5.8966i, -1.6547+2.8864i, -9.0069-2.4278i, 8.0543+6.2316i, 8.8957+0.6565i, -0.1827-2.9855i, -0.2149+8.7800i, -3.2456+7.5189i, 8.0011+1.0031i, -2.6151+2.4495i, -7.7759+1.7409i, 5.6050-5.8452i, -2.2052-3.9751i, -5.1662-0.5815i, -1.9218-5.3902i, -8.0709+6.8862i};

 std::vector<std::complex<double>> colMajB    = {-6.1047+1.8979i, -5.4816-4.7558i, -6.5858+2.0569i, -5.4467+4.2243i, -1.2860-5.5651i, -3.7780-7.6516i, 8.4676-4.0665i, -1.3959-3.6244i, -6.3037-1.5167i, 8.0976+0.1572i, 9.5950-8.2897i, -1.2226-4.7504i, -7.7776+6.0203i, -4.8387-9.4156i, -1.8256+8.5771i};

 std::vector<double> colMajCRef = {74.92997043000+60.41177740000i, 68.68128451000+-11.33054285000i, -25.11192585000+-144.81250496000i, 95.52346107000+-103.31332342000i, 73.24681851000+55.92558929000i, -36.99845809000+-69.59977720000i, -102.60633765000+-20.00364712000i, 59.83337972000+-29.39511641000i, -96.71910555000+186.79008085000i, 112.66095147000+135.59545175000i, -13.73630099000+35.46105722000i, -7.75695628000+63.87618727000i, 153.51847755000+-52.83320611000i, 61.57782105000+80.95033363000i, -58.17323841000+30.34237712000i, 63.72613459000+-5.94065997000i, 48.04782842000+-147.59750358000i, -27.25546966000+73.00649468000i, -235.04215973000+-99.67046561000i, -209.67549601000+128.62316201000i, -14.58470454000+-65.44681352000i, 31.57340509000+185.82008578000i, 183.04524314000+135.87391121000i, -15.89947478000+67.86778036000i, 89.62708261000+-129.31182216000i, -144.84031490000+55.75960501000i, 41.56047283000+-82.08987665000i, 48.11566773000+2.87984341000i, -32.82382050000+-190.53171849000i, -38.33508665000+-180.36641389000i};

 std::vector<double> C(Cm * Cn, 0.0);

 int lda = Am, ldb = Bm, ldc = Cm;

 double alpha = 1.0, beta = 0.0;

 blasQueueType<Host> queue;

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

 for(unsigned int i = 0; i < C.size(); ++i)
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

 for (auto i : C) i = 0;

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

 for(unsigned int i = 0; i < C.size(); ++i)
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
