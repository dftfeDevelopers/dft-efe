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

#include <vector>

int
main(int argc, char **argv) {

 size_type Am = 10, An = 5, Bm = An, Bn = 3, Cm = Am, Cn = Bn;

 std::vector<double> colMajA = {-4.47950, 3.59405, 3.10196, -6.74777, -7.62005, -0.03272, 9.19488, -3.19229, 1.70536, -5.52376, 5.02534, -4.89810, 0.11914, 3.98153, 7.81807, 9.18583, 0.94431, -7.22751, -7.01412, -4.84983, 6.81435, -4.91436, 6.28570, -5.12950, 8.58527, -3.00032, -6.06809, -4.97832, 2.32089, -0.53422, -2.96681, 6.61657, 1.70528, 0.99447, 8.34387, -4.28322, 5.14400, 5.07458, -2.39108, 1.35643, -8.48291, -8.92100, 0.61595, 5.58334, 8.68021, -7.40188, 1.37647, -0.61219, -9.76196, -3.25755};

 int lda = Am, ldb = Bm, ldc = Cm;
 std::vector<double> A(Am*An), B(Bm*Bn), C(Cm*Cn, 0.0);
 double alpha = 1.5, beta = 1.1;
 for (auto i = 0; i < A.size(); ++i) {
     A[i] = i;
   }
 //  std::cout << "mat A: " << std::endl;
 //  printMatrix(A, Am, An);

 for (auto i = 0; i < B.size(); ++i) {
     B[i] = A.size() + i;
   }
 //  std::cout << "mat A: " << std::endl;
 //  printMatrix(B, Bm, Bn);

 dftefe::linearAlgebra::blasWrapper::blasQueueType<dftefe::utils::MemorySpace::HOST> a;

 dftefe::linearAlgebra::gemm(dftefe::linearAlgebra::blasWrapper::Layout::ColMajor,
                             dftefe::linearAlgebra::blasWrapper::Op::NoTrans,
                             dftefe::linearAlgebra::blasWrapper::Op::NoTrans,
                             Am,
                             Cn,
                             An,
                             alpha,
                             A.data(),
                             lda,
                             B.data(),
                             ldb,
                             beta,
                             C.data(),
                             ldc,
                             a);

 std::cout << "mat C: " << std::endl;
 for (auto i : C) {
     printf("%.4f ", i);
   }
}