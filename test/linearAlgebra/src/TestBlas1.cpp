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

#include <iostream>
#include <fstream>

#include "linearAlgebra/BlasWrappers.h"

/**
* @brief print out column major matrix in a readable format
* @param mat matrix input
* @param m number of rows
* @param n number of columns
*/
void printMatrix(const std::vector<double> &mat, int64_t m, int64_t n) {
 for (auto i = 0; i < m; ++i) {
     for (auto j = 0; j < n; ++j) {
         printf("%.8f, ", mat[i+j*m]);
       }
     printf("\n");
   }
}

int
main(int argc, char **argv) {
 std::string filename = "TestBlas1.out";
 std::fstream fout(filename);

 int Am = 10, An = 5, Bm = An, Bn = 3, Cm = Am, Cn = Bn;
 int lda = Am, ldb = Bm, ldc = Cm;
 std::vector<double> A(Am*An), B(Bm*Bn), C(Cm*Cn, 0.0);
 double alpha = 1.5, beta = 0.0;
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
 for (auto i : C) {printf("%.4f ", i);}
}