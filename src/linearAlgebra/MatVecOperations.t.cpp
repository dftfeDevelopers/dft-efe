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
 * @author Sambit Das
 */

#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatVecOperations::computeOverlapMatrixAConjugateTransposeA(
      const MultiVector<ValueType, memorySpace> &A,
      HermitianMatrix<ValueType, memorySpace> &  S,
      LinAlgOpContext<memorySpace> &             context,
      const size_type                            vectorsBlockSize)
    {
      const size_type N = A.numVectors();

      const size_type maxEntriesInBlock =
        40000 * 400; // about 250 MB in FP64 datatype
      const size_type vectorsBlockSizeUsed =
        vectorsBlockSize == 0 ? (maxEntriesInBlock / N) : vectorsBlockSize;

      dftefe::utils::MemoryStorage<ValueType, memorySpace> overlapMatrixBlock(
        N * vectorsBlockSizeUsed, 0);

      for (size_type ivec = 0; ivec < N; ivec += vectorsBlockSizeUsed)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const size_type B = std::min(vectorsBlockSizeUsed, N - ivec);

          const blasLapack::Op
            transA = blasLapack::Op::NoTrans,
            transB = std::is_same<ValueType, std::complex<double>>::value ?
                       blasLapack::Op::ConjTrans :
                       blasLapack::Op::Trans;
          const ValueType scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;

          // member function to be created
          overlapMatrixBlock = 0.;

          const unsigned int D = N - ivec;

          // Comptute local XTrunc^{T}*XcBlock.
          blasLapack::gemm<ValueType, memorySpace>(
            blasLapack::Layout::ColMajor,
            transA,
            transB,
            D,
            B,
            A.getMPIPatternP2P()->localOwnedSize(),
            scalarCoeffAlpha,
            A.data() + ivec,
            N,
            A.data() + ivec,
            N,
            scalarCoeffBeta,
            overlapMatrixBlock.data(),
            D,
            context);


          // Sum local XTrunc^{T}*XcBlock across MPI tasks
          utils::mpi::MPIAllreduce<memorySpace>(
            MPI_IN_PLACE,
            overlapMatrixBlock.data(),
            D * B * sizeof(ValueType),
            utils::mpi::MPIByte,
            utils::mpi::MPISum,
            A.getMPIPatternP2P()->mpiCommunicator());


          // Copy only the lower triangular part to the SLATE
          // overlap matrix
        }
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatVecOperations::choleskyOrthogonalization(
      const MultiVector<ValueType, memorySpace> &A,
      MultiVector<ValueType, memorySpace> &      B,
      LinAlgOpContext<memorySpace> &             context)
    {}
  } // namespace linearAlgebra
} // namespace dftefe
