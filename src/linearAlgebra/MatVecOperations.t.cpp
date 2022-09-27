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
    MatVecOperations::computeAMultiVecConjTransTimesAMultiVec(
      const MultiVector<ValueType, memorySpace> &A,
      HermitianMatrix<ValueType, memorySpace> &  S,
      LinAlgOpContext<memorySpace> &             context,
      const size_type                            NBlockSize)
    {
      const size_type N = A.numVectors();

      const size_type maxEntriesInBlock =
        40000 * 400; // about 250 MB in FP64 datatype
      const size_type NBlockSizeUsed =
        NBlockSize == 0 ? (maxEntriesInBlock / N) : NBlockSize;

      dftefe::utils::MemoryStorage<ValueType, memorySpace> overlapMatrixBlock(
        N * NBlockSizeUsed, 0);

      for (size_type ivec = 0; ivec < N; ivec += NBlockSizeUsed)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const size_type B = std::min(NBlockSizeUsed, N - ivec);

          const blasLapack::Op
            transA = blasLapack::Op::NoTrans,
            transB = std::is_same<ValueType, std::complex<double>>::value ?
                       blasLapack::Op::ConjTrans :
                       blasLapack::Op::Trans;
          const ValueType scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;

          // member function to be created
          overlapMatrixBlock = 0.;

          const unsigned int D = N - ivec;

          // Block1: (0,M-1,ivec-1,N-1)
          // Block2: (0,M-1,ivec-1,ivec+B-1)
          // Comptute local ABlock1^{Trans}*ABlock2Conj.
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


          // Sum local ABlock1^{Trans}*ABlock2Conj across MPI tasks
          utils::mpi::MPIAllreduce<memorySpace>(
            MPI_IN_PLACE,
            overlapMatrixBlock.data(),
            D * B * sizeof(ValueType),
            utils::mpi::MPIByte,
            utils::mpi::MPISum,
            A.getMPIPatternP2P()->mpiCommunicator());


          // Copy only the lower/upper triangular part to the Hermitian
          // overlap matrix, the matrix object already assumed to be created
          // outside this function
          // FIXME:The setValues function should work for the case where
          // the Hermitian matrix is actually storing upper triangular values
          // and the setValues function tries to set for the lower triangular
          // part
          S.setValues(
            ivec - 1, N - 1, ivec - 1, ivec + B - 1, overlapMatrixBlock.data());
        }

      // FIXME: to be implemented (not complex conjugate transpose)
      // S.complexConjugate();
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    computeAMultiVecTimesTriangularMat(
      MultiVector<ValueType, memorySpace> &           A,
      const TriangularMatrix<ValueType, memorySpace> &T,
      LinAlgOpContext<memorySpace> &                  context,
      const size_type                                 MBlockSize = 0,
      const size_type                                 NBlockSize = 0)
    {}


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    computeAMultiVecTimesGeneralMat(
      MultiVector<ValueType, memorySpace> &        A,
      const GeneralMatrix<ValueType, memorySpace> &T,
      LinAlgOpContext<memorySpace> &               context,
      const size_type                              MBlockSize = 0,
      const size_type                              NBlockSize = 0)
    {}


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatVecOperations::choleskyOrthogonalization(
      MultiVector<ValueType, memorySpace> &A,
      LinAlgOpContext<memorySpace> &       context)
    {}
  } // namespace linearAlgebra
} // namespace dftefe
