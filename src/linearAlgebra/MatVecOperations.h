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

#ifndef dftefeMatVecOperations_h
#define dftefeMatVecOperations_h

#include <memory>
#include <blas.hh>
#include <linearAlgebra/BlasLapackTypedef.h>
#include <linearAlgebra/BlasLapack.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/Matrix.h>
#include <utils/MemorySpaceType.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    namespace MatVecOperations
    {
      /**
       * @brief Compute \f$ {\bf S}= {\bf A}^{\dagger} {\bf A} \f$ where
       * \f$ {\bf A} \f$ is a dense matrix of size M times N with distributed
       * memory parallelization over the rows. \f$ {\bf A} \f$ is stored
       * as a MultiVector with number of vectors to be N and row major
       * storage. \f$ {\bf S} \f$ is the N times N Hermitian overlap matrix
       * stored in block-cyclic format. The parallelization of \f$ {\bf S} \f$
       * is independent of that of \f$ {\bf A} \f$
       * @param[in] A the dense matrix stored as a MultiVector
       * @param[in,out] S preallocated Hermitian overlap matrix
       * @param[in] context LinAlg context
       * @param[in] NBlockSize determines the block size for
       * blocked loop over the N vectors during the computation
       * of \f$ {\bf S} \f$, for default value of 0 it is heuristically
       * determined. This aspect to required for reducing the local MPI task
       * peak memory scaling to \f$ \approx \f$ N times vectorsBlockSize
       * instead of N times N
       */
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      computeAMultiVecConjTransTimesAMultiVec(
        const MultiVector<ValueType, memorySpace> &A,
        HermitianMatrix<ValueType, memorySpace> &  S,
        LinAlgOpContext<memorySpace> &             context,
        const size_type                            NBlockSize = 0);

      /**
       * @brief In-place computation of \f$ {\bf A}= {\bf A} {\bf T} \f$ where
       * \f$ {\bf A} \f$ is a dense matrix of size M times N with distributed
       * memory parallelization over the rows. \f$ {\bf A} \f$ is stored
       * as a MultiVector with number of vectors to be N and row major
       * storage. \f$ {\bf T} \f$ is either an upper or lower triangular
       * N times N matrix stored in block cyclic format. The parallelization
       * of \f$ {\bf T} \f$ is independent of that of \f$ {\bf A} \f$
       * @param[in,out] A the dense matrix stored as a MultiVector
       * @param[in] T preallocated triangular matrix
       * @param[in] context LinAlg context
       * @param[in] MBlockSize determines the block size for
       * blocked loop over the M vector size during the in-place computation.
       * Default value of 0 sets block size heuristically.
       * @param[in] NBlockSize determines the block size for
       * blocked loop over the N vectors. Default value of 0 sets
       * block size heuristically.
       */
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      computeAMultiVecTimesTriangularMat(
        MultiVector<ValueType, memorySpace> &           A,
        const TriangularMatrix<ValueType, memorySpace> &T,
        LinAlgOpContext<memorySpace> &                  context,
        const size_type                                 MBlockSize = 0,
        const size_type                                 NBlockSize = 0);


      /**
       * @brief In-place computation of \f$ {\bf A}= {\bf A} {\bf G} \f$ where
       * \f$ {\bf A} \f$ is a dense matrix of size M times N with distributed
       * memory parallelization over the rows. \f$ {\bf A} \f$ is stored
       * as a MultiVector with number of vectors to be N and row major
       * storage. \f$ {\bf G} \f$ is a N times N general matrix
       * stored in block cyclic format. The parallelization
       * of \f$ {\bf G} \f$ is independent of that of \f$ {\bf A} \f$
       * @param[in,out] A the dense matrix stored as a MultiVector
       * @param[in] G preallocated general matrix
       * @param[in] context LinAlg context
       * @param[in] MBlockSize determines the block size for
       * blocked loop over the M vector size during the in-place computation
       * Default value of 0 sets block size heuristically.
       * @param[in] NBlockSize determines the block size for
       * blocked loop over the N vectors. Default value of 0 sets block size
       * heuristically.
       */
      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      computeAMultiVecTimesGeneralMat(
        MultiVector<ValueType, memorySpace> &        A,
        const GeneralMatrix<ValueType, memorySpace> &G,
        LinAlgOpContext<memorySpace> &               context,
        const size_type                              MBlockSize = 0,
        const size_type                              NBlockSize = 0);


      template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
      void
      choleskyOrthogonalization(MultiVector<ValueType, memorySpace> &A,
                                LinAlgOpContext<memorySpace> &       context);

    } // namespace MatVecOperations
  }   // namespace linearAlgebra

} // namespace dftefe

#include "MatVecOperations.t.cpp"
#endif // dftefeMatVecOperations_h
