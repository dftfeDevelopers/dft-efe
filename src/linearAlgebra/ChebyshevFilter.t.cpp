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

#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <string>
#include <chrono>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    ChebyshevFilter(
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &A,
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &                                         BInv,
      MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess,
      const size_type                             polynomialDegree,
      const double                                wantedSpectrumLowerBound,
      const double                                wantedSpectrumUpperBound,
      const double                                unWantedSpectrumUpperBound,
      MultiVector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                  memorySpace> &                  filteredSubspace)
    {
      /* taken from "Lin CC, Gavini V. TTDFT: A GPU accelerated Tucker
       *  tensor DFT code for large-scale Kohn-Sham DFT calculations. Computer
       * Physics Communications. 2023 Jan 1;282:108516.*/

      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;

      const size_type locallyOwnedMultivecSize =
        eigenSubspaceGuess.locallyOwnedSize() *
        eigenSubspaceGuess.getNumberComponents();

      const double e =
        (0.5) * (unWantedSpectrumUpperBound - wantedSpectrumUpperBound);
      const double c =
        (0.5) * (unWantedSpectrumUpperBound + wantedSpectrumUpperBound);
      double sigma = e / (wantedSpectrumLowerBound - c);

      const double sigma1 = sigma;
      const double gamma  = (2.0 / sigma1);
      double       sigma2;

      // MultiVector<ValueType, memorySpace> filteredSubspaceNew(
      //   eigenSubspaceGuess, (ValueType)0);
      MultiVector<ValueType, memorySpace> scratch1(eigenSubspaceGuess,
                                                   (ValueType)0);
      MultiVector<ValueType, memorySpace> scratch2(eigenSubspaceGuess,
                                                   (ValueType)0);

      // Compute B^-1AX
      A.apply(eigenSubspaceGuess, scratch1);
      BInv.apply(scratch1, scratch2);

      // filteredSubspace = (\sigma1/e)(B^-1A eigenSubspaceGuess - c
      // eigenSubspaceGuess)
      blasLapack::axpby<ValueType, ValueTypeOperand, memorySpace>(
        locallyOwnedMultivecSize,
        sigma1 / e,
        scratch2.data(),
        -sigma1 / e * c,
        eigenSubspaceGuess.data(),
        filteredSubspace.data(),
        *eigenSubspaceGuess.getLinAlgOpContext());

      for (size_type i = 2; i <= polynomialDegree; i++)
        {
          sigma2 = 1.0 / (gamma - sigma);

          // Compute B^-1A filteredSubspace
          A.apply(filteredSubspace, scratch1);
          BInv.apply(scratch1, scratch2);

          // temp = (2\sigma2/e)(B^-1A filteredSubspace - c filteredSubspace)
          blasLapack::axpby<ValueType, ValueType, memorySpace>(
            locallyOwnedMultivecSize,
            2.0 * sigma2 / e,
            scratch2.data(),
            -2.0 * sigma2 / e * c,
            filteredSubspace.data(),
            scratch1.data(),
            *eigenSubspaceGuess.getLinAlgOpContext());

          // filteredSubspaceNew = temp - \sigma*\sigma2*eigenSubspaceGuess
          // Note: works if axpby is capable of z being same as either of x or y
          blasLapack::axpby<ValueType, ValueTypeOperand, memorySpace>(
            locallyOwnedMultivecSize,
            (ValueType)1.0,
            scratch1.data(),
            -sigma * sigma2,
            eigenSubspaceGuess.data(),
            eigenSubspaceGuess /*filteredSubspaceNew*/.data(),
            *eigenSubspaceGuess.getLinAlgOpContext());

          // eigenSubspaceGuess = filteredSubspace;

          // filteredSubspace = filteredSubspaceNew;

          swap(eigenSubspaceGuess, filteredSubspace);

          sigma = sigma2;
        }

      eigenSubspaceGuess = filteredSubspace;
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
