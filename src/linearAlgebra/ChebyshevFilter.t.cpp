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
                  memorySpace>
        &filteredSubspace) // remove this and put X (in/out)
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
      A.apply(eigenSubspaceGuess, scratch1, true, false);
      BInv.apply(scratch1, scratch2, false, false);

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

      for (size_type degree = 2; degree <= polynomialDegree; degree++)
        {
          sigma2 = 1.0 / (gamma - sigma);

          // Compute B^-1A filteredSubspace
          A.apply(filteredSubspace, scratch1, true, false);
          BInv.apply(scratch1, scratch2, false, false);

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

      eigenSubspaceGuess = filteredSubspace; // remove this and put X (in/out)
    }

    // filtering for AX = \lambda BX
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    ChebyshevFilterGEP(
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &A,
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &B,
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &                                         BInv,
      MultiVector<ValueTypeOperand, memorySpace> &eigenSubspaceGuess,
      const size_type                             polynomialDegree,
      const double                                wantedSpectrumLowerBound,
      const double                                wantedSpectrumUpperBound,
      const double                                unWantedSpectrumUpperBound,
      MultiVector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                  memorySpace>
        &filteredSubspace) // remove this and put X (in/out)
    {
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

      MultiVector<ValueType, memorySpace> scratch1(eigenSubspaceGuess,
                                                   (ValueType)0);
      MultiVector<ValueType, memorySpace> scratch2(eigenSubspaceGuess,
                                                   (ValueType)0);

      B.apply(eigenSubspaceGuess, filteredSubspace, true, false);

      BInv.apply(filteredSubspace, scratch1, true, false);
      A.apply(scratch1, scratch2, true, false);

      // eigenSubspaceGuess = (\sigma1/e)(AB^-1 filteredSubspace - c
      // filteredSubspace)
      blasLapack::axpby<ValueType, ValueTypeOperand, memorySpace>(
        locallyOwnedMultivecSize,
        sigma1 / e,
        scratch2.data(),
        -sigma1 / e * c,
        filteredSubspace.data(),
        eigenSubspaceGuess.data(),
        *eigenSubspaceGuess.getLinAlgOpContext());

      swap(eigenSubspaceGuess, filteredSubspace);

      for (size_type degree = 2; degree <= polynomialDegree; degree++)
        {
          sigma2 = 1.0 / (gamma - sigma);

          BInv.apply(filteredSubspace, scratch1, true, false);
          A.apply(scratch1, scratch2, true, false);

          // temp = (2\sigma2/e)(AB^-1 filteredSubspace - c filteredSubspace)
          blasLapack::axpby<ValueType, ValueType, memorySpace>(
            locallyOwnedMultivecSize,
            2.0 * sigma2 / e,
            scratch2.data(),
            -2.0 * sigma2 / e * c,
            filteredSubspace.data(),
            scratch1.data(),
            *eigenSubspaceGuess.getLinAlgOpContext());

          // Note: works if axpby is capable of z being same as either of x or y
          blasLapack::axpby<ValueType, ValueTypeOperand, memorySpace>(
            locallyOwnedMultivecSize,
            (ValueType)1.0,
            scratch1.data(),
            -sigma * sigma2,
            eigenSubspaceGuess.data(),
            eigenSubspaceGuess.data(),
            *eigenSubspaceGuess.getLinAlgOpContext());

          swap(eigenSubspaceGuess, filteredSubspace);

          sigma = sigma2;
        }
      BInv.apply(filteredSubspace, eigenSubspaceGuess, true, true);

      filteredSubspace = eigenSubspaceGuess; // remove this and put X (in/out)
    }


    // filtering for AX = \lambda BX
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace>
    void
    ResidualChebyshevFilterGEP(
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &A,
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &B,
      const OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
        &BInv,
      std::vector<blasLapack::real_type<
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>>>
        &                                         eigenvalues,
      MultiVector<ValueTypeOperand, memorySpace> &X,
      const size_type                             polynomialDegree,
      const double                                wantedSpectrumLowerBound,
      const double                                wantedSpectrumUpperBound,
      const double                                unWantedSpectrumUpperBound,
      MultiVector<blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>,
                  memorySpace> &                  Y)
    {
      using ValueType =
        blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
      using RealType = blasLapack::real_type<ValueType>;
      LinAlgOpContext<memorySpace> &linAlgOpContext = *X.getLinAlgOpContext();

      const double e =
        (0.5) * (unWantedSpectrumUpperBound - wantedSpectrumUpperBound);
      const double c =
        (0.5) * (unWantedSpectrumUpperBound + wantedSpectrumUpperBound);
      double sigma = e / (wantedSpectrumLowerBound - c);

      const double sigma1 = sigma;
      const double gamma  = (2.0 / sigma1);
      double       sigma2;

      //============scratch spaces================
      MultiVector<ValueType, memorySpace> scratch1(X, (ValueType)0);
      MultiVector<ValueType, memorySpace> scratch2(X, (ValueType)0);
      MultiVector<ValueType, memorySpace> scratch3(X, (ValueType)0);

      MultiVector<ValueType, memorySpace> Residual(X, (ValueType)0);
      MultiVector<ValueType, memorySpace> ResidualNew(X, (ValueType)0);
      //============scratch spaces================

      utils::MemoryStorage<RealType, memorySpace> eigenValuesFiltered(
        eigenvalues.size()),
        eigenValuesFiltered1, eigenValuesFiltered2;
      const utils::MemoryStorage<RealType, memorySpace> ones(eigenvalues.size(),
                                                             (RealType)1.0);

      eigenValuesFiltered.copyFrom(eigenvalues);
      eigenValuesFiltered1 = eigenValuesFiltered;
      eigenValuesFiltered2 = eigenValuesFiltered;
      eigenValuesFiltered1.setValue(1.0);

      double alpha1 = sigma1 / e, alpha2 = -c;

      B.apply(X, Y, true, false);
      A.apply(X, scratch3, true, false);
      linearAlgebra::blasLapack::
        axpbyBlocked<ValueType, ValueType, memorySpace>(
          X.locallyOwnedSize(),
          X.getNumberComponents(),
          1,
          ones.data(),
          scratch3.data(),
          -1,
          eigenValuesFiltered.data(),
          Y.data(),
          Y.data(),
          linAlgOpContext); // Y = AX - \lambda BX

      ResidualNew = Y;

      Residual.setValue(0.0);

      // //m=1 operations
      //========= finding lambda1  ================
      eigenValuesFiltered2.setValue(alpha1 * alpha2);

      // eigenValuesFiltered2 = eigenValuesFiltered2 + alpha1 *
      // eigenValuesFiltered1 * eigenValuesFiltered
      linearAlgebra::blasLapack::
        axpbyBlocked<ValueType, ValueType, memorySpace>(
          1,
          eigenValuesFiltered2.size(),
          1,
          ones.data(),
          eigenValuesFiltered2.data(),
          alpha1,
          eigenValuesFiltered.data(),
          eigenValuesFiltered1.data(),
          eigenValuesFiltered2.data(),
          linAlgOpContext);
      //========= finding lambda1  ================

      //===== Note: for the first filtered eigenspace X_1 = alpha1 * A * X
      // assumed ======
      //====== Implies: BR_1 = \alpha_1 *( AX - \lambda B X) = \alpha_1 *
      // ResidualNew ========
      blasLapack::ascale(X.locallyOwnedSize() * X.getNumberComponents(),
                         alpha1,
                         ResidualNew.data(),
                         ResidualNew.data(),
                         linAlgOpContext); // ResidualNew *= alpha1

      // //
      // // polynomial loop
      // //
      for (unsigned int degree = 2; degree <= polynomialDegree; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          //======Residual = alpha1 * H * M^-1 * ResidualNew + alpha2 * Residual
          //- c *
          // alpha1 * ResidualNew======
          BInv.apply(ResidualNew, scratch1, true, false);
          A.apply(scratch1, scratch2, false, false);

          blasLapack::axpby<ValueType, ValueType, memorySpace>(
            X.locallyOwnedSize() * X.getNumberComponents(),
            alpha1,
            scratch2.data(),
            -c * alpha1,
            ResidualNew.data(),
            scratch1.data(),
            linAlgOpContext);

          blasLapack::axpby<ValueType, ValueTypeOperand, memorySpace>(
            X.locallyOwnedSize() * X.getNumberComponents(),
            (ValueType)1.0,
            scratch1.data(),
            alpha2,
            Residual.data(),
            Residual.data(),
            linAlgOpContext);
          //======Residual = alpha1 * H * M^-1 * ResidualNew + alpha2 * Residual
          //- c *
          // alpha1 * ResidualNew======

          // Residual = Residual + alpha1 * Y * eigenValuesFiltered2
          linearAlgebra::blasLapack::
            axpbyBlocked<ValueType, ValueType, memorySpace>(
              X.locallyOwnedSize(),
              X.getNumberComponents(),
              1,
              ones.data(),
              Residual.data(),
              alpha1,
              eigenValuesFiltered2.data(),
              Y.data(),
              Residual.data(),
              linAlgOpContext);

          //=======Filtering of eigenValues===============

          // eigenValuesFiltered1 = -c * alpha1 * eigenValuesFiltered2 + alpha2
          // * eigenValuesFiltered1
          linearAlgebra::blasLapack::axpby(eigenValuesFiltered2.size(),
                                           -c * alpha1,
                                           eigenValuesFiltered2.data(),
                                           alpha2,
                                           eigenValuesFiltered1.data(),
                                           eigenValuesFiltered1.data(),
                                           linAlgOpContext);

          // eigenValuesFiltered1 = eigenValuesFiltered1 + alpha1 *
          // eigenValuesFiltered2 * eigenValuesFiltered
          linearAlgebra::blasLapack::
            axpbyBlocked<ValueType, ValueType, memorySpace>(
              1,
              eigenValuesFiltered1.size(),
              1,
              ones.data(),
              eigenValuesFiltered1.data(),
              alpha1,
              eigenValuesFiltered.data(),
              eigenValuesFiltered2.data(),
              eigenValuesFiltered1.data(),
              linAlgOpContext);

          //=======Filtering of eigenValues===============

          swap(ResidualNew, Residual);
          utils::swap(eigenValuesFiltered1, eigenValuesFiltered2);

          sigma = sigma2;
        }

      BInv.apply(ResidualNew, Residual, true, true);

      linearAlgebra::blasLapack::
        axpbyBlocked<ValueType, ValueType, memorySpace>(
          X.locallyOwnedSize(),
          X.numVectors(),
          1,
          ones.data(),
          Residual.data(),
          1,
          eigenValuesFiltered2.data(),
          X.data(),
          Y.data(), // remove this and put X (in/out)
          linAlgOpContext);
    }

  } // end of namespace linearAlgebra
} // end of namespace dftefe
