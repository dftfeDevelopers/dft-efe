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
//
// @author Vishal Subramanian
//

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::MixingScheme(
      const utils::mpi::MPIComm &mpiComm)
      : d_mpiComm(mpiComm)
      , d_anyMixingParameterAdaptive(false)
    {}

    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    void
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::addMixingVariable(
      const mixingVariable mixingVariableList,
      const utils::MemoryStorage<ValueTypeWeights, utils::MemorySpace::HOST>
        &          weightDotProducts,
      const bool   performMPIReduce,
      const double mixingValue, /*param for linear mixing (a's)*/
      const bool   adaptMixingValue)
    {
      d_variableHistoryIn[mixingVariableList] =
        std::deque<std::vector<ValueTypeMixingVariable>>();
      d_variableHistoryResidual[mixingVariableList] =
        std::deque<std::vector<ValueTypeMixingVariable>>();
      d_vectorDotProductWeights[mixingVariableList] = weightDotProducts;

      d_performMPIReduce[mixingVariableList]     = performMPIReduce;
      d_mixingParameter[mixingVariableList]      = mixingValue;
      d_adaptMixingParameter[mixingVariableList] = adaptMixingValue;
      d_anyMixingParameterAdaptive =
        adaptMixingValue || d_anyMixingParameterAdaptive;
      d_adaptiveMixingParameterDecLastIteration = false;
      d_adaptiveMixingParameterDecAllIterations = true;
      d_adaptiveMixingParameterIncAllIterations = true;
      size_type weightDotProductsSize           = weightDotProducts.size();

      int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
        utils::mpi::MPIInPlace,
        &weightDotProductsSize,
        1,
        utils::mpi::Types<size_type>::getMPIDatatype(),
        utils::mpi::MPIMax,
        d_mpiComm);

      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      if (weightDotProductsSize > 0)
        {
          d_performMixing[mixingVariableList] = true;
        }
      else
        {
          d_performMixing[mixingVariableList] = false;
        }
    }

    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    void
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::
      computeMixingMatrices(
        const std::deque<std::vector<ValueTypeMixingVariable>> &inHist,
        const std::deque<std::vector<ValueTypeMixingVariable>> &residualHist,
        const utils::MemoryStorage<ValueTypeWeights, utils::MemorySpace::HOST>
          &                     weightDotProducts,
        const bool              isPerformMixing,
        const bool              isMPIAllReduce,
        std::vector<ValueType> &A,
        std::vector<ValueType> &c)
    {
      std::vector<ValueType> Adensity;
      Adensity.resize(A.size());
      std::fill(Adensity.begin(), Adensity.end(), 0.0);

      std::vector<ValueType> cDensity;
      cDensity.resize(c.size());
      std::fill(cDensity.begin(), cDensity.end(), 0.0);

      int       N             = inHist.size() - 1;
      size_type numQuadPoints = 0;
      if (N > 0)
        numQuadPoints = inHist[0].size();

      if (isPerformMixing)
        {
          DFTEFE_AssertWithMsg(
            numQuadPoints == weightDotProducts.size(),
            "DFT-EFE Error: The size of the weight dot products vec "
            "does not match the size of the vectors in history."
            "Please resize the vectors appropriately.");
          for (size_type iQuad = 0; iQuad < numQuadPoints; iQuad++)
            {
              ValueTypeMixingVariable Fn = residualHist[N][iQuad];
              for (int m = 0; m < N; m++)
                {
                  ValueTypeMixingVariable Fnm = residualHist[N - 1 - m][iQuad];
                  for (int k = 0; k < N; k++)
                    {
                      ValueTypeMixingVariable Fnk =
                        residualHist[N - 1 - k][iQuad];
                      Adensity[k * N + m] +=
                        (Fn - Fnm) * (Fn - Fnk) *
                        *(weightDotProducts.data() + iQuad); // (m,k)^th entry
                    }
                  cDensity[m] +=
                    (Fn - Fnm) * (Fn) *
                    *(weightDotProducts.data() + iQuad); // (m)^th entry
                }
            }

          size_type aSize = Adensity.size();
          size_type cSize = cDensity.size();

          std::vector<ValueType> ATotal(aSize), cTotal(cSize);
          std::fill(ATotal.begin(), ATotal.end(), 0.0);
          std::fill(cTotal.begin(), cTotal.end(), 0.0);
          if (isMPIAllReduce)
            {
              int mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                &Adensity[0],
                &ATotal[0],
                aSize,
                utils::mpi::Types<ValueType>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_mpiComm);

              std::pair<bool, std::string> mpiIsSuccessAndMsg =
                utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
              utils::throwException(mpiIsSuccessAndMsg.first,
                                    "MPI Error:" + mpiIsSuccessAndMsg.second);


              mpierr = utils::mpi::MPIAllreduce<utils::MemorySpace::HOST>(
                &cDensity[0],
                &cTotal[0],
                cSize,
                utils::mpi::Types<ValueType>::getMPIDatatype(),
                utils::mpi::MPISum,
                d_mpiComm);

              mpiIsSuccessAndMsg = utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
              utils::throwException(mpiIsSuccessAndMsg.first,
                                    "MPI Error:" + mpiIsSuccessAndMsg.second);
            }
          else
            {
              ATotal = Adensity;
              cTotal = cDensity;
            }
          for (size_type i = 0; i < aSize; i++)
            {
              A[i] += ATotal[i];
            }

          for (size_type i = 0; i < cSize; i++)
            {
              c[i] += cTotal[i];
            }
        }
    }

    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    size_type
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::lengthOfHistory()
    {
      return d_variableHistoryIn[mixingVariable::rho].size();
    }

    // Fucntion to compute the mixing coefficients (b_i,j) based on anderson
    // scheme
    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    void
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::
      computeAndersonMixingCoeff(
        const std::vector<mixingVariable> mixingVariablesList,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>
          &linAlgOpContextHost)
    {
      // initialize data structures
      // assumes rho is a mixing variable
      int N = d_variableHistoryIn[mixingVariable::rho].size() - 1;
      if (N > 0)
        {
          size_type NRHS = 1, lda = N, ldb = N;
          std::vector<linearAlgebra::blasLapack::LapackInt> ipiv(N);
          d_A.resize(lda * N);
          d_c.resize(ldb * NRHS);
          for (int i = 0; i < lda * N; i++)
            d_A[i] = 0.0;
          for (int i = 0; i < ldb * NRHS; i++)
            d_c[i] = 0.0;

          for (const auto &key : mixingVariablesList)
            {
              computeMixingMatrices(d_variableHistoryIn[key],
                                    d_variableHistoryResidual[key],
                                    d_vectorDotProductWeights[key],
                                    d_performMixing[key],
                                    d_performMPIReduce[key],
                                    d_A,
                                    d_c);
            }

          linearAlgebra::blasLapack::gesv<ValueType, utils::MemorySpace::HOST>(
            N, NRHS, &d_A[0], lda, &ipiv[0], &d_c[0], ldb, linAlgOpContextHost);
        }
      d_cFinal = 1.0;
      for (int i = 0; i < N; i++)
        d_cFinal -= d_c[i];
      computeAdaptiveAndersonMixingParameter();
    }


    // Fucntion to compute the mixing parameter (a_i) based on an adaptive
    // anderson scheme, algorithm 1 in [CPC. 292, 108865 (2023)]
    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    void
    MixingScheme<ValueTypeMixingVariable,
                 ValueTypeWeights>::computeAdaptiveAndersonMixingParameter()
    {
      ValueType ci = 1.0;
      if (d_anyMixingParameterAdaptive &&
          d_variableHistoryIn[mixingVariable::rho].size() > 1)
        {
          ValueType bii   = std::abs(d_cFinal);
          double    gbase = 1.0;
          double    gpv   = 0.02;
          double    gi =
            gpv * ((double)d_variableHistoryIn[mixingVariable::rho].size()) +
            gbase;
          ValueType x = std::abs(bii) / gi;
          if (x < 0.5)
            ci = 1.0 / (2.0 + std::log(0.5 / x));
          else if (x <= 2.0)
            ci = x;
          else
            ci = 2.0 + std::log(x / 2.0);
          double pi = 0.0;
          if (ci < 1.0 == d_adaptiveMixingParameterDecLastIteration)
            if (ci < 1.0)
              if (d_adaptiveMixingParameterDecAllIterations)
                pi = 1.0;
              else
                pi = 2.0;
            else if (d_adaptiveMixingParameterIncAllIterations)
              pi = 1.0;
            else
              pi = 2.0;
          else
            pi = 3.0;

          ci                                        = std::pow(ci, 1.0 / pi);
          d_adaptiveMixingParameterDecLastIteration = ci < 1.0;
          d_adaptiveMixingParameterDecAllIterations =
            d_adaptiveMixingParameterDecAllIterations & ci < 1.0;
          d_adaptiveMixingParameterIncAllIterations =
            d_adaptiveMixingParameterIncAllIterations & ci >= 1.0;
        }

      int mpierr = utils::mpi::MPIBcast<utils::MemorySpace::HOST>(
        &ci, 1, utils::mpi::Types<ValueType>::getMPIDatatype(), 0, d_mpiComm);

      std::pair<bool, std::string> mpiIsSuccessAndMsg =
        utils::mpi::MPIErrIsSuccessAndMsg(mpierr);
      utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);

      for (const auto &[key, value] : d_variableHistoryIn)
        if (d_adaptMixingParameter[key])
          {
            d_mixingParameter[key] *= ci;
          }
      if (d_adaptMixingParameter[mixingVariable::rho])
        std::cout << "Adaptive Anderson mixing parameter for Rho: "
                  << d_mixingParameter[mixingVariable::rho] << std::endl;
    }

    // Fucntions to add to the history
    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    template <utils::MemorySpace memorySpace>
    void
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::
      addVariableToInHist(const mixingVariable           mixingVariableName,
                          const ValueTypeMixingVariable *inputVariableToInHist,
                          const size_type                length)
    {
      d_variableHistoryIn[mixingVariableName].push_back(
        std::vector<ValueTypeMixingVariable>(length));

      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>
        memoryTransfer;

      memoryTransfer.copy(length,
                          d_variableHistoryIn[mixingVariableName].back().data(),
                          inputVariableToInHist);

      // std::memcpy(d_variableHistoryIn[mixingVariableName].back().data(),
      //             inputVariableToInHist,
      //             length * sizeof(ValueTypeMixingVariable));
    }

    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    template <utils::MemorySpace memorySpace>
    void
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::
      addVariableToResidualHist(
        const mixingVariable           mixingVariableName,
        const ValueTypeMixingVariable *inputVariableToResidualHist,
        const size_type                length)
    {
      d_variableHistoryResidual[mixingVariableName].push_back(
        std::vector<ValueTypeMixingVariable>(length));

      utils::MemoryTransfer<utils::MemorySpace::HOST, memorySpace>
        memoryTransfer;

      memoryTransfer.copy(
        length,
        d_variableHistoryResidual[mixingVariableName].back().data(),
        inputVariableToResidualHist);

      // std::memcpy(d_variableHistoryResidual[mixingVariableName].back().data(),
      //             inputVariableToResidualHist,
      //             length * sizeof(ValueTypeMixingVariable));
    }

    // Computes the new variable after mixing.
    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    template <utils::MemorySpace memorySpace>
    void
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::mixVariable(
      mixingVariable           mixingVariableName,
      ValueTypeMixingVariable *outputVariable,
      const size_type          lenVar)
    {
      size_type N = d_variableHistoryIn[mixingVariableName].size() - 1;
      // Assumes the variable is present otherwise will lead to a seg fault

      DFTEFE_AssertWithMsg(
        lenVar == d_variableHistoryIn[mixingVariableName][0].size(),
        "DFT-EFE Error: The size of the input variables in history does not match the provided size.");

      std::vector<ValueTypeMixingVariable> outputVariableHost(lenVar, 0.0);

      for (size_type iQuad = 0; iQuad < lenVar; iQuad++)
        {
          ValueType varResidualBar =
            d_cFinal * d_variableHistoryResidual[mixingVariableName][N][iQuad];
          ValueType varInBar =
            d_cFinal * d_variableHistoryIn[mixingVariableName][N][iQuad];

          for (int i = 0; i < N; i++)
            {
              varResidualBar +=
                d_c[i] *
                d_variableHistoryResidual[mixingVariableName][N - 1 - i][iQuad];
              varInBar +=
                d_c[i] *
                d_variableHistoryIn[mixingVariableName][N - 1 - i][iQuad];
            }

          // compute the next guess
          //
          outputVariableHost[iQuad] = (ValueTypeMixingVariable)(
            varInBar + d_mixingParameter[mixingVariableName] * varResidualBar);
        }

      utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>
        memoryTransfer;
      memoryTransfer.copy(lenVar, outputVariable, outputVariableHost.data());
    }

    // Clears the history
    // But it does not clear the list of variables
    // and its corresponding JxW values
    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    void
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::clearHistory()
    {
      for (const auto &[key, value] : d_variableHistoryIn)
        {
          d_variableHistoryIn[key].clear();
          d_variableHistoryResidual[key].clear();
        }
    }


    // Deletes old history.
    // This is not recursively
    // If the length is greater or equal to mixingHistory then the
    // oldest history is deleted
    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    void
    MixingScheme<ValueTypeMixingVariable, ValueTypeWeights>::popOldHistory(
      size_type mixingHistory)
    {
      if (d_variableHistoryIn[mixingVariable::rho].size() >= mixingHistory)
        {
          for (const auto &[key, value] : d_variableHistoryIn)
            {
              d_variableHistoryIn[key].pop_front();
              d_variableHistoryResidual[key].pop_front();
            }
        }
    }
  } // namespace ksdft
} // namespace dftefe
