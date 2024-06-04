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
 * @adapted from
 * "https://github.com/dftfeDevelopers/dftfe/blob/publicGithubDevelop/include/MixingScheme.h"
 */

#ifndef dftefeMixingScheme_h
#define dftefeMixingScheme_h

#include <deque>
#include <utils/MemorySpaceType.h>
#include <linearAlgebra/MultiVector.h>
#include <linearAlgebra/BlasLapack.h>

namespace dftefe
{
  namespace ksdft
  {
    /**
     * @brief Enum class that stores he list of variables that will can be
     * used in the mixing scheme
     *
     */
    enum class mixingVariable
    {
      rho
    };

    /**
     * @brief This class performs the anderson mixing in a variable agnostic way
     * This class takes can take different input variables as input in a
     * std::vector format and computes the mixing coefficients These
     * coefficients can then be used to compute the new variable at the start of
     * the SCF.
     * @author Vishal Subramanian
     */
    template <typename ValueTypeMixingVariable, typename ValueTypeWeights>
    class MixingScheme
    {
    public:
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeWeights,
                                               ValueTypeMixingVariable>;

    public:
      MixingScheme(const utils::mpi::MPIComm &mpiComm);

      size_type
      lengthOfHistory();

      /**
       * @brief Computes the mixing coefficients.
       *
       */
      void
      computeAndersonMixingCoeff(
        const std::vector<mixingVariable> mixingVariablesList,
        linearAlgebra::LinAlgOpContext<utils::MemorySpace::HOST>
          &linAlgOpContextHost);

      /**
       * @brief Computes the adaptive mixing parameter.
       *
       */
      void
      computeAdaptiveAndersonMixingParameter();

      /**
       * @brief Deletes the old history if the length exceeds max length of history
       *
       */
      void
      popOldHistory(size_type mixingHistory);

      /**
       * @brief Clears all the the history.
       *
       */
      void
      clearHistory();

      /**
       * @brief This function is used to add the mixing variables and its corresponding
       * JxW values
       * For dependent variables which are not used in mixing, the
       * weightDotProducts is set to a vector of size zero. Later the dependent
       * variables can be mixed with the same mixing coefficients.
       *
       */
      void
      addMixingVariable(
        const mixingVariable mixingVariableList,
        const utils::MemoryStorage<ValueTypeWeights, utils::MemorySpace::HOST>
          &          weightDotProducts,
        const bool   performMPIReduce,
        const double mixingValue,
        const bool   adaptMixingValue);

      /**
       * @brief Adds to the input history
       *
       */
      template <utils::MemorySpace memorySpace>
      void
      addVariableToInHist(const mixingVariable           mixingVariableName,
                          const ValueTypeMixingVariable *inputVariableToInHist,
                          const size_type                length);

      /**
       * @brief Adds to the residual history
       *
       */
      template <utils::MemorySpace memorySpace>
      void
      addVariableToResidualHist(
        const mixingVariable           mixingVariableName,
        const ValueTypeMixingVariable *inputVariableToResidualHist,
        const size_type                length);

      /**
       * @brief Computes the input for the next iteration based on the anderson coefficients
       *
       */
      template <utils::MemorySpace memorySpace>
      void
      mixVariable(const mixingVariable     mixingVariableName,
                  ValueTypeMixingVariable *outputVariable,
                  const size_type          lenVar);


    private:
      /**
       * @brief Computes the matrix A and c vector that will be needed for anderson mixing.
       * This function computes the A and c values for each variable which will
       * be then added up in computeAndersonMixingCoeff()
       */
      void
      computeMixingMatrices(
        const std::deque<std::vector<ValueTypeMixingVariable>> &inHist,
        const std::deque<std::vector<ValueTypeMixingVariable>> &outHist,
        const utils::MemoryStorage<ValueTypeWeights, utils::MemorySpace::HOST>
          &                     weightDotProducts,
        const bool              isPerformMixing,
        const bool              isMPIAllReduce,
        std::vector<ValueType> &A,
        std::vector<ValueType> &c);

      std::vector<ValueType> d_A, d_c;
      ValueType              d_cFinal;

      std::map<mixingVariable, std::deque<std::vector<ValueTypeMixingVariable>>>
        d_variableHistoryIn, d_variableHistoryResidual;
      std::map<mixingVariable,
               utils::MemoryStorage<ValueTypeWeights, utils::MemorySpace::HOST>>
                                     d_vectorDotProductWeights;
      std::map<mixingVariable, bool> d_performMPIReduce;

      const MPI_Comm &d_mpiComm;

      std::map<mixingVariable, double> d_mixingParameter;
      std::map<mixingVariable, bool>   d_adaptMixingParameter;
      bool                             d_anyMixingParameterAdaptive;
      bool                           d_adaptiveMixingParameterDecLastIteration;
      bool                           d_adaptiveMixingParameterDecAllIterations;
      bool                           d_adaptiveMixingParameterIncAllIterations;
      size_type                      d_mixingHistory;
      std::map<mixingVariable, bool> d_performMixing;
    };
  } // namespace ksdft
} //  end of namespace dftefe
#include <ksdft/MixingScheme.t.cpp>
#endif // dftefeMixingScheme_h
