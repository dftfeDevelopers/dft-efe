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
 * @author DFTFE, Avirup Sircar
 */

#ifndef FEEvaluationWrapper_h
#define FEEvaluationWrapper_h

#include <boost/preprocessor.hpp>
#include <cmath>
#include <memory>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

namespace dftefe
{
  namespace basis
  {
    class FEEvaluationWrapperBase
    {
    public:
      template <typename T>
      using distributedCPUVec =
        dealii::LinearAlgebra::distributed::Vector<T,
                                                   dealii::MemorySpace::Host>;

      /**
       * @brief Returns the total number of quadrature points in all 3 directions
       */
      virtual unsigned int
      totalNumberofQuadraturePoints() = 0;

      /**
       * @brief reinits the dealii::FEEvaluation object for the macrocellIndex
       */
      virtual void
      reinit(const unsigned int macrocell) = 0;

      /**
       * @brief Calls dealii::FEEvaluation::read_dof_values
       */
      virtual void
      readDoFValues(const distributedCPUVec<double> &tempvec) = 0;

      /**
       * @brief Calls dealii::FEEvaluation::read_dofs_values_plain
       */
      virtual void
      readDoFValuesPlain(const distributedCPUVec<double> &tempvec) = 0;

      /**
       * @brief Calls the dealii::FEEvaluation::evaluate
       */
      virtual void
      evaluate(dealii::EvaluationFlags::EvaluationFlags evaluateFlags) = 0;

      virtual void
      submitInterpolatedGradientsAndMultiply(
        dealii::VectorizedArray<double> &alpha) = 0;

      // virtual void submitInterpolatedGradientsAndMultiply(
      //   dealii::AlignedVector<
      //     dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &alpha) = 0;

      virtual void
      submitInterpolatedValuesAndMultiply(
        dealii::VectorizedArray<double> &alpha) = 0;

      virtual void
      submitInterpolatedValuesAndMultiplySquared() = 0;

      virtual void
      submitInterpolatedValuesAndMultiply(
        dealii::AlignedVector<dealii::VectorizedArray<double>> &alpha) = 0;

      virtual void
      submitValues(
        const dealii::VectorizedArray<double> &scaling,
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &alpha) = 0;

      virtual void
      submitGradients(
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &alpha) = 0;

      virtual void
      submitValues(
        dealii::AlignedVector<dealii::VectorizedArray<double>> &alpha) = 0;


      virtual dealii::VectorizedArray<double>
      integrateValue() = 0;

      virtual void
      submitValueAtQuadpoint(const unsigned int                     iQuadPoint,
                             const dealii::VectorizedArray<double> &value) = 0;

      virtual void
      alphaTimesQuadValuesPlusYFromSubCell(const unsigned int subCellIndex,
                                           const double       alpha,
                                           double *           outputVector) = 0;

      virtual void
      getQuadGradientsForSubCell(const unsigned int subCellIndex,
                                 const double       alpha,
                                 double *           outputVector) = 0;

      virtual void
      getQuadHessianForSubCell(const unsigned int subCellIndex,
                               const double       alpha,
                               double *           outputVector) = 0;


      virtual void
      submitInterpolatedValuesSubmitInterpolatedGradients(
        const dealii::VectorizedArray<double> &scaleValues,
        const bool                             scaleValuesFlag,
        const dealii::VectorizedArray<double> &scaleGradients,
        const bool                             scaleGradientsFlag) = 0;

      virtual void
      integrate(dealii::EvaluationFlags::EvaluationFlags evaluateFlags) = 0;

      virtual dealii::Point<3, dealii::VectorizedArray<double>>
      getQuadraturePoint(const unsigned int iQuadPoint) = 0;

      virtual void
      getValues(
        dealii::AlignedVector<dealii::VectorizedArray<double>> &tempVec) = 0;

      virtual void
      distributeLocalToGlobal(distributedCPUVec<double> &tempvec) = 0;
    };

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    class FEEvaluationWrapperDerived : public FEEvaluationWrapperBase
    {
    public:
      FEEvaluationWrapperDerived(
        const dealii::MatrixFree<3, double> &matrixFreeData,
        const unsigned int                   matrixFreeVectorComponent,
        const unsigned int                   matrixFreeQuadratureComponent);


      std::unique_ptr<
        dealii::FEEvaluation<3, FEOrder, num_1d_quadPoints, n_components>>
        d_dealiiFEEvaluation;


      unsigned int
      totalNumberofQuadraturePoints() override;

      void
      reinit(const unsigned int macrocell) override;

      void
      readDoFValues(const distributedCPUVec<double> &tempvec) override;

      void
      readDoFValuesPlain(const distributedCPUVec<double> &tempvec) override;

      void
      evaluate(dealii::EvaluationFlags::EvaluationFlags evaluateFlags) override;

      void
      submitInterpolatedGradientsAndMultiply(
        dealii::VectorizedArray<double> &alpha) override;

      // void submitInterpolatedGradientsAndMultiply(
      //   dealii::AlignedVector<
      //     dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &alpha)
      //     override;
      void
      submitInterpolatedValuesAndMultiply(
        dealii::VectorizedArray<double> &alpha) override;

      void
      submitInterpolatedValuesAndMultiplySquared() override;

      void
      submitInterpolatedValuesAndMultiply(
        dealii::AlignedVector<dealii::VectorizedArray<double>> &alpha) override;

      void
      submitValues(const dealii::VectorizedArray<double> &scaling,
                   dealii::AlignedVector<
                     dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
                     &alpha) override;

      void
      submitGradients(dealii::AlignedVector<
                      dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
                        &alpha) override;

      void
      submitValues(
        dealii::AlignedVector<dealii::VectorizedArray<double>> &alpha) override;

      dealii::VectorizedArray<double>
      integrateValue() override;

      void
      submitValueAtQuadpoint(
        const unsigned int                     iQuadPoint,
        const dealii::VectorizedArray<double> &value) override;

      dealii::Point<3, dealii::VectorizedArray<double>>
      getQuadraturePoint(const unsigned int iQuadPoint) override;

      void
      alphaTimesQuadValuesPlusYFromSubCell(const unsigned int subCellIndex,
                                           const double       alpha,
                                           double *outputVector) override;

      void
      getQuadGradientsForSubCell(const unsigned int subCellIndex,
                                 const double       alpha,
                                 double *           outputVector) override;

      void
      getQuadHessianForSubCell(const unsigned int subCellIndex,
                               const double       alpha,
                               double *           outputVector) override;

      void
      submitInterpolatedValuesSubmitInterpolatedGradients(
        const dealii::VectorizedArray<double> &scaleValues,
        const bool                             scaleValuesFlag,
        const dealii::VectorizedArray<double> &scaleGradients,
        const bool                             scaleGradientsFlag) override;

      void
      integrate(
        dealii::EvaluationFlags::EvaluationFlags evaluateFlags) override;

      void
      getValues(dealii::AlignedVector<dealii::VectorizedArray<double>> &tempVec)
        override;

      void
      distributeLocalToGlobal(distributedCPUVec<double> &tempvec) override;
    };



    template <unsigned int numberOfComponents>
    class DealiiFEEvaluationWrapper
    {
    public:
      /**
       * @brief constructor call for DealiiFEEvaluationWrapper
       * @param[in] fe_degree interpolating polynomial degree
       * @param[in] num_1d_quad number of quad points in 1 direction
       * @param[in] matrixFreeData MatrixFree object.
       * @param[in] matrixFreeVectorComponent  used for getting data
       * from the MatrixFree object.
       * @param[in] matrixFreeQuadratureComponent quadrature ID that index the
       * quadrature location in the MatrixFree object.
       */
      DealiiFEEvaluationWrapper(
        unsigned int                         fe_degree,
        unsigned int                         num_1d_quad,
        const dealii::MatrixFree<3, double> &matrixFreeData,
        const unsigned int                   matrixFreeVectorComponent,
        const unsigned int                   matrixFreeQuadratureComponent);

      const FEEvaluationWrapperBase &
      getFEEvaluationWrapperBase() const;

    private:
      unsigned int d_feDegree;
      unsigned int d_num1dQuad;
      unsigned int d_matrixFreeVectorComponent;
      unsigned int d_matrixFreeQuadratureComponent;

      std::unique_ptr<FEEvaluationWrapperBase> d_feEvaluationBase;
      const dealii::MatrixFree<3, double> *    d_matrix_free_data;



    }; // end of DealiiFEEvaluationWrapper

  } // end of namespace basis

} // end of namespace dftefe


#endif // FEEvaluationWrapper_h
