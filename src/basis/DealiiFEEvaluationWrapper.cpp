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
#include <basis/DealiiFEEvaluationWrapper.h>
#include <utils/Exceptions.h>

namespace dftefe
{
  namespace basis
  {
    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      FEEvaluationWrapperDerived(
        const dealii::MatrixFree<3, double> &matrixFreeData,
        const unsigned int                   matrixFreeVectorComponent,
        const unsigned int                   matrixFreeQuadratureComponent)
    {
      d_dealiiFEEvaluation = std::make_unique<
        dealii::FEEvaluation<3, FEOrder, num_1d_quadPoints, n_components>>(
        matrixFreeData,
        matrixFreeVectorComponent,
        matrixFreeQuadratureComponent);
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      ~FEEvaluationWrapperDerived()
    {
      d_dealiiFEEvaluation.reset(nullptr);
    }

    FEEvaluationWrapperBase::~FEEvaluationWrapperBase()
    {}

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    unsigned int
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      totalNumberofQuadraturePoints()
    {
      return d_dealiiFEEvaluation->n_q_points;
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      reinit(const unsigned int macrocell)
    {
      d_dealiiFEEvaluation->reinit(macrocell);
    }
    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      readDoFValues(const distributedCPUVec<double> &tempvec)
    {
      d_dealiiFEEvaluation->read_dof_values(tempvec);
    }
    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      readDoFValuesPlain(const distributedCPUVec<double> &tempvec)
    {
      d_dealiiFEEvaluation->read_dof_values_plain(tempvec);
    }
    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      evaluate(dealii::EvaluationFlags::EvaluationFlags evaluateFlags)
    {
      d_dealiiFEEvaluation->evaluate(evaluateFlags);
    }


    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitInterpolatedGradientsAndMultiply(
        dealii::VectorizedArray<double> &alpha)
    {
      for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
        {
          d_dealiiFEEvaluation->submit_gradient(
            alpha * d_dealiiFEEvaluation->get_gradient(q), q);
        }
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitInterpolatedValuesAndMultiply(
        dealii::VectorizedArray<double> &alpha)
    {
      for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
        {
          d_dealiiFEEvaluation->submit_value(
            alpha * d_dealiiFEEvaluation->get_value(q), q);
        }
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitInterpolatedValuesAndMultiplySquared()
    {
      for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
        {
          d_dealiiFEEvaluation->submit_value(
            d_dealiiFEEvaluation->get_value(q) *
              d_dealiiFEEvaluation->get_value(q),
            q);
        }
    }



    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitInterpolatedValuesAndMultiply(
        dealii::AlignedVector<dealii::VectorizedArray<double>> &alpha)
    {
      for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
        {
          d_dealiiFEEvaluation->submit_value(
            alpha[q] * d_dealiiFEEvaluation->get_value(q), q);
        }
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitValues(
        const dealii::VectorizedArray<double> &scaling,
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &alpha)
    {
      if constexpr (n_components == 3)
        for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
          {
            d_dealiiFEEvaluation->submit_value(scaling * alpha[q], q);
          }
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitInterpolatedValuesSubmitInterpolatedGradients(
        const dealii::VectorizedArray<double> &scaleValues,
        const bool                             scaleValuesFlag,
        const dealii::VectorizedArray<double> &scaleGradients,
        const bool                             scaleGradientsFlag)
    {
      if (scaleValuesFlag)
        {
          if (scaleGradientsFlag)
            {
              for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points;
                   ++q)
                {
                  d_dealiiFEEvaluation->submit_value(
                    scaleValues * d_dealiiFEEvaluation->get_value(q), q);
                  d_dealiiFEEvaluation->submit_gradient(
                    scaleGradients * d_dealiiFEEvaluation->get_gradient(q), q);
                }
            }
          else
            {
              for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points;
                   ++q)
                {
                  d_dealiiFEEvaluation->submit_value(
                    scaleValues * d_dealiiFEEvaluation->get_value(q), q);
                  d_dealiiFEEvaluation->submit_gradient(
                    d_dealiiFEEvaluation->get_gradient(q), q);
                }
            }
        }
      else
        {
          if (scaleGradientsFlag)
            {
              for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points;
                   ++q)
                {
                  d_dealiiFEEvaluation->submit_value(
                    d_dealiiFEEvaluation->get_value(q), q);
                  d_dealiiFEEvaluation->submit_gradient(
                    scaleGradients * d_dealiiFEEvaluation->get_gradient(q), q);
                }
            }
          else
            {
              for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points;
                   ++q)
                {
                  d_dealiiFEEvaluation->submit_value(
                    d_dealiiFEEvaluation->get_value(q), q);
                  d_dealiiFEEvaluation->submit_gradient(
                    d_dealiiFEEvaluation->get_gradient(q), q);
                }
            }
        }
    }


    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitGradients(
        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>> &alpha)
    {
      for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
        {
          d_dealiiFEEvaluation->submit_gradient(alpha[q], q);
        }
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    dealii::VectorizedArray<double>
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      integrateValue()
    {
      if constexpr (n_components == 1)
        return d_dealiiFEEvaluation->integrate_value();
    }
    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      alphaTimesQuadValuesPlusYFromSubCell(const unsigned int subCellIndex,
                                           const double       alpha,
                                           double *           outputVector)
    {
      if constexpr (n_components == 1)
        {
          for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
            {
              *(outputVector + q) +=
                alpha * d_dealiiFEEvaluation->get_value(q)[subCellIndex];
            }
        }
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      getQuadGradientsForSubCell(const unsigned int subCellIndex,
                                 const double       alpha,
                                 double *           outputVector)
    {
      if constexpr (n_components == 1)
        {
          for (unsigned int q_point = 0;
               q_point < d_dealiiFEEvaluation->n_q_points;
               ++q_point)
            {
              const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
                &gradVals = d_dealiiFEEvaluation->get_gradient(q_point);
              *(outputVector + 3 * q_point + 0) +=
                alpha * gradVals[0][subCellIndex];
              *(outputVector + 3 * q_point + 1) +=
                alpha * gradVals[1][subCellIndex];
              *(outputVector + 3 * q_point + 2) +=
                alpha * gradVals[2][subCellIndex];
            }
        }
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      getQuadHessianForSubCell(const unsigned int subCellIndex,
                               const double       alpha,
                               double *           outputVector)
    {
      if constexpr (n_components == 1)
        {
          for (unsigned int q_point = 0;
               q_point < d_dealiiFEEvaluation->n_q_points;
               ++q_point)
            {
              const dealii::Tensor<2, 3, dealii::VectorizedArray<double>>
                &hessianVals = d_dealiiFEEvaluation->get_hessian(q_point);
              for (unsigned int i = 0; i < 3; i++)
                for (unsigned int j = 0; j < 3; j++)
                  *(outputVector + 9 * q_point + 3 * i + j) +=
                    alpha * hessianVals[i][j][subCellIndex];
            }
        }
    }

    // template <int          FEOrder,
    //           unsigned int num_1d_quadPoints,
    //           unsigned int n_components>
    // void FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints,
    // n_components>::
    //   submitGradients(
    //     dealii::AlignedVector<
    //       dealii::Tensor<2, 3, dealii::VectorizedArray<double>>> &alpha,
    //     const dealii::VectorizedArray<double> &                   scaling,
    //     bool positiveFlag)
    // {
    //   if constexpr (n_components == 3)
    //     {
    //       for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
    //         {
    //           d_dealiiFEEvaluation->submit_gradient(scaling * alpha[q], q);
    //         }
    //     }
    //   else
    //     {
    //       DFTEFE_AssertWithMsg(
    //         n_components == 3, "DFT-EFE Error: This type of submitGradient
    //         can be called only
    //           for n_components = 3");
    //     }
    // }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitValues(
        dealii::AlignedVector<dealii::VectorizedArray<double>> &alpha)

    {
      if constexpr (n_components == 1)
        for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
          {
            d_dealiiFEEvaluation->submit_value(alpha[q], q);
          }
      else
        DFTEFE_AssertWithMsg(
          n_components == 1,
          "DFT-EFE Error: Incorrect call with number of components");
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      getValues(dealii::AlignedVector<dealii::VectorizedArray<double>> &tempVec)

    {
      if constexpr (n_components == 1)
        for (unsigned int q = 0; q < d_dealiiFEEvaluation->n_q_points; ++q)
          {
            tempVec[q] = d_dealiiFEEvaluation->get_value(q);
          }
      else
        DFTEFE_AssertWithMsg(
          n_components == 1,
          "DFT-EFE Error: Incorrect call with number of components");
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      integrate(dealii::EvaluationFlags::EvaluationFlags evaluateFlags)
    {
      d_dealiiFEEvaluation->integrate(evaluateFlags);
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      submitValueAtQuadpoint(const unsigned int                     iQuadPoint,
                             const dealii::VectorizedArray<double> &value)
    {
      if constexpr (n_components == 1)
        d_dealiiFEEvaluation->submit_value(value, iQuadPoint);
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    dealii::Point<3, dealii::VectorizedArray<double>>
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      getQuadraturePoint(const unsigned int iQuadPoint)
    {
      return d_dealiiFEEvaluation->quadrature_point(iQuadPoint);
    }

    template <int          FEOrder,
              unsigned int num_1d_quadPoints,
              unsigned int n_components>
    void
    FEEvaluationWrapperDerived<FEOrder, num_1d_quadPoints, n_components>::
      distributeLocalToGlobal(distributedCPUVec<double> &tempvec)
    {
      d_dealiiFEEvaluation->distribute_local_to_global(tempvec);
    }
// #ifdef DFTEFE_MINIMAL_COMPILE
// #  define RANGE_FEORDER ((1)(2)(3)(4)(5)(6)(7))
// #  define RANGE_QUADRATURE ((2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14))
// #else
#define RANGE_FEORDER \
  ((3)(4)(5)(6)(7)) //(1)(2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14)
#define RANGE_QUADRATURE \
  ((10)(12)(14)(16)(18)( \
    20)) //(2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14)(15)(16)(18)(19)(20)
// #endif
#define MACRO(r, p)                                                  \
  template class FEEvaluationWrapperDerived<BOOST_PP_SEQ_ELEM(0, p), \
                                            BOOST_PP_SEQ_ELEM(1, p), \
                                            1>;
    BOOST_PP_SEQ_FOR_EACH_PRODUCT(MACRO, RANGE_FEORDER RANGE_QUADRATURE)
#undef MACRO
    template class FEEvaluationWrapperDerived<-1, 1, 1>;
// #define MACRO(r, p)                                                  \
//   template class FEEvaluationWrapperDerived<BOOST_PP_SEQ_ELEM(0, p), \
//                                             BOOST_PP_SEQ_ELEM(1, p), \
//                                             3>;
//     BOOST_PP_SEQ_FOR_EACH_PRODUCT(MACRO, RANGE_FEORDER RANGE_QUADRATURE)
// #undef MACRO
#undef RANGE_FEORDER
#undef RANGE_QUADRATURE
    template <unsigned int numberOfComponents>
    DealiiFEEvaluationWrapper<numberOfComponents>::DealiiFEEvaluationWrapper(
      unsigned int                         fe_degree,
      unsigned int                         num_1d_quad,
      const dealii::MatrixFree<3, double> &matrixFreeData,
      const unsigned int                   matrixFreeVectorComponent,
      const unsigned int                   matrixFreeQuadratureComponent)
    {
      d_feDegree                      = fe_degree;
      d_num1dQuad                     = num_1d_quad;
      d_matrixFreeVectorComponent     = matrixFreeVectorComponent;
      d_matrixFreeQuadratureComponent = matrixFreeQuadratureComponent;
      d_matrix_free_data              = &matrixFreeData;
// #ifdef DFTEFE_MINIMAL_COMPILE
// #  define RANGE_FEORDER ((1)(2)(3)(4)(5)(6)(7))
// #  define RANGE_QUADRATURE ((2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14))
// #else
#define RANGE_FEORDER \
  ((3)(4)(5)(6)(7)) //(1)(2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14)
#define RANGE_QUADRATURE \
  ((10)(12)(14)(16)(18)( \
    20)) //(2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14)(15)(16)(18)(19)(20)
// #endif
#define MACRO(r, p)                                                   \
  if (BOOST_PP_SEQ_ELEM(0, p) == d_feDegree &&                        \
      BOOST_PP_SEQ_ELEM(1, p) == d_num1dQuad)                         \
    d_feEvaluationBase = std::make_unique<FEEvaluationWrapperDerived< \
      BOOST_PP_TUPLE_REM_CTOR(2, BOOST_PP_SEQ_TO_TUPLE(p)),           \
      numberOfComponents>>(*d_matrix_free_data,                       \
                           d_matrixFreeVectorComponent,               \
                           d_matrixFreeQuadratureComponent);
      BOOST_PP_SEQ_FOR_EACH_PRODUCT(MACRO, RANGE_FEORDER RANGE_QUADRATURE)
#undef MACRO
      else
      {
        d_feEvaluationBase = std::make_unique<
          FEEvaluationWrapperDerived<-1, 1, numberOfComponents>>(
          *d_matrix_free_data,
          d_matrixFreeVectorComponent,
          d_matrixFreeQuadratureComponent);
      }
#undef RANGE_FEORDER
#undef RANGE_QUADRATURE
    }

    template <unsigned int numberOfComponents>
    DealiiFEEvaluationWrapper<numberOfComponents>::~DealiiFEEvaluationWrapper()
    {
      if (d_feEvaluationBase.get() != nullptr)
        {
          d_feEvaluationBase.reset(nullptr);
        }
    }

    template <unsigned int numberOfComponents>
    FEEvaluationWrapperBase &
    DealiiFEEvaluationWrapper<numberOfComponents>::getFEEvaluationWrapperBase()
      const
    {
      return *d_feEvaluationBase;
    }

    template class DealiiFEEvaluationWrapper<1>;
    // template class DealiiFEEvaluationWrapper<3>;

  } // End of namespace basis
} // End of namespace dftefe
