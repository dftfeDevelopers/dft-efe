
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
 * @author Bikash Kanungo, Vishal Subramanian
 */

#include <utils/Exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
namespace dftefe
{
  namespace basis
  {
    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    FEBasisDataStorageDealii<ValueType, memorySpace, dim>::
      FEBasisDataStorageDealii(
        std::shared_ptr<const FEBasisManagerDealii>           feBM,
        std::vector<std::shared_ptr<const ConstraintsDealii>> constraintsVec,
        const std::vector<
          typename BasisDataStorage<ValueType, memorySpace>::QuadratureRuleType>
          &quadRuleTypeVec)
    {
      const size_type numConstraints  = constraintsVec.size();
      const size_type numQuadRuleType = quadRuleTypeVec.size();
      std::shared_ptr<const dealii::DoFHandler<dim>> dofHandler =
        feBM->getDoFHandler();
      std::vector<const dealii::DoFHandler<dim> *> dofHandlerVec(
        numConstraints, dofHandler.get());
      std::vector<const dealii::AffineConstraints<ValueType> *>
        dealiiAffineConstraintsVec(numConstraints, nullptr);
      for (size_type i = 0; i < numConstraints; ++i)
        {
          dealiiAffineConstraintsVec[i] =
            (constraintsVec[i]->getAffineConstraints()).get();
        }

      std::vector<dealii::QuadratureType> dealiiQuadratureTypeVec(0);
      for (size_type i = 0; i < numQuadRuleType; ++i)
        {
          size_type num1DQuadPoints =
            quadrature::get1DQuadNumPoints(quadRuleTypeVec[i]);
          quadrature::QuadratureFamily quadFamily =
            quadrature::getQuadratureFamily(quadRuleTypeVec[i]);
          if (quadFamily == quadrature::QuadFamily : GAUSS)
            dealiiQuadratureTypeVec.push_back(
              dealii::QGauss<1>(num1DQuadPoints));
          else if (quadFamily == quadrature::QuadFamily : GLL)
            dealiiQuadratureTypeVec.push_back(
              dealii::QGaussLobatto<1>(num1DQuadPoints));
          else
            utils::throwException<utils::InvalidArgument>(
              false,
              "Quadrature family is undefined. Currently, only Gauss and GLL quadrature families are supported.")
        }

      d_dealiiMatrixFree =
        std::make_shared<dealii::MatrixFree<dim, ValueType>>();
      d_dealiiMatrixFree->clear();
      d_dealiiMatrixFree->reinit(dofHandlerVec,
                                 dealiiAffineConstraintsVec,
                                 dealiiQuadratureTypeVec, );
    }

  } // end of namespace basis
} // end of namespace dftefe
