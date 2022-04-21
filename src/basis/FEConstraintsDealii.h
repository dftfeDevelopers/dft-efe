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
* @author Vishal Subramanian
 */

#ifndef dftefeFEConstraintsDealii_h
#define dftefeFEConstraintsDealii_h

#include "FEConstraintsBase.h"
#include "FEBasisManager.h"
#include <utils/TypeConfig.h>
#include <deal.II/lac/affine_constraints.h>
#include "FEBasisManagerDealii.h"
namespace dftefe
{
  namespace basis
  {

    template <size_type dim,typename ValueType>
    class FEConstraintsDealii : public FEConstraintsBase<ValueType>
    {
    public:
      FEConstraintsDealii();
      ~FEConstraintsDealii();
      void clear () override ;

      void makeHangingNodeConstraint(
        std::shared_ptr<FEBasisManager> feBasis) override;

      void addLine(size_type lineId) override;

      void setInhomogeneity(size_type lineId,
                        ValueType constraintValue) override;
      bool isConstrained( size_type nodeId) override;
      void close() override;

      bool isClosed() override;

      void setHomogeneousDirichletBC() override;

      //dealii specific fucntions
      //std::shared_ptr <dealii::AffineConstraints<ValueType>>
      //const dealii::AffineConstraints<ValueType> *
     const dealii::AffineConstraints<ValueType> &
     getAffineConstraints() const;

    private:
      std::shared_ptr <dealii::AffineConstraints<ValueType>> d_constraintMatrix;
      std::shared_ptr <const FEBasisManagerDealii<dim>> d_dofHandler;
      bool d_isCleared;
      bool d_isClosed;

    };

  }
}
#include "FEConstraintsDealii.t.cpp"
#endif // dftefeFEConstraintsDealii_h
