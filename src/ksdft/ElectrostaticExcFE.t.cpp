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

#include <utils/DataTypeOverloads.h>
#include <ksdft/Defaults.h>

namespace dftefe
{
  namespace ksdft
  {
    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                       ValueTypeElectrostaticsBasis,
                       ValueTypeWaveFunctionCoeff,
                       ValueTypeWaveFunctionBasis,
                       memorySpace,
                       dim>::
      ElectrostaticExcFE(
        std::shared_ptr<const ElectrostaticFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>       electroHamiltonian,
        std::shared_ptr<const ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                    ValueTypeWaveFunctionCoeff,
                                                    memorySpace,
                                                    dim>> excHamiltonian)
    {
      reinit(electroHamiltonian, excHamiltonian);
    }


    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                       ValueTypeElectrostaticsBasis,
                       ValueTypeWaveFunctionCoeff,
                       ValueTypeWaveFunctionBasis,
                       memorySpace,
                       dim>::
      reinit(
        std::shared_ptr<const ElectrostaticFE<ValueTypeElectrostaticsBasis,
                                              ValueTypeElectrostaticsCoeff,
                                              ValueTypeWaveFunctionBasis,
                                              memorySpace,
                                              dim>>       electroHamiltonian,
        std::shared_ptr<const ExchangeCorrelationFE<ValueTypeWaveFunctionBasis,
                                                    ValueTypeWaveFunctionCoeff,
                                                    memorySpace,
                                                    dim>> excHamiltonian)
    {
      d_electroHamiltonian = electroHamiltonian;
      d_excHamiltonian     = excHamiltonian;
    }

    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                       ValueTypeElectrostaticsBasis,
                       ValueTypeWaveFunctionCoeff,
                       ValueTypeWaveFunctionBasis,
                       memorySpace,
                       dim>::getLocal(Storage &cellWiseStorage) const
    {
      std::shared_ptr<quadrature::QuadratureValuesContainer<ValueType, memorySpace>>
        elextroxcPotentialQuad = nullptr;

      if(d_electroHamiltonian->hasLocalComponent() && !d_excHamiltonian->hasLocalComponent())
      {
        elextroxcPotentialQuad  =
          std::make_shared<quadrature::QuadratureValuesContainer<ValueType, memorySpace>>
            (d_electroHamiltonian->getFunctionalDerivative());
      }
      else if(!d_electroHamiltonian->hasLocalComponent() && d_excHamiltonian->hasLocalComponent())
      {
        elextroxcPotentialQuad  =
          std::make_shared<quadrature::QuadratureValuesContainer<ValueType, memorySpace>>
            (d_excHamiltonian->getFunctionalDerivative());
      }
      else
      {
        elextroxcPotentialQuad  =
          std::make_shared<quadrature::QuadratureValuesContainer<ValueType, memorySpace>>
            (d_electroHamiltonian->getFunctionalDerivative());

        quadrature::add((ValueType)1.0,
                      d_excHamiltonian->getFunctionalDerivative(),
                      (ValueType)1.0,
                      *elextroxcPotentialQuad,
                      *d_excHamiltonian->getLinAlgOpContext());
      }

      d_excHamiltonian->getHamiltonianFEBasisOperations()->computeFEMatrices(
        basis::realspace::LinearLocalOp::IDENTITY,
        basis::realspace::VectorMathOp::MULT,
        basis::realspace::VectorMathOp::MULT,
        basis::realspace::LinearLocalOp::IDENTITY,
        *elextroxcPotentialQuad,
        cellWiseStorage,
        *d_excHamiltonian->getLinAlgOpContext());
    }

    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    typename ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                                ValueTypeElectrostaticsBasis,
                                ValueTypeWaveFunctionCoeff,
                                ValueTypeWaveFunctionBasis,
                                memorySpace,
                                dim>::RealType
    ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                       ValueTypeElectrostaticsBasis,
                       ValueTypeWaveFunctionCoeff,
                       ValueTypeWaveFunctionBasis,
                       memorySpace,
                       dim>::getEnergy() const
    {
      return d_excHamiltonian->getEnergy() + d_electroHamiltonian->getEnergy();
    }

    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    void
    ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                       ValueTypeElectrostaticsBasis,
                       ValueTypeWaveFunctionCoeff,
                       ValueTypeWaveFunctionBasis,
                       memorySpace,
                       dim>::applyNonLocal(linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace> &X, 
                    linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace> &Y,
                    bool updateGhostX,
                    bool updateGhostY) const
     {
        if(d_excHamiltonian->hasNonLocalComponent() && !d_electroHamiltonian->hasNonLocalComponent())
          d_excHamiltonian->applyNonLocal(X,Y,updateGhostX,updateGhostY);
        if(!d_excHamiltonian->hasNonLocalComponent() && d_electroHamiltonian->hasNonLocalComponent())
          d_electroHamiltonian->applyNonLocal(X,Y,updateGhostX,updateGhostY);
        else
        {
          linearAlgebra::MultiVector<ValueTypeWaveFunctionCoeff, memorySpace> Z(X, (ValueTypeWaveFunctionCoeff)0);
          d_excHamiltonian->applyNonLocal(X,Z,updateGhostX,updateGhostY);
          d_electroHamiltonian->applyNonLocal(X,Y,updateGhostX,updateGhostY);
          linearAlgebra::add<ValueTypeWaveFunctionCoeff, 
          ValueTypeWaveFunctionCoeff, 
            memorySpace>((ValueTypeWaveFunctionCoeff)1.0, Y, 
                          (ValueTypeWaveFunctionCoeff)1.0, Z, Y);
        }
     }
     
 
    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                       ValueTypeElectrostaticsBasis,
                       ValueTypeWaveFunctionCoeff,
                       ValueTypeWaveFunctionBasis,
                       memorySpace,
                       dim>::hasLocalComponent() const
     {
        return (bool)(d_excHamiltonian->hasLocalComponent() || 
          d_electroHamiltonian->hasLocalComponent());
     }

    template <typename ValueTypeElectrostaticsCoeff,
              typename ValueTypeElectrostaticsBasis,
              typename ValueTypeWaveFunctionCoeff,
              typename ValueTypeWaveFunctionBasis,
              utils::MemorySpace memorySpace,
              size_type          dim>
    bool
    ElectrostaticExcFE<ValueTypeElectrostaticsCoeff,
                       ValueTypeElectrostaticsBasis,
                       ValueTypeWaveFunctionCoeff,
                       ValueTypeWaveFunctionBasis,
                       memorySpace,
                       dim>::hasNonLocalComponent() const
     {
        return (bool)(d_excHamiltonian->hasNonLocalComponent() || 
          d_electroHamiltonian->hasNonLocalComponent());
     }

  } // end of namespace ksdft
} // end of namespace dftefe
