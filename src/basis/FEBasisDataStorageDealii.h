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

#ifndef dftefeFEBasisDataStorageDealii_h
#define dftefeFEBasisDataStorageDealii_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <basis/BasisDataStorage.h>
#include <basis/FEBasisManagerDealii.h>
#include <basis/ConstraintsDealii.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <memory>
#include <map>
#include <vector>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class to store and access data for a given basis,
     * such as the basis function values on a quadrature grid, the overlap
     * matrix of the basis, etc.
     */
    template <typename ValueType, utils::MemorySpace memorySpace, size_type dim>
    class FEBasisDataStorageDealii
      : public BasisDataStorage<ValueType, memorySpace>
    {
    public:
      using QuadratureAttributes =
        BasisDataStorage<ValueType, memorySpace>::QuadratureAttributes;
      using QuadratureRuleType =
        BasisDataStorage<ValueType, memorySpace>::QuadratureRuleType;
      using Storage = BasisDataStorage<ValueType, memorySpace>::Storage;

      FEBasisDataStorageDealii(
        std::shared_ptr<const FEBasisManagerDealii>           feBM,
        std::vector<std::shared_ptr<const ConstraintsDealii>> constraintsVec,
        const std::vector<QuadratureRuleType> &               quadRuleTypeVec,
        const bool                                            storeValues,
        const bool                                            storeGradient,
        const bool                                            storeHessian,
        const bool                                            storeJxW,
        const bool storeQuadRealPoints);
      ~FEBasisDataStorageDealii();
      evaluateBasisData(
        std::shared_ptr<const quadrature::CellQuadratureContainer>
                                 quadratureContainer,
        const QuadratureRuleType quadRuleType,
        const bool               storeGradient,
        const bool               storeHessian,
        const bool               storeOverlap) override;
      deleteBasisData(const QuadratureRuleType quadRuleType) override;

      std::shared_ptr<const quadrature::CellQuadratureContainer>
      getCellQuadratureRuleContainer(
        const QuadratureRuleType quadRuleType) const override;

      // functions to get data for a basis function on a given quad point in a
      // cell
      ValueType
      getBasisData(const QuadratureAttributes &attributes,
                   const size_type             basisId) const override;
      Storage
      getBasisGradientData(const QuadratureAttributes &attributes,
                           const size_type             basisId) const override;
      Storage
      getBasisHessianData(const QuadratureAttributes &attributes,
                          const size_type             basisId) const override;

      // functions to get data for a basis function on all quad points in a cell
      Storage
      getBasisDataInCell(const QuadratureRuleType quadRuleType,
                         const size_type          cellId,
                         const size_type          basisId) const override;
      Storage
      getBasisGradientDataInCell(const QuadratureRuleType quadRuleType,
                                 const size_type          cellId,
                                 const size_type basisId) const override;
      Storage
      getBasisHessianDataInCell(const QuadratureRuleType quadRuleType,
                                const size_type          cellId,
                                const size_type basisId) const override;

      // functions to get data for all basis functions on all quad points in a
      // cell
      Storage
      getBasisDataInCell(const QuadratureRuleType quadRuleType,
                         const size_type          cellId) const override;
      Storage
      getBasisGradientDataInCell(const QuadratureRuleType quadRuleType,
                                 const size_type cellId) const override;
      Storage
      getBasisHessianDataInCell(const QuadratureRuleType quadRuleType,
                                const size_type          cellId) const override;

      // functions to get data for all basis functions on all quad points in all
      // cells
      const Storage &
      getBasisDataInAllCells(
        const QuadratureRuleType quadRuleType) const override;
      const Storage &
      getBasisGradientDataInAllCells(
        const QuadratureRuleType quadRuleType) const override;
      const Storage &
      getBasisHessianDataInAllCells(
        const QuadratureRuleType quadRuleType) const override;

      // get overlap of two basis functions in a cell
      ValueType
      getBasisOverlap(const QuadratureRuleType quadRuleType,
                      const size_type          cellId,
                      const size_type          basisId1,
                      const size_type          basisId2) const override;

      // get overlap of all the basis functions in a cell
      Storage
      getBasisOverlapInCell(const QuadratureRuleType quadRuleType,
                            const size_type          cellId) const override;

      // get overlap of all the basis functions in all cells
      const Storage &
      getBasisOverlapInAllCells(
        const QuadratureRuleType quadRuleType) const override;

    private:
      std::shared_ptr<const FEBasisManagerDealii>            d_feBM;
      std::map<QuadratureRuleType, std::shared_ptr<Storage>> d_basisQuadStorage;
      std::map<QuadratureRuleType, std::shared_ptr<Storage>>
        d_basisGradientQuadStorage;
      std::map<QuadratureRuleType, std::shared_ptr<Storage>>
        d_basisHessianQuadStorage;
      std::map<QuadratureRuleType, std::shared_ptr<Storage>> d_basisOverlap;
      std::shared_ptr<dealii::MatrixFree<dim, ValueType>>    d_dealiiMatrixFree;

    }; // end of FEBasisDataStorageDealii
  }    // end of namespace basis
} // end of namespace dftefe
#include <basis/FEBasisDataStorageDealii.t.cpp>
#endif // dftefeFEBasisDataStorageDealii_h
