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
 * @author Bikash Kanungo
 */
#ifndef dftefeQuadratureValuesContainer_h
#define dftefeQuadratureValuesContainer_h

#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <linearAlgebra/LinAlgOpContext.h>
#include <quadrature/QuadratureRuleContainer.h>
namespace dftefe
{
  namespace quadrature
  {
    template <typename ValueType, utils::MemorySpace memorySpace>
    class QuadratureValuesContainer
    {
    public:
      using Storage = dftefe::utils::MemoryStorage<ValueType, memorySpace>;
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;
      using pointer =
        typename dftefe::utils::MemoryStorage<ValueType, memorySpace>::pointer;
      using reference =
        typename dftefe::utils::MemoryStorage<ValueType,
                                              memorySpace>::reference;
      using const_reference =
        typename dftefe::utils::MemoryStorage<ValueType,
                                              memorySpace>::const_reference;
      using iterator =
        typename dftefe::utils::MemoryStorage<ValueType, memorySpace>::iterator;
      using const_iterator =
        typename dftefe::utils::MemoryStorage<ValueType,
                                              memorySpace>::const_iterator;

    public:
      QuadratureValuesContainer();
      QuadratureValuesContainer(
        std::shared_ptr<const QuadratureRuleContainer> quadratureRuleContainer,
        const size_type                                numberComponents,
        const ValueType                                initVal = ValueType());

      void
      reinit(
        std::shared_ptr<const QuadratureRuleContainer> quadratureRuleContainer,
        const size_type                                numberComponents,
        const ValueType                                initVal = ValueType());

      QuadratureValuesContainer(const QuadratureValuesContainer &u);
      QuadratureValuesContainer(QuadratureValuesContainer &&u);

      QuadratureValuesContainer &
      operator=(const QuadratureValuesContainer &rhs);

      QuadratureValuesContainer &
      operator=(QuadratureValuesContainer &&rhs);

      template <utils::MemorySpace memorySpaceSrc>
      void
      setCellValues(const size_type cellId, const ValueType *values);

      template <utils::MemorySpace memorySpaceSrc>
      void
      setCellQuadValues(const size_type  cellId,
                        const size_type  componentId,
                        const ValueType *values);

      template <utils::MemorySpace memorySpaceDst>
      void
      getCellValues(const size_type cellId, ValueType *values) const;

      template <utils::MemorySpace memorySpaceDst>
      void
      getCellQuadValues(const size_type cellId,
                        const size_type componentId,
                        ValueType *     values) const;

      std::shared_ptr<const QuadratureRuleContainer>
      getQuadratureRuleContainer() const;

      size_type
      getNumberComponents() const;

      size_type
      nCells() const;
      size_type
      nQuadraturePoints() const;
      size_type
      nEntries() const;

      size_type
      nCellQuadraturePoints(const size_type cellId) const;
      size_type
      nCellEntries(const size_type cellId) const;
      size_type
      cellStartId(const size_type cellId) const;

      const SizeTypeVector &
      getCellStartIds() const;
      const SizeTypeVector &
      getNumberCellEntries() const;

      iterator
      begin();

      const_iterator
      begin() const;

      iterator
      end();

      const_iterator
      end() const;

      iterator
      begin(const size_type cellId);

      const_iterator
      begin(const size_type cellId) const;

      iterator
      end(const size_type cellId);

      const_iterator
      end(const size_type cellId) const;

      // ValueType
      // dotProduct() const;

    private:
      size_type                                      d_numberComponents;
      SizeTypeVector                                 d_cellStartIds;
      SizeTypeVector                                 d_numCellEntries;
      Storage                                        d_storage;
      std::shared_ptr<const QuadratureRuleContainer> d_quadratureRuleContainer;
    }; // end of QuadratureValuesContainer


    //
    // Helper functions
    //

    /**
     * @brief Perform \f$ w = a*u + b*v \f$
     * @param[in] a scalar
     * @param[in] u QuadratureValuesContainer
     * @param[in] b scalar
     * @param[in] v QuadratureValuesContainer
     * @param[out] w Resulting QuadratureValuesContainer
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                                                a,
        const QuadratureValuesContainer<ValueType, memorySpace> &u,
        ValueType                                                b,
        const QuadratureValuesContainer<ValueType, memorySpace> &v,
        QuadratureValuesContainer<ValueType, memorySpace> &      w,
        const linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext);

    /**
     * @brief Perform \f$ w = a*u + b*v \f$
     * @param[in] a scalar
     * @param[in] u QuadratureValuesContainer
     * @param[in] b scalar
     * @param[in] v QuadratureValuesContainer
     * @return Resulting QuadratureValuesContainer w
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    QuadratureValuesContainer<ValueType, memorySpace>
    add(ValueType                                                a,
        const QuadratureValuesContainer<ValueType, memorySpace> &u,
        ValueType                                                b,
        const QuadratureValuesContainer<ValueType, memorySpace> &v,
        const linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext);

    // FIXME: Uncomment the following once ascale is implemented in
    // linearAlgebra::blaslapack

    //    /**
    //     * @brief Perform \f$ w = a*u\f$
    //     * @param[in] a scalar
    //     * @param[in] u QuadratureValuesContainer
    //     * @param[out] w Resulting QuadratureValuesContainer
    //     */
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //    void
    //    scale(ValueType                                                a,
    //          const QuadratureValuesContainer<ValueType, memorySpace> &u,
    //          QuadratureValuesContainer<ValueType, memorySpace> &      w,
    //          const linearAlgebra::LinAlgOpContext<memorySpace>
    //          &linAlgOpContext);
    //
    //    /**
    //     * @brief Perform \f$ w = a*u\f$
    //     * @param[in] a scalar
    //     * @param[in] u QuadratureValuesContainer
    //     * @return Resulting QuadratureValuesContainer w
    //     */
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //    void
    //    scale(ValueType                                          a,
    //          QuadratureValuesContainer<ValueType, memorySpace> &u,
    //          const linearAlgebra::LinAlgOpContext<memorySpace> &
    //          linAlgOpContext);

  } // end of namespace quadrature
} // end of namespace dftefe
#include <quadrature/QuadratureValuesContainer.t.cpp>
#endif // dftefeQuadratureValuesContainer_h
