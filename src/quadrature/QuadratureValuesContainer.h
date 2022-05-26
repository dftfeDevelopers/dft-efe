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
        dftefe::utils::MemoryStorage<ValueType, memorySpace>::pointer;
      using reference =
        dftefe::utils::MemoryStorage<ValueType, memorySpace>::reference;
      using const_reference =
        dftefe::utils::MemoryStorage<ValueType, memorySpace>::const_reference;
      using iterator =
        dftefe::utils::MemoryStorage<ValueType, memorySpace>::iterator;
      using const_iterator =
        dftefe::utils::MemoryStorage<ValueType, memorySpace>::const_iterator;

    public:
      QuadratureValuesContainer();
      QuadratureValuesContainer(
        const QuadratureRuleContainer &quadratureRuleContainer,
        const size_type                numberComponents,
        const ValueType                initVal = ValueType());

      reinit(const QuadratureRuleContainer &quadratureRuleContainer,
             const size_type                numberComponents,
             const ValueType                initVal = ValueType());

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
                        const size_type  quadId,
                        const ValueType *values);

      template <utils::MemorySpace memorySpaceDst>
      void
      getCellValues(const size_type cellId, const ValueType *values);

      template <utils::MemorySpace memorySpaceDst>
      void
      getCellQuadValues(const size_type  cellId,
                        const size_type  quadId,
                        const ValueType *values);

      const QuadratureRuleContainer &
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
      getNumberCellEntries() const

        iterator begin();

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

    private:
      size_type                      d_numberComponents;
      SizeTypeVector                 d_cellStartIds;
      SizeTypeVector                 d_numCellEntries;
      Storage                        d_storage;
      const QuadratureRuleContainer *d_quadratureRuleContainer;
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
        const linearAlgebra::LinAlgOpContext &linAlgOpContext);

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
        const linearAlgebra::LinAlgOpContext &linAlgOpContext);

    /**
     * @brief Perform \f$ w = a*u\f$
     * @param[in] a scalar
     * @param[in] u QuadratureValuesContainer
     * @param[out] w Resulting QuadratureValuesContainer
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    scale(ValueType                                                a,
          const QuadratureValuesContainer<ValueType, memorySpace> &u,
          QuadratureValuesContainer<ValueType, memorySpace> &      w,
          const linearAlgebra::LinAlgOpContext &linAlgOpContext);

    /**
     * @brief Perform \f$ w = a*u\f$
     * @param[in] a scalar
     * @param[in] u QuadratureValuesContainer
     * @return Resulting QuadratureValuesContainer w
     */
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    scale(ValueType                                          a,
          QuadratureValuesContainer<ValueType, memorySpace> &u,
          const linearAlgebra::LinAlgOpContext &             linAlgOpContext);

  } // end of namespace quadrature
} // end of namespace dftefe
#include <quadrature/QuadratureValuesContainer.t.cpp>
#endif // dftefeQuadratureValuesContainer_h
