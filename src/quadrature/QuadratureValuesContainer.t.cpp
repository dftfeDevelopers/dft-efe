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
#include <utils/MemoryTransfer.h>
#include <utils/Exceptions.h>
#include <linearAlgebra/BlasLapack.h>
namespace dftefe
{
  namespace quadrature
  {
    namespace QuadratureValuesContainerInternal
    {
      template <typename ValueType, utils::MemorySpace memorySpace>
      void
      initialize(
        const quadrature::QuadratureRuleContainer *quadratureRuleContainer,
        const size_type                            numberComponents,
        const ValueType                            initVal,
        typename QuadratureValuesContainer<ValueType,
                                           memorySpace>::SizeTypeVector
          &cellStartIds,
        typename QuadratureValuesContainer<ValueType,
                                           memorySpace>::SizeTypeVector
          &numCellEntries,
        typename QuadratureValuesContainer<ValueType, memorySpace>::Storage
          &storage)
      {
        const size_type numberCells = quadratureRuleContainer->nCells();
        const size_type numberTotalQuadraturePoints =
          quadratureRuleContainer->nQuadraturePoints();

        // resize containers
        cellStartIds.resize(numberCells, 0);
        numCellEntries.resize(numberCells, 0);
        storage.resize(numberTotalQuadraturePoints * numberComponents, initVal);

        // create temporary STL containers
        std::vector<size_type> cellStartIdsTmp(numberCells, 0);
        std::vector<size_type> numCellEntriesTmp(numberCells, 0);

        for (size_type iCell = 0; iCell < numberCells; ++iCell)
          {
            cellStartIdsTmp[iCell] =
              numberComponents *
              (quadratureRuleContainer->getCellQuadStartId(iCell));
            numCellEntriesTmp[iCell] =
              numberComponents *
              (quadratureRuleContainer->nCellQuadraturePoints(iCell));
          }

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          numberCells, cellStartIds.data(), cellStartIdsTmp.data());

        utils::MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
          numberCells, numCellEntries.data(), numCellEntriesTmp.data());
      }

    } // end of namespace QuadratureValuesContainerInternal

    //
    // Default Constructor
    //
    template <typename ValueType, utils::MemorySpace memorySpace>
    QuadratureValuesContainer<ValueType,
                              memorySpace>::QuadratureValuesContainer()
      : d_quadratureRuleContainer(nullptr)
      , d_numberComponents(0)
      , d_cellStartIds(0)
      , d_numCellEntries(0)
      , d_storage(0)
    {}


    //
    // Constructor
    //
    template <typename ValueType, utils::MemorySpace memorySpace>
    QuadratureValuesContainer<ValueType, memorySpace>::
      QuadratureValuesContainer(
        const quadrature::QuadratureRuleContainer &quadratureRuleContainer,
        const size_type                            numberComponents,
        const ValueType                            initVal /*= ValueType()*/)
      : d_quadratureRuleContainer(&quadratureRuleContainer)
      , d_numberComponents(numberComponents)
      , d_cellStartIds(0)
      , d_numCellEntries(0)
      , d_storage(0)
    {
      QuadratureValuesContainerInternal::initialize(d_quadratureRuleContainer,
                                                    d_numberComponents,
                                                    initVal,
                                                    d_cellStartIds,
                                                    d_numCellEntries,
                                                    d_storage);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    QuadratureValuesContainer<ValueType, memorySpace>::reinit(
      const quadrature::QuadratureRuleContainer &quadratureRuleContainer,
      const size_type                            numberComponents,
      const ValueType                            initVal /*= ValueType()*/)
    {
      d_quadratureRuleContainer = &quadratureRuleContainer;
      QuadratureValuesContainerInternal::initialize(d_quadratureRuleContainer,
                                                    d_numberComponents,
                                                    initVal,
                                                    d_cellStartIds,
                                                    d_numCellEntries,
                                                    d_storage);
    }

    //
    // Copy Constructor
    //
    template <typename ValueType, utils::MemorySpace memorySpace>
    QuadratureValuesContainer<ValueType, memorySpace>::
      QuadratureValuesContainer(
        const QuadratureValuesContainer<ValueType, memorySpace> &u)
      : d_quadratureRuleContainer(u.d_quadratureRuleContainer)
      , d_numberComponents(u.d_numberComponents)
      , d_cellStartIds(u.d_cellStartIds)
      , d_numCellEntries(u.d_numCellEntries)
      , d_storage(u.d_storage)
    {}

    //
    // Move Constructor
    //
    template <typename ValueType, utils::MemorySpace memorySpace>
    QuadratureValuesContainer<ValueType, memorySpace>::
      QuadratureValuesContainer(
        QuadratureValuesContainer<ValueType, memorySpace> &&u)
      : d_quadratureRuleContainer(std::move(u.d_quadratureRuleContainer))
      , d_numberComponents(std::move(u.d_numberComponents))
      , d_cellStartIds(std::move(u.d_cellStartIds))
      , d_numCellEntries(std::move(u.d_numCellEntries))
      , d_storage(std::move(u.d_storage))
    {}

    //
    // Copy Assignment
    //
    template <typename ValueType, utils::MemorySpace memorySpace>
    QuadratureValuesContainer<ValueType, memorySpace> &
    QuadratureValuesContainer<ValueType, memorySpace>::operator=(
      const QuadratureValuesContainer<ValueType, memorySpace> &rhs)
    {
      d_quadratureRuleContainer = rhs.d_quadratureRuleContainer;
      d_numberComponents        = rhs.d_numberComponents;
      d_cellStartIds            = rhs.d_cellStartIds;
      d_numCellEntries          = rhs.d_numCellEntries;
      d_storage                 = rhs.d_storage;
      return *this;
    }

    //
    // Move Assignment
    //
    template <typename ValueType, utils::MemorySpace memorySpace>
    QuadratureValuesContainer<ValueType, memorySpace> &
    QuadratureValuesContainer<ValueType, memorySpace>::operator=(
      QuadratureValuesContainer<ValueType, memorySpace> &&rhs)
    {
      d_quadratureRuleContainer = std::move(rhs.d_quadratureRuleContainer);
      d_numberComponents        = std::move(rhs.d_numberComponents);
      d_cellStartIds            = std::move(rhs.d_cellStartIds);
      d_numCellEntries          = std::move(rhs.d_numCellEntries);
      d_storage                 = std::move(rhs.d_storage);
      return *this;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    template <utils::MemorySpace memorySpaceSrc>
    void
    QuadratureValuesContainer<ValueType, memorySpace>::setCellValues(
      const size_type  cellId,
      const ValueType *values)
    {
      size_type size   = nCellQuadraturePoints(cellId) * d_numberComponents;
      size_type offset = cellStartId(cellId);
      d_storage.copyFrom<memorySpaceSrc>(values, size, 0, offset);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    template <utils::MemorySpace memorySpaceSrc>
    void
    QuadratureValuesContainer<ValueType, memorySpace>::setCellQuadValues(
      const size_type  cellId,
      const size_type  quadId,
      const ValueType *values)
    {
      size_type size   = d_numberComponents;
      size_type offset = cellStartId(cellId) + quadId * d_numberComponents;
      d_storage.copyFrom<memorySpaceSrc>(values, size, 0, offset);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    template <utils::MemorySpace memorySpaceDst>
    void
    QuadratureValuesContainer<ValueType, memorySpace>::getCellValues(
      const size_type cellId,
      ValueType *     values) const
    {
      size_type size   = nCellQuadraturePoints(cellId) * d_numberComponents;
      size_type offset = cellStartId(cellId);
      d_storage.copyTo<memorySpaceDst>(values, size, offset, 0);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    template <utils::MemorySpace memorySpaceDst>
    void
    QuadratureValuesContainer<ValueType, memorySpace>::getCellQuadValues(
      const size_type cellId,
      const size_type quadId,
      ValueType *     values) const
    {
      size_type size   = d_numberComponents;
      size_type offset = cellStartId(cellId) + quadId * d_numberComponents;
      d_storage.copyTo<memorySpaceDst>(values, size, offset, 0);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    const QuadratureRuleContainer &
    QuadratureValuesContainer<ValueType,
                              memorySpace>::getQuadratureRuleContainer() const
    {
      return *d_quadratureRuleContainer;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    QuadratureValuesContainer<ValueType, memorySpace>::getNumberComponents()
      const
    {
      return d_numberComponents;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    QuadratureValuesContainer<ValueType, memorySpace>::nCells() const
    {
      return d_quadratureRuleContainer->nCells();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    QuadratureValuesContainer<ValueType, memorySpace>::nQuadraturePoints() const
    {
      return d_quadratureRuleContainer->nQuadraturePoints();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    QuadratureValuesContainer<ValueType, memorySpace>::nEntries() const
    {
      return d_numberComponents *
             (d_quadratureRuleContainer->nQuadraturePoints());
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    QuadratureValuesContainer<ValueType, memorySpace>::nCellQuadraturePoints(
      const size_type cellId) const
    {
      return d_quadratureRuleContainer->nCellQuadraturePoints(cellId);
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    QuadratureValuesContainer<ValueType, memorySpace>::nCellEntries(
      const size_type cellId) const
    {
      return d_numberComponents *
             (d_quadratureRuleContainer->nCellQuadraturePoints(cellId));
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    size_type
    QuadratureValuesContainer<ValueType, memorySpace>::cellStartId(
      const size_type cellId) const
    {
      return d_numberComponents *
             (d_quadratureRuleContainer->getCellQuadStartId(cellId));
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    const typename QuadratureValuesContainer<ValueType,
                                             memorySpace>::SizeTypeVector &
    QuadratureValuesContainer<ValueType, memorySpace>::getCellStartIds() const
    {
      return d_cellStartIds;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    const typename QuadratureValuesContainer<ValueType,
                                             memorySpace>::SizeTypeVector &
    QuadratureValuesContainer<ValueType, memorySpace>::getNumberCellEntries()
      const
    {
      return d_numCellEntries;
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename QuadratureValuesContainer<ValueType, memorySpace>::iterator
    QuadratureValuesContainer<ValueType, memorySpace>::begin()
    {
      return d_storage.begin();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename QuadratureValuesContainer<ValueType, memorySpace>::const_iterator
    QuadratureValuesContainer<ValueType, memorySpace>::begin() const
    {
      return d_storage.begin();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename QuadratureValuesContainer<ValueType, memorySpace>::iterator
    QuadratureValuesContainer<ValueType, memorySpace>::end()
    {
      return d_storage.end();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename QuadratureValuesContainer<ValueType, memorySpace>::const_iterator
    QuadratureValuesContainer<ValueType, memorySpace>::end() const
    {
      return d_storage.end();
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename QuadratureValuesContainer<ValueType, memorySpace>::iterator
    QuadratureValuesContainer<ValueType, memorySpace>::begin(
      const size_type cellId)
    {
      return (d_storage.begin() + cellStartId(cellId));
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename QuadratureValuesContainer<ValueType, memorySpace>::const_iterator
    QuadratureValuesContainer<ValueType, memorySpace>::begin(
      const size_type cellId) const
    {
      return (d_storage.begin() + cellStartId(cellId));
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename QuadratureValuesContainer<ValueType, memorySpace>::iterator
    QuadratureValuesContainer<ValueType, memorySpace>::end(
      const size_type cellId)
    {
      return (d_storage.begin() + cellStartId(cellId) + nCellEntries(cellId));
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    typename QuadratureValuesContainer<ValueType, memorySpace>::const_iterator
    QuadratureValuesContainer<ValueType, memorySpace>::end(
      const size_type cellId) const
    {
      return (d_storage.begin() + cellStartId(cellId) + nCellEntries(cellId));
    }


    //
    // Helper functions
    //

    //
    // w = a*u + b*v
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    add(ValueType                                                a,
        const QuadratureValuesContainer<ValueType, memorySpace> &u,
        ValueType                                                b,
        const QuadratureValuesContainer<ValueType, memorySpace> &v,
        QuadratureValuesContainer<ValueType, memorySpace> &      w,
        const linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
      utils::throwException<utils::LengthError>(
        u.nEntries() == v.nEntries(),
        "Mismatch in sizes of the two input QuadratureValuesContainer passed"
        " for addition");
      utils::throwException<utils::LengthError>(
        u.nEntries() == w.nEntries(),
        "Mismatch in sizes of input and output QuadratureValuesContainer passed"
        " for addition");
      linearAlgebra::blasLapack::axpby(u.nEntries(),
                                       a,
                                       u.begin(),
                                       b,
                                       v.begin(),
                                       w.begin(),
                                       linAlgOpContext.getBlasQueue());
    }

    //
    // u = a*u + b*v
    //
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    QuadratureValuesContainer<ValueType, memorySpace>
    add(ValueType                                                a,
        const QuadratureValuesContainer<ValueType, memorySpace> &u,
        ValueType                                                b,
        const QuadratureValuesContainer<ValueType, memorySpace> &v,
        const linearAlgebra::LinAlgOpContext<memorySpace> &linAlgOpContext)
    {
      utils::throwException<utils::LengthError>(
        u.nEntries() == v.nEntries(),
        "Mismatch in sizes of the two input QuadratureValuesContainer passed"
        " for addition");
      QuadratureValuesContainer<ValueType, memorySpace> w(u);
      linearAlgebra::blasLapack::axpby(u.nEntries(),
                                       a,
                                       u.begin(),
                                       b,
                                       v.begin(),
                                       w.begin(),
                                       linAlgOpContext.getBlasQueue());
      return w;
    }

    // FIXME: Uncomment the scale functions after ascale has been implemented in
    // linearAlgebra:blasLapack

    //    //
    //    // w = a*u
    //    //
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //    void
    //    scale(ValueType                                                a,
    //          const QuadratureValuesContainer<ValueType, memorySpace> &u,
    //          QuadratureValuesContainer<ValueType, memorySpace> &      w,
    //          const linearAlgebra::LinAlgOpContext<memorySpace>
    //          &linAlgOpContext)
    //    {
    //      w = u;
    //      linearAlgebra::blasLapack::ascale(w.nEntries(),
    //                                        a,
    //                                        w.begin(),
    //                                        linAlgOpContext.getBlasQueue());
    //    }
    //
    //    //
    //    // u = a*u
    //    //
    //    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    //    void
    //    scale(ValueType                                          a,
    //          QuadratureValuesContainer<ValueType, memorySpace> &u,
    //          const linearAlgebra::LinAlgOpContext<memorySpace> &
    //          linAlgOpContext)
    //    {
    //      linearAlgebra::blasLapack::ascale(u.nEntries(),
    //                                        a,
    //                                        u.begin(),
    //                                        linAlgOpContext.getBlasQueue());
    //    }
  } // namespace quadrature
} // namespace dftefe
