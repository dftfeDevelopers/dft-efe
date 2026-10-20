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

#ifndef dftefeBasisScratch_h
#define dftefeBasisScratch_h

#include <memory>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <utils/Exceptions.h>
#include <linearAlgebra/LinAlgOpContext.h>

namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Scratch buffers shared by objects that never run at the same
     * time, e.g. the per k-point EFE basis data storages and the
     * FEBasisOperations built on them, so that the scratch is held once
     * instead of once per object.
     *
     * Every buffer starts empty and is grown by its users; none of them
     * holds data that has to survive from one call to the next, except the
     * cell-range basis data of FEBasisOperations, whose cache is valid only
     * for the object recorded as its owner. A FEBasisOperations uses these
     * only when its union value type is the basis value type.
     *
     * Two groups of buffers, one for the basis data storages and one for the
     * FEBasisOperations, are each claimed by one object at a time
     * (acquire/release, re-entrant for that object) and must be used on one
     * LinAlgOpContext, so that work queued on the device stays ordered.
     */
    template <typename ValueTypeBasisData, utils::MemorySpace memorySpace>
    class BasisScratch
    {
    public:
      using Storage = utils::MemoryStorage<ValueTypeBasisData, memorySpace>;

      BasisScratch()
        : d_enrichValues(std::make_shared<Storage>(0))
        , d_enrichGradients(std::make_shared<Storage>(0))
        , d_classValues(std::make_shared<Storage>(0))
        , d_classGradients(std::make_shared<Storage>(0))
        , d_fieldCellValues(std::make_shared<Storage>(0))
        , d_fxJxWxNBlock(std::make_shared<Storage>(0))
        , d_JxWxNBlock(std::make_shared<Storage>(0))
        , d_cellBasisValues(std::make_shared<Storage>(0))
        , d_cellBasisGradients(std::make_shared<Storage>(0))
        , d_cellBasisValuesOwner(nullptr)
        , d_cellBasisGradientsOwner(nullptr)
        , d_storageUser(nullptr)
        , d_storageUseDepth(0)
        , d_operationsUser(nullptr)
        , d_operationsUseDepth(0)
        , d_linAlgOpContext(nullptr)
      {}

      ~BasisScratch() = default;

      // --- one user at a time per group ---

      void
      acquireStorageScratch(const void *user,
                            const linearAlgebra::LinAlgOpContext<memorySpace>
                              &linAlgOpContext) const
      {
        checkLinAlgOpContext(linAlgOpContext);
        utils::throwException(
          d_storageUser == nullptr || d_storageUser == user,
          "BasisScratch: the basis data storage buffers are in use by another "
          "storage.");
        d_storageUser = user;
        d_storageUseDepth += 1;
      }

      void
      releaseStorageScratch() const
      {
        d_storageUseDepth -= 1;
        if (d_storageUseDepth == 0)
          d_storageUser = nullptr;
      }

      void
      acquireOperationsScratch(const void *user,
                               const linearAlgebra::LinAlgOpContext<memorySpace>
                                 &linAlgOpContext) const
      {
        checkLinAlgOpContext(linAlgOpContext);
        utils::throwException(
          d_operationsUser == nullptr || d_operationsUser == user,
          "BasisScratch: the FEBasisOperations buffers are in use by another "
          "FEBasisOperations.");
        d_operationsUser = user;
        d_operationsUseDepth += 1;
      }

      void
      releaseOperationsScratch() const
      {
        d_operationsUseDepth -= 1;
        if (d_operationsUseDepth == 0)
          d_operationsUser = nullptr;
      }

      // the first context seen is the only one allowed: a buffer written by
      // work on one device stream and read by work on another would race
      void
      checkLinAlgOpContext(const linearAlgebra::LinAlgOpContext<memorySpace>
                             &linAlgOpContext) const
      {
        if (d_linAlgOpContext == nullptr)
          d_linAlgOpContext = &linAlgOpContext;
        utils::throwException(
          d_linAlgOpContext == &linAlgOpContext,
          "BasisScratch: every user must run on the same LinAlgOpContext.");
      }

      // --- basis data storage: one cell block of enrichment and classical
      // values / gradients ---

      std::shared_ptr<Storage>
      getEnrichValues() const
      {
        return d_enrichValues;
      }

      std::shared_ptr<Storage>
      getEnrichGradients() const
      {
        return d_enrichGradients;
      }

      std::shared_ptr<Storage>
      getClassValues() const
      {
        return d_classValues;
      }

      std::shared_ptr<Storage>
      getClassGradients() const
      {
        return d_classGradients;
      }

      // --- FEBasisOperations ---

      std::shared_ptr<Storage>
      getFieldCellValues() const
      {
        return d_fieldCellValues;
      }

      std::shared_ptr<Storage>
      getFxJxWxNBlock() const
      {
        return d_fxJxWxNBlock;
      }

      std::shared_ptr<Storage>
      getJxWxNBlock() const
      {
        return d_JxWxNBlock;
      }

      std::shared_ptr<Storage>
      getCellBasisValues() const
      {
        return d_cellBasisValues;
      }

      std::shared_ptr<Storage>
      getCellBasisGradients() const
      {
        return d_cellBasisGradients;
      }

      // the object whose cached cell range is in getCellBasisValues() /
      // getCellBasisGradients(); any other object's cache is stale
      void
      setCellBasisValuesOwner(const void *owner) const
      {
        d_cellBasisValuesOwner = owner;
      }

      const void *
      getCellBasisValuesOwner() const
      {
        return d_cellBasisValuesOwner;
      }

      void
      setCellBasisGradientsOwner(const void *owner) const
      {
        d_cellBasisGradientsOwner = owner;
      }

      const void *
      getCellBasisGradientsOwner() const
      {
        return d_cellBasisGradientsOwner;
      }

    private:
      std::shared_ptr<Storage> d_enrichValues;
      std::shared_ptr<Storage> d_enrichGradients;
      std::shared_ptr<Storage> d_classValues;
      std::shared_ptr<Storage> d_classGradients;
      std::shared_ptr<Storage> d_fieldCellValues;
      std::shared_ptr<Storage> d_fxJxWxNBlock;
      std::shared_ptr<Storage> d_JxWxNBlock;
      std::shared_ptr<Storage> d_cellBasisValues;
      std::shared_ptr<Storage> d_cellBasisGradients;
      mutable const void *     d_cellBasisValuesOwner;
      mutable const void *     d_cellBasisGradientsOwner;
      mutable const void *     d_storageUser;
      mutable size_type        d_storageUseDepth;
      mutable const void *     d_operationsUser;
      mutable size_type        d_operationsUseDepth;
      mutable const linearAlgebra::LinAlgOpContext<memorySpace>
        *d_linAlgOpContext;
    }; // end of class BasisScratch

  } // end of namespace basis
} // end of namespace dftefe

#endif // dftefeBasisScratch_h
