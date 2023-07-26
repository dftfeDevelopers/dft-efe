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

#ifndef dftefeEFEBasisHandlerDealii_h
#define dftefeEFEBasisHandlerDealii_h

#include <basis/EFEBasisHandler.h>
#include <basis/BasisManager.h>
#include <basis/EFEBasisManagerDealii.h>
#include <basis/Constraints.h>
#include <basis/EFEConstraintsDealii.h>
#include <utils/MPIPatternP2P.h>
#include <memory>
#include <map>
#include <deal.II/matrix_free/matrix_free.h>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class to encapsulate the partitioning
     * of a finite element basis across multiple processors
     */
    template <typename ValueTypeBasisCoeff,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    class EFEBasisHandlerDealii
      : public EFEBasisHandler<ValueTypeBasisCoeff, memorySpace, dim>
    {
      //
      // typedefs
      //
    public:
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;
      using GlobalSizeTypeVector =
        utils::MemoryStorage<global_size_type, memorySpace>;
      using LocalIndexIter       = typename SizeTypeVector::iterator;
      using const_LocalIndexIter = typename SizeTypeVector::const_iterator;
      using GlobalIndexIter      = typename GlobalSizeTypeVector::iterator;
      using const_GlobalIndexIter =
        typename GlobalSizeTypeVector::const_iterator;

    public:
      EFEBasisHandlerDealii(
        std::shared_ptr<const BasisManager> basisManager,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>
                                   constraints,
        const utils::mpi::MPIComm &mpiComm);
      void
      reinit(
        std::shared_ptr<const BasisManager> basisManager,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>
                                   constraints,
        const utils::mpi::MPIComm &mpiComm);

      EFEBasisHandlerDealii(
        std::shared_ptr<const BasisManager> basisManager,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>
                                  constraints);
      void
      reinit(
        std::shared_ptr<const BasisManager> basisManager,
        std::shared_ptr<const Constraints<ValueTypeBasisCoeff, memorySpace>>
                                  constraints);

      ~EFEBasisHandlerDealii() = default;

      const BasisManager &
      getBasisManager() const override;

      bool
      isDistributed() const override;

      const Constraints<ValueTypeBasisCoeff, memorySpace> &
      getConstraints() const override;

      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
      getMPIPatternP2P() const override;

      std::vector<std::pair<global_size_type, global_size_type>>
      getLocallyOwnedRanges() const override;

      size_type
      nLocallyOwned() const override;

      const GlobalSizeTypeVector &
      getGhostIndices() const override;

      size_type
      nLocal() const override;


      size_type
      nGhost() const override;

      std::pair<bool, size_type>
      inLocallyOwnedRanges(const global_size_type globalId) const override;

      std::pair<bool, size_type>
      isGhostEntry(const global_size_type ghostId) const override;

      size_type
      globalToLocalIndex(const global_size_type globalId) const override;

      global_size_type
      localToGlobalIndex(const size_type   localId) const override;

      void
      getBasisCenters(const size_type       localId,
                      dftefe::utils::Point &basisCenter) const override;

      //
      // FE specific functions
      //
      size_type
      nLocallyOwnedCells() const override;

      size_type
      nLocallyOwnedCellDofs(const size_type cellId) const override;

      size_type
      nCumulativeLocallyOwnedCellDofs() const override;

      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsBegin() const override;

      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsBegin(const size_type   cellId) const override;

      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsEnd(const size_type   cellId) const override;

      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsBegin() const override;

      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsBegin(const size_type   cellId) const override;

      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsEnd(const size_type   cellId) const override;

      void
      getCellDofsLocalIds(
        const size_type         cellId,
        std::vector<size_type> &vecLocalNodeId) const override;

      //
      // dealii specific functions
      //

    private:
      std::shared_ptr<const EFEBasisManagerDealii<dim>> d_efeBMDealii;
      std::shared_ptr<
          const EFEConstraintsDealii<ValueTypeBasisCoeff, memorySpace, dim>>
                          d_efeConstraintsDealiiOpt;
      utils::mpi::MPIComm d_mpiComm;
      bool                d_isDistributed;
      std::vector<std::pair<global_size_type, global_size_type>>
                             d_locallyOwnedRanges; 
      std::vector<size_type> d_locallyOwnedCellStartIds;
      GlobalSizeTypeVector   d_locallyOwnedCellGlobalIndices;
      std::vector<size_type> d_numLocallyOwnedCellDofs;

      // constraints dependent data
      std::shared_ptr<GlobalSizeTypeVector> d_ghostIndices;
      std::shared_ptr<utils::mpi::MPIPatternP2P<memorySpace>> d_mpiPatternP2P;
      std::shared_ptr<SizeTypeVector> d_locallyOwnedCellLocalIndices;
      std::map<global_size_type, utils::Point> d_supportPoints;
    };

  } // end of namespace basis
} // end of namespace dftefe
#include <basis/EFEBasisHandlerDealii.t.cpp>
#endif // dftefeEFEBasisHandlerDealii_h
