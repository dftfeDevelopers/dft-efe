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

#ifndef dftefeFEBasisHandlerDealii_h
#define dftefeFEBasisHandlerDealii_h

#ifdef DFTEFE_WITH_MPI
#  include <mpi.h>
#endif // DFTEFE_WITH_MPI

#include <basis/FEBasisHandler.h>
#include <basis/BasisManager.h>
#include <basis/FEBasisManagerDealii.h>
#include <basis/Constraints.h>
#include <utils/MPIPatternP2P.h>
#include <memory>
namespace dftefe
{
  namespace basis
  {
    /**
     * @brief An abstract class to encapsulate the partitioning
     * of a finite element basis across multiple processors
     */
    template <typename ValueType,
              dftefe::utils::MemorySpace memorySpace,
              size_type                  dim>
    class FEBasisHandlerDealii : public FEBasisHandler<memorySpace>
    {
      //
      // typedefs
      //
    public:
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;
      using GlobalSizeTypeVector =
        utils::MemoryStorage<global_size_type, memorySpace>;
      using LocalIndexIter = SizeTypeVector::iterator;
      using const_LocalIndexIter = SizeTypeVector::const_iterator;
      using GlobalIndexIter = GlobalSizeTypeVector::iterator;
      using const_GlobalIndexIter = GlobalSizeTypeVector::const_iterator;

    public:
#ifdef DFTEFE_WITH_MPI
      FEBasisHandlerDealii(
        std::shared_ptr<const BasisManager> basisManager,
        std::map<std::string, std::shared_ptr<const Constraints>>
                        constraintsMap,
        const MPI_Comm &mpiComm);
      reinit(std::shared_ptr<const BasisManager> basisManager,
             std::map<std::string, std::shared_ptr<const Constraints>>
                             constraintsMap,
             const MPI_Comm &mpiComm);
#else
      FEBasisHandlerDealii(
        std::shared_ptr<const BasisManager> basisManager,
        std::map<std::string, std::shared_ptr<const Constraints>>
          constraintsMap);
      reinit(std::shared_ptr<const BasisManager> basisManager,
             std::map<std::string, std::shared_ptr<const Constraints>>
               constraintsMap);
#endif // DFTEFE_WITH_MPI

      ~FEBasisHandlerDealii() = default;

      std::pair<global_size_type, global_size_type>
      getLocallyOwnedRange() const override;

      size_type
      nLocallyOwned() const override;

      const GlobalSizeTypeVector &
      getGhostIndices(const std::string constraintsName) const override;

      size_type
      nLocal(const std::string constraintsName) const override;


      size_type
      nGhost(const std::string constraintsName) const override;

      bool
      inLocallyOwnedRange(const global_size_type globalId,
                          const std::string constraintsName) const override;

      bool
      isGhostEntry(const global_size_type ghostId,
                   const std::string      constraintsName) const override;

      size_type
      globalToLocalIndex(const global_size_type globalId,
                         const std::string      constraintsName) const override;

      global_size_type
      localToGlobalIndex(const size_type   localId,
                         const std::string constraintsName) const override;

      //
      // FE specific functions
      //
      size_type
	numLocallyOwnedCellDofs(const size_type cellId) const override;
      
      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsBegin(const std::string constraintsName) const override;

      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsBegin(const size_type   cellId,
	  const std::string constraintsName) const override;
      
      const_GlobalIndexIter
      locallyOwnedCellGlobalDofIdsEnd(const size_type   cellId,
	  const std::string constraintsName) const override;
      
      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsBegin(const std::string constraintsName) const override;
      
      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsBegin(const size_type   cellId,
	  const std::string constraintsName) const override;
      
      const_LocalIndexIter
      locallyOwnedCellLocalDofIdsEnd(const size_type   cellId,
	  const std::string constraintsName) const override;

      //
      // dealii specific functions
      //

    private:
      std::shared_ptr<const FEBasisManagerDealii<dim>> d_feBMDealii;
      std::map<std::string,
               std::shared_ptr<const FEBasisConstraintsDealii<dim, ValueType>>>
        d_feConstraintsDealiiMap;
#ifdef DFTEFE_WITH_MPI
      MPI_Comm d_mpiComm;
#endif // DFTEFE_WITH_MPI
      std::pair<global_size_type, global_size_type> d_locallyOwnedRange;
      SizeTypeVector                                d_locallyOwnedCellStartIds;
      GlobalSizeTypeVector d_locallyOwnedCellGlobalIndices;
      std::vector<size_type> d_numLocallyOwnedCellDofs;

      // constraints dependent data
      std::map<std::string, std::shared_ptr<GlobalSizeTypeVector>>
        d_ghostIndicesMap;
      std::map < std::string,
        std::shared_ptr<utils::MPIPatternP2P<memorySpace>> d_mpiPatternP2PMap;
      std::map<std::string, std::shared_ptr<SizeTypeVector>>
        d_locallyOwnedCellLocalIndicesMap;
    };

  } // end of namespace basis
} // end of namespace dftefe
#include <basis/FEBasisHandlerDealii.t.cpp>
#endif // dftefeFEBasisHandlerDealii_h
