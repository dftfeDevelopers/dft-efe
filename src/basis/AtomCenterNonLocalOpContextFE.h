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

#ifndef dftefeAtomCenterNonLocalOpContextFE_h
#define dftefeAtomCenterNonLocalOpContextFE_h

#include <utils/MemorySpaceType.h>
#include <basis/FEBasisManager.h>
#include <basis/FEBasisOperations.h>
#include <basis/FEBasisDofHandler.h>
#include <linearAlgebra/OperatorContext.h>
#include <atoms/AtomSphericalDataContainer.h>
#include <basis/EnrichmentIdsPartition.h>
#include <vector>
#include <memory>

namespace dftefe
{
  namespace basis
  {
    template <typename ValueTypeOperator,
              typename ValueTypeOperand,
              utils::MemorySpace memorySpace,
              size_type          dim>
    class AtomCenterNonLocalOpContextFE
      : public linearAlgebra::
          OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
    {
    public:
      using ValueType =
        linearAlgebra::blasLapack::scalar_type<ValueTypeOperator,
                                               ValueTypeOperand>;
      using RealType = linearAlgebra::blasLapack::real_type<ValueType>;

    public:
      /**
       * @brief Constructor Class does : \sum_atoms \sum_lpm CVC^T X = Y where V = coupling matrix
       * C_lpm,j = \integral_\omega \beta_lp Y_lm N_j
       */
      AtomCenterNonLocalOpContextFE(
        const FEBasisManager<ValueTypeOperand,
                             ValueTypeOperator,
                             memorySpace,
                             dim> &feBasisManager,
        const basis::FEBasisDataStorage<ValueTypeOperator, memorySpace>
          &feBasisDataStorage,
        std::shared_ptr<const atoms::AtomSphericalDataContainer>
                                         atomSphericalDataContainer,
        const double                     atomPartitionTolerance,
        const std::vector<std::string> & atomSymbolVec,
        const std::vector<utils::Point> &atomCoordinatesVec,
        const size_type                  maxCellBlock,
        const size_type                  maxFieldBlock,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                   linAlgOpContext,
        const utils::mpi::MPIComm &comm);

      /**
       *@brief Default Destructor
       *
       */
      ~AtomCenterNonLocalOpContextFE() = default;

      void
      apply(linearAlgebra::MultiVector<ValueTypeOperand, memorySpace> &X,
            linearAlgebra::MultiVector<ValueType, memorySpace> &       Y,
            bool updateGhostX = false,
            bool updateGhostY = false) const override;

    private:
      // gets the projector values with quad pts as fastest index
      // and proj Id as second index in a cell. Assumption:
      //  m values ar consecutive for all l,p pairs.
      std::vector<double>
      getProjectorValues(const size_type                          cellId,
                         const std::vector<dftefe::utils::Point> &points) const;

      // // size is  proj x nProj accumulated overcells
      // utils::MemoryStorage<ValueTypeOperator, utils::memorySpace::HOST>
      //   d_projectorQuadStorage; // cell->quad->proj

      // size is proj x nDofs accumulated over cells
      utils::MemoryStorage<ValueTypeOperator, memorySpace>
        d_cellWiseC; // cell->dofs->kpt->proj

      // size is localProjNum(numDofs partiitoned) x numVec(numComp)
      // mutable so that it can be reinited inside apply if block size changes
      mutable std::shared_ptr<linearAlgebra::MultiVector<ValueType, memorySpace>> d_CX;

      // size id localProjNum x localProjNum
      utils::MemoryStorage<ValueTypeOperator, memorySpace> d_V;

      dftefe::utils::MemoryStorage<size_type, memorySpace>
        d_locallyOwnedCellLocalProjectorIds;

      // num proj in cells and max proj in cell
      std::vector<size_type> d_numProjsInCells;
      size_type              d_maxProjInCell;
      size_type              d_totProjInProc;

      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
        d_mpiPatternP2PProj;

      const FEBasisManager<ValueTypeOperand,
                           ValueTypeOperator,
                           memorySpace,
                           dim> *d_feBasisManager;
      const size_type            d_maxCellBlock;
      const size_type            d_maxWaveFnBatch;
      std::shared_ptr<const EnrichmentIdsPartition<dim>>
        d_projectorIdsPartition;
      std::vector<std::vector<global_size_type>>
        d_overlappingProjectorIdsInCells;
      const std::shared_ptr<const atoms::AtomSphericalDataContainer>
        d_atomSphericalDataContainer;

      std::unordered_map<std::string, std::vector<double>>
                                           d_atomSymbolToCouplingConstVecMap;
      std::unordered_map<std::string, int> d_atomSymbolToNumProjMap;
      std::unordered_map<std::string, std::vector<int>> d_atomSymbolToBetaIndexVecMap;      

      const std::vector<std::string> & d_atomSymbolVec;
      const std::vector<utils::Point> &d_atomCoordinatesVec;
      const std::string                d_fieldNameProjector;
      std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
        d_linAlgOpContext;
    };
  } // namespace basis
} // end of namespace dftefe
#include <basis/AtomCenterNonLocalOpContextFE.t.cpp>
#endif // dftefeAtomCenterNonLocalOpContextFE_h
