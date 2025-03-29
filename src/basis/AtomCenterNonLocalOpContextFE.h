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
 
 namespace dftefe
 {
   namespace linearAlgebra
   {
     template <typename ValueTypeOperator,
               typename ValueTypeOperand,
               utils::MemorySpace memorySpace,
               size_type          dim>
     class AtomCenterNonLocalOpContextFE ::
       public linearAlgebra::OperatorContext<ValueTypeOperator, ValueTypeOperand, memorySpace>
     {
      public:
        using ValueType =
          blasLapack::scalar_type<ValueTypeOperator, ValueTypeOperand>;
        using RealType = blasLapack::real_type<ValueType>;

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
          const std::string                fieldName,
          const size_type maxCellBlock,
          const size_type maxFieldBlock,
          std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                                    linAlgOpContext,
          const utils::mpi::MPIComm &comm);
  
        /**
          *@brief Default Destructor
          *
          */
        ~AtomCenterNonLocalOpContextFE() = default;
  
        void
        apply(MultiVector<ValueTypeOperand, memorySpace> &X,
              MultiVector<ValueTypeUnion, memorySpace> & Y) const override;

      private:
        
        // // size is  proj x nProj accumulated overcells
        // utils::MemoryStorage<ValueTypeBasisData, utils::memorySpace::HOST> 
        //   d_projectorQuadStorage; // cell->quad->proj 
  
        // size is proj x nDofs accumulated over cells
        utils::MemoryStorage<ValueTypeBasisData, memorySpace> 
          d_cellWiseC; // cell->dofs->kpt->proj
        
        // size is localProjNum(numDofs partiitoned) x numVec(numComp)
        linearAlgebra::MultiVector<> d_CX;

        // size id localProjNum x localProjNum
        utils::MemoryStorage<ValueTypeBasisData, memorySpace> d_V;

        //num proj in cells and max proj in cell
        std::vector<size_type> d_numProjInCells;
        size_type d_maxProjInCell;
     };
   } // end of namespace linearAlgebra
 } // end of namespace dftefe
 #endif // dftefeAtomCenterNonLocalOpContextFE_h
 