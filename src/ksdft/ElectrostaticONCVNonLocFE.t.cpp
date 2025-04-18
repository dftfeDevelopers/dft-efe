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

 #include <ksdft/Defaults.h>
 #include <basis/FEBasisDofHandler.h>
 #include <utils/ConditionalOStream.h>
 namespace dftefe
 {
   namespace ksdft
   {
     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::
       ElectrostaticONCVNonLocFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> & atomSymbolVec,
        const std::map<std::string, std::string> &atomSymbolToPSPFilename,
        const std::vector<double> &      smearedChargeRadius,
         const quadrature::QuadratureValuesContainer<RealType, memorySpace>
           &                                               electronChargeDensity,
         std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>> feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,                                                     
         std::shared_ptr<
           const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
           feBDTotalChargeStiffnessMatrix,
         std::shared_ptr<
           const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
           feBDNuclearChargeRhs,
         std::shared_ptr<
           const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
           feBDElectronicChargeRhs,
         std::shared_ptr<
           const basis::FEBasisDataStorage<ValueTypeWaveFnBasis,
                                           memorySpace>> feBDHamiltonian,
         std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                         linAlgOpContext,
         const size_type maxCellBlock,
         const size_type maxWaveFnBlock,
         const bool      useDealiiMatrixFreePoissonSolve)
       : d_linAlgOpContext(linAlgOpContext)
       , d_numComponents(1)
       , d_rootCout(std::cout)
       , d_mpiComm(feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator())
       , d_atomSymbolVec(atomSymbolVec)
       , d_maxCellBlock(maxCellBlock)
       , d_maxWaveFnBlock(maxWaveFnBlock)
     {
       int rank;
       utils::mpi::MPICommRank(d_mpiComm, &rank);
 
       d_rootCout.setCondition(rank == 0);

      const std::vector<std::string> metadataNames = atoms::AtomSphDataPSPDefaults::METADATANAMES;
      std::vector<std::string> fieldNamesPSP = {"vlocal", "rhoatom"};

      d_atomSphericalDataContainerPSP = 
        std::make_shared<const atoms::AtomSphericalDataContainer>(
                                  atoms::AtomSphericalDataType::PSEUDOPOTENTIAL,
                                  atomSymbolToPSPFilename,
                                  fieldNamesPSP, metadataNames);

      bool isNonLocPSP = false, isNlcc = false;                           
      for(int atomSymbolId = 0 ; atomSymbolId < atomSymbolVec.size() ; atomSymbolId++) 
      {   
        int numProj = 0;         
        utils::stringOps::strToInt(d_atomSphericalDataContainerPSP->getMetadata(atomSymbolVec[atomSymbolId], "number_of_proj"),
                                                       numProj);               
        if(numProj > 0)
        {
          isNonLocPSP = true;
          bool coreCorrect = false;
          utils::stringOps::strToBool(d_atomSphericalDataContainerPSP->getMetadata(atomSymbolVec[atomSymbolId] ,"core_correction"),
                                                        coreCorrect);       
          if(coreCorrect)
          {
            isNlcc = true;
          }
          else if(isNlcc && !coreCorrect)
          {
            utils::throwException(false, "Some atoms do not have NLCC parts in UPF files while some have.");
          }
        }
        else if(!(numProj > 0) && isNlcc)
        {
          utils::throwException(false, "Some atoms do not have NONLOCAL parts in UPF files while some have.");
        }
      }

      if(isNonLocPSP)
      {
        d_atomSphericalDataContainerPSP->addFieldName("beta");
        if(isNlcc)
        {
          d_atomSphericalDataContainerPSP->addFieldName("nlcc");
        }
      }

      d_atomNonLocOpContext =
        std::make_shared<const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                      ValueTypeWaveFnCoeff,
                                                      memorySpace,
                                                      dim>>(
                                                          feBMWaveFn,
                                                          feBDHamiltonian,
                                                          d_atomSphericalDataContainerPSP,
                                                          ElectroHamiltonianDefaults::ATOM_PARTITION_TOL_BETA ,
                                                          atomSymbolVec,
                                                          atomCoordinates,
                                                          maxCellBlock,
                                                          maxWaveFnBlock,
                                                          linAlgOpContext,
                                                          d_mpiComm);

      d_atomVLocFunction =     
        std::make_shared<const atoms::AtomSevereFunction<dim>>(
                                d_atomSphericalDataContainerPSP,
                                atomSymbolVec,
                                atomCoordinates,
                                "vlocal",
                                0,
                                1);
 
       d_electrostaticLocal = 
        std::make_shared<const ElectrostaticLocalFE<ValueTypeBasisData,
                                              ValueTypeBasisCoeff,
                                              ValueTypeWaveFnBasis,
                                              memorySpace,
                                              dim>>(atomCoordinates,
                                                    atomCharges,
                                                    smearedChargeRadius,
                                                    electronChargeDensity,
                                                    feBMTotalCharge,
                                                    feBDTotalChargeStiffnessMatrix,
                                                    feBDNuclearChargeRhs,
                                                    feBDElectronicChargeRhs,
                                                    feBDHamiltonian,
                                                    *d_atomVLocFunction,
                                                    linAlgOpContext,
                                                    maxCellBlock,
                                                    useDealiiMatrixFreePoissonSolve);
     }

     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::
       ElectrostaticONCVNonLocFE(
        const std::vector<utils::Point> &atomCoordinates,
        const std::vector<double> &      atomCharges,
        const std::vector<std::string> & atomSymbolVec,
        const std::map<std::string, std::string> &atomSymbolToPSPFilename,        
        const std::vector<double> &      smearedChargeRadius,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &atomicElectronChargeDensity,
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                    memorySpace>
          &atomicTotalElecPotNuclearQuad,
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                    memorySpace>
          &atomicTotalElecPotElectronicQuad,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>>
          feBMTotalCharge, // will be same as bc of totalCharge -
                           // atomicTotalCharge
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,                              
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis,
                                          memorySpace>> feBDHamiltonian,
        std::shared_ptr<linearAlgebra::LinAlgOpContext<memorySpace>>
                        linAlgOpContext,
        const size_type maxCellBlock,
        const size_type maxWaveFnBlock,        
        const bool      useDealiiMatrixFreePoissonSolve)
       : d_linAlgOpContext(linAlgOpContext)
       , d_numComponents(1)
       , d_rootCout(std::cout)
       , d_mpiComm(feBMTotalCharge->getMPIPatternP2P()->mpiCommunicator())
       , d_atomSymbolVec(atomSymbolVec)
       , d_maxCellBlock(maxCellBlock)
       , d_maxWaveFnBlock(maxWaveFnBlock)
     {
       int rank;
       utils::mpi::MPICommRank(d_mpiComm, &rank);
 
       d_rootCout.setCondition(rank == 0);

      const std::vector<std::string> metadataNames = atoms::AtomSphDataPSPDefaults::METADATANAMES;
      std::vector<std::string> fieldNamesPSP = {"vlocal", "rhoatom"};

      d_atomSphericalDataContainerPSP = 
        std::make_shared<const atoms::AtomSphericalDataContainer>(
                                  atoms::AtomSphericalDataType::PSEUDOPOTENTIAL,
                                  atomSymbolToPSPFilename,
                                  fieldNamesPSP, metadataNames);

      bool isNonLocPSP = false, isNlcc = false;                           
      for(int atomSymbolId = 0 ; atomSymbolId < atomSymbolVec.size() ; atomSymbolId++) 
      {   
        int numProj = 0;         
        utils::stringOps::strToInt(d_atomSphericalDataContainerPSP->getMetadata(atomSymbolVec[atomSymbolId], "number_of_proj"),
                                                       numProj);               
        if(numProj > 0)
        {
          isNonLocPSP = true;
          bool coreCorrect = false;
          utils::stringOps::strToBool(d_atomSphericalDataContainerPSP->getMetadata(atomSymbolVec[atomSymbolId] ,"core_correction"),
                                                        coreCorrect);       
          if(coreCorrect)
          {
            isNlcc = true;
          }
          else if(isNlcc && !coreCorrect)
          {
            utils::throwException(false, "Some atoms do not have NLCC parts in UPF files while some have.");
          }
        }
        else if(!(numProj > 0) && isNlcc)
        {
          utils::throwException(false, "Some atoms do not have NONLOCAL parts in UPF files while some have.");
        }
      }

      if(isNonLocPSP)
      {
        d_atomSphericalDataContainerPSP->addFieldName("beta");
        if(isNlcc)
        {
          d_atomSphericalDataContainerPSP->addFieldName("nlcc");
        }
      }

      d_atomNonLocOpContext =
        std::make_shared<const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                      ValueTypeWaveFnCoeff,
                                                      memorySpace,
                                                      dim>>(
                                                          feBMWaveFn,
                                                          feBDHamiltonian,
                                                          d_atomSphericalDataContainerPSP,
                                                          ElectroHamiltonianDefaults::ATOM_PARTITION_TOL_BETA ,
                                                          atomSymbolVec,
                                                          atomCoordinates,
                                                          maxCellBlock,
                                                          maxWaveFnBlock,
                                                          linAlgOpContext,
                                                          d_mpiComm);

      d_atomVLocFunction =     
        std::make_shared<const atoms::AtomSevereFunction<dim>>(
                                d_atomSphericalDataContainerPSP,
                                atomSymbolVec,
                                atomCoordinates,
                                "vlocal",
                                0,
                                1);
 
       d_electrostaticLocal = 
        std::make_shared<const ElectrostaticLocalFE<ValueTypeBasisData,
                                              ValueTypeBasisCoeff,
                                              ValueTypeWaveFnBasis,
                                              memorySpace,
                                              dim>>(atomCoordinates,
                                                atomCharges,
                                                smearedChargeRadius,
                                                atomicElectronChargeDensity,
                                                atomicTotalElecPotNuclearQuad,
                                                atomicTotalElecPotElectronicQuad,
                                                feBMTotalCharge, // will be same as bc of totalCharge -
                                                                   // atomicTotalCharge
                                                feBDTotalChargeStiffnessMatrix,
                                                feBDNuclearChargeRhs,
                                                feBDElectronicChargeRhs,
                                                feBDHamiltonian,
                                                *d_atomVLocFunction,
                                                linAlgOpContext,
                                                maxCellBlock,
                                                useDealiiMatrixFreePoissonSolve);
     }

     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     void
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::
       reinitBasis(
         const std::vector<utils::Point>                        & atomCoordinates,
         std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                     ValueTypeBasisData,
                                                     memorySpace,
                                                     dim>> feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis,
                                          memorySpace>> feBDHamiltonian)
     {
        d_atomNonLocOpContext =
          std::make_shared<const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                        ValueTypeWaveFnCoeff,
                                                        memorySpace,
                                                        dim>>(
                                                            feBMWaveFn,
                                                            feBDHamiltonian,
                                                            d_atomSphericalDataContainerPSP,
                                                            ElectroHamiltonianDefaults::ATOM_PARTITION_TOL_BETA ,
                                                            d_atomSymbolVec,
                                                            atomCoordinates,
                                                            d_maxCellBlock,
                                                            d_maxWaveFnBlock,
                                                            d_linAlgOpContext,
                                                            d_mpiComm);

        d_atomVLocFunction = 
          std::make_shared<const atoms::AtomSevereFunction<dim>>(
                                  d_atomSphericalDataContainerPSP,
                                  d_atomSymbolVec,
                                  atomCoordinates,
                                  "vlocal",
                                  0,
                                  1);

        d_electrostaticLocal->reinitBasis(atomCoordinates,
                                          feBMTotalCharge,
                                          feBDTotalChargeStiffnessMatrix,
                                          feBDNuclearChargeRhs,
                                          feBDElectronicChargeRhs,
                                          feBDHamiltonian,
                                          *d_atomVLocFunction);
     }
 

     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     void
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::
       reinitBasis(
        const std::vector<utils::Point> &atomCoordinates,
        const quadrature::QuadratureValuesContainer<RealType, memorySpace>
          &atomicElectronChargeDensity,
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                    memorySpace>
          &atomicTotalElecPotNuclearQuad,
        const quadrature::QuadratureValuesContainer<ValueTypeBasisCoeff,
                                                    memorySpace>
          &atomicTotalElecPotElectronicQuad,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeBasisCoeff,
                                                    ValueTypeBasisData,
                                                    memorySpace,
                                                    dim>> feBMTotalCharge,
        std::shared_ptr<const basis::FEBasisManager<ValueTypeWaveFnCoeff,
                                                    ValueTypeWaveFnBasis,
                                                    memorySpace,
                                                    dim>> feBMWaveFn,                                                    
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDTotalChargeStiffnessMatrix,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDNuclearChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeBasisData, memorySpace>>
          feBDElectronicChargeRhs,
        std::shared_ptr<
          const basis::FEBasisDataStorage<ValueTypeWaveFnBasis,
                                          memorySpace>> feBDHamiltonian)
     {
        d_atomNonLocOpContext =
          std::make_shared<const basis::AtomCenterNonLocalOpContextFE<ValueTypeWaveFnBasis,
                                                        ValueTypeWaveFnCoeff,
                                                        memorySpace,
                                                        dim>>(
                                                            feBMWaveFn,
                                                            feBDHamiltonian,
                                                            d_atomSphericalDataContainerPSP,
                                                            ElectroHamiltonianDefaults::ATOM_PARTITION_TOL_BETA ,
                                                            d_atomSymbolVec,
                                                            atomCoordinates,
                                                            d_maxCellBlock,
                                                            d_maxWaveFnBlock,
                                                            d_linAlgOpContext,
                                                            d_mpiComm);

        d_atomVLocFunction = 
          std::make_shared<const atoms::AtomSevereFunction<dim>>(
                                  d_atomSphericalDataContainerPSP,
                                  d_atomSymbolVec,
                                  atomCoordinates,
                                  "vlocal",
                                  0,
                                  1);

        d_electrostaticLocal->reinitBasis(atomCoordinates,
                                          atomicElectronChargeDensity,
                                          atomicTotalElecPotNuclearQuad,
                                          atomicTotalElecPotElectronicQuad,
                                          feBMTotalCharge,
                                          feBDTotalChargeStiffnessMatrix,
                                          feBDNuclearChargeRhs,
                                          feBDElectronicChargeRhs,
                                          feBDHamiltonian,
                                          *d_atomVLocFunction);
     }
     
     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     void
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::
       reinitField(
         const quadrature::QuadratureValuesContainer<RealType, memorySpace>
           &electronChargeDensity)
     {
        d_electrostaticLocal->reinitField(electronChargeDensity);
     }
 
     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     void
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::getLocal(Storage &cellWiseStorage) const
     {
        d_electrostaticLocal->getLocal(cellWiseStorage);
     }

     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     void
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::applyNonLocal(linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &X, 
                    linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &Y) const
     {
        d_atomNonLocOpContext->apply(X,Y);
     }
     
 
     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     bool
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::hasLocalComponent() const
     {
        return true;
     }

     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     bool
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::hasNonLocalComponent() const
     {
        return true;
     }


     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     void
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::evalEnergy(const std::vector<RealType> &                  occupation,
                            const linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &waveFn, 
                            const linearAlgebra::MultiVector<ValueTypeWaveFnCoeff, memorySpace> &VnonLocxWaveFn)
     {
        d_electrostaticLocal->evalEnergy();
        d_energy = d_electrostaticLocal->getEnergy();
        std::vector<linearAlgebra::blasLapack::scalar_type<ValueTypeWaveFnCoeff, ValueTypeWaveFnCoeff>> dotProds(waveFn.getNumberComponents());
        linearAlgebra::dot(waveFn,
                          VnonLocxWaveFn,
                          dotProds,
                          linearAlgebra::blasLapack::ScalarOp::Conj,
                          linearAlgebra::blasLapack::ScalarOp::Identity);

        RealType nonLocEnergy = 0;                  
        for(int i = 0 ; i< dotProds.size() ; i++)
          nonLocEnergy +=  dotProds[i] * occupation[i]; 
        
        d_energy += nonLocEnergy;
     }
 
     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     typename ElectrostaticFE<ValueTypeBasisData,
                              ValueTypeBasisCoeff,
                              ValueTypeWaveFnBasis,
                              memorySpace,
                              dim>::RealType
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::getEnergy() const
     {
       return d_energy;
     }
 
     template <typename ValueTypeBasisData,
               typename ValueTypeBasisCoeff,
               typename ValueTypeWaveFnBasis,
               typename ValueTypeWaveFnCoeff,
               utils::MemorySpace memorySpace,
               size_type          dim>
     const quadrature::QuadratureValuesContainer<
       typename ElectrostaticFE<ValueTypeBasisData,
                                ValueTypeBasisCoeff,
                                ValueTypeWaveFnBasis,
                                memorySpace,
                                dim>::ValueType,
       memorySpace> &
     ElectrostaticONCVNonLocFE<ValueTypeBasisData,
                          ValueTypeBasisCoeff,
                          ValueTypeWaveFnBasis,
                          ValueTypeWaveFnCoeff,
                          memorySpace,
                          dim>::getFunctionalDerivative() const
     {
       return d_electrostaticLocal->getFunctionalDerivative();
     }
 
   } // end of namespace ksdft
 } // end of namespace dftefe
 