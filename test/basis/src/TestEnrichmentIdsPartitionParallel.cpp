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
* @author 
*/

// For the Base class
#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/AtomIdsPartition.h>
#include <basis/FEBasisManagerDealii.h>
#include <basis/FEBasisManager.h>
#include <basis/EnrichmentIdsPartition.h>

// Header for the utils class
#include <utils/PointImpl.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>

// Header for the dealii
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cfloat>

int main()
{
#ifdef DFTEFE_WITH_MPI

    const unsigned int dim = 3;

    dftefe::utils::mpi::MPIComm mpi_communicator = dftefe::utils::mpi::MPICommWorld;

    // initialize the MPI environment
    dftefe::utils::mpi::MPIInit(NULL, NULL);

    // Get the number of processes
    int numProcs;
    dftefe::utils::mpi::MPICommSize(mpi_communicator, &numProcs);

    // Get the rank of the process
    int rank;
    dftefe::utils::mpi::MPICommRank(mpi_communicator, &rank);

    std::vector<std::vector<dftefe::utils::Point>> cellVerticesVector;
    std::vector<dftefe::utils::Point> cellVertices;
    std::string inputFileName = "/home/avirup/dft-efe/test/basis/src/AtomData.in";
    std::fstream fstream;

    // Set up Triangulation
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(mpi_communicator);
    std::vector<unsigned int>         subdivisions = {5, 5, 5};
    std::vector<bool>                 isPeriodicFlags(dim, false);
    std::vector<dftefe::utils::Point> domainVectors(dim, dftefe::utils::Point(dim, 0.));

    double xmax = 5.0;
    double ymax = 5.0;
    double zmax = 5.0;

    domainVectors[0][0] = xmax;
    domainVectors[1][1] = ymax;
    domainVectors[2][2] = zmax;

    // Initialize the triangulation
    triangulationBase->initializeTriangulationConstruction();
    triangulationBase->createUniformParallelepiped(subdivisions, domainVectors, isPeriodicFlags);
    triangulationBase->finalizeTriangulationConstruction();

    dftefe::size_type feOrder = 1;

    //Create the febasismanager object
    std::shared_ptr<dftefe::basis::FEBasisManager> feBM =
        std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase,feOrder);

    dftefe::size_type numLocallyOwnedCells  = feBM->nLocallyOwnedCells();

    std::cout<<"--------------Hello Tester from rank "<<rank<<"-----------------"<<numLocallyOwnedCells;

    auto feBMCellIter = feBM->beginLocallyOwnedCells();
    unsigned int atomcount = 0;
    fstream.open(inputFileName, std::fstream::out | std::fstream::trunc);
    dealii::Point<dim> coords;
    //get the cellvertices vector
    for( ; feBMCellIter != feBM->endLocallyOwnedCells(); feBMCellIter++)
    {
        (*feBMCellIter)->getVertices(cellVertices);
        cellVerticesVector.push_back(cellVertices);

        std::shared_ptr<dftefe::basis::FECellDealii<dim>> fecelldealiiobjptr = 
            std::dynamic_pointer_cast<dftefe::basis::FECellDealii<dim>>(*feBMCellIter);

        dealii::DoFHandler<dim>::active_cell_iterator dealiicelliter =
            fecelldealiiobjptr->getDealiiFECellIter();

        for (auto face_index : dealii::GeometryInfo<dim>::face_indices())
        {
            auto neighboriter = 
                dealiicelliter->neighbor(face_index);
            if(neighboriter->state() == dealii::IteratorState::valid && neighboriter->is_ghost())
            {
                for(auto vertex_index : dealiicelliter->face(face_index)->vertex_indices())
                {
                    bool flag = true;
                    coords = dealiicelliter->face(face_index)->vertex(vertex_index);
                    for(unsigned int i = 0; i < dim ; i++)
                    {
                        if(coords[i]>=domainVectors[i][i] || coords[i]<=0)
                        {
                            flag = false;
                            break;
                        }
                    }
                    if(flag)
                    {
                        atomcount += 1;

                    }
                    if(atomcount >=1) break;
                }
            }
            if(atomcount >=1) break;
        }
        if(atomcount >=1) break;
    }

    dftefe::utils::mpi::MPIBarrier(mpi_communicator);
    if(rank == 0)
    {
        fstream <<"C ";
        for(unsigned int i = 0; i < dim ; i++)
            fstream <<coords[i]<<" ";
        fstream <<"\n";
    }

    fstream.close();

    fstream.open(inputFileName, std::fstream::in);

    std::vector<double> minbound;
    std::vector<double> maxbound;
    maxbound.resize(dim,0);
    minbound.resize(dim,0);

        
    for( unsigned int k=0;k<dim;k++)
    {
        double maxtmp = -DBL_MAX,mintmp = DBL_MAX;
        auto cellIter = cellVerticesVector.begin();
        for ( ; cellIter != cellVerticesVector.end(); ++cellIter)
        {

            auto cellVertices = cellIter->begin(); 
            for( ; cellVertices != cellIter->end(); ++cellVertices)
            {
                if(maxtmp<=*(cellVertices->begin()+k)) maxtmp = *(cellVertices->begin()+k);
                if(mintmp>=*(cellVertices->begin()+k)) mintmp = *(cellVertices->begin()+k);
            }

        }
        maxbound[k]=maxtmp;
        minbound[k]=mintmp;
    }

    // read the input file and create atomsymbol vector and atom coordinates vector.
    std::vector<dftefe::utils::Point> atomCoordinatesVec;
    std::vector<double> coordinates;
    coordinates.resize(dim,0.);
    std::vector<std::string> atomSymbol;
    std::string symbol;
    atomSymbol.resize(0);
    std::string line;
    while (std::getline(fstream, line)){
        std::stringstream ss(line);
        ss >> symbol; 
        for(unsigned int i=0 ; i<dim ; i++){
            ss >> coordinates[i]; 
        }
        atomCoordinatesVec.push_back(coordinates);
        atomSymbol.push_back(symbol);
    }
    dftefe::utils::mpi::MPIBarrier(mpi_communicator);
        
    std::map<std::string, std::string> atomSymbolToFilename;
    for (auto i:atomSymbol )
    {
        atomSymbolToFilename[i] = i+".xml";
    }
        
    // assume the tolerance value
    double tolerance = 1e-6;

    // Create the AtomSphericaldataContainer object

    std::vector<std::string> fieldNames{ "density", "vhartree", "vnuclear", "vtotal", "orbital" };
    std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
    std::shared_ptr<dftefe::atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
        std::make_shared<dftefe::atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                        fieldNames,
                                                        metadataNames);
    
    // Create the atomidsPartition object
    std::shared_ptr<dftefe::basis::AtomIdsPartition<dim>> atomIdsPartition =
        std::make_shared<dftefe::basis::AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                        minbound,
                                                        maxbound,
                                                        cellVerticesVector,
                                                        tolerance,
                                                        mpi_communicator);

    // Create the enrichemntIdsPartition object
    std::string fieldName = "density";  // Each fieldname will have own set of enrichment ids
    std::shared_ptr<dftefe::basis::EnrichmentIdsPartition<dim>> enrichmentIdsPartition =
        std::make_shared<dftefe::basis::EnrichmentIdsPartition<dim>>(atomSphericalDataContainer,
                                                        atomIdsPartition,
                                                        atomSymbol,
                                                        atomCoordinatesVec,
                                                        fieldName,                   
                                                        minbound,  
                                                        maxbound,
                                                        cellVerticesVector,
                                                        mpi_communicator);    

    std::cout<<"\nnewAtomIdToEnrichmentIdOffset:\n";
    std::vector<dftefe::size_type> offset =
        enrichmentIdsPartition->newAtomIdToEnrichmentIdOffset();
    for(auto i:offset ) { std::cout<<"rank "<<rank<<" : "<<i<<"\n";}

    std::cout<<"\nrank "<<rank<<"->overlappingEnrichmentIdsInCells:\n";
    std::vector<std::vector<dftefe::size_type>> epartition =
        enrichmentIdsPartition->overlappingEnrichmentIdsInCells();
    auto iter2 = epartition.begin();
    for( ; iter2 != epartition.end() ; iter2++)
    {
      std::cout<<"{";
        auto iter1 = iter2->begin();
        for( ; iter1 != iter2->end() ; iter1++)
        {
            std::cout<<*(iter1)<<",";
        }
        std::cout<<"}, ";
    }

    std::cout<<"\nlocallyOwnedEnrichmentIds:\n";
    std::pair<dftefe::size_type,dftefe::size_type> localeid =
        enrichmentIdsPartition->locallyOwnedEnrichmentIds();
    std::cout<<"rank "<<rank<<" : "<<localeid.first<<" "<<localeid.second<<"\n";

    std::cout<<"\nghostEnrichmentIds:\n";
    std::vector<dftefe::size_type> ghosteid =
        enrichmentIdsPartition->ghostEnrichmentIds();
    for(auto i:ghosteid ) { std::cout<<"rank "<<rank<<" : "<<i<<"\n";}

    std::cout<<"\nenrichmentIdToNewAtomIdMap:\n";
    std::map<dftefe::size_type,dftefe::size_type> eidtonatomid =
        enrichmentIdsPartition->enrichmentIdToNewAtomIdMap();
    for(auto i:eidtonatomid ) { std::cout<<"rank "<<rank<<":"<<i.first<<"->"<<i.second<<"\n";}   

    std::cout<<"\nenrichmentIdToQuantumIdMap:";
    std::map<dftefe::size_type,dftefe::size_type> eidtoqid =
        enrichmentIdsPartition->enrichmentIdToQuantumIdMap();
    for(auto i:eidtoqid ) { std::cout<<"rank "<<rank<<":"<<i.first<<"->"<<i.second<<"\n";} 

    dftefe::utils::mpi::MPIFinalize();
#endif
}
