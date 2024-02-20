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
#include <basis/TriangulationDealiiSerial.h>
#include <basis/AtomIdsPartition.h>
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

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>

int main()
{
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
    std::vector<double> maxbound(dim,0.);
    std::vector<double> minbound(dim,0.);

    // Set up Triangulation
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiSerial<dim>>();
    std::vector<unsigned int>         subdivisions = {5, 5, 5};
    std::vector<bool>                 isPeriodicFlags(dim, false);
    std::vector<dftefe::utils::Point> domainVectors(dim, dftefe::utils::Point(dim, 0.));

    double xmin = 10.0;
    double ymin = 10.0;
    double zmin = 10.0;

    domainVectors[0][0] = xmin;
    domainVectors[1][1] = ymin;
    domainVectors[2][2] = zmin;
    // initialize the vector domain

    // Initialize the triangulation
    triangulationBase->initializeTriangulationConstruction();
    triangulationBase->createUniformParallelepiped(subdivisions, domainVectors, isPeriodicFlags);
    triangulationBase->finalizeTriangulationConstruction();

    dftefe::size_type numLocallyOwnedCells  = triangulationBase->nLocallyOwnedCells();

    auto triaCellIter = triangulationBase->beginLocal();

    //get the cellvertices vector
    for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
    {
        (*triaCellIter)->getVertices(cellVertices);
        cellVerticesVector.push_back(cellVertices);
    }

    auto cellIter = cellVerticesVector.begin();
    double maxtmp = -DBL_MAX , mintmp = DBL_MAX;
    for( unsigned int k=0;k<dim;k++)
    {
        for ( ; cellIter != cellVerticesVector.end(); ++cellIter)
        {
            auto cellVerticesIter = cellIter->begin();
                for( ; cellVerticesIter != cellIter->end(); ++cellVerticesIter)
                {
                    if(maxtmp<*(cellVerticesIter->begin()+k)) maxtmp = *(cellVerticesIter->begin()+k);
                    if(mintmp>*(cellVerticesIter->begin()+k)) mintmp = *(cellVerticesIter->begin()+k);
                }
        }
        maxbound[k]=maxtmp;
        minbound[k]=mintmp;
    }

    // read the input file and create atomsymbol vector and atom coordinates vector.
    std::vector<dftefe::utils::Point> atomCoordinatesVec;
    std::string inputFileName = "AtomData.in";
    std::ifstream fstream;
    fstream.open(inputFileName);
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

    std::map<std::string, std::string> atomSymbolToFilename;
    for (auto i:atomSymbol )
    {
        atomSymbolToFilename[i] = i+".xml";
    }
        
    // assume the tolerance value
    double tolerance = 1e-6;

    std::cout<<"--------------Hello Tester from rank "<<rank<<"-----------------";

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
                                                        0,
                                                        cellVerticesVector,
                                                        mpi_communicator);    

    std::cout<<"\nnewAtomIdToEnrichmentIdOffset:\n";
    std::vector<dftefe::global_size_type> offset =
        enrichmentIdsPartition->newAtomIdToEnrichmentIdOffset();
    for(auto i:offset ) { std::cout<<i<<"\n";}

    std::cout<<"\noverlappingEnrichmentIdsInCells:\n";
    std::vector<std::vector<dftefe::global_size_type>> epartition =
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
    std::pair<dftefe::global_size_type,dftefe::global_size_type> localeid =
        enrichmentIdsPartition->locallyOwnedEnrichmentIds();
    std::cout<<"\n"<<localeid.first<<" "<<localeid.second;

    std::cout<<"\nghostEnrichmentIds:\n";
    std::vector<dftefe::global_size_type> ghosteid =
        enrichmentIdsPartition->ghostEnrichmentIds();
    for(auto i:ghosteid ) { std::cout<<i<<"\n";}

    // std::cout<<"\nenrichmentIdToNewAtomIdMap:\n";
    // std::map<dftefe::size_type,dftefe::size_type> eidtonatomid =
    //     enrichmentIdsPartition->enrichmentIdToNewAtomIdMap();
    // for(auto i:eidtonatomid ) { std::cout<<i.first<<"->"<<i.second<<"\n";}   

    // std::cout<<"\nenrichmentIdToQuantumIdMap:";
    // std::map<dftefe::size_type,dftefe::size_type> eidtoqid =
    //     enrichmentIdsPartition->enrichmentIdToQuantumIdMap();
    // for(auto i:eidtoqid ) { std::cout<<i.first<<"->"<<i.second<<"\n";} 

}
