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

    // Set up Triangulation
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(mpi_communicator);
    std::vector<unsigned int>         subdivisions = {5, 5, 5};
    std::vector<bool>                 isPeriodicFlags(dim, false);
    std::vector<dftefe::utils::Point> domainVectors(dim, dftefe::utils::Point(dim, 0.));

    double xmin = 3.0;
    double ymin = 3.0;
    double zmin = 3.0;

    domainVectors[0][0] = xmin;
    domainVectors[1][1] = ymin;
    domainVectors[2][2] = zmin;

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

    //get the cellvertices vector
    for( ; feBMCellIter != feBM->endLocallyOwnedCells(); feBMCellIter++)
    {
        (*feBMCellIter)->getVertices(cellVertices);
        cellVerticesVector.push_back(cellVertices);
    }

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
        
    // assume the tolerance value
    double tolerance = 1e-6;

    // Use the atomidsPartition object

    std::vector<dftefe::size_type> nAtoms;

    std::shared_ptr<dftefe::basis::AtomIdsPartition<dim>> atomIdsPartition =
        std::make_shared<dftefe::basis::AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                        minbound,
                                                        maxbound,
                                                        cellVerticesVector,
                                                        tolerance,
                                                        mpi_communicator,
                                                        numProcs);

    for( auto i:atomIdsPartition->oldAtomIds())
        std::cout<<"oldAtomIds of "<<rank<<"->"<<i<<"\n";
    std::cout<<"\n";
    for( auto i:atomIdsPartition->newAtomIds())
        std::cout<<"newAtomIds of "<<rank<<"->"<<i<<"\n";
    std::cout<<rank<<"->"<<"\n";
    for( auto i:atomIdsPartition->nAtomIdsInProcessorCumulative())
        std::cout<<"nAtomIdsInProcessorCumulative of "<<rank<<"->"<<i<<"\n";
    std::cout<<"\n";
    for( auto i:atomIdsPartition->nAtomIdsInProcessor())
        std::cout<<"nAtomIdsInProcessor "<<"->"<<i<<",";
    std::cout<<"\n";
    for( auto i:atomIdsPartition->locallyOwnedAtomIds())
        std::cout<<"locallyOwnedAtomIds of "<<rank<<"->"<<i<<"\n";

    dftefe::utils::mpi::MPIFinalize();
#endif
}
