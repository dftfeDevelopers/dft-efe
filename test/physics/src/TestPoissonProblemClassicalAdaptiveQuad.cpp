    #include <basis/TriangulationBase.h>
    #include <basis/TriangulationDealiiParallel.h>
    #include <basis/CellMappingBase.h>
    #include <basis/LinearCellMappingDealii.h>
    #include <basis/FEBasisManagerDealii.h>
    #include <basis/FEConstraintsDealii.h>
    #include <basis/FEBasisDataStorageDealii.h>
    #include <basis/FEBasisOperations.h>
    #include <basis/FEConstraintsDealii.h>
    #include <basis/FEBasisHandlerDealii.h>
    #include <quadrature/QuadratureAttributes.h>
    #include <quadrature/QuadratureRuleGauss.h>
    #include <quadrature/QuadratureRuleContainer.h>
    #include <quadrature/QuadratureValuesContainer.h>
    #include <basis/FECellWiseDataOperations.h>
    #include <utils/Point.h>
    #include <utils/TypeConfig.h>
    #include <utils/MemorySpaceType.h>
    #include <utils/MemoryStorage.h>
    #include <vector>
    #include <cmath>
    #include <memory>
    #include <linearAlgebra/LinearSolverFunction.h>
    #include <physics/PoissonLinearSolverFunctionFE.h>
    #include <linearAlgebra/LinearAlgebraProfiler.h>
    #include <linearAlgebra/CGLinearSolver.h>

    #include <deal.II/grid/tria.h>
    #include <deal.II/grid/grid_generator.h>
    #include <deal.II/dofs/dof_handler.h>
    #include <deal.II/fe/fe_q.h>
    #include <deal.II/fe/fe_values.h>
    #include <deal.II/base/quadrature_lib.h>
    #include <deal.II/fe/mapping_q1.h>
    #include <deal.II/dofs/dof_tools.h>

    #include <iostream>

    // operator - nabla^2 in weak form
    // operand - V_H
    // memoryspace - HOST

    double rho(dftefe::utils::Point &point, std::vector<dftefe::utils::Point> &origin, double rc)
    {
    double ret = 0;
    // The function should have homogeneous dirichlet BC
    for (unsigned int i = 0 ; i < origin.size() ; i++ )
    {
    double r = 0;
    for (unsigned int j = 0 ; j < point.size() ; j++ )
    {
        r += std::pow((point[j]-origin[i][j]),2);
    }
    r = std::sqrt(r);
    if( r > rc )
        ret += 0;
    else
        ret += -21*std::pow((r-rc),3)*(6*r*r + 3*r*rc + rc*rc)/(5*M_PI*std::pow(rc,8))*4*M_PI;
    }
    return ret;
    }

    double potential(dftefe::utils::Point &point, std::vector<dftefe::utils::Point> &origin, double rc)
    {
    double ret = 0;
    // The function should have homogeneous dirichlet BC
    for (unsigned int i = 0 ; i < origin.size() ; i++ )
    {
    double r = 0;
    for (unsigned int j = 0 ; j < point.size() ; j++ )
    {
        r += std::pow((point[j]-origin[i][j]),2);
    }
    r = std::sqrt(r);
    if( r > rc )
        ret += 1/r;
    else
        ret += (9*std::pow(r,7)-30*std::pow(r,6)*rc
        +28*std::pow(r,5)*std::pow(rc,2)-14*std::pow(r,2)*std::pow(rc,5)
        +12*std::pow(rc,7))/(5*std::pow(rc,8));
    }
    return ret;
    } 

    int main()
    {

    std::cout<<" Entering test poisson problem classical adaptive quad \n";

    // Required to solve : \nabla^2 V_H = g(r,r_c) Solve using CG in linearAlgebra
    // In the weak form the eqn is:
    // (N_i,N_j)*V_H = (N_i, g(r,r_c))
    // Input to CG are : linearSolverFnction. Reqd to create a derived class of the base.
    // For the nabla : LaplaceOperatorContextFE to get \nabla^2(A)*x = y

    //initialize MPI

    int mpiInitFlag = 0;
    dftefe::utils::mpi::MPIInitialized(&mpiInitFlag);
    if(!mpiInitFlag)
    {
    dftefe::utils::mpi::MPIInit(NULL, NULL);
    }

    dftefe::utils::mpi::MPIComm comm = dftefe::utils::mpi::MPICommWorld;

    int rank;
    dftefe::utils::mpi::MPICommRank(comm, &rank);

    int blasQueue = 0;
    dftefe::linearAlgebra::blasLapack::BlasQueue<dftefe::utils::MemorySpace::HOST> *blasQueuePtr = &blasQueue;

    std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext =
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(blasQueuePtr);

    // Set up Triangulation
    const unsigned int dim = 3;
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
    std::vector<unsigned int>         subdivisions = {10, 10 ,10};
    std::vector<bool>                 isPeriodicFlags(dim, false);
    std::vector<dftefe::utils::Point> domainVectors(dim,
                                                    dftefe::utils::Point(dim, 0.0));

    double xmax = 10.0;
    double ymax = 10.0;
    double zmax = 10.0;
    double rc = 0.5;
    unsigned int numComponents = 1;
    double hMin = 2.0;
    dftefe::size_type maxIter = 2e7;
    double absoluteTol = 1e-10;
    double relativeTol = 1e-12;
    double divergenceTol = 1e10;
    double refineradius = 3*rc;

    domainVectors[0][0] = xmax;
    domainVectors[1][1] = ymax;
    domainVectors[2][2] = zmax;

    // initialize the triangulation
    triangulationBase->initializeTriangulationConstruction();
    triangulationBase->createUniformParallelepiped(subdivisions,
                                                    domainVectors,
                                                    isPeriodicFlags);
    triangulationBase->finalizeTriangulationConstruction();

    // Enrichment data file consisting of g(r,\theta,\phi) = f(r)*Y_lm(\theta, \phi)
    std::string sourceDir = "/home/avirup/dft-efe/test/physics/src/";
    std::string atomDataFile = "AtomData.in";
    std::string inputFileName = sourceDir + atomDataFile;
    std::fstream fstream;

    fstream.open(inputFileName, std::fstream::in);

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
    dftefe::utils::mpi::MPIBarrier(comm);
        
    std::map<std::string, std::string> atomSymbolToFilename;
    for (auto i:atomSymbol )
    {
        atomSymbolToFilename[i] = sourceDir + i + ".xml";
    }

    std::vector<std::string> fieldNames{"vnuclear"};
    std::vector<std::string> metadataNames{ "symbol", "Z", "charge", "NR", "r" };
    std::shared_ptr<dftefe::atoms::AtomSphericalDataContainer>  atomSphericalDataContainer = 
        std::make_shared<dftefe::atoms::AtomSphericalDataContainer>(atomSymbolToFilename,
                                                        fieldNames,
                                                        metadataNames);

    std::string fieldName = "vnuclear";
    double atomPartitionTolerance = 1e-6;

    int flag = 1;
    int mpiReducedFlag = 1;
    bool radiusRefineFlag = true;
    while(mpiReducedFlag)
    {
    flag = 0;
    radiusRefineFlag = false;
    auto triaCellIter = triangulationBase->beginLocal();
    for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
    {
        (*triaCellIter)->clearRefineFlag();
        dftefe::utils::Point centerPoint(dim, 0.0); 
        (*triaCellIter)->center(centerPoint);
        for ( unsigned int i=0 ; i<atomCoordinatesVec.size() ; i++)
        {
        double dist = 0;
        for (unsigned int j = 0 ; j < dim ; j++ )
        {
            dist += std::pow((centerPoint[j]-atomCoordinatesVec[i][j]),2);
        }
        dist = std::sqrt(dist);
        if(dist < refineradius)
            radiusRefineFlag = true;
        }
        if (radiusRefineFlag && (*triaCellIter)->diameter() > hMin)
        {
        (*triaCellIter)->setRefineFlag();
        flag = 1;
        }
    }
    triangulationBase->executeCoarseningAndRefinement();
    triangulationBase->finalizeTriangulationConstruction();
    // Mpi_allreduce that all the flags are 1 (mpi_max)
    int err = dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        &flag,
        &mpiReducedFlag,
        1,
        dftefe::utils::mpi::MPIInt,
        dftefe::utils::mpi::MPIMax,
        comm);
    std::pair<bool, std::string> mpiIsSuccessAndMsg =
        dftefe::utils::mpi::MPIErrIsSuccessAndMsg(err);
    dftefe::utils::throwException(mpiIsSuccessAndMsg.first,
                            "MPI Error:" + mpiIsSuccessAndMsg.second);
    }

    // initialize the basis Manager

    unsigned int feDegree = 3;

    std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feDegree);
    std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
    basisManager->getBasisCenters(dofCoords);

    std::cout << "Locally owned cells : " << basisManager->nLocallyOwnedCells() << "\n";
    std::cout << "Total Number of dofs : " << basisManager->nGlobalNodes() << "\n";

    // Set the constraints

    std::string constraintHanging = "HangingNodeConstraint"; //give BC to rho
    std::string constraintHomwHan = "HomogeneousWithHanging"; // use this to solve the laplace equation
    std::string constraintPotential = "InHomogeneosWithHangingPotential"; // this is for getting analytical solution
    std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec;
    constraintsVec.resize(3);
    for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
    constraintsVec[i] = std::make_shared<dftefe::basis::FEConstraintsDealii<double, dftefe::utils::MemorySpace::HOST, dim>>();

    constraintsVec[0]->clear();
    constraintsVec[0]->makeHangingNodeConstraint(basisManager);
    constraintsVec[0]->close();

    constraintsVec[1]->clear();
    constraintsVec[1]->makeHangingNodeConstraint(basisManager);
    constraintsVec[1]->setHomogeneousDirichletBC();
    constraintsVec[1]->close();

    constraintsVec[2]->clear();
    constraintsVec[2]->makeHangingNodeConstraint(basisManager);
    const unsigned int dofs_per_cell =
    basisManager->nCellDofs(0);
    const unsigned int faces_per_cell =
    dealii::GeometryInfo<dim>::faces_per_cell;
    const unsigned int dofs_per_face =
    std::pow((basisManager->getFEOrder(0)+1),2);
    std::vector<dftefe::global_size_type> cellGlobalDofIndices(dofs_per_cell);
    std::vector<dftefe::global_size_type> iFaceGlobalDofIndices(dofs_per_face);
    std::vector<bool> dofs_touched(basisManager->nGlobalNodes(), false);
    auto              icell = basisManager->beginLocallyOwnedCells();
    dftefe::utils::Point basisCenter(dim, 0);
    for (; icell != basisManager->endLocallyOwnedCells(); ++icell)
    {
        (*icell)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
        for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
        {
            (*icell)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
            const dftefe::size_type boundaryId = (*icell)->getFaceBoundaryId(iFace);
            if (boundaryId == 0)
            {
                for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                    ++iFaceDof)
                {
                    const dftefe::global_size_type nodeId =
                    iFaceGlobalDofIndices[iFaceDof];
                    if (dofs_touched[nodeId])
                    continue;
                    dofs_touched[nodeId] = true;
                    if (!constraintsVec[2]->isConstrained(nodeId))
                    {
                        basisCenter = dofCoords.find(nodeId)->second;
                        double constraintValue = potential(basisCenter, atomCoordinatesVec, rc); 
                        constraintsVec[2]->setInhomogeneity(nodeId, constraintValue);
                    } // non-hanging node check
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned
    constraintsVec[2]->close();

    std::map<std::string,
            std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

    constraintsMap[constraintHanging] = constraintsVec[0];
    constraintsMap[constraintHomwHan] = constraintsVec[1];
    constraintsMap[constraintPotential] = constraintsVec[2];

    // Set up quad attribute
    dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::ADAPTIVE,false);

    dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;
    basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreQuadRealPoints] = true;

    // Set up the FE Basis Data Storage
    std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::FEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, quadAttr, basisAttrMap);

    //--------- For the atomSevereFunction one needs enrichmentIdsPartition to reduce the number of 
    // enriched ids one has to add for in the functions. Due to this, the objct is created from scrach from,
    // the FEBasisManagerDealii triangulation information-----------------------------------------//

    //get the cellvertices vector
    auto feBMCellIter = feBM->beginLocallyOwnedCells();
    for( ; feBMCellIter != feBM->endLocallyOwnedCells(); feBMCellIter++)
    {
        (*feBMCellIter)->getVertices(cellVertices);
        cellVerticesVector.push_back(cellVertices);
    }

    // create  minbound and maxbound
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

    // Create the atomidsPartition object
    std::shared_ptr<dftefe::basis::AtomIdsPartition<dim>> atomIdsPartition =
        std::make_shared<dftefe::basis::AtomIdsPartition<dim>>(atomCoordinatesVec,
                                                        minbound,
                                                        maxbound,
                                                        cellVerticesVector,
                                                        atomPartitionTolerance,
                                                        comm);

    // Create the enrichemntIdsPartition object
    std::shared_ptr<dftefe::basis::EnrichmentIdsPartition<dim>> enrichmentIdsPartition =
        std::make_shared<dftefe::basis::EnrichmentIdsPartition<dim>>(atomSphericalDataContainer,
                                                        atomIdsPartition,
                                                        atomSymbol,
                                                        atomCoordinatesVec,
                                                        fieldName,                   
                                                        minbound,  
                                                        maxbound,
                                                        cellVerticesVector,
                                                        comm);  

    // Set up the vector of scalarSpatialRealFunctions for adaptive quadrature
    std::vector<std::shared_ptr<const utils::ScalarSpatialFunctionReal>> functionsVec;
    unsigned int numfun = 2;
    functionsVec.resize(numfun); // Do the first and second derivative of the enrichemnt function
    std::vector<double> tolerances(numfun);
    std::vector<double> integralThresholds(numfun);
    for ( unsigned int i=0 ;i < functionsVec.size() ; i++ )
    {
        functionsVec[i] = std::make_shared<dftefe::atoms::AtomSevereFunction<dim>>(        
            enrichmentIdsPartition,
            atomSphericalDataContainer,
            atomSymbolVec,
            atomCoordinatesVec,
            fieldName,
            i+1);
        tolerances[i] = 1e7;
        integralThresholds[i] = 1e-16;
    }

    double smallestCellVolume = 1e-4;
    unsigned int maxRecursion = 1e3;

    // Set up base quadrature rule for adaptive quadrature 
    unsigned int num1DGaussSize = 4;
    std::shared_ptr<dftefe::quadrature::QuadratureRule> quadRule =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);
    
    // evaluate basis data
    feBasisData->evaluateBasisData(quadAttr,  quadRule, functionsVec,
         tolerances, integralThresholds, smallestCellVolume, maxRecursion, basisAttrMap);

    // Set up BasisHandler
    std::shared_ptr<dftefe::basis::FEBasisHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisHandler =
    std::make_shared<dftefe::basis::FEBasisHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap, comm);

    // Set up basis Operations
    dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

    // set up MPIPatternP2P for the constraints
    auto mpiPatternP2PHanging = basisHandler->getMPIPatternP2P(constraintHanging);
    auto mpiPatternP2PHomwHan = basisHandler->getMPIPatternP2P(constraintHomwHan);
    auto mpiPatternP2PPotential = basisHandler->getMPIPatternP2P(constraintPotential);

    // set up different multivectors - rho, vh with inhomogeneous BC, vh
    std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
    dens = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
        mpiPatternP2PHomwHan, linAlgOpContext, numComponents, double());

    std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
    vh = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
        mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

    std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
    vhNHDB = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
        mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

    std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
    solution = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
        mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

    std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
    error = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
        mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

    //populate the value of the Density at the nodes for interpolating to quad points
    auto numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
    auto itField  = dens->begin();
    dftefe::utils::Point nodeLoc(dim,0.0);
    for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
        // get cell dof global ids
        std::vector<dftefe::global_size_type> cellGlobalNodeIds;
        basisManager->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

        // loop over nodes of a cell
        for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
        // If node not constrained then get the local id and coordinates of the node
        dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
        if( !basisHandler->getConstraints(constraintHomwHan).isConstrained(globalId))
        {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintHomwHan) ;
            basisHandler->getBasisCenters(localId,constraintHomwHan,nodeLoc);
            *(itField + localId )  = rho(nodeLoc, atomCoordinatesVec, rc);
        }
        }
    }
    dens->updateGhostValues();
    basisHandler->getConstraints(constraintHomwHan).distributeParentToChild(*dens, numComponents);


    // vector for lhs

    numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
    itField  = vhNHDB->begin();
    dofs_touched.clear();
    dofs_touched.resize(basisManager->nGlobalNodes(), false);
    icell = basisManager->beginLocallyOwnedCells();
    for (; icell != basisManager->endLocallyOwnedCells(); ++icell)
    {
        (*icell)->cellNodeIdtoGlobalNodeId(cellGlobalDofIndices);
        for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
        {
            (*icell)->getFaceDoFGlobalIndices(iFace, iFaceGlobalDofIndices);
            const dftefe::size_type boundaryId = (*icell)->getFaceBoundaryId(iFace);
            if (boundaryId == 0)
            {
                for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
                    ++iFaceDof)
                {
                    const dftefe::global_size_type globalId =
                    iFaceGlobalDofIndices[iFaceDof];
                    if (dofs_touched[globalId])
                    continue;
                    dofs_touched[globalId] = true;
                    if (!basisHandler->getConstraints(constraintHanging).isConstrained(globalId))
                    {
                        dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintHanging) ;
                        basisHandler->getBasisCenters(localId,constraintHanging,nodeLoc);
                        *(itField + localId )  = potential(nodeLoc, atomCoordinatesVec, rc);
                    } // non-hanging node check
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned
    vhNHDB->updateGhostValues();
    basisHandler->getConstraints(constraintHanging).distributeParentToChild(*vhNHDB, numComponents);


    //populate the value of the Potential at the nodes for the analytic expressions

    numLocallyOwnedCells  = basisManager->nLocallyOwnedCells();
    itField  = vh->begin();
    for (dftefe::size_type iCell = 0; iCell < numLocallyOwnedCells ; iCell++)
    {
        // get cell dof global ids
        std::vector<dftefe::global_size_type> cellGlobalNodeIds;
        basisManager->getCellDofsGlobalIds(iCell, cellGlobalNodeIds);

        // loop over nodes of a cell
        for ( dftefe::size_type iNode = 0 ; iNode < cellGlobalNodeIds.size() ; iNode++)
        {
            // If node not constrained then get the local id and coordinates of the node
            dftefe::global_size_type globalId = cellGlobalNodeIds[iNode];
            if( !basisHandler->getConstraints(constraintPotential).isConstrained(globalId))
            {
            dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintPotential) ;
            basisHandler->getBasisCenters(localId,constraintPotential,nodeLoc);
            *(itField + localId )  = potential(nodeLoc, atomCoordinatesVec, rc);
            }
        }
    }
    vh->updateGhostValues();
    basisHandler->getConstraints(constraintPotential).distributeParentToChild(*vh, numComponents);

    // create the quadrature Value Container

    dftefe::basis::LinearCellMappingDealii<dim> linearCellMappingDealii;
    dftefe::quadrature::QuadratureRuleContainer quadRuleContainer( quadAttr, quadRule, triangulationBase,
                                                                    linearCellMappingDealii);

    dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, numComponents);

    for(dftefe::size_type i = 0 ; i < quadValuesContainer.nCells() ; i++)
    {
    dftefe::size_type quadId = 0;
    for (auto j : quadRuleContainer.getCellRealPoints(i))
    {
        double a = rho( j[0], j[1], j[2], Rx, Ry, Rz, rc);
        double *b = &a;
        quadValuesContainer.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, quadId, b);
        quadId = quadId + 1;
    }
    }

    //feBasisOp.interpolate( *dens, constraintHomwHan, *basisHandler, quadAttr, quadValuesContainer);

    std::shared_ptr<dftefe::linearAlgebra::LinearSolverFunction<double,
                                                    double,
                                                    dftefe::utils::MemorySpace::HOST>> linearSolverFunction =
    std::make_shared<dftefe::physics::PoissonLinearSolverFunctionFE<double,
                                                    double,
                                                    dftefe::utils::MemorySpace::HOST,
                                                    dim>>
                                                    (basisHandler,
                                                    feBasisOp,
                                                    feBasisData,
                                                    quadValuesContainer,
                                                    quadAttr,
                                                    constraintHanging,
                                                    constraintHomwHan,
                                                    *vhNHDB,
                                                    dftefe::linearAlgebra::PreconditionerType::JACOBI,
                                                    linAlgOpContext,
                                                    50);

    dftefe::linearAlgebra::LinearAlgebraProfiler profiler;

    std::shared_ptr<dftefe::linearAlgebra::LinearSolverImpl<double,
                                                    double,
                                                    dftefe::utils::MemorySpace::HOST>> CGSolve =
    std::make_shared<dftefe::linearAlgebra::CGLinearSolver<double,
                                                    double,
                                                    dftefe::utils::MemorySpace::HOST>>
                                                    ( maxIter,
                                                    absoluteTol,
                                                    relativeTol,
                                                    divergenceTol,
                                                    profiler);

    CGSolve->solve(*linearSolverFunction);

    linearSolverFunction->getSolution(*solution);

    std::vector<double> ones(0);
    ones.resize(numComponents, (double)1.0);
    std::vector<double> nOnes(0);
    nOnes.resize(numComponents, (double)-1.0);

    std::cout<<"solution norm: "<<solution->l2Norms()[0]<<", potential analytical norm: "<<vh->l2Norms()[0]<<"\n";

    dftefe::linearAlgebra::add(ones, *vh, nOnes, *solution, *error);

    std::cout<<"No of dofs: "<< basisManager->nGlobalNodes() <<", error norm: "<<error->l2Norms()[0]<<", relative error: "<<(error->l2Norms()[0]/vh->l2Norms()[0])<<"\n";

    //gracefully end MPI

    int mpiFinalFlag = 0;
    dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
    if(!mpiFinalFlag)
    {
    dftefe::utils::mpi::MPIFinalize();
    }
    }
