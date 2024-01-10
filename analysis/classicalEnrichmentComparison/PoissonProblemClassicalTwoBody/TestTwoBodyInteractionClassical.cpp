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

template<typename T>
T readParameter(std::string ParamFile, std::string param)
{
  T t(0);
  std::string line;
  std::fstream fstream;
  fstream.open(ParamFile, std::fstream::in);
  int count = 0;
  while (std::getline(fstream, line))
  {
    for (int i = 0; i < line.length(); i++)
    {
        if (line[i] == ' ')
        {
            line.erase(line.begin() + i);
            i--;
        }
    }
    std::istringstream iss(line);
    std::string type;
    std::getline(iss, type, '=');
    if (type.compare(param) == 0)
    {
      iss >> t;
      count = 1;
      break;
    }
  }
  if(count == 0)
  {
    dftefe::utils::throwException(false, "The parameter is not found: "+ param);
  }
  fstream.close();
  return t;
}


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
      ret += -21*std::pow((r-rc),3)*(6*r*r + 3*r*rc + rc*rc)/(5*M_PI*std::pow(rc,8));
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

int main(int argc, char** argv)
{
// argv[1] = "PoissonProblemClassicalTwoBody/param.in"

  std::cout<<" Entering test two body interaction \n";

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


  // Read the parameter files and atom coordinate files
  std::string sourceDir = "/home/avirup/dft-efe/analysis/classicalEnrichmentComparison/";
  std::string atomDataFile = "TwoSmearedCharge_dist1.5.in";
  std::string paramDataFile = argv[1];
  std::string inputFileName = sourceDir + atomDataFile;
  std::string parameterInputFileName = sourceDir + paramDataFile;

  double xmax = readParameter<double>(parameterInputFileName, "xmax");
  double ymax = readParameter<double>(parameterInputFileName, "ymax");
  double zmax = readParameter<double>(parameterInputFileName, "zmax");
  unsigned int subdivisionx = readParameter<unsigned int>(parameterInputFileName, "subdivisionx");
  unsigned int subdivisiony = readParameter<unsigned int>(parameterInputFileName, "subdivisiony");
  unsigned int subdivisionz = readParameter<unsigned int>(parameterInputFileName, "subdivisionz");
  double rc = readParameter<double>(parameterInputFileName, "rc");
  double hMin = readParameter<double>(parameterInputFileName, "hMin");
  unsigned int maxIter = readParameter<unsigned int>(parameterInputFileName, "maxIter");
  double absoluteTol = readParameter<double>(parameterInputFileName, "absoluteTol");
  double relativeTol = readParameter<double>(parameterInputFileName, "relativeTol");
  double divergenceTol = readParameter<double>(parameterInputFileName, "divergenceTol");
  double refineradius = readParameter<double>(parameterInputFileName, "refineradius");
  unsigned int num1DGaussSize = readParameter<unsigned int>(parameterInputFileName, "num1DQuadratureSize");
  unsigned int feOrder = readParameter<unsigned int>(parameterInputFileName, "feOrder");

  // Set up Triangulation
  const unsigned int dim = 3;
    std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
        std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<unsigned int>         subdivisions = {subdivisionx, subdivisiony ,subdivisionz};
  std::vector<bool>                 isPeriodicFlags(dim, false);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  domainVectors[0][0] = xmax;
  domainVectors[1][1] = ymax;
  domainVectors[2][2] = zmax;

  std::vector<double> origin(0);
  origin.resize(dim);
  for(unsigned int i = 0 ; i < dim ; i++)
    origin[i] = -domainVectors[i][i]*0.5;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  triangulationBase->shiftTriangulation(dftefe::utils::Point(origin));
  triangulationBase->finalizeTriangulationConstruction();

  std::fstream fstream;
  fstream.open(inputFileName, std::fstream::in);
  
  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<dftefe::utils::Point> atomCoordinatesVec(0,dftefe::utils::Point(dim, 0.0));
    std::vector<double> coordinates;
  coordinates.resize(dim,0.);
  std::vector<std::string> atomSymbolVec;
  std::string symbol;
  atomSymbolVec.resize(0);
  std::string line;
  while (std::getline(fstream, line)){
      std::stringstream ss(line);
      ss >> symbol; 
      for(unsigned int i=0 ; i<dim ; i++){
          ss >> coordinates[i]; 
      }
      atomCoordinatesVec.push_back(coordinates);
      atomSymbolVec.push_back(symbol);
  }
  dftefe::utils::mpi::MPIBarrier(comm);

  const unsigned int nAtoms = atomCoordinatesVec.size(); 
  const unsigned int numComponents = nAtoms+1;

  int flag = 1;
  int mpiReducedFlag = 1;
  bool radiusRefineFlag = true;
  while(mpiReducedFlag)
  {
    flag = 0;
    auto triaCellIter = triangulationBase->beginLocal();
    for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
    {
      radiusRefineFlag = false;
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

  std::shared_ptr<dftefe::basis::FEBasisManager> basisManager =   std::make_shared<dftefe::basis::FEBasisManagerDealii<dim>>(triangulationBase, feOrder);
  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisManager->getBasisCenters(dofCoords);

  std::cout << "Locally owned cells : " << basisManager->nLocallyOwnedCells() << "\n";
  std::cout << "Total Number of dofs : " << basisManager->nGlobalNodes() << "\n";

  // Set the constraints

  std::string constraintHanging = "HangingNodeConstraint"; //give BC to rho
  std::string constraintHomwHan = "HomogeneousWithHanging"; // use this to solve the laplace equation

  std::vector<std::shared_ptr<dftefe::basis::FEConstraintsBase<double, dftefe::utils::MemorySpace::HOST>>>
    constraintsVec;
  constraintsVec.resize(2);
  for ( unsigned int i=0 ;i < constraintsVec.size() ; i++ )
   constraintsVec[i] = std::make_shared<dftefe::basis::FEConstraintsDealii<double, dftefe::utils::MemorySpace::HOST, dim>>();

  constraintsVec[0]->clear();
  constraintsVec[0]->makeHangingNodeConstraint(basisManager);
  constraintsVec[0]->close();

  constraintsVec[1]->clear();
  constraintsVec[1]->makeHangingNodeConstraint(basisManager);
  constraintsVec[1]->setHomogeneousDirichletBC();
  constraintsVec[1]->close();

  std::map<std::string,
           std::shared_ptr<const dftefe::basis::Constraints<double, dftefe::utils::MemorySpace::HOST>>> constraintsMap;

  constraintsMap[constraintHanging] = constraintsVec[0];
  constraintsMap[constraintHomwHan] = constraintsVec[1];

  // Set up the quadrature rule

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::FEBasisDataStorageDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, quadAttr, basisAttrMap);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // Set up BasisHandler
  std::shared_ptr<dftefe::basis::FEBasisHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisHandler =
    std::make_shared<dftefe::basis::FEBasisHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisManager, constraintsMap, comm);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PHanging = basisHandler->getMPIPatternP2P(constraintHanging);
  auto mpiPatternP2PHomwHan = basisHandler->getMPIPatternP2P(constraintHomwHan);

  // set up different multivectors - rho, vh with inhomogeneous BC, vh
  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   dens = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHomwHan, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   vhNHDB = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   solution = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PHanging, linAlgOpContext, numComponents, double());

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
          dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintHanging) ;
          basisHandler->getBasisCenters(localId,constraintHanging,nodeLoc);
          for (unsigned int j = 0 ; j < nAtoms ; j++ )
          {
            std::vector<dftefe::utils::Point> coord{atomCoordinatesVec[j]};
            *(itField + (localId)*(numComponents) + j)  = rho(nodeLoc, coord, rc);
          }
          *(itField + (localId)*(numComponents) + nAtoms)  = rho(nodeLoc, atomCoordinatesVec, rc) ;  
        }
      }
    }
  dens->updateGhostValues();
  basisHandler->getConstraints(constraintHomwHan).distributeParentToChild(*dens, numComponents);


  // vector for lhs

  itField  = vhNHDB->begin();
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
                  const dftefe::global_size_type globalId =
                    iFaceGlobalDofIndices[iFaceDof];
                  if (dofs_touched[globalId])
                    continue;
                  dofs_touched[globalId] = true;
                  if (!basisHandler->getConstraints(constraintHanging).isConstrained(globalId))
                    {
                      dftefe::size_type localId = basisHandler->globalToLocalIndex(globalId,constraintHanging) ;
                      basisHandler->getBasisCenters(localId,constraintHanging,nodeLoc);
                      for (unsigned int j = 0 ; j < nAtoms ; j++ )
                      {
                        std::vector<dftefe::utils::Point> coord{atomCoordinatesVec[j]};
                        *(itField + (localId)*(numComponents) + j)  = potential(nodeLoc, coord, rc);
                      }
                      *(itField + (localId)*(numComponents) + nAtoms)  = potential(nodeLoc, atomCoordinatesVec, rc) ;          
                    } // non-hanging node check
                }     // Face dof loop
            }
        } // Face loop
    }     // cell locally owned


  // create the quadrature Value Container

  std::shared_ptr<dftefe::quadrature::QuadratureRule> quadRule =
    std::make_shared<dftefe::quadrature::QuadratureRuleGauss>(dim, num1DGaussSize);

  dftefe::basis::LinearCellMappingDealii<dim> linearCellMappingDealii;
    std::shared_ptr<const dftefe::quadrature::QuadratureRuleContainer> quadRuleContainer =  
                feBasisData->getQuadratureRuleContainer();

  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, numComponents);
  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainerNumerical(quadRuleContainer, numComponents);

  // Calculate the offset of the smeared charge so that the total charge in the domain is one
  std::vector<double> chargeDensity(nAtoms, 0.0), mpiReducedChargeDensity(chargeDensity.size(), 0.0);
  for(dftefe::size_type i = 0 ; i < quadValuesContainer.nCells() ; i++)
  {
    std::vector<double> JxW = quadRuleContainer->getCellJxW(i);
    dftefe::size_type quadId = 0;
    for (auto j : quadRuleContainer->getCellRealPoints(i))
    {
      for(unsigned int k = 0 ; k < nAtoms ; k++)
      {
        std::vector<dftefe::utils::Point> coord{atomCoordinatesVec[k]};
        chargeDensity[k] += rho( j, coord, rc) * JxW[quadId];
      }
      quadId = quadId + 1;
    }
  }
  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        chargeDensity.data(),
        mpiReducedChargeDensity.data(),
        chargeDensity.size(),
        dftefe::utils::mpi::MPIDouble,
        dftefe::utils::mpi::MPISum,
        comm);

  std::cout << rank << "," << mpiReducedChargeDensity[0] << "," << mpiReducedChargeDensity[1] << std::endl;

  // Store the charge density at the quadrature points for the poisson problem

  for(dftefe::size_type i = 0 ; i < quadValuesContainer.nCells() ; i++)
  {
    dftefe::size_type quadId = 0;
    for (auto j : quadRuleContainer->getCellRealPoints(i))
    {
      std::vector<double> a(numComponents, 0);
      for (unsigned int k = 0 ; k < nAtoms ; k++)
      {
        std::vector<dftefe::utils::Point> coord{atomCoordinatesVec[k]};
      a[k] = rho( j, coord, rc) * (4*M_PI) * (1.0/mpiReducedChargeDensity[k]);
      }
      a[nAtoms] = rho( j, atomCoordinatesVec, rc) * (4*M_PI) * nAtoms/(std::accumulate(mpiReducedChargeDensity.begin(),mpiReducedChargeDensity.end(),0.0));
      double *b = a.data();
      quadValuesContainer.setCellQuadValues<dftefe::utils::MemorySpace::HOST> (i, quadId, b);
      quadId = quadId + 1;
    }
  }

  std::shared_ptr<dftefe::linearAlgebra::LinearSolverFunction<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST>> linearSolverFunction =
    std::make_shared<dftefe::physics::PoissonLinearSolverFunctionFE<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST,
                                                   dim>>
                                                   (basisHandler,
                                                    feBasisData,
                                                    feBasisData,
                                                    quadValuesContainer,
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

  // perform integral rho vh 

  feBasisOp.interpolate( *solution, constraintHanging, *basisHandler, quadValuesContainerNumerical);

  auto iter1 = quadValuesContainer.begin();
  auto iter2 = quadValuesContainerNumerical.begin();
  dftefe::size_type numQuadraturePoints = quadRuleContainer->nQuadraturePoints(), mpinumQuadraturePoints=0;
  const std::vector<double> JxW = quadRuleContainer->getJxW();
  std::vector<double> energy(numComponents, 0.0), mpiReducedEnergy(energy.size(), 0.0);
  for (unsigned int i = 0 ; i < numQuadraturePoints ; i++ )
  {
    for (unsigned int j = 0 ; j < numComponents ; j++ )
    {
      energy[j] += *(i*numComponents+j+iter1) * *(i*numComponents+j+iter2) * JxW[i] * 0.5/(4*M_PI);
    }
  }

        dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
            &numQuadraturePoints,
            &mpinumQuadraturePoints,
            1,
            dftefe::utils::mpi::MPIUnsigned,
            dftefe::utils::mpi::MPISum,
            comm);

  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
        energy.data(),
        mpiReducedEnergy.data(),
        energy.size(),
        dftefe::utils::mpi::MPIDouble,
        dftefe::utils::mpi::MPISum,
        comm);

  double Ig = 10976./(17875*rc);
  double analyticalSelfEnergy = 0, numericalSelfEnergy = 0;
  for (unsigned int i = 0 ; i < nAtoms ; i++)
  {
    std::vector<dftefe::utils::Point> coord{atomCoordinatesVec[i]};
    analyticalSelfEnergy += 0.5 * (Ig - potential(atomCoordinatesVec[i], coord, rc));
    numericalSelfEnergy += mpiReducedEnergy[i];
  }

    double dist = 0;
    for (unsigned int j = 0 ; j < dim ; j++ )
    {
      dist += std::pow((atomCoordinatesVec[0][j]-atomCoordinatesVec[1][j]),2);
    }
    dist = std::sqrt(dist);
    
    std::cout << "The error in electrostatic energy: " << (mpiReducedEnergy[2] + analyticalSelfEnergy) - 1.0/dist << "\n";

        if(rank == 0)
        {
        std::ofstream myfile;
        std::stringstream ss;
        ss << "CFE"<<"domain_"<<xmax<<"x"<<ymax<<"x"<<zmax<<
        "subdiv_"<<subdivisionx<<"x"<<subdivisiony<<"x"<<subdivisionz<<
        "feOrder_"<<feOrder<<"nQuad_"<<num1DGaussSize<<"hMin_"<<hMin<<".out";
        std::string outputFile = ss.str();
        myfile.open (outputFile, std::ios::out | std::ios::trunc);
          myfile << "Total Number of dofs : " << basisManager->nGlobalNodes() << "\n";
          myfile << "No. of quad points: "<< mpinumQuadraturePoints << "\n";
          myfile << "Integral of b smear over volume: "<< std::accumulate(mpiReducedChargeDensity.begin(),mpiReducedChargeDensity.end(),0.0) << "\n";
          myfile << "The electrostatic energy (analy/num) : "<< (mpiReducedEnergy[nAtoms] + analyticalSelfEnergy) << "," << (mpiReducedEnergy[nAtoms] - numericalSelfEnergy)  << "\n";
          myfile << "The error in electrostatic energy from analytical self potential: " << (mpiReducedEnergy[nAtoms] + analyticalSelfEnergy) - 1.0/dist << " Relative error: "<<((mpiReducedEnergy[nAtoms] + analyticalSelfEnergy) - 1.0/dist)*dist<<"\n";
          myfile << "The error in electrostatic energy from numerical self potntial : " << (mpiReducedEnergy[nAtoms] - numericalSelfEnergy) - 1.0/dist << " Relative error: "<<((mpiReducedEnergy[nAtoms] - numericalSelfEnergy) - 1.0/dist)*dist<<"\n";
        myfile.close();
        }

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
