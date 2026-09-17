#include <basis/TriangulationBase.h>
#include <basis/TriangulationDealiiParallel.h>
#include <basis/CellMappingBase.h>
#include <basis/LinearCellMappingDealii.h>
#include <basis/CFEBasisDofHandlerDealii.h>
#include <basis/CFEBasisDataStorageDealii.h>
#include <basis/FEBasisOperations.h>
#include <basis/CFEConstraintsLocalDealii.h>
#include <basis/FEBasisManager.h>
#include <quadrature/QuadratureAttributes.h>
#include <quadrature/QuadratureRuleGauss.h>
#include <quadrature/QuadratureRuleContainer.h>
#include <quadrature/QuadratureValuesContainer.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <utils/ScalarZeroFunctionReal.h>
#include <utils/Exceptions.h>
#include <utils/MPITypes.h>
#include <utils/MPIWrapper.h>
#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <functional>
#include <electrostatics/PoissonSolverDealiiMatrixFreeFE.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>

#include <iostream>
const dftefe::utils::MemorySpace memorySpace = dftefe::utils::MemorySpace::HOST;
// operator - nabla^2 in weak form
// operand - V_H
// memoryspace - HOST

// The solved equation is -\nabla^2 \phi = \rho, as the Dirichlet pair below
// satisfies and as the periodic pair is built to satisfy.

const double domainLength = 5.0;

//
// Case 1: inhomogeneous Dirichlet. Unchanged from the original test.
//
double rhoDirichlet(double x, double y, double z)
{
  //0.0; // 1.0;
  // The function should have inhomogeneous dirichlet BC
    return (2.0 * (x*(x-5)*y*(y-5)) / 6.0 +
          2.0 * (x*(x-5)*z*(z-5)) / 6.0 +
          2.0 * (y*(y-5)*z*(z-5)) / 6.0);
}

double potentialDirichlet(double x, double y, double z)
{
  //1.0 ; //-((x)*(x) + (y)*(y) + (z)*(z))/(6.0);
  // The function should have inhomogeneous dirichlet BC
    return -((x)*(x-5)*(y)*(y-5)*(z)*(z-5))/6.0;
}

//
// Case 2: fully periodic, which is what exercises the mean-value constraint.
//
// With no Dirichlet boundary anywhere the Poisson operator is singular: any
// constant is in its null space. The solver pins that null space by imposing
// \int_\Omega \phi d\Omega = 0, so a test solution is only well posed if it is
// periodic AND already has zero mean.
//
//   \phi   = sin(kx) sin(ky) sin(kz),   k = 2 pi / L
//   \rho   = -\nabla^2 \phi = 3 k^2 \phi
//
// Each factor runs over a whole period across the box, so \phi and \rho both
// integrate to zero over the domain. Zero mean \rho is the compatibility
// condition without which the periodic problem has no solution at all; zero
// mean \phi makes this exact solution the very representative the constraint
// selects, so the computed field can be compared against it directly rather
// than up to a constant.
//
double potentialPeriodic(double x, double y, double z)
{
  const double k = 2.0 * M_PI / domainLength;
  return std::sin(k * x) * std::sin(k * y) * std::sin(k * z);
}

double rhoPeriodic(double x, double y, double z)
{
  const double k = 2.0 * M_PI / domainLength;
  return 3.0 * k * k * potentialPeriodic(x, y, z);
}

 class ScalarSpatialPotentialFunctionReal : public dftefe::utils::ScalarSpatialFunctionReal
  {
    public:
    ScalarSpatialPotentialFunctionReal(
      const std::function<double(double, double, double)> &potential)
      : d_potential(potential)
    {}

    double
    operator()(const dftefe::utils::Point &point) const
    {
      return d_potential(point[0], point[1], point[2]);
    }

    std::vector<double>
    operator()(const std::vector<dftefe::utils::Point> &points) const
    {
      std::vector<double> ret(0);
      ret.resize(points.size());
      for (unsigned int i = 0 ; i < points.size() ; i++)
      {
        ret[i] = d_potential(points[i][0], points[i][1], points[i][2]);
      }
      return ret;
    }

    private:
    std::function<double(double, double, double)> d_potential;
  };

//
// Solves one case and checks it. Returns the number of failures so that main
// can run both cases and report on all of them rather than stopping at the
// first.
//
// relativeErrorTol bounds the discretisation error of this mesh and degree,
// not the solver tolerance; the measured value is printed so it can be
// tightened against a real run.
//
dftefe::size_type
runCase(const std::string &                                  name,
        const bool                                           isPeriodic,
        const unsigned int                                   nSubdivisions,
        const bool                                           doRefine,
        const std::function<double(double, double, double)> &rho,
        const std::function<double(double, double, double)> &potential,
        const double                                         relativeErrorTol,
        const dftefe::utils::mpi::MPIComm &                  comm,
        std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<
          dftefe::utils::MemorySpace::HOST>>                 linAlgOpContext,
        std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<memorySpace>>
                  linAlgOpContextDevice,
        const int rank)
{
  dftefe::size_type nFailures = 0;

  const unsigned int dim = 3;
  unsigned int numComponents = 1;
  dftefe::size_type maxIter = 2e5;
  double absoluteTol = 1e-10;

  if (rank == 0)
    std::cout << "\n---- " << name << " ----\n" << std::flush;

  // Set up Triangulation
  std::shared_ptr<dftefe::basis::TriangulationBase> triangulationBase =
      std::make_shared<dftefe::basis::TriangulationDealiiParallel<dim>>(comm);
  std::vector<dftefe::size_type>    subdivisions(dim, nSubdivisions);
  std::vector<bool>                 isPeriodicFlags(dim, isPeriodic);
  std::vector<dftefe::utils::Point> domainVectors(dim,
                                                  dftefe::utils::Point(dim, 0.0));

  for (unsigned int i = 0; i < dim; i++)
    domainVectors[i][i] = domainLength;

  // initialize the triangulation
  triangulationBase->initializeTriangulationConstruction();
  triangulationBase->createUniformParallelepiped(subdivisions,
                                                 domainVectors,
                                                 isPeriodicFlags);
  // Registers the matched face pairs with dealii, which is what turns the
  // periodic flags into actual constraints on the DofHandler. A no-op when
  // nothing is flagged periodic, and it has to happen before any refinement.
  triangulationBase->markPeriodicFaces(isPeriodicFlags, domainVectors);
  triangulationBase->finalizeTriangulationConstruction();

  if (doRefine)
  {
    auto triaCellIter = triangulationBase->beginLocal();

    for( ; triaCellIter != triangulationBase->endLocal(); triaCellIter++)
    {
      dftefe::utils::Point centerPoint(dim, 0.0);
      (*triaCellIter)->center(centerPoint);
      double dist = (centerPoint[0] - 2.5) * (centerPoint[0] - 2.5);
      dist += (centerPoint[1] - 2.5) * (centerPoint[1] - 2.5);
      dist += (centerPoint[2] - 2.5) * (centerPoint[2] - 2.5);
      dist = std::sqrt(dist);
      if ( (centerPoint[0] < 1.0) || (dist < 1.0) )
      {
       (*triaCellIter)->setRefineFlag();
      }
    }

    triangulationBase->executeCoarseningAndRefinement();
    triangulationBase->finalizeTriangulationConstruction();
  }

  // initialize the basis Manager

  unsigned int feDegree = 3;

  std::shared_ptr<dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =
   std::make_shared<dftefe::basis::CFEBasisDofHandlerDealii<double, dftefe::utils::MemorySpace::HOST,dim>>(triangulationBase, feDegree, comm);

  std::map<dftefe::global_size_type, dftefe::utils::Point> dofCoords;
  basisDofHandler->getBasisCenters(dofCoords);

  if (rank == 0)
  {
    std::cout << "Locally owned cells : " <<basisDofHandler->nLocallyOwnedCells() << "\n";
    std::cout << "Total Number of dofs : " << basisDofHandler->nGlobalNodes() << "\n";
  }

  // Set up the quadrature rule
  unsigned int num1DGaussSize = 4;

  dftefe::quadrature::QuadratureRuleAttributes quadAttr(dftefe::quadrature::QuadratureFamily::GAUSS,true,num1DGaussSize);

  dftefe::basis::BasisStorageAttributesBoolMap basisAttrMap;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreValues] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradient] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreHessian] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreOverlap] = false;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreGradNiGradNj] = true;
  basisAttrMap[dftefe::basis::BasisStorageAttributes::StoreJxW] = true;

  // Set up the FE Basis Data Storage
  std::shared_ptr<dftefe::basis::FEBasisDataStorage<double, dftefe::utils::MemorySpace::HOST>> feBasisData =
    std::make_shared<dftefe::basis::CFEBasisDataStorageDealii<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, quadAttr, basisAttrMap, *linAlgOpContext);

  // evaluate basis data
  feBasisData->evaluateBasisData(quadAttr, basisAttrMap);

  // Under full periodicity every face is matched to its partner, so there is
  // no Dirichlet boundary to carry inhomogeneous data and the field manager
  // takes the DofHandler's intrinsic constraints alone.
  std::shared_ptr<const dftefe::utils::ScalarSpatialFunctionReal>
    potentialFunction = nullptr;
  if (!isPeriodic)
    potentialFunction =
      std::make_shared<ScalarSpatialPotentialFunctionReal>(potential);

  // // Set up BasisManager
  std::shared_ptr<const dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>> basisManager =
    std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
    (basisDofHandler, potentialFunction);

  // Set up basis Operations
  dftefe::basis::FEBasisOperations<double, double, dftefe::utils::MemorySpace::HOST,dim> feBasisOp(feBasisData,50);

  //
  // The constraint machinery itself, independent of the solver: assembling
  // w_i = \int N_i and electing the pinned dof. The solver builds its own
  // homogeneous manager internally and does this there, where the test cannot
  // see it, so it is exercised once here on a manager of our own. The private
  // copy that enableMeanValueConstraint takes keeps this off the manager the
  // solve below uses.
  //
  if (isPeriodic)
  {
    std::shared_ptr<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>> basisManagerHomo =
      std::make_shared<dftefe::basis::FEBasisManager<double, double, dftefe::utils::MemorySpace::HOST,dim>>
      (basisDofHandler,
       std::make_shared<const dftefe::utils::ScalarZeroFunctionReal>());

    basisManagerHomo->enableMeanValueConstraint(feBasisData,
                                                linAlgOpContext,
                                                comm);

    if (!basisManagerHomo->getConstraints().hasMeanValueConstraint())
    {
      nFailures++;
      if (rank == 0)
        std::cout << "FAILURE: enableMeanValueConstraint left the constraint "
                     "inactive on a fully periodic mesh\n";
    }
    else if (rank == 0)
    {
      std::cout << "Mean-value constraint pinned at global dof "
                << basisManagerHomo->getConstraints()
                     .getMeanValueConstraintNodeIdGlobal()
                << " on processor "
                << basisManagerHomo->getConstraints()
                     .getMeanValueConstraintProcId()
                << "\n";
    }
  }

  // set up MPIPatternP2P for the constraints
  auto mpiPatternP2PPotential = basisManager->getMPIPatternP2P();

  std::shared_ptr<dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>
   solution = std::make_shared<
    dftefe::linearAlgebra::MultiVector<double, dftefe::utils::MemorySpace::HOST>>(
      mpiPatternP2PPotential, linAlgOpContext, numComponents, double());

  solution->setValue(0);

  // create the quadrature Value Container

  std::shared_ptr<const dftefe::quadrature::QuadratureRuleContainer> quadRuleContainer =
                feBasisData->getQuadratureRuleContainer();

  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainer(quadRuleContainer, numComponents);

  // Accumulated while the source is laid down, and checked after the reduction
  // below: under full periodicity \int_\Omega \rho d\Omega = 0 is the
  // compatibility condition, not a nicety. See the check for why.
  double rhoIntegral = 0.0, rhoAbsIntegral = 0.0;

  for(dftefe::size_type i = 0 ; i < quadValuesContainer.nCells() ; i++)
  {
    const std::vector<double> cellJxW = quadRuleContainer->getCellJxW(i);
    for(dftefe::size_type iComp = 0 ; iComp < numComponents ; iComp ++)
    {
      dftefe::size_type quadId = 0;
      std::vector<double> a(quadRuleContainer->nCellQuadraturePoints(i));
      for (auto j : quadRuleContainer->getCellRealPoints(i))
      {
        a[quadId] = rho( j[0], j[1], j[2]);
        if (iComp == 0)
        {
          rhoIntegral += a[quadId] * cellJxW[quadId];
          rhoAbsIntegral += std::abs(a[quadId]) * cellJxW[quadId];
        }
        quadId = quadId + 1;
      }
      double *b = a.data();
      quadValuesContainer.setCellValues<dftefe::utils::MemorySpace::HOST> (i, b);
    }
  }

  std::shared_ptr<dftefe::electrostatics::PoissonSolverDealiiMatrixFreeFE<double,
                                                   double,
                                                   memorySpace,
                                                   dim>> poissonSolveDealiiMatrixFree =
    std::make_shared<dftefe::electrostatics::PoissonSolverDealiiMatrixFreeFE<double,
                                                   double,
                                                   memorySpace,
                                                   dim>>
                                                   (basisManager,
                                                    feBasisData,
                                                    feBasisData,
                                                    quadValuesContainer,
                                                    dftefe::linearAlgebra::PreconditionerType::JACOBI,
                                                    linAlgOpContextDevice);

  poissonSolveDealiiMatrixFree->solve(absoluteTol, maxIter);

  poissonSolveDealiiMatrixFree->getSolution(*solution);

  //
  // Everything is measured on the quadrature points rather than on the nodes.
  // The error there is the L2 error of the field itself, and it sidesteps
  // having to reason about which nodal values are constrained.
  //
  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST>
    quadSolution(quadRuleContainer, numComponents);

  feBasisOp.interpolate(*solution, *basisManager, quadSolution);

  // errSq: \int (phi_h - phi)^2, exactSq: \int phi^2, computed: \int phi_h and
  // \int phi_h^2 and the volume, which together give the mean the constraint
  // is supposed to have driven to zero.
  double local[7] = {0.0, 0.0, 0.0, 0.0, 0.0, rhoIntegral, rhoAbsIntegral};
  for (dftefe::size_type iCell = 0; iCell < quadSolution.nCells(); iCell++)
  {
    const std::vector<double> jxw = quadRuleContainer->getCellJxW(iCell);
    const std::vector<dftefe::utils::Point> points =
      quadRuleContainer->getCellRealPoints(iCell);
    std::vector<double> values(quadRuleContainer->nCellQuadraturePoints(iCell));
    quadSolution.getCellValues<dftefe::utils::MemorySpace::HOST>(iCell,
                                                                 values.data());

    for (dftefe::size_type iQuad = 0; iQuad < values.size(); iQuad++)
    {
      const double exact =
        potential(points[iQuad][0], points[iQuad][1], points[iQuad][2]);
      const double diff = values[iQuad] - exact;
      local[0] += diff * diff * jxw[iQuad];
      local[1] += exact * exact * jxw[iQuad];
      local[2] += values[iQuad] * jxw[iQuad];
      local[3] += values[iQuad] * values[iQuad] * jxw[iQuad];
      local[4] += jxw[iQuad];
    }
  }

  double global[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
    local,
    global,
    7,
    dftefe::utils::mpi::MPIDouble,
    dftefe::utils::mpi::MPISum,
    comm);

  const double relativeError = std::sqrt(global[0] / global[1]);
  const double volume        = global[4];
  const double meanValue     = global[2] / volume;
  // Root mean square of the computed field, to measure the mean against
  // something with the same units instead of against an absolute number.
  const double rms         = std::sqrt(global[3] / volume);
  const double meanOverRms = std::abs(meanValue) / rms;
  // Measured against the integral of |rho| rather than against an absolute
  // number, so the criterion does not depend on how strong the source is.
  const double rhoNeutrality =
    (global[6] > 0.0) ? std::abs(global[5]) / global[6] : 0.0;

  if (rank == 0)
  {
    std::cout << "Relative L2 error against the analytic solution: "
              << relativeError << " (bound " << relativeErrorTol << ")\n";
    std::cout << "Integral of the computed potential: " << global[2]
              << ", domain volume: " << volume << "\n";
    std::cout << "Mean value: " << meanValue << ", rms: " << rms
              << ", ratio: " << meanOverRms << "\n";
    std::cout << "Integral of the source: " << global[5]
              << ", of its magnitude: " << global[6]
              << ", ratio: " << rhoNeutrality << "\n"
              << std::flush;
  }

  if (!(relativeError < relativeErrorTol))
  {
    nFailures++;
    if (rank == 0)
      std::cout << "FAILURE: the computed potential does not match the "
                   "analytic solution\n";
  }

  if (isPeriodic)
  {
    // Solvability, checked before anything is concluded from the solution.
    //
    // With every face periodic the constant function is in the left null space
    // of the operator, so the weak form can only be satisfied if the source is
    // orthogonal to it, i.e. \int_\Omega \rho d\Omega = 0. The mean-value
    // constraint does not rescue a source that violates this: it removes the
    // null space on the solution side, while the inconsistency sits on the
    // right hand side. Pinning a dof leaves the system square and nonsingular,
    // so CG converges contentedly to the solution of that modified system,
    // which does not satisfy the original weak form -- the residual is simply
    // dumped into the pinned row. The answer comes back plausible and wrong.
    //
    // rho above integrates to zero because each sine spans whole periods, but
    // that is a property of the chosen function, so it is checked rather than
    // trusted: anyone swapping in a source without zero mean gets told so
    // instead of watching the error check fail for no visible reason.
    if (!(rhoNeutrality < 1e-10))
    {
      nFailures++;
      if (rank == 0)
        std::cout << "FAILURE: the source is not charge neutral over the "
                     "domain, so the fully periodic problem has no solution. "
                     "Pick a source with zero mean.\n";
    }

    // What the mean-value constraint exists to enforce. The integral is taken
    // on the same quadrature the constraint was assembled on, so agreement is
    // limited by the solver tolerance rather than by the discretisation.
    if (!(meanOverRms < 1e-8))
    {
      nFailures++;
      if (rank == 0)
        std::cout << "FAILURE: the fully periodic solution does not have zero "
                     "mean, so the mean-value constraint did not take\n";
    }
  }
  else
  {
    // The negative control. This solution has a large non-zero mean, so if the
    // constraint had been applied where it does not belong the solve would
    // have come back shifted and this would catch it.
    if (!(meanOverRms > 1e-3))
    {
      nFailures++;
      if (rank == 0)
        std::cout << "FAILURE: the Dirichlet solution came back with zero "
                     "mean, so the mean-value constraint was applied where it "
                     "does not belong\n";
    }
  }

  return nFailures;
}

int main()
{

  std::cout<<" Entering test poisson problem classical \n";

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

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<memorySpace>> linAlgOpContextDevice =
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<memorySpace>>(10);

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>> linAlgOpContext =
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext<dftefe::utils::MemorySpace::HOST>>(0);

  dftefe::size_type nFailures = 0;

  // The original case, kept as is: adaptively refined, inhomogeneous Dirichlet
  // data on every face, mean-value constraint inactive.
  nFailures += runCase("Inhomogeneous Dirichlet",
                       false,
                       5,
                       true,
                       rhoDirichlet,
                       potentialDirichlet,
                       1e-2,
                       comm,
                       linAlgOpContext,
                       linAlgOpContextDevice,
                       rank);

  // Fully periodic, which makes the operator singular and turns the
  // mean-value constraint on inside the solver. Uniform mesh: hanging nodes
  // across a periodic face are a separate concern and would blur what a
  // failure here means.
  nFailures += runCase("Fully periodic, mean value pinned",
                       true,
                       10,
                       false,
                       rhoPeriodic,
                       potentialPeriodic,
                       1e-2,
                       comm,
                       linAlgOpContext,
                       linAlgOpContextDevice,
                       rank);

  if (rank == 0)
  {
    if (nFailures == 0)
      std::cout << "\nTestPoissonSolverDealiiMatrixFreeFE passed.\n"
                << std::flush;
    else
      std::cout << "\n" << nFailures << " check(s) did not pass.\n"
                << std::flush;
  }

  dftefe::utils::throwException(nFailures == 0,
                                "TestPoissonSolverDealiiMatrixFreeFE found " +
                                  std::to_string(nFailures) +
                                  " failing check(s); see the messages above.");

  //gracefully end MPI

  int mpiFinalFlag = 0;
  dftefe::utils::mpi::MPIFinalized(&mpiFinalFlag);
  if(!mpiFinalFlag)
  {
    dftefe::utils::mpi::MPIFinalize();
  }
}
