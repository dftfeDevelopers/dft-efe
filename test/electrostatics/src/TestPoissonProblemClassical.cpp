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
#include <basis/FECellWiseDataOperations.h>
#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <utils/MemoryStorage.h>
#include <utils/ScalarZeroFunctionReal.h>
#include <utils/Exceptions.h>
#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <functional>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <linearAlgebra/LinearSolverFunction.h>
#include <electrostatics/PoissonLinearSolverFunctionFE.h>
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
//
// The solved equation is -\nabla^2 \phi = \rho, with no 1/(4 pi) anywhere in
// the solver. The smeared charge pair below carries its own factor of 4 pi in
// rho, which is why the analytic potential is the bare 1/r outside r_c; the
// periodic pair is built to the same convention.

//
// Case 1: smeared nuclear charge with inhomogeneous Dirichlet data. Unchanged.
//
    double rho(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin, double rc)
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

    double potential(const dftefe::utils::Point &point, const std::vector<dftefe::utils::Point> &origin, double rc)
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

//
// Case 2: fully periodic, which is what exercises the mean-value constraint.
//
// With no Dirichlet boundary anywhere the Poisson operator is singular: any
// constant is in its null space. The solver pins that null space by imposing
// \int_\Omega \phi d\Omega = 0, so a test solution is only well posed if it is
// periodic AND already has zero mean.
//
//   \phi = sin(kx) sin(ky) sin(kz),   k = 2 pi / L
//   \rho = -\nabla^2 \phi = 3 k^2 \phi
//
// Each factor runs over a whole period across the box, so \phi and \rho both
// integrate to zero over the domain. Zero mean \rho is the compatibility
// condition without which the periodic problem has no solution at all; zero
// mean \phi makes this exact solution the very representative the constraint
// selects, so the computed field can be compared against it directly rather
// than up to a constant.
//
    double potentialPeriodic(const dftefe::utils::Point &point, double L)
    {
      const double k = 2.0 * M_PI / L;
      return std::sin(k * point[0]) * std::sin(k * point[1]) *
             std::sin(k * point[2]);
    }

    double rhoPeriodic(const dftefe::utils::Point &point, double L)
    {
      const double k = 2.0 * M_PI / L;
      return 3.0 * k * k * potentialPeriodic(point, L);
    }

  class ScalarSpatialPotentialFunctionReal : public dftefe::utils::ScalarSpatialFunctionReal
  {
    public:
    ScalarSpatialPotentialFunctionReal(
      const std::function<double(const dftefe::utils::Point &)> &potential)
    :d_potential(potential)
    {}

    double
    operator()(const dftefe::utils::Point &point) const
    {
      return d_potential(point);
    }

    std::vector<double>
    operator()(const std::vector<dftefe::utils::Point> &points) const
    {
      std::vector<double> ret(0);
      ret.resize(points.size());
      for (unsigned int i = 0 ; i < points.size() ; i++)
      {
        ret[i] = d_potential(points[i]);
      }
      return ret;
    }

    private:
    std::function<double(const dftefe::utils::Point &)> d_potential;
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
// refineAroundPoints empty means no adaptive refinement. When it is not, the
// original radius-driven sweep is run verbatim, hMin included.
//
dftefe::size_type
runCase(const std::string &                                        name,
        const bool                                                 isPeriodic,
        const dftefe::size_type                                    nSubdivisions,
        const double                                               domainLength,
        const std::function<double(const dftefe::utils::Point &)> &rhoFn,
        const std::function<double(const dftefe::utils::Point &)> &potentialFn,
        const double                             relativeErrorTol,
        const std::vector<dftefe::utils::Point> &refineAroundPoints,
        const double                             refineRadius,
        const double                             hMin,
        const dftefe::utils::mpi::MPIComm &      comm,
        std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext<
          dftefe::utils::MemorySpace::HOST>>     linAlgOpContext,
        const int                                rank)
{
  dftefe::size_type nFailures = 0;

  const unsigned int dim = 3;
  unsigned int numComponents = 1;
  dftefe::size_type maxIter = 2e7;
  double absoluteTol = 1e-10;
  double relativeTol = 1e-12;
  double divergenceTol = 1e10;

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

  if (!refineAroundPoints.empty())
  {
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
        for ( unsigned int i=0 ; i<refineAroundPoints.size() ; i++)
        {
        double dist = 0;
        for (unsigned int j = 0 ; j < dim ; j++ )
        {
            dist += std::pow((centerPoint[j]-refineAroundPoints[i][j]),2);
        }
        dist = std::sqrt(dist);
        if(dist < refineRadius)
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
  }

  unsigned int feDegree = 3;

  std::shared_ptr<const dftefe::basis::FEBasisDofHandler<double, dftefe::utils::MemorySpace::HOST,dim>> basisDofHandler =
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
      std::make_shared<ScalarSpatialPotentialFunctionReal>(potentialFn);

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
  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainerAnalytical(quadRuleContainer, numComponents);
  dftefe::quadrature::QuadratureValuesContainer<double, dftefe::utils::MemorySpace::HOST> quadValuesContainerNumerical(quadRuleContainer, numComponents);

  for(dftefe::size_type i = 0 ; i < quadValuesContainer.nCells() ; i++)
  {
    for(dftefe::size_type iComp = 0 ; iComp < numComponents ; iComp ++)
    {
      dftefe::size_type quadId = 0;
      std::vector<double> a(quadRuleContainer->nCellQuadraturePoints(i));
      for (auto j : quadRuleContainer->getCellRealPoints(i))
      {
        a[quadId] = rhoFn( j );
        quadId = quadId + 1;
      }
      double *b = a.data();
      quadValuesContainer.setCellValues<dftefe::utils::MemorySpace::HOST> (i, b);
    }
  }

  std::shared_ptr<dftefe::linearAlgebra::LinearSolverFunction<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST>> linearSolverFunction =
    std::make_shared<dftefe::electrostatics::PoissonLinearSolverFunctionFE<double,
                                                   double,
                                                   dftefe::utils::MemorySpace::HOST,
                                                   dim>>
                                                   (basisManager,
                                                    feBasisData,
                                                    feBasisData,
                                                    quadValuesContainer,
                                                    dftefe::linearAlgebra::PreconditionerType::JACOBI ,
                                                    linAlgOpContext,
                                                    50,
                                                    numComponents);

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

  for(dftefe::size_type i = 0 ; i < quadValuesContainerAnalytical.nCells() ; i++)
  {
    for(dftefe::size_type iComp = 0 ; iComp < numComponents ; iComp ++)
    {
      dftefe::size_type quadId = 0;
      std::vector<double> a(quadRuleContainer->nCellQuadraturePoints(i));
      for (auto j : quadRuleContainer->getCellRealPoints(i))
      {
        a[quadId] = potentialFn( j );
        quadId = quadId + 1;
      }
      double *b = a.data();
      quadValuesContainerAnalytical.setCellValues<dftefe::utils::MemorySpace::HOST> (i, b);
    }
  }

  feBasisOp.interpolate( *solution, *basisManager, quadValuesContainerNumerical);

  //
  // Everything is measured on the quadrature points. The error there is the L2
  // error of the field itself, and it sidesteps having to reason about which
  // nodal values are constrained.
  //
  auto iterPotAnalytic = quadValuesContainerAnalytical.begin();
  auto iterPotNumeric = quadValuesContainerNumerical.begin();
  auto iterRho = quadValuesContainer.begin();
  dftefe::size_type numQuadraturePoints = quadRuleContainer->nQuadraturePoints(), mpinumQuadraturePoints=0;
  const std::vector<double> JxW = quadRuleContainer->getJxW();
  std::vector<double> integral(8, 0.0), mpiReducedIntegral(integral.size(), 0.0);

  for (dftefe::size_type i = 0 ; i < numQuadraturePoints ; i++ )
  {
      integral[0] += std::pow((*(i+iterPotAnalytic) - *(i+iterPotNumeric)),2) * JxW[i];
      integral[1] += *(i+iterRho) * *(i+iterPotNumeric) * JxW[i] * 0.5/(4*M_PI);
      integral[2] += *(i+iterRho) * JxW[i]/(4*M_PI);
      // The analytic field, and the computed field's mean, mean square and the
      // volume they are taken over.
      integral[3] += std::pow(*(i+iterPotAnalytic),2) * JxW[i];
      integral[4] += *(i+iterPotNumeric) * JxW[i];
      integral[5] += std::pow(*(i+iterPotNumeric),2) * JxW[i];
      integral[6] += JxW[i];
      // The source and its magnitude, for the solvability check below.
      integral[7] += std::abs(*(i+iterRho)) * JxW[i];
  }

  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
      &numQuadraturePoints,
      &mpinumQuadraturePoints,
      1,
      dftefe::utils::mpi::MPIUnsignedLong,
      dftefe::utils::mpi::MPISum,
      comm);

  dftefe::utils::mpi::MPIAllreduce<dftefe::utils::MemorySpace::HOST>(
      integral.data(),
      mpiReducedIntegral.data(),
      integral.size(),
      dftefe::utils::mpi::MPIDouble,
      dftefe::utils::mpi::MPISum,
      comm);

  const double relativeError =
    std::sqrt(mpiReducedIntegral[0] / mpiReducedIntegral[3]);
  const double volume    = mpiReducedIntegral[6];
  const double meanValue = mpiReducedIntegral[4] / volume;
  // Root mean square of the computed field, to measure the mean against
  // something with the same units instead of against an absolute number.
  const double rms         = std::sqrt(mpiReducedIntegral[5] / volume);
  const double meanOverRms = std::abs(meanValue) / rms;
  // The source integral, likewise measured against the integral of its
  // magnitude so the criterion does not depend on how strong the source is.
  const double rhoIntegral = 4 * M_PI * mpiReducedIntegral[2];
  const double rhoNeutrality =
    (mpiReducedIntegral[7] > 0.0) ?
      std::abs(rhoIntegral) / mpiReducedIntegral[7] :
      0.0;

  if (rank == 0)
  {
    std::cout << "No. of quad points: "<< mpinumQuadraturePoints<<"\n";
    std::cout << "Integral of b over volume: "<< mpiReducedIntegral[2]<<"\n";
    std::cout << "The integral L2 norm of potential: " << std::sqrt(mpiReducedIntegral[0]) << "\n";
    std::cout << "Relative L2 error against the analytic solution: "
              << relativeError << " (bound " << relativeErrorTol << ")\n";
    std::cout << "Integral of the computed potential: "
              << mpiReducedIntegral[4] << ", domain volume: " << volume << "\n";
    std::cout << "Mean value: " << meanValue << ", rms: " << rms
              << ", ratio: " << meanOverRms << "\n";
    std::cout << "Integral of the source: " << rhoIntegral
              << ", of its magnitude: " << mpiReducedIntegral[7]
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
    // The negative control. A smeared positive charge in a box has a large
    // positive mean potential, so if the constraint had been applied where it
    // does not belong the solve would have come back shifted and this would
    // catch it.
    if (!(meanOverRms > 1e-3))
    {
      nFailures++;
      if (rank == 0)
        std::cout << "FAILURE: the Dirichlet solution came back with zero "
                     "mean, so the mean-value constraint was applied where it "
                     "does not belong\n";
    }

    // The smeared-charge self energy, kept as a diagnostic rather than a
    // check: on this mesh r_c is comparable to the element size, so the
    // quantity is not converged enough for a bound to mean much. Printed so
    // the periodic work can be seen not to have disturbed it.
    const double Ig = 10976./(17875*0.5);
    if (rank == 0)
      std::cout << "Self energy integral: " << mpiReducedIntegral[1]
                << " (analytic I_g term " << Ig << ")\n"
                << std::flush;
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

  // Get the rank of the process
  int rank;
  dftefe::utils::mpi::MPICommRank(comm, &rank);

  // Get nProcs
  int numProcs;
  dftefe::utils::mpi::MPICommSize(comm, &numProcs);

  std::shared_ptr<dftefe::linearAlgebra::LinAlgOpContext
    <dftefe::utils::MemorySpace::HOST>> linAlgOpContext =
    std::make_shared<dftefe::linearAlgebra::LinAlgOpContext
    <dftefe::utils::MemorySpace::HOST>>(0);

  const unsigned int dim = 3;
  const double domainLength = 20.0;
  const double rc = 0.5;
  const double refineradius = 3*rc;
  // Left at the original value, at which the refinement condition
  // diameter() > hMin never fires, so the Dirichlet mesh stays uniform.
  const double hMin = 1e6;

  // Enrichment data file consisting of g(r,\theta,\phi) = f(r)*Y_lm(\theta, \phi)
  char* dftefe_path = getenv("DFTEFE_PATH");
  std::string sourceDir;
  // if executes if a non null value is returned
  // otherwise else executes
  if (dftefe_path != NULL)
  {
    sourceDir = (std::string)dftefe_path + "/test/electrostatics/src/";
  }
  else
  {
    dftefe::utils::throwException(false,
                          "dftefe_path does not exist!");
  }
  std::string atomDataFile = "SingleAtomData.in";
  std::string inputFileName = sourceDir + atomDataFile;
  std::fstream fstream;

  fstream.open(inputFileName, std::fstream::in);

  // read the input file and create atomsymbol vector and atom coordinates vector.
  std::vector<dftefe::utils::Point> atomCoordinatesVec;
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

  dftefe::size_type nFailures = 0;

  // The original case, kept as is: smeared nuclear charge, inhomogeneous
  // Dirichlet data on every face, mean-value constraint inactive.
  nFailures += runCase(
    "Smeared charge, inhomogeneous Dirichlet",
    false,
    20,
    domainLength,
    [&](const dftefe::utils::Point &p) { return rho(p, atomCoordinatesVec, rc); },
    [&](const dftefe::utils::Point &p) { return potential(p, atomCoordinatesVec, rc); },
    1e-1,
    atomCoordinatesVec,
    refineradius,
    hMin,
    comm,
    linAlgOpContext,
    rank);

  // Fully periodic, which makes the operator singular and turns the
  // mean-value constraint on inside the solver. Uniform and coarser: hanging
  // nodes across a periodic face are a separate concern, and the sine is
  // resolved far more cheaply than the smeared charge.
  nFailures += runCase(
    "Fully periodic, mean value pinned",
    true,
    10,
    domainLength,
    [&](const dftefe::utils::Point &p) { return rhoPeriodic(p, domainLength); },
    [&](const dftefe::utils::Point &p) { return potentialPeriodic(p, domainLength); },
    1e-2,
    std::vector<dftefe::utils::Point>(),
    refineradius,
    hMin,
    comm,
    linAlgOpContext,
    rank);

  if (rank == 0)
  {
    if (nFailures == 0)
      std::cout << "\nTestPoissonProblemClassical passed.\n" << std::flush;
    else
      std::cout << "\n" << nFailures << " check(s) did not pass.\n"
                << std::flush;
  }

  dftefe::utils::throwException(nFailures == 0,
                                "TestPoissonProblemClassical found " +
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
