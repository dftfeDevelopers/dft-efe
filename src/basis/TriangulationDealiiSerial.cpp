#include <utils/Exceptions.h>
#include <utils/DealiiConversions.h>
#include "TriangulationDealiiSerial.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/point.h>

namespace dftefe
{
  namespace basis
  {
    
    <template unsigned int dim>
      TriangulationDealiiSerial<dim>::TriangulationDealiiSerial():
	  isInitialized(false),
	  isFinalized(false)
      {

      }

    <template unsigned int dim>
      TriangulationDealiiSerial<dim>::~TriangulationDealiiSerial()
      {

      }

    <template unsigned int dim>
      void
      TriangulationDealiiSerial<dim>::initializeTriangulationConstruction()
    {
      isInitialized = true;
      isFinalized   = false;

      for (unsigned int iCell = 0; iCell < nLocalCells(); iCell++)
        {
          delete d_triaVectorCell[iCell];
        }
      d_triangulationDealii.clear();
    }
    <template unsigned int dim> TriangulationDealiiSerial<
      dim>::finalizeTriangulationConstruction()
    {
      isInitialized      = false;
      isFinalized        = true;
      unsigned int iCell = 0;
      d_triaVectorCell.resize(nLocalCells());

      for (unsigned int iLevel = 0;
           iLevel < d_triangulationDealii.n_global_levels();
           iLevel++)
        {
          for (auto cellPtr : d_triangulationDealii.begin_active(iLevel))
            {
              d_triaVectorCell[iCell] =
                std::make_shared<TriaCellDealii>(cellPtr);
              iCell++;
            }
        }

  utils::throwException(iCell == nLocalCells() , "Number of active cells is not matching in Finalize.
	    You can not do anyhting now, Vishal has to rectify this");
    }

    <template unsigned int dim> void
    TriangulationDealiiSerial<dim>::createUniformParallelepiped(
      const std::vector<unsigned int> &subdivisions,
      const std::vector<utils::Point> &domainVectors,
      const std::vector<bool> &        isPeriodicFlags)
    {
  utils::throwException(isInitialized && !isFinalized, "Cannot create triangulation without calling
	    initializeTriangulationConstruction");
	utils::Assert(dim == domainVectors.size(),
	    "Mismatch of dimension for dealii and the domain vectors");
	dealii::Point<dim, double> * points = new dealii::Point<dim, double>[dim];
	for(unsigned int i = 0; i < dim; ++i)
	{
         utils::convertToDealiiPoint<dim>(domainVectors[i], points[i]);
	}

	dealii::GridGenerator::subdivided_parallelepiped<dim>(d_triangulationDealii,
	    &subdivisions[0],
	    points);
	markPeriodicFaces(isPeriodicFlags, domainVectors);
    }

    <template unsigned int dim> void
    TriangulationDealiiSerial<dim>::shiftTriangulation(
      const utils::Point &origin)
    {
      utils::throwException<utils::LogicError>(
        isInitialized && !isFinalized,
        "Cannot shift triangulation without calling initializeTriangulationConstruction");
      utils::Assert(dim == origin.size(),
                    "Mismatch of dimension for dealii and the origin");
      dealii::Point<dim, double> dealiiOrigin ;
      utils::convertToDealiiPoint<dim>(origin, dealiiOrigin);
      dealii::GridTools::shift(dealiiOrigin, d_triangulationDealii);
    }

    <template unsigned int dim> void
    TriangulationDealiiSerial<dim>::markPeriodicFaces(
      const std::vector<bool> &        isPeriodicFlags,
      const std::vector<utils::Point> &domainVectors)
    {
      utils::throwException<utils::LogicError>(
        isInitialized && !isFinalized,
        "Cannot mark periodic faces in triangulation without calling initializeTriangulationConstruction");

      utils::Assert(dim == isPeriodicFlags.size(),
                    "Mismatch of dimension for dealii and the isPeriodicFlags");

      utils::Assert(dim == domainVectors.size(),
                    "Mismatch of dimension for dealii and the domainVectors");

  utils::Assert(d_triangulationDealii.n_global_levels == 1,
	    "Cannot mark periodic faces after refinement. This has to be done
	    at the coarsest level");

	for(unsigned int i = 0; i < dim; ++i)
	{
        if (isPeriodicFlags[i] == true)
          {
      utils::throwException<utils::InvalidArgument>(false, "The markPeriodicFaces has not
		yet been implemented for periodic problems. Please ask Vishal to implement it.");
          }
	}
    }

    <template unsigned int dim> void
    TriangulationDealiiSerial<dim>::refineGlobal(const unsigned int times = 1)
    {
  utils::throwException<utils::LogicError>(isInitialized && !isFinalized, "Cannot refine triangulation without calling
	    initializeTriangulationConstruction");
	d_triangulationDealii.refine_global(times);
    }

    <template unsigned int dim> void
    TriangulationDealiiSerial<dim>::coarsenGlobal(const unsigned int times = 1)
    {
  utils::throwException<utils::LogicError>(isInitialized && !isFinalized, "Cannot coarsen triangulation without calling
	    initializeTriangulationConstruction");
	d_triangulationDealii.coarsen_global(times);
    }

    <template unsigned int dim> void
    TriangulationDealiiSerial<dim>::clearUserFlags()
    {
  utils::throwException<utils::LogicError>(isInitialized && !isFinalized, "Cannot clear user flags triangulation without calling
	    initializeTriangulationConstruction");
	d_triangulationDealii.clear_user_flags();
    }

    <template unsigned int dim> void
    TriangulationDealiiSerial<dim>::executeCoarseningAndRefinement()
    {
  utils::throwException<utils::LogicError>(isInitialized && !isFinalized, "Cannot execute coarsening or refinement of triangulation without calling
	    initializeTriangulationConstruction");
	d_triangulationDealii.execute_coarsening_and_refinement();
    }

    <template unsigned int dim> size_type
    TriangulationDealiiSerial<dim>::nLocalCells() const
    {
      return d_triangulationDealii.n_active_cells();
    }

    <template unsigned int dim> size_type
    TriangulationDealiiSerial<dim>::nGlobalCells() const
    {
      return d_triangulationDealii.n_global_active_cells();
    }

    <template unsigned int dim>
      std::vector< size_type >
      TriangulationDealiiSerial<dim>::getBoundaryIds () const
      {
	utils::throwException<utils::LogicError>(isInitialized && !isFinalized, "Cannot execute coarsening or refinement of triangulation without calling
	    initializeTriangulationConstruction");
	utils::throwException("The getBoundaryIds() in TriangulationDealiiSerial has not
	    yet been implemented. Please ask Vishal to implement it.");
      }

    <template unsigned int dim>
      unsigned int
      TriangulationDealiiSerial<dim>::getDim () const
    {
      return dim;
    }
  } // namespace basis

} // namespace dftefe
