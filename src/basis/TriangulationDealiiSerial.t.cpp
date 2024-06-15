#include <utils/Exceptions.h>
#include "DealiiConversions.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/grid_out.h>
#include <fstream>

namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    TriangulationDealiiSerial<dim>::TriangulationDealiiSerial()
      : isInitialized(false)
      , isFinalized(false)
      , d_isPeriodicFlags(0)
    {}

    template <unsigned int dim>
    TriangulationDealiiSerial<dim>::~TriangulationDealiiSerial()
    {
      d_triangulationDealii.clear();
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::initializeTriangulationConstruction()
    {
      isInitialized = true;
      isFinalized   = false;
      for (unsigned int iCell = 0; iCell < nLocallyOwnedCells(); iCell++)
        {
          // delete
          d_triaVectorCell[iCell].reset();
        }

      d_triaVectorCell.resize(0);
      d_triangulationDealii.clear();
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::finalizeTriangulationConstruction()
    {
      isInitialized      = false;
      isFinalized        = true;
      unsigned int iCell = 0;
      d_triaVectorCell.resize(nLocallyOwnedCells());

      for (unsigned int iLevel = 0;
           iLevel < d_triangulationDealii.n_global_levels();
           iLevel++)
        {
          for (auto &cellPtr :
               d_triangulationDealii.active_cell_iterators_on_level(iLevel))
            {
              d_triaVectorCell[iCell] =
                std::make_shared<TriangulationCellDealii<dim>>(cellPtr);
              iCell++;
            }
        }

      utils::throwException(
        iCell == nLocallyOwnedCells(),
        "Number of active cells is not matching in Finalize."
        "Kneel at the altar of Lord Vishal and"
        "he may grant your wish to rectify this error.");
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::createUniformParallelepiped(
      const std::vector<unsigned int> &subdivisions,
      const std::vector<utils::Point> &domainVectors,
      const std::vector<bool> &        isPeriodicFlags)
    {
      utils::throwException(isInitialized && !isFinalized,
                            "Cannot create triangulation without calling"
                            "initializeTriangulationConstruction");
      DFTEFE_AssertWithMsg(
        dim == domainVectors.size(),
        "Mismatch of dimension for dealii and the domain vectors");
      dealii::Point<dim, double> dealiiPoints[dim];

      for (unsigned int i = 0; i < dim; ++i)
        {
          convertToDealiiPoint<dim>(domainVectors[i], dealiiPoints[i]);
        }

      unsigned int dealiiSubdivisions[dim];
      std::copy(subdivisions.begin(), subdivisions.end(), dealiiSubdivisions);


      dealii::GridGenerator::subdivided_parallelepiped<dim>(
        d_triangulationDealii, dealiiSubdivisions, dealiiPoints);
      markPeriodicFaces(isPeriodicFlags, domainVectors);

      d_isPeriodicFlags.resize(dim);
      d_isPeriodicFlags = isPeriodicFlags;

      d_domainVectors = domainVectors;
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::createSingleCellTriangulation(
      const std::vector<utils::Point> &vertices)
    {
      utils::throwException(isInitialized && !isFinalized,
                            "Cannot create triangulation without calling"
                            "initializeTriangulationConstruction");
      DFTEFE_AssertWithMsg(dim == vertices[0].size(),
                           "Mismatch of dimension for dealii and the vertices");
      const unsigned int                      numPoints = vertices.size();
      std::vector<dealii::Point<dim, double>> dealiiVertices(numPoints);
      for (unsigned int i = 0; i < numPoints; ++i)
        {
          convertToDealiiPoint<dim>(vertices[i], dealiiVertices[i]);
        }

      dealii::GridGenerator::general_cell(d_triangulationDealii,
                                          dealiiVertices);
    }


    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::shiftTriangulation(
      const utils::Point &origin)
    {
      utils::throwException<utils::LogicError>(
        isInitialized && !isFinalized,
        "Cannot shift triangulation without calling"
        "initializeTriangulationConstruction");
      DFTEFE_AssertWithMsg(dim == origin.size(),
                           "Mismatch of dimension for dealii and the origin");
      dealii::Point<dim, double> dealiiOrigin;
      convertToDealiiPoint<dim>(origin, dealiiOrigin);
      dealii::GridTools::shift(dealiiOrigin, d_triangulationDealii);
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::markPeriodicFaces(
      const std::vector<bool> &        isPeriodicFlags,
      const std::vector<utils::Point> &domainVectors)
    {
      utils::throwException<utils::LogicError>(
        isInitialized && !isFinalized,
        "Cannot mark periodic faces in triangulation without"
        "calling initializeTriangulationConstruction");

      DFTEFE_AssertWithMsg(
        dim == isPeriodicFlags.size(),
        "Mismatch of dimension for dealii and the isPeriodicFlags");

      DFTEFE_AssertWithMsg(
        dim == domainVectors.size(),
        "Mismatch of dimension for dealii and the domainVectors");

      DFTEFE_AssertWithMsg(d_triangulationDealii.n_global_levels() == 1,
                           "Cannot mark periodic faces after refinement."
                           "This has to be done at the coarsest level");

      for (unsigned int i = 0; i < dim; ++i)
        {
          if (isPeriodicFlags[i] == true)
            {
              utils::throwException<utils::InvalidArgument>(
                false,
                "The markPeriodicFaces has not yet been implemented for periodic problems."
                "Please ask Vishal to implement it.");
            }
        }
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::refineGlobal(
      const unsigned int times /* = 1 */)
    {
      utils::throwException<utils::LogicError>(
        isInitialized && !isFinalized,
        "Cannot refine triangulation without calling"
        "initializeTriangulationConstruction");
      d_triangulationDealii.refine_global(times);
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::coarsenGlobal(
      const unsigned int times /* = 1 */)
    {
      utils::throwException<utils::LogicError>(
        isInitialized && !isFinalized,
        "Cannot coarsen triangulation without calling"
        "initializeTriangulationConstruction");
      d_triangulationDealii.coarsen_global(times);
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::clearUserFlags()
    {
      utils::throwException<utils::LogicError>(
        isInitialized && !isFinalized,
        "Cannot clear user flags triangulation without calling"
        "initializeTriangulationConstruction");
      d_triangulationDealii.clear_user_flags();
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::executeCoarseningAndRefinement()
    {
      /*
       utils::throwException<utils::LogicError>(
         isInitialized && !isFinalized,
         "Cannot execute coarsening or refinement of triangulation without
       calling" "initializeTriangulationConstruction");*/
      d_triangulationDealii.execute_coarsening_and_refinement();
    }

    template <unsigned int dim>
    size_type
    TriangulationDealiiSerial<dim>::nLocallyOwnedCells() const
    {
      return d_triangulationDealii.n_active_cells();
    }

    template <unsigned int dim>
    size_type
    TriangulationDealiiSerial<dim>::nGlobalCells() const
    {
      return d_triangulationDealii.n_global_active_cells();
    }

    template <unsigned int dim>
    size_type
    TriangulationDealiiSerial<dim>::nLocalCells() const
    {
      return d_triangulationDealii.n_cells();
    }

    template <unsigned int dim>
    std::vector<size_type>
    TriangulationDealiiSerial<dim>::getBoundaryIds() const
    {
      utils::throwException<utils::LogicError>(
        isInitialized && !isFinalized,
        "Cannot execute coarsening or refinement of triangulation without calling"
        "initializeTriangulationConstruction");
      utils::throwException(
        false,
        "The getBoundaryIds() in TriangulationDealiiSerial has not"
        "yet been implemented. Please ask Vishal to implement it.");

      return std::vector<size_type>(0);
    }

    template <unsigned int dim>
    double
    TriangulationDealiiSerial<dim>::maxElementLength() const
    {
      double maxElemLength = 0.0;
      auto   cell          = beginLocal();
      auto   endc          = endLocal();
      for (; cell != endc; ++cell)
        {
          if ((*cell)->minimumVertexDistance() > maxElemLength)
            maxElemLength = (*cell)->minimumVertexDistance();
        }

      return maxElemLength;
    }

    template <unsigned int dim>
    double
    TriangulationDealiiSerial<dim>::minElementLength() const
    {
      double minElemLength = 1e8;
      auto   cell          = beginLocal();
      auto   endc          = endLocal();
      for (; cell != endc; ++cell)
        {
          if ((*cell)->minimumVertexDistance() < minElemLength)
            minElemLength = (*cell)->minimumVertexDistance();
        }

      return minElemLength;
    }

    template <unsigned int dim>
    TriangulationBase::TriangulationCellIterator
    TriangulationDealiiSerial<dim>::beginLocal()
    {
      return d_triaVectorCell.begin();
    }

    template <unsigned int dim>
    TriangulationBase::TriangulationCellIterator
    TriangulationDealiiSerial<dim>::endLocal()
    {
      return d_triaVectorCell.end();
    }

    template <unsigned int dim>
    TriangulationBase::const_TriangulationCellIterator
    TriangulationDealiiSerial<dim>::beginLocal() const
    {
      return d_triaVectorCell.begin();
    }

    template <unsigned int dim>
    TriangulationBase::const_TriangulationCellIterator
    TriangulationDealiiSerial<dim>::endLocal() const
    {
      return d_triaVectorCell.end();
    }

    template <unsigned int dim>
    unsigned int
    TriangulationDealiiSerial<dim>::getDim() const
    {
      return dim;
    }

    template <unsigned int dim>
    std::vector<bool>
    TriangulationDealiiSerial<dim>::getPeriodicFlags() const
    {
      return d_isPeriodicFlags;
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::saveRefineFlags(std::vector<bool> &v) const
    {
      d_triangulationDealii.save_refine_flags(v);
    }

    template <unsigned int dim>
    void
    TriangulationDealiiSerial<dim>::writeToVtkFile(std::ostream &out) const
    {
      dealii::GridOut grid_out;
      grid_out.write_vtk(d_triangulationDealii, out);
    }

    template <unsigned int dim>
    std::vector<utils::Point>
    TriangulationDealiiSerial<dim>::getDomainVectors() const
    {
      return d_domainVectors;
    }

    template <unsigned int dim>
    const dealii::Triangulation<dim> &
    TriangulationDealiiSerial<dim>::returnDealiiTria() const
    {
      return d_triangulationDealii;
    }
  } // namespace basis

} // namespace dftefe
