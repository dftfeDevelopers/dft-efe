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
 * @author Bikash Kanungo, Vishal Subramanian
 */


#include "DealiiConversions.h"
#include <utils/Exceptions.h>
#include <deal.II/base/geometry_info.h>

namespace dealii
{
  namespace basis
  {
    template <unsigned int dim>
    FECellDealii<dim>::FECellDealii(
      dealii::DoFHandler<dim>::active_cell_iterator dealiiFECellIter)
    {
      // TODO check if this is correct and wont lead to seg faults
      d_dealiiFECellIter = dealiiFECellIter;
    }

    template <unsigned int dim>
    std::vector<std::shared_ptr<dftefe::utils::Point>>
    FECellDealii<dim>::getVertices() const
    {
      const unsigned int nVertices =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      std::vector<std::shared_ptr<dftefe::utils::Point>> points(
        nVertices, std::make_shared<dftefe::utils::Point>(dim, 0.0));
      std::vector<dealii::Point<dim, double>> pointsDealii;
      pointsDealii.resize(nVertices);
      for (unsigned int iVertex = 0; iVertex < nVertices; iVertex++)
        {
          pointsDealii[iVertex] = d_dealiiFECellIter->vertex(iVertex);
          convertToDftefePoint<dim>(d_dealiiFECellIter->vertex(iVertex),
                                    points[iVertex]);
        }

      return points;
    }

    template <unsigned int dim>
    std::shared_ptr<dftefe::utils::Point>
    FECellDealii<dim>::getVertex(size_type i) const
    {
      std::shared_ptr<dftefe::utils::Point> point =
        std::make_shared<dftefe::utils::Point>(dim, 0.0);
      ;

      convertToDftefePoint<dim>(d_dealiiFECellIter->vertex(i), point);

      return (point);
    }

    template <unsigned int dim>
    std::vector<std::shared_ptr<dftefe::utils::Point>>
    FECellDealii<dim>::getNodalPoints() const
    {
      utils::throwException(
        false, "getNodalPoints() in FECellDealii not yet implemented.");
      return 0;
    }

    template <unsigned int dim>
    size_type
    FECellDealii<dim>::getId() const
    {
      utils::throwException(false,
                            "getId() in FECellDealii not yet implemented.");
      return 0;
    }

    template <unsigned int dim>
    bool
    FECellDealii<dim>::isPointInside(
      std::shared_ptr<const dftefe::utils::Point> point) const
    {
      dealii::Point<dim, double> dealiiPoint;
      convertToDealiiPoint<dim>(point, dealiiPoint);
      return d_dealiiFECellIter->point_inside(dealiiPoint);
    }

    template <unsigned int dim>
    bool
    FECellDealii<dim>::isAtBoundary(const unsigned int i) const
    {
      return d_dealiiFECellIter->at_boundary(i);
    }

    template <unsigned int dim>
    bool
    FECellDealii<dim>::isAtBoundary() const
    {
      return d_dealiiFECellIter->at_boundary();
    }

    template <unsigned int dim>
    void
    FECellDealii<dim>::setRefineFlag()
    {
      d_dealiiFECellIter->set_refine_flag();
    }

    template <unsigned int dim>
    void
    FECellDealii<dim>::clearRefineFlag()
    {
      d_dealiiFECellIter->clear_refine_flag();
    }

    template <unsigned int dim>
    void
    FECellDealii<dim>::setCoarsenFlag()
    {
      d_dealiiFECellIter->set_coarsen_flag();
    }

    template <unsigned int dim>
    void
    FECellDealii<dim>::clearCoarsenFlag()
    {
      d_dealiiFECellIter->clear_coarsen_flag();
    }

    template <unsigned int dim>
    bool
    FECellDealii<dim>::isActive() const
    {
      d_dealiiFECellIter->is_active();
    }

    template <unsigned int dim>
    bool
    FECellDealii<dim>::isLocallyOwned() const
    {
      d_dealiiFECellIter->is_locally_owned();
    }

    template <unsigned int dim>
    bool
    FECellDealii<dim>::isGhost() const
    {
      d_dealiiFECellIter->is_ghost();
    }

    template <unsigned int dim>
    bool
    FECellDealii<dim>::isArtificial() const
    {
      d_dealiiFECellIter->is_artificial();
    }

    template <unsigned int dim>
    int
    FECellDealii<dim>::getDim() const
    {
      return dim;
    }

    template <unsigned int dim>
    std::shared_ptr<dftefe::utils::Point>
    FECellDealii<dim>::getParametricPoint(
      std::shared_ptr<const Point> realPoint,
      const CellMappingBase &      cellMapping) const
    {
      std::shared_ptr<dftefe::utils::Point> parametricPoint =
        std::make_shared<dftefe::utils::Point>(dim, 0.0);
      bool isPointInside;
      cellMapping.getParametricPoint(realPoint,
                                     *this,
                                     parametricPoint,
                                     isPointInside);

      return parametricPoint;
    }

    template <unsigned int dim>
    std::shared_ptr<Point>
    FECellDealii<dim>::getRealPoint(
      std::shared_ptr<const Point> parametricPoint,
      const CellMappingBase &      cellMapping) const
    {
      utils::throwException(
        false, "getRealPoint() in FECellDealii not yet implemented.");
    }

    template <unsigned int dim>
    global_size_type
    FECellDealii<dim>::getLocalToGlobalDoFId(size_type i) const
    {}

    template <unsigned int dim>
    size_type
    FECellDealii<dim>::getFEOrder() const
    {
      return (d_dealiiFECellIter->get_fet().degree());
    }

    template <unsigned int dim>
    dealii::DoFHandler<dim>::active_cell_iterator &
    FECellDealii<dim>::getDealiiFECellIter()
    {
      return d_dealiiFECellIter;
    }

  } // namespace basis
} // namespace dealii
