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

namespace dftefe
{
  namespace basis
  {
    template <size_type dim>
    FECellDealii<dim>::FECellDealii(
      typename dealii::DoFHandler<dim>::active_cell_iterator dealiiFECellIter)
    {
      // TODO check if this is correct and wont lead to seg faults
      d_dealiiFECellIter = dealiiFECellIter;
    }

    template <size_type dim>
    void
    FECellDealii<dim>::getVertices(std::vector<utils::Point> &points) const
    {
      const size_type nVertices =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      points.resize(nVertices, utils::Point(dim));
      std::vector<dealii::Point<dim, double>> pointsDealii;
      pointsDealii.resize(nVertices);
      for (size_type iVertex = 0; iVertex < nVertices; iVertex++)
        {
          pointsDealii[iVertex] = d_dealiiFECellIter->vertex(iVertex);
          convertToDftefePoint<dim>(d_dealiiFECellIter->vertex(iVertex),
                                    points[iVertex]);
        }
    }

    template <size_type dim>
    void
    FECellDealii<dim>::getVertex(size_type i, utils::Point &point) const
    {
      convertToDftefePoint<dim>(d_dealiiFECellIter->vertex(i), point);
    }

    template <size_type dim>
    std::vector<std::shared_ptr<dftefe::utils::Point>>
    FECellDealii<dim>::getNodalPoints() const
    {
      utils::throwException(
        false, "getNodalPoints() in FECellDealii can not be implemented.");

      std::vector<std::shared_ptr<dftefe::utils::Point>> returnPoint;
      return returnPoint;
    }

    template <size_type dim>
    size_type
    FECellDealii<dim>::getId() const
    {
      utils::throwException(false,
                            "getId() in FECellDealii not yet implemented.");
      return 0;
    }

    template <size_type dim>
    bool
    FECellDealii<dim>::isPointInside(const utils::Point &point) const
    {
      dealii::Point<dim, double> dealiiPoint;
      convertToDealiiPoint<dim>(point, dealiiPoint);
      return d_dealiiFECellIter->point_inside(dealiiPoint);
    }

    template <size_type dim>
    bool
    FECellDealii<dim>::isAtBoundary(const size_type i) const
    {
      return d_dealiiFECellIter->at_boundary(i);
    }

    template <size_type dim>
    bool
    FECellDealii<dim>::hasPeriodicNeighbor(const size_type i) const
    {
      return d_dealiiFECellIter->has_periodic_neighbor(i);
    }

    template <size_type dim>
    bool
    FECellDealii<dim>::isAtBoundary() const
    {
      return d_dealiiFECellIter->at_boundary();
    }
    template <size_type dim>
    double
    FECellDealii<dim>::diameter() const
    {
      return d_dealiiFECellIter->diameter();
    }

    template <size_type dim>
    void
    FECellDealii<dim>::center(dftefe::utils::Point &centerPoint) const
    {
      dealii::Point<dim, double> dealiiPoint;
      dealiiPoint = d_dealiiFECellIter->center();
      convertToDftefePoint<dim>(dealiiPoint, centerPoint);
    }


    template <size_type dim>
    void
    FECellDealii<dim>::setRefineFlag()
    {
      d_dealiiFECellIter->set_refine_flag();
    }

    template <size_type dim>
    void
    FECellDealii<dim>::clearRefineFlag()
    {
      d_dealiiFECellIter->clear_refine_flag();
    }

    template <size_type dim>
    double
    FECellDealii<dim>::minimumVertexDistance() const
    {
      return d_dealiiFECellIter->minimum_vertex_distance();
    }

    template <size_type dim>
    double
    FECellDealii<dim>::distanceToUnitCell(
      dftefe::utils::Point &parametricPoint) const
    {
      dealii::Point<dim, double> dealiiPoint;
      convertToDealiiPoint<dim>(parametricPoint, dealiiPoint);
      return dealii::GeometryInfo<dim>::distance_to_unit_cell(dealiiPoint);
    }

    template <size_type dim>
    void
    FECellDealii<dim>::setCoarsenFlag()
    {
      d_dealiiFECellIter->set_coarsen_flag();
    }

    template <size_type dim>
    void
    FECellDealii<dim>::clearCoarsenFlag()
    {
      d_dealiiFECellIter->clear_coarsen_flag();
    }

    template <size_type dim>
    bool
    FECellDealii<dim>::isActive() const
    {
      return d_dealiiFECellIter->is_active();
    }

    template <size_type dim>
    bool
    FECellDealii<dim>::isLocallyOwned() const
    {
      return d_dealiiFECellIter->is_locally_owned();
    }

    template <size_type dim>
    bool
    FECellDealii<dim>::isGhost() const
    {
      return d_dealiiFECellIter->is_ghost();
    }

    template <size_type dim>
    bool
    FECellDealii<dim>::isArtificial() const
    {
      return d_dealiiFECellIter->is_artificial();
    }

    template <size_type dim>
    size_type
    FECellDealii<dim>::getDim() const
    {
      return dim;
    }

    template <size_type dim>
    void
    FECellDealii<dim>::getParametricPoint(const utils::Point &   realPoint,
                                          const CellMappingBase &cellMapping,
                                          utils::Point &parametricPoint) const
    {
      bool isPointInside;
      cellMapping.getParametricPoint(realPoint,
                                     *this,
                                     parametricPoint,
                                     isPointInside);
    }

    template <size_type dim>
    void
    FECellDealii<dim>::getRealPoint(const utils::Point &   parametricPoint,
                                    const CellMappingBase &cellMapping,
                                    utils::Point &         realPoint) const
    {
      bool isPointInside;
      cellMapping.getRealPoint(parametricPoint, *this, realPoint);
    }

    template <size_type dim>
    void
    FECellDealii<dim>::cellNodeIdtoGlobalNodeId(
      std::vector<global_size_type> &vecId) const
    {
      d_dealiiFECellIter->get_dof_indices(vecId);
    }

    template <size_type dim>
    size_type
    FECellDealii<dim>::getFaceBoundaryId(size_type faceId) const
    {
      return d_dealiiFECellIter->face(faceId)->boundary_id();
    }

    template <size_type dim>
    void
    FECellDealii<dim>::getFaceDoFGlobalIndices(
      size_type                      faceId,
      std::vector<global_size_type> &vecNodeId) const
    {
      d_dealiiFECellIter->face(faceId)->get_dof_indices(vecNodeId);
    }

    template <size_type dim>
    size_type
    FECellDealii<dim>::getFEOrder() const
    {
      return (d_dealiiFECellIter->get_fe().degree);
    }

    template <size_type dim>
    typename dealii::DoFHandler<dim>::active_cell_iterator &
    FECellDealii<dim>::getDealiiFECellIter()
    {
      return d_dealiiFECellIter;
    }

  } // namespace basis
} // namespace dftefe
