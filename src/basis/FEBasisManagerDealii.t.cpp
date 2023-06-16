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
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_tools.h>
#include "TriangulationDealiiParallel.h"
#include "TriangulationDealiiSerial.h"
#include "FECellDealii.h"


namespace dftefe
{
  namespace basis
  {
    template <size_type dim>
    FEBasisManagerDealii<dim>::FEBasisManagerDealii(
      std::shared_ptr<const TriangulationBase> triangulation,
      const size_type                          feOrder)
      : d_isHPRefined(false)
    {
      d_dofHandler = std::make_shared<dealii::DoFHandler<dim>>();
      reinit(triangulation, feOrder);
    }

    template <size_type dim>
    double
    FEBasisManagerDealii<dim>::getBasisFunctionValue(
      const size_type     basisId,
      const utils::Point &point) const
    {
      utils::throwException(
        false,
        "getBasisFunctionValue() in FEBasisManagerDealii not yet implemented.");
      return 0;
    }

    template <size_type dim>
    std::vector<double>
    FEBasisManagerDealii<dim>::getBasisFunctionDerivative(
      const size_type     basisId,
      const utils::Point &point,
      const size_type     derivativeOrder) const
    {
      utils::throwException(
        false,
        "getBasisFunctionDerivative() in FEBasisManagerDealii not yet implemented.");

      std::vector<double> vecReturn;
      return vecReturn;
    }

    template <size_type dim>
    void
    FEBasisManagerDealii<dim>::reinit(
      std::shared_ptr<const TriangulationBase> triangulation,
      const size_type                          feOrder)
    {
      dealii::FE_Q<dim>                       feElem(feOrder);
      const TriangulationDealiiParallel<dim> *dealiiParallelTria =
        dynamic_cast<const TriangulationDealiiParallel<dim> *>(
          triangulation.get());

      if (!(dealiiParallelTria == nullptr))
        {
          //          d_dofHandler->initialize(dealiiParallelTria->returnDealiiTria(),
          //          feElem);
          d_dofHandler->reinit(dealiiParallelTria->returnDealiiTria());
          d_dofHandler->distribute_dofs(feElem);
        }
      else
        {
          const TriangulationDealiiSerial<dim> *dealiiSerialTria =
            dynamic_cast<const TriangulationDealiiSerial<dim> *>(
              triangulation.get());

          if (!(dealiiSerialTria == nullptr))
            {
              //              d_dofHandler->initialize(dealiiSerialTria->returnDealiiTria(),
              //              feElem);
              d_dofHandler->reinit(dealiiSerialTria->returnDealiiTria());
              d_dofHandler->distribute_dofs(feElem);
            }
          else
            {
              utils::throwException(
                false,
                "reinit() in FEBasisManagerDealii is not able to re cast the Triangulation.");
            }
        }

      // TODO check how to pass the triangulation to dofHandler
      d_triangulation = triangulation;

      typename dealii::DoFHandler<dim>::active_cell_iterator cell =
        d_dofHandler->begin_active();
      typename dealii::DoFHandler<dim>::active_cell_iterator endc =
        d_dofHandler->end();

      cell = d_dofHandler->begin_active();
      endc = d_dofHandler->end();

      for (; cell != endc; cell++)
        if (cell->is_locally_owned())
          {
            std::shared_ptr<FECellDealii<dim>> cellDealii =
              std::make_shared<FECellDealii<dim>>(cell);

            d_localCells.push_back(cellDealii);
            d_locallyOwnedCells.push_back(cellDealii);
          }


      cell = d_dofHandler->begin_active();
      for (; cell != endc; cell++)
        if (cell->is_ghost())
          {
            std::shared_ptr<FECellDealii<dim>> cellDealii =
              std::make_shared<FECellDealii<dim>>(cell);
            d_localCells.push_back(cellDealii);
          }

      d_numCumulativeLocallyOwnedCellDofs = 0;
      d_numCumulativeLocalCellDofs        = 0;
      for (size_type iCell = 0; iCell < d_locallyOwnedCells.size(); ++iCell)
        d_numCumulativeLocallyOwnedCellDofs += nCellDofs(iCell);

      for (size_type iCell = 0; iCell < d_localCells.size(); ++iCell)
        d_numCumulativeLocalCellDofs += nCellDofs(iCell);
    }

    template <size_type dim>
    std::shared_ptr<const TriangulationBase>
    FEBasisManagerDealii<dim>::getTriangulation() const
    {
      return d_triangulation;
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nLocalCells() const
    {
      return d_localCells.size();
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.size();
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nGlobalCells() const
    {
      return d_triangulation->nGlobalCells();
    }

    // TODO put an assert condition to check if p refined is false
    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::getFEOrder(size_type cellId) const
    {
      return (d_dofHandler->get_fe().degree);
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nCellDofs(size_type cellId) const
    {
      return d_dofHandler->get_fe().n_dofs_per_cell();
    }

    template <size_type dim>
    bool
    FEBasisManagerDealii<dim>::isHPRefined() const
    {
      return d_isHPRefined;
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nLocalNodes() const
    {
      return d_dofHandler->n_locally_owned_dofs();
    }

    template <size_type dim>
    global_size_type
    FEBasisManagerDealii<dim>::nGlobalNodes() const
    {
      return d_dofHandler->n_dofs();
    }

    // template <size_type dim>
    // std::pair<global_size_type, global_size_type>
    // FEBasisManagerDealii<dim>::getLocallyOwnedRange() const
    // {
    //   auto             dealiiIndexSet = d_dofHandler->locally_owned_dofs();
    //   global_size_type startId        = *(dealiiIndexSet.begin());
    //   global_size_type endId = startId +
    //   d_dofHandler->n_locally_owned_dofs(); std::pair<global_size_type,
    //   global_size_type> returnValue =
    //     std::make_pair(startId, endId);
    //   return returnValue;
    // }

    template <size_type dim>
    std::vector<std::pair<global_size_type, global_size_type>>
    FEBasisManagerDealii<dim>::getLocallyOwnedRanges() const
    {
      std::vector<std::pair<global_size_type, global_size_type>> returnValue(0);
      auto             dealiiIndexSet = d_dofHandler->locally_owned_dofs();
      global_size_type startId        = *(dealiiIndexSet.begin());
      global_size_type endId = startId + d_dofHandler->n_locally_owned_dofs();
      std::pair<global_size_type, global_size_type> classicalRange =
        std::make_pair(startId, endId);

      returnValue.push_back(classicalRange);

      return returnValue;
    }

    template <size_type dim>
    std::map<BasisIdAttribute, size_type>
    FEBasisManagerDealii<dim>::getBasisAttributeToRangeIdMap() const
    {
      std::map<BasisIdAttribute, size_type> returnValue;
      returnValue[BasisIdAttribute::CLASSICAL] = 0;
      return returnValue;
    }

    template <size_type dim>
    std::vector<size_type>
    FEBasisManagerDealii<dim>::getLocalNodeIds(size_type cellId) const
    {
      utils::throwException(
        false,
        "getLocalNodeIds() in FEBasisManagerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;
      /// implement this now
    }

    template <size_type dim>
    std::vector<size_type>
    FEBasisManagerDealii<dim>::getGlobalNodeIds() const
    {
      utils::throwException(
        false,
        "getGlobalNodeIds() in FEBasisManagerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;

      /// implement this now
    }

    template <size_type dim>
    void
    FEBasisManagerDealii<dim>::getCellDofsGlobalIds(
      size_type                      cellId,
      std::vector<global_size_type> &vecGlobalNodeId) const
    {
      vecGlobalNodeId.resize(nCellDofs(cellId), 0);

      d_locallyOwnedCells[cellId]->cellNodeIdtoGlobalNodeId(vecGlobalNodeId);
    }

    template <size_type dim>
    std::vector<size_type>
    FEBasisManagerDealii<dim>::getBoundaryIds() const
    {
      utils::throwException(
        false,
        "getBoundaryIds() in FEBasisManagerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;
      //// implement this now ?
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    FEBasisManagerDealii<dim>::beginLocallyOwnedCells()
    {
      return d_locallyOwnedCells.begin();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    FEBasisManagerDealii<dim>::endLocallyOwnedCells()
    {
      return d_locallyOwnedCells.end();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    FEBasisManagerDealii<dim>::beginLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.begin();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    FEBasisManagerDealii<dim>::endLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.end();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    FEBasisManagerDealii<dim>::beginLocalCells()
    {
      return d_localCells.begin();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    FEBasisManagerDealii<dim>::endLocalCells()
    {
      return d_localCells.end();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    FEBasisManagerDealii<dim>::beginLocalCells() const
    {
      return d_localCells.begin();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    FEBasisManagerDealii<dim>::endLocalCells() const
    {
      return d_localCells.end();
    }

    template <size_type dim>
    unsigned int
    FEBasisManagerDealii<dim>::getDim() const
    {
      return dim;
    }

    //
    // dealii specific functions
    //
    template <size_type dim>
    std::shared_ptr<const dealii::DoFHandler<dim>>
    FEBasisManagerDealii<dim>::getDoFHandler() const
    {
      return d_dofHandler;
    }

    template <size_type dim>
    const dealii::FiniteElement<dim> &
    FEBasisManagerDealii<dim>::getReferenceFE(const size_type cellId) const
    {
      //
      // NOTE: The implementation is only restricted to
      // h-refinement (uniform p) and hence the reference FE
      // is same for all cellId. As a result, we pass index
      // 0 to dealii's dofHandler
      //
      if (d_isHPRefined)
        {
          utils::throwException(
            false,
            "Support for hp-refined finite element mesh is not supported yet.");
        }

      return d_dofHandler->get_fe(0);
    }

    template <size_type dim>
    void
    FEBasisManagerDealii<dim>::getBasisCenters(
      std::map<global_size_type, utils::Point> &dofCoords) const
    {
      // TODO if the creation of linear mapping is inefficient, then this has to
      // be improved
      std::map<global_size_type, dealii::Point<dim, double>> dealiiDofCoords;
      dealii::MappingQ1<dim, dim>                            mappingQ1;
      dealii::DoFTools::map_dofs_to_support_points<dim, dim>(
        mappingQ1, *(d_dofHandler.get()), dealiiDofCoords);

      convertToDftefePoint<dim>(dealiiDofCoords, dofCoords);
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nCumulativeLocallyOwnedCellDofs() const
    {
      return d_numCumulativeLocallyOwnedCellDofs;
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nCumulativeLocalCellDofs() const
    {
      return d_numCumulativeLocalCellDofs;
    }

  } // namespace basis
} // namespace dftefe
