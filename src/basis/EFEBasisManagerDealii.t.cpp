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
 * @author Avirup Sircar
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
    EFEBasisManagerDealii<dim>::EFEBasisManagerDealii(
        std::shared_ptr<const TriangulationBase>     triangulation,
        std::shared_ptr<const atoms::AtomSphericalDataContainer> atomSphericalDataContainer,
        const size_type                              feOrder,
        const double                                 atomPartitionTolerance,
        const std::vector<std::string> &             atomSymbolVec,
        const std::vector<utils::Point> &            atomCoordinatesVec,
        const std::string                            fieldName,
        const utils::mpi::MPIComm &                  comm)
      : d_isHPRefined(false),
        d_atomSphericalDataContainer(
          std::make_shared<const atoms::AtomSphericalDataContainer> atomSphericalDataContainer),
        d_atomSymbolVec(atomSymbolVec),
        d_atomCoordinatesVec(atomCoordinatesVec),
        d_fieldName(fieldName)
    {
      d_dofHandler = std::make_shared<dealii::DoFHandler<dim>>();
      // making the classical and enriched dofs in the dealii mesh here
      reinit(triangulation, 
        feOrder, 
        atomPartitionTolerance,
        comm);
    }

    template <size_type dim>
    void
    EFEBasisManagerDealii<dim>::reinit(
        std::shared_ptr<const TriangulationBase>     triangulation,
        const size_type                              feOrder,
        const double                                 atomPartitionTolerance,
        const utils::mpi::MPIComm &                  comm)
    {
      // Create Classical FE dof_handler
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
                "reinit() in EFEBasisManagerDealii is not able to re cast the Triangulation.");
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

      //----------------ENRICHEMNT ADD --------------------------------------------------------//

      // Add enriched FE dofs with on top of the classical dofs.
      // Note that the enriched FE dofs are already partitioned elswhere 
      // This implies we pass mpi communicator here also.
      // So we only work with local enrichement ids which are created.

      cell = d_dofHandler->begin_active();

      for (; cell != endc; cell++)
        if (cell->is_locally_owned())
        {
          std::shared_ptr<FECellDealii<dim>> cellDealii =
            std::make_shared<FECellDealii<dim>>(cell);

            cellDealii->getVertices(cellVertices);
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

      // Create atomIdsPartition Object.
      d_atomIdsPartition = make_shared<const atomIdsPartition>(
        d_atomCoordinatesVec,
        minbound,
        maxbound,
        cellVerticesVector,
        atomPartitionTolerance,
        comm );

      // Create enrichmentIdsPartition Object.
      d_enrichmentIdsPartition = make_shared<const enrichmentIdsPartition>(
        d_atomSphericalDataContainer,
        d_atomIdsPartition,
        d_atomSymbolVec,
        d_atomCoordinatesVec,
        d_fieldName,
        minbound,
        maxbound,
        cellVerticesVector,
        comm );

      d_overlappingEnrichmentIdsInCells = d_enrichmentIdsPartition->overlappingEnrichmentIdsInCells();
    }

    template <size_type dim>
    double
    EFEBasisManagerDealii<dim>::getBasisFunctionValue(
      const size_type     basisId,
      const utils::Point &point) const
    {
      utils::throwException(
        false,
        "getBasisFunctionValue() in EFEBasisManagerDealii not yet implemented.");
      return 0;
    }

    template <size_type dim>
    std::vector<double>
    EFEBasisManagerDealii<dim>::getBasisFunctionDerivative(
      const size_type     basisId,
      const utils::Point &point,
      const size_type     derivativeOrder) const
    {
      utils::throwException(
        false,
        "getBasisFunctionDerivative() in EFEBasisManagerDealii not yet implemented.");

      std::vector<double> vecReturn;
      return vecReturn;
    }

    template <size_type dim>
    std::shared_ptr<const TriangulationBase>
    EFEBasisManagerDealii<dim>::getTriangulation() const
    {
      return d_triangulation;
    }

    template <size_type dim>
    size_type
    EFEBasisManagerDealii<dim>::nLocalCells() const
    {
      return d_localCells.size();
    }

    template <size_type dim>
    size_type
    EFEBasisManagerDealii<dim>::nLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.size();
    }

    template <size_type dim>
    size_type
    EFEBasisManagerDealii<dim>::nGlobalCells() const
    {
      return d_triangulation->nGlobalCells();
    }

    // TODO put an assert condition to check if p refined is false
    template <size_type dim>
    size_type
    EFEBasisManagerDealii<dim>::getFEOrder(size_type cellId) const
    {
      return (d_dofHandler->get_fe().degree);
    }

    template <size_type dim>
    size_type
    EFEBasisManagerDealii<dim>::nCellDofs(size_type cellId) const
    {
      size_type classicalDofs = d_dofHandler->get_fe().n_dofs_per_cell();
      size_type enrichedDofs = d_overlappingEnrichmentIdsInCells[cellId].size();
      return (classicalDofs + enrichedDofs);
    }

    template <size_type dim>
    bool
    EFEBasisManagerDealii<dim>::isHPRefined() const
    {
      return d_isHPRefined;
    }

    template <size_type dim>
    size_type
    EFEBasisManagerDealii<dim>::nLocalNodes() const
    {
      return (d_dofHandler->n_locally_owned_dofs() + d_enrichmentIdsPartition->nLocallyOwnedEnrichmentIds());
    }

    template <size_type dim>
    global_size_type
    EFEBasisManagerDealii<dim>::nGlobalNodes() const
    {
      return (d_dofHandler->n_dofs() + d_enrichmentIdsPartition->nTotalEnrichmentIds()) ;
    }

    std::vector<std::pair<global_size_type, global_size_type>>
    getLocallyOwnedRanges()
    {
      std::vector<std::pair<global_size_type, global_size_type>> returnValue(0);
      auto             dealiiIndexSet = d_dofHandler->locally_owned_dofs();
      global_size_type startId        = *(dealiiIndexSet.begin());
      global_size_type endId = startId + d_dofHandler->n_locally_owned_dofs();
      std::pair<global_size_type, global_size_type> classicalRange =
        std::make_pair(startId, endId);

      returnValue.push_back(classicalRange);
      returnValue.push_back(d_enrichmentIdsPartition->locallyOwnedEnrichmentIds());

      return returnValue;
    }

    std::map < BasisIdAttribute basisIdAttribute , std::pair<global_size_type, global_size_type> >
    getLocallyOwnedRangeMap()
    {
      std::map < BasisIdAttribute basisIdAttribute , std::pair<global_size_type, global_size_type> > returnValue;

      std::vector<std::pair<global_size_type, global_size_type>> locallyOwnedRangeVec(0);
      locallyOwnedRangeVec = getLocallyOwnedRanges();
      returnValue[BasisIdAttribute::CLASSICAL] = locallyOwnedRangeVec[0];
      returnValue[BasisIdAttribute::ENRICHED] =  locallyOwnedRangeVec[1];

      return returnValue;
    }

    template <size_type dim>
    std::vector<size_type>
    EFEBasisManagerDealii<dim>::getLocalNodeIds(size_type cellId) const
    {
      utils::throwException(
        false,
        "getLocalNodeIds() in EFEBasisManagerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;
      /// implement this now
    }

    template <size_type dim>
    std::vector<size_type>
    EFEBasisManagerDealii<dim>::getGlobalNodeIds() const
    {
      utils::throwException(
        false,
        "getGlobalNodeIds() in EFEBasisManagerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;

      /// implement this now
    }

    template <size_type dim>
    void
    EFEBasisManagerDealii<dim>::getCellDofsGlobalIds(
      size_type                      cellId,
      std::vector<global_size_type> &vecGlobalNodeId) const
    {
      vecGlobalNodeId.resize(nCellDofs(cellId), 0);
      vecGlobalClassicalNodeId.resize(d_dofHandler->get_fe().n_dofs_per_cell(), 0);

      d_locallyOwnedCells[cellId]->cellNodeIdtoGlobalNodeId(vecGlobalClassicalNodeId);

      size_type count = 0, classicalcount = 0, enrichedcount = 0;
      std::vector<global_size_type> vecGlobalEnrichedNodeId(0);
      vecGlobalEnrichedNodeId = d_overlappingEnrichmentIdsInCells[cellId];

      for( auto i:vecGlobalNodeId )
      {
        if ( i < d_dofHandler->get_fe().n_dofs_per_cell())
        {
          vecGlobalNodeId[count] = vecGlobalClassicalNodeId[classicalcount];
          classicalcount += 1;
        }
        else
        {
          vecGlobalNodeId[count] = vecGlobalEnrichedNodeId[enrichedcount];
          enrichedcount += 1;
        }
        count = count + 1;
      }
    }

    template <size_type dim>
    std::vector<size_type>
    EFEBasisManagerDealii<dim>::getBoundaryIds() const
    {
      utils::throwException(
        false,
        "getBoundaryIds() in EFEBasisManagerDealii is not be implemented.");
      std::vector<size_type> vec;
      return vec;
      //// implement this now ?
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    EFEBasisManagerDealii<dim>::beginLocallyOwnedCells()
    {
      return d_locallyOwnedCells.begin();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    EFEBasisManagerDealii<dim>::endLocallyOwnedCells()
    {
      return d_locallyOwnedCells.end();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    EFEBasisManagerDealii<dim>::beginLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.begin();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    EFEBasisManagerDealii<dim>::endLocallyOwnedCells() const
    {
      return d_locallyOwnedCells.end();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    EFEBasisManagerDealii<dim>::beginLocalCells()
    {
      return d_localCells.begin();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::iterator
    EFEBasisManagerDealii<dim>::endLocalCells()
    {
      return d_localCells.end();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    EFEBasisManagerDealii<dim>::beginLocalCells() const
    {
      return d_localCells.begin();
    }

    template <size_type dim>
    std::vector<std::shared_ptr<FECellBase>>::const_iterator
    EFEBasisManagerDealii<dim>::endLocalCells() const
    {
      return d_localCells.end();
    }

    template <size_type dim>
    unsigned int
    EFEBasisManagerDealii<dim>::getDim() const
    {
      return dim;
    }

    //
    // dealii specific functions
    //
    template <size_type dim>
    std::shared_ptr<const dealii::DoFHandler<dim>>
    EFEBasisManagerDealii<dim>::getDoFHandler() const
    {
      return d_dofHandler;
    }

    template <size_type dim>
    const dealii::FiniteElement<dim> &
    EFEBasisManagerDealii<dim>::getReferenceFE(const size_type cellId) const
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
    EFEBasisManagerDealii<dim>::getBasisCenters(
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
    EFEBasisManagerDealii<dim>::nCumulativeLocallyOwnedCellDofs() const
    {
      return d_numCumulativeLocallyOwnedCellDofs;
    }

    template <size_type dim>
    size_type
    EFEBasisManagerDealii<dim>::nCumulativeLocalCellDofs() const
    {
      return d_numCumulativeLocalCellDofs;
    }

    // Enrichment functions with dealii mesh. The enrichedid is the cell local id.
    double
    getEnrichmentValue(
      const size_type cellId,
      const size_type cellLocalEnrichmentId,
      const dftefe::utils::Point & point) const
    {
      double retValue = 0;
      if(!= d_overlappingEnrichmentIdsInCells[cellId].empty())
      {
        if(d_overlappingEnrichmentIdsInCells[cellId].size() > cellLocalEnrichmentId)
        {
          double  polarAngleTolerance = 0; //Change it
          size_type globalEnrichmentId = d_overlappingEnrichmentIdsInCells[cellId][cellLocalEnrichmentId];
          size_type atomId = d_enrichmentIdsPartition->getAtomId(globalEnrichmentId);
          size_type qNumberId = 
            (d_enrichmentIdsPartition->getEnrichmentIdAttribute(globalEnrichmentId)).localIdInAtom;
          std::string atomSymbol = d_atomSymbolVec[atomId];
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          std::vector<int> qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol, d_fieldName);
          std::allocator<atoms::sphericalData> alloc;
          d_sphericalData =
            std::allocate_shared<atoms::sphericalData> 
            (alloc, d_atomSphericalDataContainer->getSphericalData(
              atomSymbol,
              d_fieldName, 
              qNumbers[qNumberId]));
          d_sphericalData->initSpline();
          retValue = d_sphericalData->getValue(point, origin, polarAngleTolerance);
        }
        else
        {
          utils::throwException(
          false,
          "The requested cell local enrichment id does not exist.");
        }
      }
      else
      {
        utils::throwException(
        false,
        "The requested cell does not have any enrichment ids overlapping with it.");
      }
      return retValue;
    }

    std::vector<double>
    getEnrichmentDerivative(
      const size_type cellId,
      const size_type cellLocalEnrichmentId,
      const dftefe::utils::Point & point) const 
    {
      std::vector<double> retValue(0);
      if(!= d_overlappingEnrichmentIdsInCells[cellId].empty())
      {
        if(d_overlappingEnrichmentIdsInCells[cellId].size() > cellLocalEnrichmentId)
        {
          double  polarAngleTolerance = 0; //Change it
          double  cutoffTolerance = 0; //Change it
          size_type globalEnrichmentId = d_overlappingEnrichmentIdsInCells[cellId][cellLocalEnrichmentId];
          size_type atomId = d_enrichmentIdsPartition->getAtomId(globalEnrichmentId);
          size_type qNumberId = 
            (d_enrichmentIdsPartition->getEnrichmentIdAttribute(globalEnrichmentId)).localIdInAtom;
          std::string atomSymbol = d_atomSymbolVec[atomId];
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          std::vector<int> qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol, d_fieldName);
          std::allocator<atoms::sphericalData> alloc;
          d_sphericalData =
            std::allocate_shared<atoms::sphericalData> 
            (alloc, d_atomSphericalDataContainer->getSphericalData(
              atomSymbol,
              d_fieldName, 
              qNumbers[qNumberId]));
          d_sphericalData->initSpline();
          retValue = d_sphericalData->getGradientValue(point, origin, polarAngleTolerance, cutoffTolerance);
        }
        else
        {
          utils::throwException(
          false,
          "The requested cell local enrichment id does not exist.");
        }
      }
      else
      {
        utils::throwException(
        false,
        "The requested cellid does not have any enrichment ids overlapping with it.");
      }
      return retValue;
    }

    std::vector<double>
    getEnrichmentHessian(     
      const size_type cellId,
      const size_type cellLocalEnrichmentId,
      const dftefe::utils::Point & point) const
    {
      std::vector<double> retValue(0);
      if(!= d_overlappingEnrichmentIdsInCells[cellId].empty())
      {
        if(d_overlappingEnrichmentIdsInCells[cellId].size() > cellLocalEnrichmentId)
        {
          double  polarAngleTolerance = 0; //Change it
          double  cutoffTolerance = 0; //Change it
          size_type globalEnrichmentId = d_overlappingEnrichmentIdsInCells[cellId][cellLocalEnrichmentId];
          size_type atomId = d_enrichmentIdsPartition->getAtomId(globalEnrichmentId);
          size_type qNumberId = 
            (d_enrichmentIdsPartition->getEnrichmentIdAttribute(globalEnrichmentId)).localIdInAtom;
          std::string atomSymbol = d_atomSymbolVec[atomId];
          utils::Point origin(d_atomCoordinatesVec[atomId]);
          std::vector<int> qNumbers = d_atomSphericalDataContainer->getQNumbers(atomSymbol, d_fieldName);
          std::allocator<atoms::sphericalData> alloc;
          d_sphericalData =
            std::allocate_shared<atoms::sphericalData> 
            (alloc, d_atomSphericalDataContainer->getSphericalData(
              atomSymbol,
              d_fieldName, 
              qNumbers[qNumberId]));
          d_sphericalData->initSpline();
          retValue = d_sphericalData->getHessianValue(point, origin, polarAngleTolerance, cutoffTolerance);
        }
        else
        {
          utils::throwException(
          false,
          "The requested cell local enrichment id does not exist.");
        }
      }
      else
      {
        utils::throwException(
        false,
        "The requested cell does not have any enrichment ids overlapping with it.");
      }
      return retValue;
    }

  } // namespace basis
} // namespace dftefe
