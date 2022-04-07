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

namespace dftefe
{
  namespace basis
  {
    template <size_type dim>
    FEBasisManagerDealii<dim>::FEBasisManagerDealii(TriangulationBase &tria)
      : d_isHPRefined(false)
    {

      d_triangulation = tria;
      d_dofHandler = std::make_shared<dealii::DoFHandler< dim>();

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
    }

    template <size_type dim>
    std::vector<double>
    FEBasisManagerDealii<dim>::getBasisFunctionDerivative(
      const size_type     basisId,
      const utils::Point &point,
      const size_type     derivativeOrder = 1) const
    {
      utils::throwException(
        false,
        "getBasisFunctionDerivative() in FEBasisManagerDealii not yet implemented.");
    }

    template <size_type dim>
    void
    FEBasisManagerDealii<dim>::reinit(const TriangulationBase &triangulation,
                                      const size_type          feOrder)
    {
      const TriangulationDealiiParallel<dim> dealiiParallelTria =
        dynamic_cast<const TriangulationDealiiParallel<dim> &>(triangulation);

      if (!(dealiiParallelTria == nullptr) )
        {
          d_dofHandler->initialize(dealiiParallelTria->returnDealiiTria(), feOrder);
        }
      else
        {
          const TriangulationDealiiSerial<dim> dealiiSerialTria =
            dynamic_cast<const TriangulationDealiiSerial<dim> &>(triangulation);

          if (!(dealiiParallelTria == nullptr) )
            {
              d_dofHandler->initialize(dealiiSerialTria->returnDealiiTria(), feOrder);
            }
          else
            {
              utils::throwException(
                false, "reinit() in FEBasisManagerDealii is not able to re cast the Triangulation.");
            }

        }

      // TODO check how to pass the triangulation to dofHandler
      d_triangulation = triangulation;
      d_dofHandler->initialize(triangulation, feOrder);
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nLocallyActiveCells() const
    {
      utils::throwException(
        false,
        "nLocallyActiveCells() in FEBasisManagerDealii not yet implemented.");
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nLocallyOwnedCells() const
    {
      utils::throwException(
        false,
        "nLocallyOwnedCells() in FEBasisManagerDealii not yet implemented.");
    }

    template <size_type dim>
    size_type
    FEBasisManagerDealii<dim>::nGlobalCells() const
    {
      utils::throwException(
        false, "nGlobalCells() in FEBasisManagerDealii not yet implemented.");
    }

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
      d_dofHandler->n_locally_owned_dofs();
    }

    template <size_type dim>
    global_size_type
    FEBasisManagerDealii<dim>::nGlobalNodes() const
    {
      return d_dofHandler->n_dofs();
    }

    template <size_type dim>
    std::vector<size_type>
    FEBasisManagerDealii<dim>::getLocalNodeIds(size_type cellId) const
    {}

    template <size_type dim>
    std::vector<size_type>
    FEBasisManagerDealii<dim>::getGlobalNodeIds() const
    {}

    template <size_type dim>
    std::vector<size_type>
    FEBasisManagerDealii<dim>::getCellDofsLocalIds(size_type cellId) const
    {}

    template <size_type dim>
    std::vector<size_type>
    FEBasisManagerDealii<dim>::getBoundaryIds() const
    {}

    template <size_type dim>
    FEBasisManager<dim>::std::vector<std::shared_ptr<FECellBase>>::iterator
    FEBasisManagerDealii<dim>::beginLocallyOwnedCells()
    {
      d_dofHandler->begin();
    }

    template <size_type dim>
    FEBasisManager<dim>::std::vector<std::shared_ptr<FECellBase>>::iterator
    FEBasisManagerDealii<dim>::endLocallyOwnedCells()
    {
      d_dofHandler->end();
    }

    template <size_type dim>
    FEBasisManager<dim>::std::vector<
      std::shared_ptr<FECellBase>>::const_iterator
    FEBasisManagerDealii<dim>::beginLocallyOwnedCells() const
    {
      d_dofHandler->begin();
    }

    FEBasisManager<dim>::std::vector<
      std::shared_ptr<FECellBase>>::const_iterator
    FEBasisManagerDealii<dim>::endLocallyOwnedCells() const
    {
      d_dofHandler->end();
    }

    template <size_type dim>
    FEBasisManager<dim>::std::vector<std::shared_ptr<FECellBase>>::iterator
    FEBasisManagerDealii<dim>::beginLocallyActiveCells()
    {
      d_dofHandler->begin_active();
    }

    template <size_type dim>
    FEBasisManager<dim>::std::vector<std::shared_ptr<FECellBase>>::iterator
    FEBasisManagerDealii<dim>::endLocallyActiveCells()
    {
      d_dofHandler->end_active();
    }

    template <size_type dim>
    FEBasisManager<dim>::std::vector<
      std::shared_ptr<FECellBase>>::const_iterator
    FEBasisManagerDealii<dim>::beginLocallyActiveCells() const
    {
      d_dofHandler->begin_active();
    }

    template <size_type dim>
    FEBasisManager<dim>::std::vector<
      std::shared_ptr<FECellBase>>::const_iterator
    FEBasisManagerDealii<dim>::endLocallyActiveCells() const
    {
      d_dofHandler->end_active();
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
    FEBasisManagerDealii<dim>::getDoFHandler()
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
      if(d_isHPRefined)
      {
	utils::throwException(false, 
	"Support for hp-refined finite element mesh is not supported yet.");
      }
      
      return d_dofHandler->get_fe(0);
    }

  } // namespace basis
} // namespace dftefe
