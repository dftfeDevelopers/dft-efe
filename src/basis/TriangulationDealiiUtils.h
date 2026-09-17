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

#ifndef dftefeTriangulationDealiiUtils_h
#define dftefeTriangulationDealiiUtils_h

#include <utils/TypeConfig.h>
#include <utils/Point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <vector>

namespace dftefe
{
  namespace basis
  {
    /**
     * @brief Free functions shared by TriangulationDealiiSerial and
     * TriangulationDealiiParallel (and, downstream, by the DoFHandler-level
     * periodic constraint construction in CFEBasisDofHandlerDealii and
     * EFEBasisDofHandlerDealii) to periodize a parallelepiped domain.
     *
     * These mirror the dftfe meshGenUtils namespace
     * (dftfe/include/meshGenUtils.h), including the boundary-id convention:
     * the b-th periodic direction (counted in increasing dimension index,
     * skipping the non-periodic ones) gets boundary ids 2b+1 and 2b+2 on its
     * two faces, and every non-periodic boundary face is left at boundary id
     * 0.
     */
    namespace TriangulationDealiiUtils
    {
      /**
       * @brief Returns the dimension indices that are flagged periodic, in
       * increasing order. Its position in the returned vector is the index
       * @p b used by the 2b+1 / 2b+2 boundary-id convention.
       */
      inline std::vector<size_type>
      getPeriodicDirections(const std::vector<bool> &isPeriodicFlags);

      /**
       * @brief Computes the outward normal direction of the pair of faces
       * bounding the domain along each lattice direction, as
       * @f$(a_2 \times a_3, a_3 \times a_1, a_1 \times a_2)@f$. Only
       * meaningful for dim = 3.
       */
      inline void
      computePeriodicFaceNormals(
        const std::vector<utils::Point> &domainVectors,
        std::vector<utils::Point> &      periodicFaceNormals);

      /**
       * @brief Computes the translation that maps the face with the higher
       * boundary id onto its partner, i.e. the negated lattice vectors. This
       * is the offset argument expected by
       * dealii::GridTools::collect_periodic_faces.
       */
      template <size_type dim>
      void
      computeOffsetVectors(const std::vector<utils::Point> &   domainVectors,
                           std::vector<dealii::Tensor<1, dim>> &offsetVectors);

      /**
       * @brief Cosine of the angle between a dealii Tensor and a utils::Point
       * treated as a vector.
       */
      template <size_type dim>
      double
      getCosineAngle(const dealii::Tensor<1, dim> &vector1,
                     const utils::Point &          vector2);

      /**
       * @brief Stamps the periodic boundary ids on @p triangulation and
       * registers the matched face pairs on it via add_periodicity(). Must be
       * called on the coarsest level, before any refinement.
       *
       * A no-op when no direction is flagged periodic, so the non-periodic
       * path is untouched. Takes the base dealii::Triangulation reference so
       * that the same body serves both the serial triangulation and the
       * parallel::distributed one (add_periodicity is virtual).
       *
       * @throws utils::InvalidArgument if dim != 3, or if the lattice vectors
       * are left-handed, i.e. @f$(a_1 \times a_2)\cdot a_3 \le 0@f$, in which
       * case the face pairing produced below would be wrong.
       */
      template <size_type dim>
      void
      markPeriodicFacesAndAddPeriodicity(
        dealii::Triangulation<dim> &     triangulation,
        const std::vector<bool> &        isPeriodicFlags,
        const std::vector<utils::Point> &domainVectors);

    } // namespace TriangulationDealiiUtils
  }   // namespace basis
} // namespace dftefe
#include "TriangulationDealiiUtils.t.cpp"
#endif // dftefeTriangulationDealiiUtils_h
