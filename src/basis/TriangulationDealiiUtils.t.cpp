#include <utils/Exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <cmath>
#include <string>

namespace dftefe
{
  namespace basis
  {
    namespace TriangulationDealiiUtils
    {
      inline std::vector<size_type>
      getPeriodicDirections(const std::vector<bool> &isPeriodicFlags)
      {
        size_type numPeriodicDirections = 0;
        for (size_type i = 0; i < isPeriodicFlags.size(); ++i)
          if (isPeriodicFlags[i] == true)
            numPeriodicDirections++;

        std::vector<size_type> periodicDirections(numPeriodicDirections, 0);
        size_type              index = 0;
        for (size_type i = 0; i < isPeriodicFlags.size(); ++i)
          if (isPeriodicFlags[i] == true)
            periodicDirections[index++] = i;

        return periodicDirections;
      }

      inline void
      computePeriodicFaceNormals(
        const std::vector<utils::Point> &domainVectors,
        std::vector<utils::Point> &      periodicFaceNormals)
      {
        utils::throwException<utils::InvalidArgument>(
          domainVectors.size() == 3,
          "computePeriodicFaceNormals requires three lattice vectors.");

        periodicFaceNormals.resize(3, utils::Point(3, 0.0));

        const size_type cyclic[3][2] = {{1, 2}, {2, 0}, {0, 1}};
        for (size_type d = 0; d < 3; ++d)
          {
            const utils::Point &a = domainVectors[cyclic[d][0]];
            const utils::Point &b = domainVectors[cyclic[d][1]];
            periodicFaceNormals[d][0] = a[1] * b[2] - a[2] * b[1];
            periodicFaceNormals[d][1] = a[2] * b[0] - a[0] * b[2];
            periodicFaceNormals[d][2] = a[0] * b[1] - a[1] * b[0];
          }
      }

      template <size_type dim>
      void
      computeOffsetVectors(const std::vector<utils::Point> &    domainVectors,
                           std::vector<dealii::Tensor<1, dim>> &offsetVectors)
      {
        offsetVectors.resize(domainVectors.size());
        for (size_type i = 0; i < domainVectors.size(); ++i)
          {
            for (size_type j = 0; j < dim; ++j)
              offsetVectors[i][j] = -domainVectors[i][j];
          }
      }

      template <size_type dim>
      double
      getCosineAngle(const dealii::Tensor<1, dim> &vector1,
                     const utils::Point &          vector2)
      {
        double dotProduct      = 0.0;
        double lengthVector1Sq = 0.0;
        double lengthVector2Sq = 0.0;
        for (size_type i = 0; i < dim; ++i)
          {
            dotProduct += vector1[i] * vector2[i];
            lengthVector1Sq += vector1[i] * vector1[i];
            lengthVector2Sq += vector2[i] * vector2[i];
          }

        return dotProduct / std::sqrt(lengthVector1Sq * lengthVector2Sq);
      }

      template <size_type dim>
      void
      markPeriodicFacesAndAddPeriodicity(
        dealii::Triangulation<dim> &     triangulation,
        const std::vector<bool> &        isPeriodicFlags,
        const std::vector<utils::Point> &domainVectors)
      {
        const std::vector<size_type> periodicDirections =
          getPeriodicDirections(isPeriodicFlags);

        if (periodicDirections.empty())
          return;

        utils::throwException<utils::InvalidArgument>(
          dim == 3,
          "Periodic boundary conditions are implemented only for dim = 3.");

        std::vector<utils::Point> periodicFaceNormals;
        computePeriodicFaceNormals(domainVectors, periodicFaceNormals);

        // A left-handed lattice flips the sign of every face normal computed
        // above, which would swap the two boundary ids of each pair while the
        // offset vector below keeps its sign -- the face pairing would then be
        // silently wrong rather than fail.
        double handedness = 0.0;
        for (size_type i = 0; i < 3; ++i)
          handedness += periodicFaceNormals[2][i] * domainVectors[2][i];
        utils::throwException<utils::InvalidArgument>(
          handedness > 0.0,
          "The domain bounding vectors must form a right-handed system, i.e. "
          "(a1 x a2).a3 > 0, for the periodic faces to be paired correctly. "
          "Got (a1 x a2).a3 = " +
            std::to_string(handedness) + ".");

        std::vector<dealii::Tensor<1, dim>> offsetVectors(0);
        computeOffsetVectors<dim>(domainVectors, offsetVectors);

        dealii::QGauss<dim - 1>  quadratureFaceFormula(2);
        dealii::FESystem<dim>    fe(dealii::FE_Q<dim>(
                                   dealii::QGaussLobatto<1>(2)),
                                 1);
        dealii::FEFaceValues<dim> feFaceValues(fe,
                                               quadratureFaceFormula,
                                               dealii::update_normal_vectors);

        auto cell = triangulation.begin_active();
        auto endc = triangulation.end();
        for (; cell != endc; ++cell)
          {
            for (size_type iFace = 0;
                 iFace < dealii::GeometryInfo<dim>::faces_per_cell;
                 ++iFace)
              {
                if (!cell->face(iFace)->at_boundary())
                  continue;

                feFaceValues.reinit(cell, iFace);
                const dealii::Tensor<1, dim> faceNormalVector =
                  feFaceValues.normal_vector(0);

                for (size_type b = 0; b < periodicDirections.size(); ++b)
                  {
                    const size_type d        = periodicDirections[b];
                    const double    cosAngle = getCosineAngle<dim>(
                      faceNormalVector, periodicFaceNormals[d]);
                    if (std::abs(cosAngle - 1.0) < 1.0e-05)
                      cell->face(iFace)->set_boundary_id(2 * b + 1);
                    else if (std::abs(cosAngle + 1.0) < 1.0e-05)
                      cell->face(iFace)->set_boundary_id(2 * b + 2);
                  }
              }
          }

        std::vector<dealii::GridTools::PeriodicFacePair<
          typename dealii::Triangulation<dim>::cell_iterator>>
          periodicityVector;
        for (size_type b = 0; b < periodicDirections.size(); ++b)
          {
            dealii::GridTools::collect_periodic_faces(
              triangulation,
              /*b_id1*/ 2 * b + 1,
              /*b_id2*/ 2 * b + 2,
              /*direction*/ periodicDirections[b],
              periodicityVector,
              offsetVectors[periodicDirections[b]]);
          }
        triangulation.add_periodicity(periodicityVector);
      }

    } // namespace TriangulationDealiiUtils
  }   // namespace basis
} // namespace dftefe
