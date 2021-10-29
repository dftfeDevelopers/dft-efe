#ifndef dftefeTriangulationDealiiSerial_h
#define dftefeTriangulationDealiiSerial_h

#include "TriangulationBase.h"
#include <utils/TypeConfig.h>
#include "TriaCellBase.h"
#include <deal.II/grid/tria.h>
namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    class TriangulationDealiiSerial : public TriangulationBase
    {
    public:
      TriangulationDealiiSerial();
      ~TriangulationDealiiSerial();

      void
      initializeTriangulationConstruction() override;
      void
      finalizeTriangulationConstruction() override;
      void
      createUniformParallelepiped(
        const std::vector<unsigned int> &                    subdivisions,
        const std::vector<utils::Point> &domainVectors const std::vector<bool>
          &isPeriodicFlags) override;
      void
      shiftTriangulation(const utils::Point &origin) override;
      void
      refineGlobal(const unsigned int times = 1) override;
      void
      coarsenGlobal(const unsigned int times = 1) override;
      void
      clearUserFlags() override;
      void
      executeCoarseningAndRefinement() override;
      size_type
      nLocalCells() const override;
      size_type
      nGlobalCells() const override;
      /**
       * \todo
       * TODO:
       * Implement it to get the user specified boundary Ids on different
       * faces of the triangulation
       */
      std::vector<size_type>
      getBoundaryIds() const override;
      TriaCellBase &
      beginLocal() const override;
      TriaCellBase &
      endLocal() const override;
      unsigned int
      getDim() const override;


    private:
      /**
       * \todo
       * TODO:
       * 1. Implement for periodic case
       * 2. Check if the domainvectors argument is redundant (i.e., if they can
       *  be fetched from the d_triangulationDealii)
       */
      void
      markPeriodicFaces(
        const std::<bool> &              isPeriodicFlags,
        const std::vector<utils::Point> &domainVectors) override;

    private:
      bool                                              isInitialized;
      bool                                              isFinalized;
      dealii::Triangulation<dim>                        d_triangulationDealii;
      std::vector<std::shared_ptr<TriaCellDealii<dim>>> d_triaVectorCell;

    }; // end of class TriangulationDealiiSerial

  } // end of namespace basis

} // end of namespace dftefe
#endif
