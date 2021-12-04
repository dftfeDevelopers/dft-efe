#ifndef dftefeDealiiConversions_h
#define dftefeDealiiConversions_h

#include "Point.h"
#include <deal.II/base/point.h>


namespace dftefe
{
  namespace utils
  {
    template <unsigned int dim>
    void
    convertToDealiiPoint(const utils::Point &        point,
                         dealii::Point<dim, double> &outputDealiiPoint);

    template <unsigned int dim>
    void
    convertToDealiiPoint(const std::vector<utils::Point> &        vecPoint,
                         std::vector<dealii::Point<dim>> & vecOutputDealiiPoint);

    template <unsigned int dim>
    void
    convertToDealiiPoint(const std::vector<double> & v,
                         dealii::Point<dim, double> &outputDealiiPoint);


    template <unsigned int dim>
    void
    convertToDftefePoint(const dealii::Point<dim, double> &dealiiPoint,
                         Point &                           outputDftefePoint);


  } // namespace utils
} // namespace dftefe
#include "DealiiConversions.t.cpp"
#endif
