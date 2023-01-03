#ifndef dftefeDealiiConversions_h
#define dftefeDealiiConversions_h

#include <utils/Point.h>
#include <deal.II/base/point.h>


namespace dftefe
{
  namespace basis
  {
    template <unsigned int dim>
    void
    convertToDealiiPoint(const utils::Point &        point,
                         dealii::Point<dim, double> &outputDealiiPoint);

    template <unsigned int dim>
    void
    convertToDealiiPoint(const std::vector<utils::Point> &vecPoint,
                         std::vector<dealii::Point<dim>> &vecOutputDealiiPoint);

    template <unsigned int dim>
    void
    convertToDealiiPoint(const std::vector<double> & v,
                         dealii::Point<dim, double> &outputDealiiPoint);


    template <unsigned int dim>
    void
    convertToDftefePoint(const dealii::Point<dim, double> &dealiiPoint,
                         utils::Point &                    outputDftefePoint);

    template <unsigned int dim>
    void
    convertToDftefePoint(
      const std::vector<dealii::Point<dim, double>> &dealiiPoints,
      std::vector<utils::Point> &                    points);

    template <unsigned int dim>
    void
    convertToDftefePoint(
      const std::map<global_size_type, dealii::Point<dim, double>>
        &                                       dealiiPoints,
      std::map<global_size_type, utils::Point> &points);

  } // namespace basis
} // namespace dftefe
#include "DealiiConversions.t.cpp"
#endif
