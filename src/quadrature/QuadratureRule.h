#ifndef dftefeQuadratureRule_h
#define dftefeQuadratureRule_h

#include <utils/Point.h>
#include <utils/TypeConfig.h>
#include <vector>

namespace dftefe
{
  namespace quadrature
  {
    /*
     *Class that stores the quadrature information. This class provides the
     *location of the quadrature points in parametric coordinates and its
     *corresponding weights
     */
    class QuadratureRule
    {
    public:
      /**
       * @brief Constructor to create a tensor structured quadrature with arbitrary  1D points and weights.
       * @param[in] dim the dimension of the cell.
       * @param[in] points The parametric points on which the 1D quad points are
       * located.
       * @param[in] weights  the weight of the quad points
       */
      QuadratureRule(const unsigned int               dim,
                     const std::vector<utils::Point> &points,
                     const std::vector<double> &      weights);

      /**
       * @brief A function to return the quad points in the parametric space.
       *
       * @returns  returns the vector of dftefe::utils::Point  containing the quad points
       */
      virtual const std::vector<utils::Point> &
      getPoints() const;

      /**
       * @brief A function to return the 1D quad points in the parametric space.
       * This function throws an error if this quadrature Rule is not tensor
       * structured
       *
       * @returns  returns the vector of dftefe::utils::Point  containing the 1D quad points
       */
      virtual const std::vector<utils::Point> &
      get1DPoints() const;

      /**
       * @brief A function to returns the weights of the quadrature points
       *
       * @returns  returns the vector doubles containing the weights
       */
      virtual const std::vector<double> &
      getWeights() const;
      /**
       * @brief A function to return the 1D quad points in the parametric space.
       * This function throws an error if this quadrature Rule is not tensor
       * structured
       *
       * @returns  returns the vector of dftefe::utils::Point  containing the 1D quad points
       */
      virtual const std::vector<double> &
      get1DWeights() const;

      virtual bool
      isTensorStructured() const;

      virtual size_type
      nPoints() const;

      virtual size_type
      n1DPoints() const;

      virtual unsigned int
      getDim() const;

    protected:
      QuadratureRule();

      unsigned int              d_dim;
      size_type                 d_numPoints;
      size_type                 d_num1DPoints;
      std::vector<utils::Point> d_points;
      std::vector<utils::Point> d_1DPoints;
      std::vector<double>       d_weights;
      std::vector<double>       d_1DWeights;
      bool                      d_isTensorStructured;
    };

  } // end of namespace quadrature

} // end of namespace dftefe

#endif // dftefeQuadratureRule_h
