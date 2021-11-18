#include "QuadratureRuleGauss.h"
#include <deal.II/base/quadrature_lib.h>

namespace dftefe {

  namespace quadrature {

	QuadratureRuleGauss::QuadratureRuleGauss(const unsigned int dim,
		  const unsigned int order1D)
	{
        d_dim = dim;
        
        d_num1DPoints = order1D;
        d_1DPoints.resize(order1D,utils::Point(1, 0.0));
        d_1DWeights.resize(order1D);
        d_isTensorStructured = true;
        
        d_numPoints = std::pow(order1D,dim);
        d_points.resize(d_numPoints,utils::Point(d_dim, 0.0));
        d_weights.resize(d_numPoints);
        
        dealii::QGauss<1> qgauss1D(order1D);
        
        for (unsigned int iQuad = 0 ; iQuad < order1D ; iQuad++)
        {
            d_weights[iQuad] = qgauss1D.weight(iQuad);
            d_1DPoints[iQuad][0] = qgauss1D.point(iQuad)[0];
        }
        
        unsigned int quadIndex = 0;
        for (unsigned int kQuad = 0 ; kQuad < order1D ; kQuad++)
        {
            for (unsigned int jQuad = 0 ; jQuad < order1D ; jQuad++)
            {
                for (unsigned int iQuad = 0 ; iQuad < order1D ; iQuad++)
                {
                    d_points[quadIndex][0] = d_1DPoints[iQuad][0];
                    d_points[quadIndex][1] = d_1DPoints[jQuad][0];
                    d_points[quadIndex][2] = d_1DPoints[kQuad][0];
                    
                    d_weights[quadIndex] = d_weights[iQuad]*d_weights[jQuad]*
                                           d_weights[kQuad];
                    
                    quadIndex++;
                    
                }
            }
        }
        


	}

  } // end of namespace quadrature
} // end of namespace dftefe
