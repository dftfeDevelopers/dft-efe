#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <cmath>
#include <vector>

namespace dftefe
{
  namespace atoms
  {
    void
    convertCartesianToSpherical(const std::vector<double> &x,
                                double &                   r,
                                double &                   theta,
                                double &                   phi,
                                double                     polarAngleTolerance)
    {
      r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      if (r == 0)
        {
          theta = 0.0;
          phi   = 0.0;
        }

      else
        {
          theta = acos(x[2] / r);
          //
          // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
          // If yes, assign phi = 0.0.
          // NOTE: In case theta = 0 or PI, phi is undetermined. The actual
          // value of phi doesn't matter in computing the enriched function
          // value or its gradient. We assign phi = 0.0 here just as a dummy
          // value
          //
          if (fabs(theta - 0.0) >= polarAngleTolerance &&
              fabs(theta - M_PI) >= polarAngleTolerance)
            phi = atan2(x[1], x[0]);
          else
            phi = 0.0;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////// START OF SPHERICAL HARMONICS RELATED FUNCTIONS //////////////
    ///////////////////////////////////////////////////////////////////////////

    //
    // We use the real form of spherical harmonics without the Condon-Shortley
    // phase (i.e., the (-1)^m prefactor) (see
    // https://en.wikipedia.org/wiki/Spherical_harmonics) NOTE: 1) The wikipedia
    // definition has the Condon-Shortley phase.
    //       2) The definition of the associated Legendre polynomial (P_lm) in
    //       Boost library also contains a Condon-Shortley phase.
    //          Thus, if you're using Boost library, multiply the P_lm
    //          evaluation with (-1)^m to remove the Condon-Shortley phase. Most
    //          Quantum Chemistry codes (e.g., QChem) do not include the
    //          Condon-Shortley phase. So to make it consistent, it is prefered
    //          to remove the Condon-Shortley phase, if there are any to begin
    //          with.
    //        3) From C++17 onwards, the <cmath> has the associated Legendre
    //        polynomial (see
    //        https://en.cppreference.com/w/cpp/numeric/special_functions/assoc_legendre)
    //           Thus, if you're using C++17 or beyond, you can use the C++
    //           standard's definition of associated Legendre polynomial instead
    //           of Boost. Note that, the C++ standard does not have the
    //           Condon-Shortley phase while Boost has it. So, we do not have to
    //           do anything special to remove it while using the C++ standard.
    //

    //
    // Y_lm(theta, phi) = Clm(l,m) * Dm(m) * P_lm(l,m,cos(theta)) * Qm(m,phi),
    // where theta = polar angle,
    // phi = azimuthal angle
    // P_lm is the associated Legendre polynomial of degree l and order m,
    // Qm is the real form of exp(i*m*phi),
    // C_lm is the normalization constant for P_lm,
    // D_m is the normalization constant for Q_m
    //

    //
    // For the definition of the associated Legendre polynomials i.e. P_lm and
    // their derivatives (as used for evaluating the real form of spherical
    // harmonics and their derivatives) refer:
    // @article{bosch2000computation,
    // 	   title={On the computation of derivatives of Legendre functions},
    //    	   author={Bosch, W},
    //        journal={Physics and Chemistry of the Earth, Part A: Solid Earth
    //        and Geodesy}, volume={25}, number={9-11}, pages={655--659},
    //        year={2000},
    //        publisher={Elsevier}
    //       }
    // We use the derivative definitions from the above reference because
    // finding the derivatives on the pole (i.e., theta = 0) is tricky. This is
    // because the azimuthal angles (phi) is undefined for a point on the pole.
    // However, the derivative is still well defined on the pole via the
    // L'Hospital's rule. However, one can avoid implementing tedious
    // L'Hospital's rule on pole and use much simpler expressions given in the
    // above reference.
    //

    double
    Dm(const int m)
    {
      if (m == 0)
        return 1.0 / sqrt(2 * M_PI);
      else
        return 1.0 / sqrt(M_PI);
    }

/*
    double
    Clm(const int l, const int m)
    {
      // assert(m >= 0);
      assert(std::abs(m) <= l);
      return sqrt(((2.0 * l + 1) * boost::math::factorial<double>(l - abs(m))) /
                  (2.0 * boost::math::factorial<double>(l + abs(m))));
    }
*/

    // Implement this instead of above function to remove underflow/overflow issues in factorial
    double Blm(const int l, const int m)
    {
      if (m==0)
        return sqrt((2.0*l+1)/2.0);
      else
        return Blm(l, m-1)/sqrt((l-m+1.0)*(l+m));
    }

    double Clm(const int l, const int m)
    {
      // assert(m >= 0);
      assert(std::abs(m) <= l);
      return Blm(l, abs(m));
    } 

    double
    Qm(const int m, const double phi)
    {
      double returnValue = 0.0;
      if (m > 0)
        returnValue = cos(m * phi);
      if (m == 0)
        returnValue = 1.0;
      if (m < 0)
        returnValue = sin(std::abs(m) * phi);

      return returnValue;
    }

    double
    dQmDPhi(const int m, const double phi)
    {
      if (m > 0)
        return -m * sin(m * phi);
      else if (m == 0)
        return 0.0;
      else
        return std::abs(m) * cos(std::abs(m) * phi);
    }

    double
    Plm(const int l, const int m, const double x)
    {
      if (std::abs(m) > l)
        return 0.0;
      else
        //
        // NOTE: Multiplies by {-1}^m to remove the
        // implicit Condon-Shortley factor in the associated legendre
        // polynomial implementation of boost
        // This is done to be consistent with the QChem's implementation
        return pow(-1.0, m) * boost::math::legendre_p(l, m, x);
    }

    double
    dPlmDTheta(const int l, const int m, const double theta)
    {
      const double cosTheta = cos(theta);

      if (std::abs(m) > l)
        return 0.0;

      else if (l == 0)
        return 0.0;

      else if (m < 0)
        {
          const int    modM   = std::abs(m);
          const double factor = pow(-1, m) *
                                boost::math::factorial<double>(l - modM) /
                                boost::math::factorial<double>(l + modM);
          return factor * dPlmDTheta(l, modM, theta);
        }

      else if (m == 0)
        {
          return -1.0 * Plm(l, 1, cosTheta);
        }

      else if (m == l)
        return l * Plm(l, l - 1, cosTheta);

      else
        {
          const double term1 = (l + m) * (l - m + 1) * Plm(l, m - 1, cosTheta);
          const double term2 = Plm(l, m + 1, cosTheta);
          return 0.5 * (term1 - term2);
        }
    }


    double
    d2PlmDTheta2(const int l, const int m, const double theta)
    {
      const double cosTheta = cos(theta);

      if (std::abs(m) > l)
        return 0.0;

      else if (l == 0)
        return 0.0;

      else if (m < 0)
        {
          const int    modM   = std::abs(m);
          const double factor = pow(-1, m) *
                                boost::math::factorial<double>(l - modM) /
                                boost::math::factorial<double>(l + modM);
          return factor * d2PlmDTheta2(l, modM, theta);
        }

      else if (m == 0)
        return -1.0 * dPlmDTheta(l, 1, theta);

      else if (m == l)
        return l * dPlmDTheta(l, l - 1, theta);

      else
        {
          double term1 = (l + m) * (l - m + 1) * dPlmDTheta(l, m - 1, theta);
          double term2 = dPlmDTheta(l, m + 1, theta);
          return 0.5 * (term1 - term2);
        }
    }
    ///////////////////////////////////////////////////////////////////////////
    ///////////// END OF SPHERICAL HARMONICS RELATED FUNCTIONS //////////////
    ///////////////////////////////////////////////////////////////////////////
  } // namespace atoms
} // namespace dftefe
