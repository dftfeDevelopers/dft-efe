#include <cmath>
#include <utils/Exceptions.h>
#include <vector>
#include <atoms/SphericalHarmonicFunctions.h>
// #include <boost/math/special_functions/legendre.hpp>
// #include <boost/math/special_functions/spherical_harmonic.hpp>
// #include <boost/math/special_functions/factorials.hpp>

namespace dftefe
{
  namespace atoms
  {
    namespace SphericalHarmonicFunctionsInternal
    {
      double
      Rlm(const int l, const int m)
      {
        if (m == 0)
          return 1.0;
        else
          return Rlm(l, m - 1) / ((l - m + 1.0) * (l + m));
      }

      double
      Plm(const int l, const int m, const double x)
      {
        // throw exception if x is not in [-1,1]
        DFTEFE_Assert(abs(x) <= 1.0);
        if (m < 0)
          {
            int    modM   = abs(m);
            double factor = pow((-1.0), m) * Rlm(l, modM);
            return factor * Plm(l, modM, x);
          }
        if (m > l)
          return 0.0;
        double cxM     = 1.0;
        double cxMplus = 0.0;
        double somx2   = sqrt(1.0 - x * x);
        double fact    = 1.0;
        for (double i = 0; i < m; i++)
          {
            cxM  = -cxM * fact * somx2;
            fact = fact + 2.0;
          }
        double cx = cxM;
        if (m != l)
          {
            double cxMPlus1 = x * (2 * m + 1) * cxM;
            cx              = cxMPlus1;

            double cxPrev     = cxMPlus1;
            double cxPrevPrev = cxM;
            for (double i = m + 2; i < l + 1; i++)
              {
                cx = ((2 * i - 1) * x * cxPrev + (-i - m + 1) * cxPrevPrev) /
                     (i - m);
                cxPrevPrev = cxPrev;
                cxPrev     = cx;
              }
          }
        //
        // NOTE: Multiplies by {-1}^m to remove the
        // implicit Condon-Shortley factor in the associated legendre
        // polynomial implementation of boost
        // This is done to be consistent with the QChem's implementation
        return pow((-1.0), m) * cx;
      }

      /**
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
      **/

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
            const double factor = pow(-1, m) * Rlm(l, modM);
            // boost::math::factorial<double>(l - modM) /
            // boost::math::factorial<double>(l + modM);
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
            const double term1 =
              (l + m) * (l - m + 1) * Plm(l, m - 1, cosTheta);
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
            const double factor = pow(-1, m) * Rlm(l, modM);
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

      void
      readLegendreOutput(const std::string &  filename,
                         std::vector<double> &readTheta,
                         std::vector<std::vector<std::vector<double>>> &data,
                         int &                                          lMax)
      {
        std::ifstream inFile(filename);
        if (!inFile)
          {
            std::cerr << "Error opening file for reading Associated Legendre "
                      << std::endl;
            return;
          }

        std::string line;
        std::getline(inFile, line); // Read lMax
        std::stringstream ss(line);
        ss >> lMax;

        // Read the data back into vectors
        data.clear();
        readTheta.clear();
        data.resize(lMax + 1, std::vector<std::vector<double>>(lMax + 1));

        std::getline(inFile, line); // Skip header line

        while (getline(inFile, line))
          {
            std::stringstream ss(line);
            double            thetaValue;
            ss >> thetaValue;
            readTheta.push_back(thetaValue);

            for (int l = 1; l < lMax + 1; l++)
              {
                for (int m = 0; m <= l; m++)
                  {
                    double value;
                    ss >> value;
                    data[l][m].push_back(value);
                  }
              }
          }
        inFile.close();
      }
    } // namespace SphericalHarmonicFunctionsInternal

    SphericalHarmonicFunctions::SphericalHarmonicFunctions(
      const bool isAssocLegendreSplineEval)
      : d_isAssocLegendreSplineEval(isAssocLegendreSplineEval)
      , d_assocLegendreSpline(0)
    {
      if (d_isAssocLegendreSplineEval)
        {
          // Read the Associated Legendre data
          std::vector<double>                           readTheta(0);
          std::vector<std::vector<std::vector<double>>> data(0);
          int                                           lMax = 0;
          char *      dftefe_path = getenv("DFTEFE_PATH");
          std::string sourceDir =
            (std::string)dftefe_path + "/src/atoms/AssociatedLegendreData.txt";
          SphericalHarmonicFunctionsInternal::readLegendreOutput(sourceDir,
                                                                 readTheta,
                                                                 data,
                                                                 lMax);

          d_assocLegendreSpline.clear();
          d_assocLegendreSpline.resize(
            lMax + 1,
            std::vector<std::shared_ptr<const utils::Spline>>(lMax + 1,
                                                              nullptr));
          for (int l = 1; l < lMax + 1; l++)
            {
              for (int m = 0; m <= l; m++)
                {
                  if (!data[l][m].empty())
                    {
                      d_assocLegendreSpline[l][m] =
                        std::make_shared<const utils::Spline>(readTheta,
                                                              data[l][m],
                                                              true);
                    }
                  else
                    {
                      utils::throwException<utils::InvalidArgument>(
                        false,
                        "Data for P_" + std::to_string(l) + std::to_string(m) +
                          " is empty in atoms::AssociatedLegendreData.txt");
                    }
                }
            }
        }
    }

    void
    convertCartesianToSpherical(const utils::Point &x,
                                double &            r,
                                double &            theta,
                                double &            phi,
                                double              polarAngleTolerance)
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

    void
    convertCartesianToSpherical(const std::vector<utils::Point> &points,
                                std::vector<double> &            r,
                                std::vector<double> &            theta,
                                std::vector<double> &            phi,
                                double polarAngleTolerance)
    {
      for (int i = 0; i < points.size(); i++)
        {
          utils::Point x(points[i]);
          double &     radius = r[i];
          radius              = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
          if (radius == 0)
            {
              theta[i] = 0.0;
              phi[i]   = 0.0;
            }

          else
            {
              theta[i] = acos(x[2] / radius);
              //
              // check if theta = 0 or PI (i.e, whether the point is on the
              // Z-axis) If yes, assign phi = 0.0. NOTE: In case theta = 0 or
              // PI, phi is undetermined. The actual value of phi doesn't matter
              // in computing the enriched function value or its gradient. We
              // assign phi = 0.0 here just as a dummy value
              //
              if (fabs(theta[i] - 0.0) >= polarAngleTolerance &&
                  fabs(theta[i] - M_PI) >= polarAngleTolerance)
                phi[i] = atan2(x[1], x[0]);
              else
                phi[i] = 0.0;
            }
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
          return sqrt(((2.0 * l + 1) * boost::math::factorial<double>(l -
       abs(m))) / (2.0 * boost::math::factorial<double>(l + abs(m))));
        }
    */

    // Implement this instead of above function to remove underflow/overflow
    // issues in factorial
    double
    Blm(const int l, const int m)
    {
      if (m == 0)
        return sqrt((2.0 * l + 1) / 2.0);
      else
        return Blm(l, m - 1) / sqrt((l - m + 1.0) * (l + m));
    }

    double
    Clm(const int l, const int m)
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
    SphericalHarmonicFunctions::Plm(const int    l,
                                    const int    m,
                                    const double theta) const
    {
      if (!d_isAssocLegendreSplineEval)
        {
          return SphericalHarmonicFunctionsInternal::Plm(l, m, cos(theta));
        }
      else
        {
          if (l != 0)
            {
              if (m < 0)
                {
                  int    modM = abs(m);
                  double factor =
                    pow((-1.0), m) *
                    SphericalHarmonicFunctionsInternal::Rlm(l, modM);
                  return (*d_assocLegendreSpline[l][modM])(theta)*factor;
                }
              else
                return (*d_assocLegendreSpline[l][m])(theta);
            }
          else
            return 1.0;
        }
    }

    double
    SphericalHarmonicFunctions::dPlmDTheta(const int    l,
                                           const int    m,
                                           const double theta) const
    {
      if (!d_isAssocLegendreSplineEval)
        {
          return SphericalHarmonicFunctionsInternal::dPlmDTheta(l, m, theta);
        }
      else
        {
          if (l != 0)
            {
              if (m < 0)
                {
                  int    modM = abs(m);
                  double factor =
                    pow((-1.0), m) *
                    SphericalHarmonicFunctionsInternal::Rlm(l, modM);
                  return (*d_assocLegendreSpline[l][modM]).deriv(1, theta) *
                         factor;
                }
              else
                return (*d_assocLegendreSpline[l][m]).deriv(1, theta);
            }
          else
            return 0.0;
        }
    }


    double
    SphericalHarmonicFunctions::d2PlmDTheta2(const int    l,
                                             const int    m,
                                             const double theta) const
    {
      if (!d_isAssocLegendreSplineEval)
        {
          return SphericalHarmonicFunctionsInternal::d2PlmDTheta2(l, m, theta);
        }
      else
        {
          if (l != 0)
            {
              if (m < 0)
                {
                  int    modM = abs(m);
                  double factor =
                    pow((-1.0), m) *
                    SphericalHarmonicFunctionsInternal::Rlm(l, modM);
                  return (*d_assocLegendreSpline[l][modM]).deriv(2, theta) *
                         factor;
                }
              else
                return (*d_assocLegendreSpline[l][m]).deriv(2, theta);
            }
          else
            return 0.0;
        }
    }
    ///////////////////////////////////////////////////////////////////////////
    ///////////// END OF SPHERICAL HARMONICS RELATED FUNCTIONS //////////////
    ///////////////////////////////////////////////////////////////////////////
  } // namespace atoms
} // namespace dftefe
