#ifndef dftefeSphericalHarmonicFunctions_h
#define dftefeSphericalHarmonicFunctions_h

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <utils/Point.h>
#include <sstream>
#include <utils/Spline.h>
#include <utils/DeviceKernelLauncherHelpers.h>
#include <cmath>

namespace dftefe
{
  namespace atoms
  {
    class SphericalHarmonicFunctions
    {
    public:
      SphericalHarmonicFunctions(const bool isAssocLegendreSplineEval);

      ~SphericalHarmonicFunctions() = default;

      ///////////////////////////////////////////////////////////////////////////
      ///////////// START OF SPHERICAL HARMONICS RELATED FUNCTIONS
      /////////////////
      ///////////////////////////////////////////////////////////////////////////

      //
      // We use the real form of spherical harmonics without the Condon-Shortley
      // phase (i.e., the (-1)^m prefactor) (see
      // https://en.wikipedia.org/wiki/Spherical_harmonics) NOTE: 1) The
      // wikipedia definition has the Condon-Shortley phase.
      //       2) The definition of the associated Legendre polynomial (P_lm) in
      //       Boost library also contains a Condon-Shortley phase.
      //          Thus, if you're using Boost library, multiply the P_lm
      //          evaluation with (-1)^m to remove the Condon-Shortley phase.
      //          Most Quantum Chemistry codes (e.g., QChem) do not include the
      //          Condon-Shortley phase. So to make it consistent, it is
      //          prefered to remove the Condon-Shortley phase, if there are any
      //          to begin with.
      //        3) From C++17 onwards, the <cmath> has the associated Legendre
      //        polynomial (see
      //        https://en.cppreference.com/w/cpp/numeric/special_functions/assoc_legendre)
      //           Thus, if you're using C++17 or beyond, you can use the C++
      //           standard's definition of associated Legendre polynomial
      //           instead of Boost. Note that, the C++ standard does not have
      //           the Condon-Shortley phase while Boost has it. So, we do not
      //           have to do anything special to remove it while using the C++
      //           standard.
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
      // finding the derivatives on the pole (i.e., theta = 0) is tricky. This
      // is because the azimuthal angles (phi) is undefined for a point on the
      // pole. However, the derivative is still well defined on the pole via the
      // L'Hospital's rule. However, one can avoid implementing tedious
      // L'Hospital's rule on pole and use much simpler expressions given in the
      // above reference.
      //

      // Scalar single-point evaluation — DFTEFE_HOST_DEVICE_FUNC so that it is
      // callable from both host and device (analytical path only; spline path is
      // handled in the batch template specialisations).
      DFTEFE_HOST_DEVICE_FUNC double
      Plm(const int l, const int m, const double theta) const;

      DFTEFE_HOST_DEVICE_FUNC double
      dPlmDTheta(const int l, const int m, const double theta) const;

      DFTEFE_HOST_DEVICE_FUNC double
      d2PlmDTheta2(const int l, const int m, const double theta) const;

      template <dftefe::utils::MemorySpace memorySpace>
      void
      Plm(size_type numPoints,
          const int l,
          const int m,
          const double *theta,
          double *out,
          utils::deviceStream_t streamId = utils::defaultStream) const;

      template <dftefe::utils::MemorySpace memorySpace>
      void
      dPlmDTheta(size_type numPoints,
                 const int l,
                 const int m,
                 const double *theta,
                 double *out,
                 utils::deviceStream_t streamId = utils::defaultStream) const;

      template <dftefe::utils::MemorySpace memorySpace>
      void
      d2PlmDTheta2(size_type numPoints,
                   const int l,
                   const int m,
                   const double *theta,
                   double *out,
                   utils::deviceStream_t streamId = utils::defaultStream) const;

      ///////////////////////////////////////////////////////////////////////////
      ///////////// END OF SPHERICAL HARMONICS RELATED FUNCTIONS //////////////
      ///////////////////////////////////////////////////////////////////////////

    private:
      std::vector<std::vector<std::shared_ptr<const utils::Spline>>>
           d_assocLegendreSpline;
      bool d_isAssocLegendreSplineEval;
    };

    // Host-only single-point evaluation (utils::Point overload)
    void
    convertCartesianToSpherical(const utils::Point &x,
                                double &            r,
                                double &            theta,
                                double &            phi,
                                double              polarAngleTolerance);

    // Single-point raw-pointer overload — callable from host and device
    DFTEFE_HOST_DEVICE_FUNC void
    convertCartesianToSpherical(const double *x,
                                double &      r,
                                double &      theta,
                                double &      phi,
                                double        polarAngleTolerance);

    template <dftefe::utils::MemorySpace memorySpace>
    void
    convertCartesianToSpherical(size_type             numPoints,
                                const double *        x,
                                double *              r,
                                double *              theta,
                                double *              phi,
                                double                polarAngleTolerance,
                                utils::deviceStream_t streamId =
                                  utils::defaultStream);
    double
    Dm(const int m);

    double
    Clm(const int l, const int m);

    DFTEFE_HOST_DEVICE_FUNC double
    Qm(const int m, const double phi);

    template <dftefe::utils::MemorySpace memorySpace>
    void
    Qm(size_type             numPoints,
       const int             m,
       const double *        phi,
       double *              out,
       utils::deviceStream_t streamId = utils::defaultStream);

    DFTEFE_HOST_DEVICE_FUNC double
    dQmDPhi(const int m, const double phi);

    template <dftefe::utils::MemorySpace memorySpace>
    void
    dQmDPhi(size_type             numPoints,
            const int             m,
            const double *        phi,
            double *              out,
            utils::deviceStream_t streamId = utils::defaultStream);

  } // namespace atoms
} // namespace dftefe

//=============================================================================
// Inline DFTEFE_HOST_DEVICE_FUNC definitions — embedded here so every
// translation unit (CPU or GPU) gets its own inline copy.
//=============================================================================

namespace dftefe
{
  namespace atoms
  {
    namespace
    {
      DFTEFE_HOST_DEVICE double
      Rlm(const int l, const int m)
      {
        if (m == 0)
          return 1.0;
        return Rlm(l, m - 1) / ((l - m + 1.0) * (l + m));
      }

      DFTEFE_HOST_DEVICE_FUNC double
      plm(int l, int absm, double cosTheta)
      {
        if (absm > l)
          return 0.0;
        double somx2 = sqrt(1.0 - cosTheta * cosTheta);
        double cxM   = 1.0;
        double fact  = 1.0;
        for (int i = 0; i < absm; i++)
          {
            cxM  = -cxM * fact * somx2;
            fact = fact + 2.0;
          }
        double cx = cxM;
        if (absm != l)
          {
            double cxMPlus1   = cosTheta * (2 * absm + 1) * cxM;
            cx                = cxMPlus1;
            double cxPrev     = cxMPlus1;
            double cxPrevPrev = cxM;
            for (int jj = absm + 2; jj < l + 1; jj++)
              {
                cx = ((2 * jj - 1) * cosTheta * cxPrev +
                      (-jj - absm + 1) * cxPrevPrev) /
                     (jj - absm);
                cxPrevPrev = cxPrev;
                cxPrev     = cx;
              }
          }
        return ((absm % 2 == 0) ? 1.0 : -1.0) * cx;
      }

      DFTEFE_HOST_DEVICE_FUNC double
      dplmDTheta(int l, int absm, double cosTheta)
      {
        if (absm > l)
          return 0.0;
        if (l == 0)
          return 0.0;
        if (absm == 0)
          return -1.0 * plm(l, 1, cosTheta);
        if (absm == l)
          return (double)l * plm(l, l - 1, cosTheta);
        double term1 =
          (double)((l + absm) * (l - absm + 1)) *
          plm(l, absm - 1, cosTheta);
        double term2 = plm(l, absm + 1, cosTheta);
        return 0.5 * (term1 - term2);
      }

      DFTEFE_HOST_DEVICE_FUNC double
      d2plmDTheta2(int l, int absm, double cosTheta)
      {
        if (absm > l)
          return 0.0;
        if (l == 0)
          return 0.0;
        if (absm == 0)
          return -1.0 * dplmDTheta(l, 1, cosTheta);
        if (absm == l)
          return (double)l * dplmDTheta(l, l - 1, cosTheta);
        double term1 = (double)((l + absm) * (l - absm + 1)) *
                       dplmDTheta(l, absm - 1, cosTheta);
        double term2 = dplmDTheta(l, absm + 1, cosTheta);
        return 0.5 * (term1 - term2);
      }

    } // anonymous namespace

    DFTEFE_HOST_DEVICE_FUNC void
    convertCartesianToSpherical(const double *x,
                                double &      r,
                                double &      theta,
                                double &      phi,
                                double        polarAngleTolerance)
    {
      double px = x[0];
      double py = x[1];
      double pz = x[2];
      r         = sqrt(px * px + py * py + pz * pz);
      if (r == 0.0)
        {
          theta = 0.0;
          phi   = 0.0;
        }
      else
        {
          theta = acos(pz / r);
          if (fabs(theta - 0.0) >= polarAngleTolerance &&
              fabs(theta - M_PI) >= polarAngleTolerance)
            phi = atan2(py, px);
          else
            phi = 0.0;
        }
    }

    DFTEFE_HOST_DEVICE_FUNC double
    Qm(const int m, const double phi)
    {
      double v = 0.0;
      if (m > 0)
        v = cos((double)m * phi);
      else if (m == 0)
        v = 1.0;
      else
        v = sin((double)(-m) * phi);
      return v;
    }

    DFTEFE_HOST_DEVICE_FUNC double
    dQmDPhi(const int m, const double phi)
    {
      double v;
      if (m > 0)
        v = -(double)m * sin((double)m * phi);
      else if (m == 0)
        v = 0.0;
      else
        v = (double)(-m) * cos((double)(-m) * phi);
      return v;
    }

    DFTEFE_HOST_DEVICE_FUNC double
    SphericalHarmonicFunctions::Plm(const int    l,
                                    const int    m,
                                    const double theta) const
    {
      const int    absm   = (m < 0) ? -m : m;
      const double factor = (m < 0) ? pow(-1.0, m) * Rlm(l, absm) : 1.0;
      if (absm > l)
        return 0.0;
      return factor * plm(l, absm, cos(theta));
    }

    DFTEFE_HOST_DEVICE_FUNC double
    SphericalHarmonicFunctions::dPlmDTheta(const int    l,
                                           const int    m,
                                           const double theta) const
    {
      const int absm = (m < 0) ? -m : m;
      if (absm > l || l == 0)
        return 0.0;
      const double factor = (m < 0) ? pow(-1.0, m) * Rlm(l, absm) : 1.0;
      return factor * dplmDTheta(l, absm, cos(theta));
    }

    DFTEFE_HOST_DEVICE_FUNC double
    SphericalHarmonicFunctions::d2PlmDTheta2(const int    l,
                                             const int    m,
                                             const double theta) const
    {
      const int absm = (m < 0) ? -m : m;
      if (absm > l || l == 0)
        return 0.0;
      const double factor = (m < 0) ? pow(-1.0, m) * Rlm(l, absm) : 1.0;
      return factor * d2plmDTheta2(l, absm, cos(theta));
    }

  } // namespace atoms
} // namespace dftefe

#endif // SphericalHarmonicFunctions
