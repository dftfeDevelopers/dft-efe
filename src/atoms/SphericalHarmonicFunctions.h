#ifndef dftefeSphericalHarmonicFunctions_h
#define dftefeSphericalHarmonicFunctions_h

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <utils/Point.h>
#include <sstream>
#include <utils/Spline.h>
#ifdef DFTEFE_WITH_DEVICE
  #  include <utils/DeviceKernelLauncherHelpers.h>
#endif

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

      double
      Plm(const int l, const int m, const double theta) const;

      double
      dPlmDTheta(const int l, const int m, const double theta) const;

      double
      d2PlmDTheta2(const int l, const int m, const double theta) const;

      template <dftefe::utils::MemorySpace memorySpace>
      void
      Plm(size_type numPoints,
          const int l, 
          const int m, 
          const double *theta, 
          double *out,
        utils::deviceStream_t  streamId = utils::defaultStream) const;

      template <dftefe::utils::MemorySpace memorySpace>
      void
      dPlmDTheta(size_type numPoints,
                const int l, 
                const int m, 
                const double *theta, 
                double *out,
              utils::deviceStream_t  streamId = utils::defaultStream) const;

      template <dftefe::utils::MemorySpace memorySpace>
      void
      d2PlmDTheta2(size_type numPoints,
                  const int l, 
                  const int m, 
                  const double *theta, 
                  double *out,
                utils::deviceStream_t  streamId = utils::defaultStream) const;

#ifdef DFTEFE_WITH_DEVICE
      DFTEFE_DEVICE_FUNC double
      PlmDevice(const int l, const int m, const double theta) const;

      DFTEFE_DEVICE_FUNC double
      dPlmDThetaDevice(const int l, const int m, const double theta) const;

      DFTEFE_DEVICE_FUNC double
      d2PlmDTheta2Device(const int l, const int m, const double theta) const;
#endif
      ///////////////////////////////////////////////////////////////////////////
      ///////////// END OF SPHERICAL HARMONICS RELATED FUNCTIONS //////////////
      ///////////////////////////////////////////////////////////////////////////

    private:
      std::vector<std::vector<std::shared_ptr<const utils::Spline>>>
           d_assocLegendreSpline;
      bool d_isAssocLegendreSplineEval;
    };

    // Analytical Functions
    void
    convertCartesianToSpherical(const utils::Point &x,
                                double &            r,
                                double &            theta,
                                double &            phi,
                                double              polarAngleTolerance);

    template <dftefe::utils::MemorySpace memorySpace>
    void
    convertCartesianToSpherical(size_type numPoints,
                                const double * x,
                                double *       r,
                                double *       theta,
                                double *       phi,
                                double polarAngleTolerance,
                                utils::deviceStream_t  streamId = 
                                  utils::defaultStream);
    double
    Dm(const int m);

    double
    Clm(const int l, const int m);

    double
    Qm(const int m, const double phi);

    template <dftefe::utils::MemorySpace memorySpace>
    void
    Qm(size_type numPoints, const int m, const double *phi, double *out,  utils::deviceStream_t  streamId = utils::defaultStream);

    double
    dQmDPhi(const int m, const double phi);

    template <dftefe::utils::MemorySpace memorySpace>
    void
    dQmDPhi(size_type numPoints, const int m, const double *phi, double *out,  utils::deviceStream_t  streamId = utils::defaultStream);

  } // namespace atoms
} // namespace dftefe

#ifdef DFTEFE_WITH_DEVICE
#include <atoms/SphericalHarmonicFunctionsDeviceKernels.h>
#endif

#endif // SphericalHarmonicFunctions
