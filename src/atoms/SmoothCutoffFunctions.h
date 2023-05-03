#include <vector>

namespace dftefe
{
    namespace atoms
    { 
        ///////////////////////////////////////////////////////////////////////////
        ///////////// START OF SMOOTH CUTOFF FUNCTION RELATED FUNCTIONS ///////////
        ///////////////////////////////////////////////////////////////////////////
        double f1(const double x);

        double f1Der(const double x);

        double f2(const double x);

        double f2Der(const double x, const double tolerance);

        double Y(const double x, const double r, const double d);

        double YDer(const double x, const double r, const double d);

        double smoothCutoffValue(const double x, const double r, const double d);

        double smoothCutoffDerivative(const double x, const double r, const double d, const double tolerance);

        ///////////////////////////////////////////////////////////////////////////
        ///////////// END OF SMOOTH CUTOFF FUNCTION RELATED FUNCTIONS ///////////
        ///////////////////////////////////////////////////////////////////////////
    }
}