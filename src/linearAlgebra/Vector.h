#ifndef dftefeVector_h
#define dftefeVector_h

#include <MemoryManager.h>
#include <vector>
namespace dftefe
{
  extern "C"
  {
    void
    dgemv_(const char *        TRANS,
           const unsigned int *M,
           const unsigned int *N,
           const double *      alpha,
           const double *      A,
           const unsigned int *LDA,
           const double *      X,
           const unsigned int *INCX,
           const double *      beta,
           double *            C,
           const unsigned int *INCY);

    void
    dgemm_(const char *        transA,
           const char *        transB,
           const unsigned int *m,
           const unsigned int *n,
           const unsigned int *k,
           const double *      alpha,
           const double *      A,
           const unsigned int *lda,
           const double *      B,
           const unsigned int *ldb,
           const double *      beta,
           double *            C,
           const unsigned int *ldc);
  }


  template <typename NumberType, MemorySpace memorySpace>
  class Vector
  {
  public:
    using size_type = unsigned int;

    Vector() = default;

    Vector(const size_type size, const NumberType initVal = 0);

    void
    testDgemv();

  private:
    NumberType *d_data;
  };

  template <typename NumberType, MemorySpace memorySpace>
  void
  Vector<NumberType, memorySpace>::testDgemv()
  {
    char                transA = 'N';
    char                transB = 'T';
    const double        alpha = 1.0, beta = 0.0;
    unsigned int        numberWaveFunctions = 10;
    unsigned int        numberDofs          = 10;
    std::vector<double> X, Y, output;
    X.resize(100);
    Y.resize(100);
    output.resize(100);



    dgemm_(&transA,
           &transB,
           &numberWaveFunctions,
           &numberWaveFunctions,
           &numberDofs,
           &alpha,
           &X[0],
           &numberWaveFunctions,
           &Y[0],
           &numberWaveFunctions,
           &beta,
           &output[0],
           &numberWaveFunctions);
  }


} // end of namespace dftefe



#endif
