#ifndef dftefeBlasTemplates_h
#define dftefeBlasTemplates_h

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
} // namespace dftefe
#endif
