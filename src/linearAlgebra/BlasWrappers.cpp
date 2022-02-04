#include "BlasWrappers.h"


namespace dftefe
{
  namespace linearAlgebra
  {
    gemm(blasLayoutType layout,
    blasOperationType transA,
    blasOperationType transB,
    size_type m, size_type n, size_type k,
    ValueType alpha,
    ValueType const *dA, size_type ldda,
    ValueType const *dB, size_type lddb,
    ValueType beta,
    ValueType       *dC, size_type lddc)
    {
      blas::gemm(layout , transA , transB , m , n , k, alpha ,
                 dA, ldda , dB , lddb, beta, dC, lddc  );
    }
  }

}
