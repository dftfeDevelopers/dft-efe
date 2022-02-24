#include "MatrixOperations.h"
#include <utils/DataTypeOverloads.h>

namespace dftfe
{
  namespace linearAlgebra
  {

    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations<ValueType , memorySpace>::matMulc
      (  blasLayoutType layout,
         blasOperationType transA,
         blasOperationType transB,
         size_type m, size_type n, size_type k,
         ValueType alpha,
         ValueType const *dA, size_type ldda,
         ValueType const *dB, size_type lddb,
         ValueType beta,
         ValueType       *dC, size_type lddc,
         blasWrapper::blasQueueType<memorySapce> &blasQueue)
      {
        dftfe::linearAlgebra::gemm(layout ,transA, transB, m,n,k, alpha,
                                   dA, ldda, dB, lddb,beta, dC, lddc, blasQueue );
      }

  }
}
