#include <MatrixOperations.h>
#include <utils/DataTypeOverloads.h>

namespace dftfe
{
  namespace linearAlgebra
  {

    template <typename ValueType>
    void
    MatrixOperations<ValueType ,dftefe::utils::MemorySpace::HOST >::matMulc( blasLayoutType layout,
                                                                             blasOperationType transA,
                                                                             blasOperationType transB,
                                                                             size_type m, size_type n, size_type k,
                                                                             ValueType alpha,
                                                                             ValueType const *dA, size_type ldda,
                                                                             ValueType const *dB, size_type lddb,
                                                                             ValueType beta,
                                                                             ValueType       *dC, size_type lddc)
    {
      dftfe::linearAlgebra::gemm(layout ,transA, transB, m,n,k, alpha, dA, ldda, dB, lddb,beta, cD, lddc );
    }

    template <typename ValueType>
    void
    MatrixOperations<ValueType ,dftefe::utils::MemorySpace::HOST >::matMulc( blasLayoutType layout,
      blasOperationType transA,
    blasOperationType transB,
      size_type m, size_type n, size_type k,
    ValueType alpha,
      ValueType const *dA, size_type ldda,
      ValueType const *dB, size_type lddb,
      ValueType beta,
    ValueType       *dC, size_type lddc,
    blasQueueType &queue)
    {
      DFTEFE_AssertWithMsg(false, "Not implemented.");
    }

    template <typename ValueType>
    void
    MatrixOperations<ValueType ,dftefe::utils::MemorySpace::HOST >::matMulc( blasLayoutType layout,
                                                                             blasOperationType transA,
                                                                             blasOperationType transB,
                                                                             size_type m, size_type n, size_type k,
                                                                             ValueType alpha,
                                                                             ValueType const *dA, size_type ldda,
                                                                             ValueType const *dB, size_type lddb,
                                                                             ValueType beta,
                                                                             ValueType       *dC, size_type lddc)
    {
      DFTEFE_AssertWithMsg(false, "Not implemented.");

    }

    template <typename ValueType>
    void
    MatrixOperations<ValueType ,dftefe::utils::MemorySpace::HOST >::matMulc( blasLayoutType layout,
                                                                             blasOperationType transA,
                                                                             blasOperationType transB,
                                                                             size_type m, size_type n, size_type k,
                                                                             ValueType alpha,
                                                                             ValueType const *dA, size_type ldda,
                                                                             ValueType const *dB, size_type lddb,
                                                                             ValueType beta,
                                                                             ValueType       *dC, size_type lddc,
                                                                             blasQueueType &queue)
    {
      dftfe::linearAlgebra::gemm(layout ,transA, transB, m,n,k, alpha, dA, ldda, dB, lddb,beta, cD, lddc, queue );
    }


  }
}
