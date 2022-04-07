#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations<ValueType, memorySpace>::matMulc(
      blasWrapper::Layout                      layout,
      blasWrapper::Op                          transA,
      blasWrapper::Op                          transB,
      size_type                                m,
      size_type                                n,
      size_type                                k,
      ValueType                                alpha,
      ValueType const *                        dA,
      size_type                                ldda,
      ValueType const *                        dB,
      size_type                                lddb,
      ValueType                                beta,
      ValueType *                              dC,
      size_type                                lddc,
      blasWrapper::blasQueueType<memorySpace> &blasQueue)
    {
      dftefe::linearAlgebra::blasWrapper::gemm(layout,
                                               transA,
                                               transB,
                                               m,
                                               n,
                                               k,
                                               alpha,
                                               dA,
                                               ldda,
                                               dB,
                                               lddb,
                                               beta,
                                               dC,
                                               lddc,
                                               blasQueue);
    }

  } // namespace linearAlgebra
} // namespace dftefe
