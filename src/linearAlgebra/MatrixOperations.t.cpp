#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations<ValueType, memorySpace>::matMulc(
      blasLapack::Layout                                  layout,
      blasLapack::Op                                      transA,
      blasLapack::Op                                      transB,
      size_type                                           m,
      size_type                                           n,
      size_type                                           k,
      ValueType                                           alpha,
      ValueType const *                                   dA,
      size_type                                           ldda,
      ValueType const *                                   dB,
      size_type                                           lddb,
      ValueType                                           beta,
      ValueType *                                         dC,
      size_type                                           lddc,
      std::shared_ptr<blasLapack::BlasQueue<memorySpace>> BlasQueue)
    {
      dftefe::linearAlgebra::blasLapack::gemm(layout,
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
                                              *BlasQueue);
    }

  } // namespace linearAlgebra
} // namespace dftefe
