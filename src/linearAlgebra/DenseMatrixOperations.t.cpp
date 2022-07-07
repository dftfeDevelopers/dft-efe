#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DenseMatrixOperations::gemm(ValueType alpha,
           DenseMatrix<ValueType,memorySpace> & A,
           DenseMatrix<ValueType,memorySpace> & B,
           ValueType beta,
           DenseMatrix<ValueType,memorySpace> & C);
    {
       auto AT = slate::transpose( A );
       auto BT = slate::conjTranspose( B );
       slate::gemm( alpha, AT, BT, beta, C );
    }

  } // namespace linearAlgebra
} // namespace dftefe
