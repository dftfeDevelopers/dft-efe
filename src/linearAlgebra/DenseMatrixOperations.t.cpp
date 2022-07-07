#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    DenseMatrixOperations::multiply(ValueType alpha,
           DenseMatrix<ValueType,memorySpace> & A,
           DenseMatrix<ValueType,memorySpace> & B,
           ValueType beta,
           DenseMatrix<ValueType,memorySpace> & C);
    {

       if (memorySpace==dftefe::utils::MemorySpace::DEVICE)
       {
          slate::gemm( alpha, A.getSlateMatrix(), B.getSlateMatrix(), beta, {slate::Option::Target, slate::Target::Devices});
       }
    }

  } // namespace linearAlgebra
} // namespace dftefe
