#include "BlasWrappers.h"


namespace dftefe
{
  namespace linearAlgebra
  {

    template<typename ValueType , dftefe::utils::MemorySpace memorySpace>
    blasWrapper::real_type< ValueType >
    asum (size_type n, ValueType const *x, size_type incx)
    {
      DFTEFE_AssertWithMsg(
        if (!std::is_same<memorySpace, dftefe::utils::MemorySpace::DEVICE>::value),
        "blas::asum() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
      blasWrapper::real_type< ValueType > output;
      output = blas::asum(n,x,incx);
      return output;
    }

    template<typename ValueType1, typename ValueType2 ,dftefe::utils::MemorySpace memorySpace>
    void axpy 	(
      size_type  	n,
      blasWrapper::scalar_type< ValueType1, ValueType2 >  	alpha,
      ValueType1 const *  	x,
      size_type  	incx,
      ValueType2 *  	y,
      size_type  	incy )
    {
      DFTEFE_AssertWithMsg(
        if (!std::is_same<memorySpace, dftefe::utils::MemorySpace::DEVICE>::value),
        "blas::axpy() is not implemented for dftefe::utils::MemorySpace::DEVICE .... ");
      blas::axpy(n,alpha, x, incx, y, incy);
    }

    template<typename ValueType1 , typename ValueType2,dftefe::utils::MemorySpace memorySpace >
    blasWrapper::scalar_type<ValueType1, ValueType2> dot
      ( 	size_type  	n,
       ValueType1 const *  	x,
       size_type  	incx,
       ValueType2 const *  	y,
       size_type  	incy
       )
    {

      DFTEFE_AssertWithMsg(
        if (!std::is_same<memorySpace, dftefe::utils::MemorySpace::DEVICE>::value),
        "blas::dot() is notimplemented for dftefe::utils::MemorySpace::DEVICE .... ");

      blasWrapper::real_type< ValueType > output;
      output = blas::dot(n ,x,incx, y, incy);
      return output;

    }

    template<typename ValueType >
    blasWrapper::real_type<ValueType> nrm2 	( 	size_type  	n,
         ValueType const *  	x,
         size_type  	incx
    )
    {
      DFTEFE_AssertWithMsg(
        if (!std::is_same<memorySpace, dftefe::utils::MemorySpace::DEVICE>::value),
        "blas::nrm2() is notimplemented for dftefe::utils::MemorySpace::DEVICE .... ");
      blasWrapper::real_type< ValueType > output;
      output = blas::nrm2(n ,x,incx);
      return output;
    }


    template<typename ValueType>
    void gemm(blasWrapper::Layout layout,
    blasOperationType transA,
    blasOperationType transB,
    size_type m, size_type n, size_type k,
    ValueType alpha,
    ValueType const *dA, size_type ldda,
    ValueType const *dB, size_type lddb,
    ValueType beta,
    ValueType       *dC, size_type lddc,
    blasWrapper::blasQueueType<dftefe::utils::MemorySpace::HOST> &blasQueue    )
    {
      blas::gemm(layout , transA , transB , m , n , k, alpha ,
                 dA, ldda , dB , lddb, beta, dC, lddc  );
    }

    template<typename ValueType>
    void gemm(blasWrapper::Layout layout,
         blasWrapper::Op transA,
         blasWrapper::Op transB,
         size_type m, size_type n, size_type k,
         ValueType alpha,
         ValueType const *dA, size_type ldda,
         ValueType const *dB, size_type lddb,
         ValueType beta,
         ValueType       *dC, size_type lddc,
         blasWrapper::blasQueueType<dftefe::utils::MemorySpace::DEVICE> &blasQueue    )
    {
      blas::gemm(layout , transA , transB , m , n , k, alpha ,
                 dA, ldda , dB , lddb, beta, dC, lddc, blasQueue );
    }

  }

}
