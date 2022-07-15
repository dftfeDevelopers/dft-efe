#include <utils/DataTypeOverloads.h>

namespace dftefe
{
  namespace linearAlgebra
  {
    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::multiply(ValueType                              alpha,
                               GeneralMatrix<ValueType, memorySpace> &A,
                               GeneralMatrix<ValueType, memorySpace> &B,
                               ValueType                              beta,
                               GeneralMatrix<ValueType, memorySpace> &C)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::HostTask});
        }
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::multiply(ValueType                                alpha,
                               HermitianMatrix<ValueType, memorySpace> &A,
                               GeneralMatrix<ValueType, memorySpace> &  B,
                               ValueType                                beta,
                               GeneralMatrix<ValueType, memorySpace> &  C)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::HostTask});
        }
    }



    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::multiply(ValueType                                alpha,
                               GeneralMatrix<ValueType, memorySpace> &  A,
                               HermitianMatrix<ValueType, memorySpace> &B,
                               ValueType                                beta,
                               GeneralMatrix<ValueType, memorySpace> &  C)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(alpha,
                          A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          beta,
                          {slate::Option::Target, slate::Target::HostTask});
        }
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::triangular_multiply(
      TriangularMatrix<ValueType, memorySpace> &A,
      Matrix<ValueType, memorySpace> &          B)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          {slate::Option::Target, slate::Target::HostTask});
        }
    }



    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::triangular_multiply(
      Matrix<ValueType, memorySpace> &          A,
      TriangularMatrix<ValueType, memorySpace> &B)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::multiply(A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          {slate::Option::Target, slate::Target::Devices});
        }
      else
        {
          slate::multiply(A.getSlateMatrix(),
                          B.getSlateMatrix(),
                          {slate::Option::Target, slate::Target::HostTask});
        }
    }


    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::triangular_solve(
      TriangularMatrix<ValueType, memorySpace> &A,
      Matrix<ValueType, memorySpace> &          B)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::triangular_solve(A.getSlateMatrix(),
                                  B.getSlateMatrix(),
                                  {slate::Option::Target,
                                   slate::Target::Devices});
        }
      else
        {
          slate::triangular_solve(A.getSlateMatrix(),
                                  B.getSlateMatrix(),
                                  {slate::Option::Target,
                                   slate::Target::HostTask});
        }
    }



    template <typename ValueType, dftefe::utils::MemorySpace memorySpace>
    void
    MatrixOperations::triangular_solve(
      Matrix<ValueType, memorySpace> &          A,
      TriangularMatrix<ValueType, memorySpace> &B)
    {
      if (memorySpace == dftefe::utils::MemorySpace::DEVICE)
        {
          slate::triangular_solve(A.getSlateMatrix(),
                                  B.getSlateMatrix(),
                                  {slate::Option::Target,
                                   slate::Target::Devices});
        }
      else
        {
          slate::triangular_solve(A.getSlateMatrix(),
                                  B.getSlateMatrix(),
                                  {slate::Option::Target,
                                   slate::Target::HostTask});
        }
    }

  } // namespace linearAlgebra
} // namespace dftefe
