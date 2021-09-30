#include "Vector.h"
#include "blasTemplates.h"
#include <typeConfig.h>
#include <MemoryManager.h>

namespace dftefe
{
  template <typename NumberType, MemorySpace memorySpace>
  void
  Vector<NumberType, memorySpace>::testDgemv()
  {
    char                transA = 'N';
    char                transB = 'T';
    const double        alpha = 1.0, beta = 0.0;
    unsigned int        numberWaveFunctions = 10;
    unsigned int        numberDofs          = 10;
    std::vector<double> X, Y, output;
    X.resize(100);
    Y.resize(100);
    output.resize(100);



    dgemm_(&transA,
           &transB,
           &numberWaveFunctions,
           &numberWaveFunctions,
           &numberDofs,
           &alpha,
           &X[0],
           &numberWaveFunctions,
           &Y[0],
           &numberWaveFunctions,
           &beta,
           &output[0],
           &numberWaveFunctions);
  }

  //
  // Constructor
  //
  template <typename NumberType, MemorySpace memorySpace>
  Vector<NumberType, memorySpace>::Vector(const size_type  size,
                                          const NumberType initVal)
    : d_size(size)
    , d_data(MemoryManager<NumberType, memorySpace>::allocate(size))
  {
    // FIXME : initVal is not used
  }
  template <typename NumberType, MemorySpace memorySpace>
  void
  Vector<NumberType, memorySpace>::resize(const size_type  size,
                                          const NumberType initVal)
  {
    MemoryManager<NumberType, memorySpace>::deallocate(d_data);
    d_size = size;
    d_data = MemoryManager<NumberType, memorySpace>::allocate(size);

    // FIXME : initVal is not used
  }

  //
  // Destructor
  //
  template <typename NumberType, MemorySpace memorySpace>
  Vector<NumberType, memorySpace>::~Vector()
  {
    MemoryManager<NumberType, memorySpace>::deallocate(d_data);
  }

  template <typename NumberType, MemorySpace memorySpace>
  Vector<NumberType, memorySpace>::Vector(const Vector &v)
  {
    // FIXME :  not implemented
  }

  template <typename NumberType, MemorySpace memorySpace>
  size_type
  Vector<NumberType, memorySpace>::size() const
  {
    return d_size;
  }

  template class Vector<double, MemorySpace::HOST>;


} // namespace dftefe
