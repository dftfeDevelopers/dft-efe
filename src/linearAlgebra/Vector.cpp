#include "Vector.h"
#include "blasTemplates.h"

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

  template class Vector<double, MemorySpace::HOST>;


} // namespace dftefe
