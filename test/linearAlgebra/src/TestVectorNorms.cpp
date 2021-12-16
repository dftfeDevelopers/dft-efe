#include <complex>
#include <fstream>
#include <iomanip>
#include "linearAlgebra/Vector.h"

int
main()
{
  std::string filename = "TestVectorNorms.out";
  std::ofstream fout(filename);

  unsigned int vSize = 3;
  // test double


  std::vector<double> dVecStd = {
    3.1, 5.3, 1.7};



  dftefe::linearAlgebra::Vector<double, dftefe::utils::MemorySpace::HOST> dVec(vSize, 0);

  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            dVec.data(),
                                            dVecStd.data());


  fout << "double vec norms ";
  fout << std::fixed << std::setprecision(8) <<dVec.l2Norm() << ", ";
  fout << std::fixed << std::setprecision(8) <<dVec.lInfNorm();  
  fout << std::endl;


  // test complex
  using namespace std::complex_literals;
  std::vector<std::complex<double>> zVecStd = {-1.54 + 16.80i,
                                               -1.79 + 20.79i,
                                               20.22 + 15.64i};


  dftefe::linearAlgebra::Vector<std::complex<double>, dftefe::utils::MemorySpace::HOST> zVec(vSize, 0);

  
  dftefe::utils::MemoryTransfer<
    dftefe::utils::MemorySpace::HOST,
    dftefe::utils::MemorySpace::HOST>::copy(vSize,
                                            zVec.data(),
                                            zVecStd.data());


  fout << "complex double vec norms ";
  fout << std::fixed << std::setprecision(8) <<zVec.l2Norm() << ", ";
  fout << std::fixed << std::setprecision(8) <<zVec.lInfNorm();  
  fout << std::endl;
  
  return 0;
}
