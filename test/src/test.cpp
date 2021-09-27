#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
extern "C" {
  int daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
}
int main() {
  int n = 10;
  std::vector<double> x(n, 1.0), y(n, 2.0);
  double a = 3.0;
  int incx = 1, incy = 1;
//  std::for_each(x.begin(), x.end(), print); std::cout << std::endl;
//  std::for_each(y.begin(), y.end(), print); std::cout << std::endl;
  std::ofstream out;
  out.open("out_test1");
  out << "Values:";
  auto print = [&out](const int& n) { out << " " << n; };
  daxpy_(&n, &a, x.data(), &incx, y.data(), &incy);
  std::for_each(y.begin(), y.end(), print); 
  out.close();
  return 0;
}
