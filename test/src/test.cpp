#include <iostream>
#include <vector>
#include <algorithm>
extern "C" {
  int daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
}
int main() {
  int n = 10;
  std::vector<double> x(n, 1.0), y(n, 2.0);
  double a = 3.0;
  int incx = 1, incy = 1;
  auto print = [](const int& n) { std::cout << " " << n; };
//  std::for_each(x.begin(), x.end(), print); std::cout << std::endl;
//  std::for_each(y.begin(), y.end(), print); std::cout << std::endl;
  daxpy_(&n, &a, x.data(), &incx, y.data(), &incy);
  std::for_each(y.begin(), y.end(), print); std::cout << std::endl;
  return 0;
}
