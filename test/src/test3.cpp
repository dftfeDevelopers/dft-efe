#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

int main() {
    int n = 10;
    std::vector<double> x(n, 1.0), y(n, 2.0);
    double a = 3.0;
    int incx = 1, incy = 1;
    std::ofstream out;
    out.open("out_test1");
    out << "Values:";
    auto printFile = [&out](const int& n) { out << " " << n; };
    std::cout << "Values:";
    auto printStd = [](const int& n) { std::cout << " " << n; };
    std::transform(x.begin(), x.end(), y.begin(), x.begin(), std::plus<int>());
    std::for_each(y.begin(), y.end(), printStd);
    std::for_each(y.begin(), y.end(), printFile);
    out.close();
    return 0;
}
