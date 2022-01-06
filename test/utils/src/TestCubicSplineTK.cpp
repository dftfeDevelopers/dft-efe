#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <iomanip>
#include <exception>
#include <algorithm>
#include <chrono>
#include "splineTK.h"

namespace {
  void readFile(std::string filename,
      std::vector<std::vector<double>> & data,
      const unsigned int numColumns)
  {

    std::vector<double> rowData(numColumns, 0.0);
    std::ifstream readFile;
    readFile.open(filename.c_str());
    std::string readLine;
    std::string word;
    int columnCount;
    if(readFile.is_open())
    {
      while (std::getline(readFile, readLine))
      {
	std::istringstream iss(readLine);

	columnCount = 0; 

	while(iss >> word && columnCount < numColumns)
	  rowData[columnCount++] = atof(word.c_str());

	data.push_back(rowData);
      }
    }
    else
    {
      throw std::invalid_argument("File: " + filename + " not found");
    }

    readFile.close();
    return;
  }
}

int main()
{

  std::string filename = "RadialFunction";
  std::vector<std::vector<double>> data(0);
  const unsigned int numColumns = 2;
  readFile(filename, data, numColumns);
  int numRows = data.size();
  const int numTestPoints = 10000;

  std::vector<double> xData(numRows,0.0);
  std::vector<double> yData(numRows,0.0);

  for (int iRow = 0; iRow < numRows; ++iRow)
  {
    xData[iRow] = data[iRow][0];
    yData[iRow] = data[iRow][1];
  }

  const double xmin = *std::min_element(xData.begin(), xData.end());
  const double xmax = *std::max_element(xData.begin(), xData.end());
  std::vector<double> testXData(numTestPoints,0.0);
  std::vector<double> testYData(numTestPoints,0.0);
  std::vector<double> testYDerData(numTestPoints,0.0);
  std::vector<double> testYSecondDerData(numTestPoints,0.0);
  for(unsigned int i = 0; i < numTestPoints; ++i)
    testXData[i] = xmin + ((1.0*std::rand())/RAND_MAX)*(xmax-xmin);

  auto start = std::chrono::high_resolution_clock::now();
  tk::spline splineTK(xData,yData);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  //std::ofstream out;
  //out.open("outTKCSplineTestPoints");
  //out << std::setprecision(16);
  start = std::chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < numTestPoints; ++i)
  {
    testYData[i] = splineTK(testXData[i]);
    //out << testXData[i] << " " << testYData[i] << std::endl;
  }

  stop = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
  //std::ofstream out1;
  //out1.open("outTKCSplineDerivativeTestPoints");
  //out1 << std::setprecision(16);
  start = std::chrono::high_resolution_clock::now();
  for(unsigned int i = 0; i < numTestPoints; ++i)
  {
    testYDerData[i] = splineTK.deriv(1, testXData[i]);
    testYSecondDerData[i] = splineTK.deriv(2, testXData[i]);
    //out1 << testXData[i] << " " << testYData[i] << " " << testYDerData[i] << " " << 
    //  testYSecondDerData[i] << std::endl;
  }

  stop = std::chrono::high_resolution_clock::now();
  auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::ofstream outTiming;
  outTiming.open("outTimingTKCSpline");
  outTiming << "Spline building time: " << duration1.count() << " microseconds" << std::endl;
  outTiming << "Spline evaulation time for " << numTestPoints << " : " << duration2.count() << " microseconds" << std::endl;
  outTiming << "Spline derivative evaulation time for " << numTestPoints << " : " << duration3.count() << " microseconds" << std::endl;

  outTiming.close(); 
 
  //out1.close()
  //out.close();
}
