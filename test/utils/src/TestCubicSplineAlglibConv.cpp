#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <iomanip>
#include <exception>
#include <algorithm>
#include <chrono>
#include <utils/alglib/interpolation.h>

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
  const int numTestPoints = 100000;

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
  for(unsigned int i = 0; i < numTestPoints; ++i)
    testXData[i] = xmin + ((1.0*std::rand())/RAND_MAX)*(xmax-xmin);

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<std::pair<double, unsigned int>> testXDataAndIndex(numTestPoints);
  for(unsigned int i = 0; i < numTestPoints; ++i)
    testXDataAndIndex[i] = std::make_pair(testXData[i],i);

  std::sort(testXDataAndIndex.begin(), testXDataAndIndex.end());
  std::vector<double> testXDataSorted(numTestPoints,0.0);
  for(unsigned int i = 0; i < numTestPoints; ++i)
    testXDataSorted[i] = testXDataAndIndex[i].first;

  alglib::real_1d_array alglibX;
  alglibX.setcontent(numRows, &xData[0]);

  alglib::real_1d_array alglibY;  
  alglibY.setcontent(numRows, &yData[0]);


  alglib::real_1d_array alglibTestX;
  alglib::real_1d_array alglibTestY;
  alglib::real_1d_array alglibTestYDer;
  alglib::real_1d_array alglibTestYSecondDer;
  alglibTestX.setcontent(numTestPoints, &testXDataSorted[0]);
  alglibTestY.setlength(numTestPoints);
  alglibTestYDer.setlength(numTestPoints);
  alglibTestYSecondDer.setlength(numTestPoints);
  
  alglib::ae_int_t natural_boundary_type = 0;
  alglib::spline1dconvdiff2cubic(
      alglibX,
      alglibY,
      numRows,
      natural_boundary_type,
      0.0,
      natural_boundary_type,
      0.0,
      alglibTestX,
      numTestPoints,
      alglibTestY,
      alglibTestYDer,
      alglibTestYSecondDer);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::ofstream outTiming;
  outTiming.open("outTimingAlglibCSplineConv");
  outTiming << "Spline building and evaluation time for " << numTestPoints << ": " << duration.count() << " microseconds" << std::endl;

  outTiming.close(); 
}
