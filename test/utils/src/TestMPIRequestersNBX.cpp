/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Bikash Kanungo
 */

#ifdef DFTEFE_WITH_MPI
#include <mpi.h>
#endif

#include <utils/TypeConfig.h>
#include <utils/Exceptions.h>
#include <utils/MPIRequestersNBX.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>


// Returns true if s is a number else false
bool isNumber(const std::string& s)
{
  int N = s.size();
  std::string scopy(s);
  if (s[0]=='-' || s[0]=='+')
    scopy = s.substr(1,N);

  return !scopy.empty() && std::find_if(scopy.begin(), 
      scopy.end(), [](unsigned char c) { return !std::isdigit(c); }) == scopy.end();
}

void readMatrix(std::vector<std::vector<int> > & mat, 
    const std::string fileName)
{    
  std::ifstream readFile;
  std::string readLine;
  readFile.open(fileName.c_str());
  assert(readFile.is_open());
  while(getline(readFile, readLine))
  {
    if(!readLine.empty())
    {
      std::vector<int> rowVals(0);
      std::istringstream lineString(readLine);
      std::string word;
      while(lineString >> word)
      {
	if(isNumber(word))
	{
	  rowVals.push_back(std::stoi(word));
	}
	else
	  dftefe::utils::throwException(false, "Undefined behavior. Value: " + 
	      word + " read in file " + fileName +  " is not an integer");
      }
      mat.push_back(rowVals);
    }
    else
      dftefe::utils::throwException(false, "Empty line found in file" + fileName); 
  }
  readFile.close();
}

int main()
{
#ifdef DFTEFE_WITH_MPI
  
  std::cout << "Running with MPI" << std::endl;
  // initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int numProcs;
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // for rank 0 write an NxN (N=numProcs) matrix
  // where the entries are either 0 or 1
  // Later on this matrix will be read by all the ranks
  std::string fileName = "MPIInteractionMatrix";
  if(rank == 0)
  {
    std::ofstream file;
    file.open(fileName.c_str());

    // randomly entries of A to 0 or 1 
    for(unsigned int i = 0; i < numProcs; ++i)
    {
      for(unsigned int j = 0; j < numProcs; ++j)
      {
	int val = rand()%2; 
	file << val << " ";	
      }
      file << std::endl;
    }
    file.close();
  }

  //
  // Let all processors wait so that the MPIInteractionMatrix 
  // file is available for all to read
  //
  MPI_Barrier(MPI_COMM_WORLD);

  // read the matrix from file MPIInteractionMatrix 
  std::vector<std::vector<int>> A(0);
  readMatrix(A, fileName);

  //
  // sanity checks for the size of A
  //
  int numRows = A.size();
  std::string errMsg =  "The number of rows (i.e.," + std::to_string(numRows) + 
    ") of matrix A does not match the number of processors (i.e., " + 
    std::to_string(numProcs) +")";
  dftefe::utils::throwException(numRows==numProcs, errMsg);

  for(unsigned int i = 0; i < numProcs; ++i)
  {
    int numCols= A[i].size();
    errMsg = "The number of cols (i.e.," + std::to_string(numCols) + ") in the "
      + std::to_string(i) + " row of matrix A does not match the number " 
      " of processors (i.e., " + std::to_string(numProcs) +")";
    dftefe::utils::throwException(numRows==numProcs, errMsg);
  }

  // 
  // Store the target IDs  and the requesting IDs 
  // for the current rank.
  // The target IDs are those entries in the i-th 
  // row, where i is the rank of the current processor,
  // for which A_ij are 1
  //
  // The requesting IDs are those entries in the i-th 
  // column, where i is the rank of the current processor,
  // for which A_ji are 1
  //
  std::vector<dftefe::size_type> targetIDs(0); 
  std::vector<dftefe::size_type> requestingIDs(0); 
  for(unsigned int i = 0; i < numProcs; ++i)
  {
    if(A[rank][i] == 1)
      targetIDs.push_back(i);

    if(A[i][rank] == 1)
      requestingIDs.push_back(i);
  }
  int numRequestingIDs = requestingIDs.size();

  dftefe::utils::MPIRequestersNBX mpiRequestersNBX(targetIDs, MPI_COMM_WORLD);
  std::vector<dftefe::size_type> requestingIDsFromNBX = 
    mpiRequestersNBX.getRequestingRankIds();
  int numRequestingIDsNBX = requestingIDsFromNBX.size();
  errMsg = "Mismatch in the number of requesting IDs expected(i.e.," + 
    std::to_string(numRequestingIDs) + ") and the number of requesting IDs" 
    " returned by MPIRequestersNBX (i.e.," + 
    std::to_string(numRequestingIDsNBX) + ") for rank " + std::to_string(rank); 
  dftefe::utils::throwException(numRequestingIDs==numRequestingIDsNBX, errMsg);

  //
  // to compare requestingIDs and requestingIDsFromNBX sort them 
  // and then compare element wise
  std::sort(requestingIDs.begin(), requestingIDs.end());
  std::sort(requestingIDsFromNBX.begin(), requestingIDsFromNBX.end());
  bool match = (requestingIDs==requestingIDsFromNBX);
  if(match==false)
  {
    std::string s1="";
    std::string s2="";
    for(unsigned int i = 0; i < numRequestingIDs; ++i)
    {
      s1 += std::to_string(requestingIDs[i]) + " ";
      s2 += std::to_string(requestingIDsFromNBX[i]) + " ";
      errMsg = "Mismatch of requesting IDs in rank " + std::to_string(rank) + 
	". \n Expected Requesting IDs: " + s1 + "\n NBX Requesting IDs: " +  s2;
      dftefe::utils::throwException(false, errMsg);
    }
  }

  MPI_Finalize();
#endif
}
