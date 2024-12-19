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

#ifndef dftefeMPIWrapper_h
#define dftefeMPIWrapper_h

#include <utils/MPITypes.h>
#include <utils/TypeConfig.h>
#include <utils/MemorySpaceType.h>
#include <string>
#include <utility>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
      template <typename MPIDatatype>
      struct MinMaxAvg
      {
        MPIDatatype min;
        MPIDatatype max;
        MPIDatatype avg;
      };

      int
      MPITypeContiguous(int count, MPIDatatype oldtype, MPIDatatype *newtype);

      template <MemorySpace memorySpace>
      int
      MPIAllreduce(const void *sendbuf,
                   void *      recvbuf,
                   int         count,
                   MPIDatatype datatype,
                   MPIOp       op,
                   MPIComm     comm);

      template <MemorySpace memorySpace>
      int
      MPIAllgather(const void *sendbuf,
                   int         sendcount,
                   MPIDatatype sendtype,
                   void *      recvbuf,
                   int         recvcount,
                   MPIDatatype recvtype,
                   MPIComm     comm);

      template <MemorySpace memorySpace>
      int
      MPIAllgatherv(const void *sendbuf,
                    int         sendcount,
                    MPIDatatype sendtype,
                    void *      recvbuf,
                    const int * recvcounts,
                    const int * displs,
                    MPIDatatype recvtype,
                    MPIComm     comm);

      int
      MPIBarrier(MPIComm comm);
      int
      MPIIbarrier(MPIComm comm, MPIRequest *request);

      template <MemorySpace memorySpace>
      int
      MPIBcast(void *      buffer,
               int         count,
               MPIDatatype datatype,
               int         root,
               MPIComm     comm);

      int
      MPICommCreateGroup(MPIComm  comm,
                         MPIGroup group,
                         int      tag,
                         MPIComm *newcomm);

      int
      MPICommFree(MPIComm *comm);
      int
      MPICommGroup(MPIComm comm, MPIGroup *group);
      int
      MPICommRank(MPIComm comm, int *rank);
      int
      MPICommSize(MPIComm comm, int *size);

      int
      MPIGroupFree(MPIGroup *group);

      int
      MPIGroupIncl(MPIGroup  group,
                   int       n,
                   const int ranks[],
                   MPIGroup *newgroup);

      int
      MPIGroupTranslateRanks(MPIGroup  group1,
                             int       n,
                             const int ranks1[],
                             MPIGroup  group2,
                             int       ranks2[]);

      int
      MPIInit(int *argc, char ***argv);

      int
      MPIInitThread(int *argc, char ***argv, int required, int *provided);

      int
      MPIInitialized(int *flag);

      int
      MPIIprobe(int        source,
                int        tag,
                MPIComm    comm,
                int *      flag,
                MPIStatus *status);

      int
      MPITest(MPIRequest *request, int *flag, MPIStatus *status);
      int
      MPITestall(int         count,
                 MPIRequest *requests,
                 int *       flag,
                 MPIStatus * statuses);

      template <MemorySpace memorySpace>
      int
      MPIIrecv(void *      buf,
               int         count,
               MPIDatatype datatype,
               int         source,
               int         tag,
               MPIComm     comm,
               MPIRequest *request);

      template <MemorySpace memorySpace>
      int
      MPIIsend(const void *buf,
               int         count,
               MPIDatatype datatype,
               int         dest,
               int         tag,
               MPIComm     comm,
               MPIRequest *request);

      template <MemorySpace memorySpace>
      int
      MPIRecv(void *      buf,
              int         count,
              MPIDatatype datatype,
              int         source,
              int         tag,
              MPIComm     comm,
              MPIStatus * status);

      int
      MPIOpCreate(MPIUserFunction *userfn, int commute, MPIOp *op);

      int
      MPIOpFree(MPIOp *op);

      template <MemorySpace memorySpace>
      int
      MPIReduce(void *      sendbuf,
                void *      recvbuf,
                int         count,
                MPIDatatype datatype,
                MPIOp       op,
                int         root,
                MPIComm     comm);

      int
      MPIRequestFree(MPIRequest *request);

      template <MemorySpace memorySpace>
      int
      MPISend(const void *buf,
              int         count,
              MPIDatatype datatype,
              int         dest,
              int         tag,
              MPIComm     comm);

      template <MemorySpace memorySpace>
      int
      MPISendrecv(const void *sendbuf,
                  int         sendcount,
                  MPIDatatype sendtype,
                  int         dest,
                  int         sendtag,
                  void *      recvbuf,
                  int         recvcount,
                  MPIDatatype recvtype,
                  int         source,
                  int         recvtag,
                  MPIComm     comm,
                  MPIStatus * status);

      template <MemorySpace memorySpace>
      int
      MPIIssend(const void *buf,
                int         count,
                MPIDatatype datatype,
                int         dest,
                int         tag,
                MPIComm     comm,
                MPIRequest *request);

      template <MemorySpace memorySpace>
      int
      MPISsend(const void *buf,
               int         count,
               MPIDatatype datatype,
               int         dest,
               int         tag,
               MPIComm     comm);

      int
      MPITypeCommit(MPIDatatype *datatype);

      int
      MPITypeFree(MPIDatatype *datatype);

      int
      MPITypeVector(int          count,
                    int          blocklength,
                    int          stride,
                    MPIDatatype  oldtype,
                    MPIDatatype *newtype);

      int
      MPIWait(MPIRequest *request, MPIStatus *status);

      int
      MPIWaitall(int count, MPIRequest requests[], MPIStatus statuses[]);

      int
      MPIErrorString(int errorcode, char *string, int *resultlen);

      int
      MPIFinalize(void);

      double
      MPIWtime(void);

      int
      MPIFinalized(int *flag);

      // template <typename T>
      // MPIDatatype
      // MPIGetDatatype();

      bool
      MPIErrIsSuccess(int errCode);

      std::string
      MPIErrMsg(int errCode);

      std::pair<bool, std::string>
      MPIErrIsSuccessAndMsg(int errCode);

      template <typename T, MemorySpace memorySpace>
      MinMaxAvg<T>
      MPIAllreduceMinMaxAvg(const T &data, MPIComm comm);

    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftefe
#include "MPIWrapper.t.cpp"
#endif // dftefeMPIWrapper_h
