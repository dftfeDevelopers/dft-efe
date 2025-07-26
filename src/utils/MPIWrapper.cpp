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
#include <utils/MPIWrapper.h>
#include <utils/Exceptions.h>
#include <utils/MemoryTransfer.h>
#include <complex>
#include <ctime>
namespace dftefe
{
  namespace utils
  {
    namespace mpi
    {
#ifdef DFTEFE_WITH_MPI
      int
      MPITypeContiguous(int count, MPIDatatype oldtype, MPIDatatype *newtype)
      {
        return ::MPI_Type_contiguous(count, oldtype, newtype);
      }

      int
      MPIBarrier(MPIComm comm)
      {
        return ::MPI_Barrier(comm);
      }

      int
      MPIIbarrier(MPIComm comm, MPIRequest *request)
      {
        return ::MPI_Ibarrier(comm, request);
      }


      int
      MPICommCreateGroup(MPIComm  comm,
                         MPIGroup group,
                         int      tag,
                         MPIComm *newcomm)
      {
        return ::MPI_Comm_create_group(comm, group, tag, newcomm);
      }

      int
      MPICommFree(MPIComm *comm)
      {
        return ::MPI_Comm_free(comm);
      }

      int
      MPICommGroup(MPIComm comm, MPIGroup *group)
      {
        return ::MPI_Comm_group(comm, group);
      }

      int
      MPICommRank(MPIComm comm, int *rank)
      {
        return ::MPI_Comm_rank(comm, rank);
      }

      int
      MPICommSize(MPIComm comm, int *size)
      {
        return ::MPI_Comm_size(comm, size);
      }

      int
      MPIGroupFree(MPIGroup *group)
      {
        return ::MPI_Group_free(group);
      }

      int
      MPIGroupIncl(MPIGroup group, int n, const int ranks[], MPIGroup *newgroup)
      {
        return ::MPI_Group_incl(group, n, ranks, newgroup);
      }

      int
      MPIGroupTranslateRanks(MPIGroup  group1,
                             int       n,
                             const int ranks1[],
                             MPIGroup  group2,
                             int       ranks2[])
      {
        return ::MPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2);
      }

      int
      MPIInit(int *argc, char ***argv)
      {
        return ::MPI_Init(argc, argv);
      }

      int
      MPIInitialized(int *flag)
      {
        return ::MPI_Initialized(flag);
      }

      int
      MPIIprobe(int source, int tag, MPIComm comm, int *flag, MPIStatus *status)
      {
        return ::MPI_Iprobe(source, tag, comm, flag, status);
      }

      int
      MPITest(MPIRequest *request, int *flag, MPIStatus *status)
      {
        return ::MPI_Test(request, flag, status);
      }

      int
      MPITestall(int         count,
                 MPIRequest *requests,
                 int *       flag,
                 MPIStatus * statuses)
      {
        return ::MPI_Testall(count, requests, flag, statuses);
      }

      int
      MPIInitThread(int *argc, char ***argv, int required, int *provided)
      {
        return ::MPI_Init_thread(argc, argv, required, provided);
      }


      int
      MPIOpCreate(MPIUserFunction *user_fn, int commute, MPI_Op *op)
      {
        return ::MPI_Op_create(user_fn, commute, op);
      }

      int
      MPIOpFree(MPIOp *op)
      {
        return ::MPI_Op_free(op);
      }


      int
      MPIRequestFree(MPIRequest *request)
      {
        return ::MPI_Request_free(request);
      }


      int
      MPITypecommit(MPIDatatype *datatype)
      {
        return ::MPI_Type_commit(datatype);
      }

      int
      MPITypeFree(MPIDatatype *datatype)
      {
        return ::MPI_Type_free(datatype);
      }

      int
      MPITypeVector(int          count,
                    int          blocklength,
                    int          stride,
                    MPIDatatype  oldtype,
                    MPIDatatype *newtype)
      {
        return ::MPI_Type_vector(count, blocklength, stride, oldtype, newtype);
      }

      int
      MPIWait(MPIRequest *request, MPIStatus *status)
      {
        return ::MPI_Wait(request, status);
      }

      int
      MPIWaitall(int count, MPIRequest requests[], MPIStatus statuses[])
      {
        return ::MPI_Waitall(count, requests, statuses);
      }

      int
      MPIErrorString(int errorcode, char *string, int *resultlen)
      {
        return ::MPI_Error_string(errorcode, string, resultlen);
      }

      int
      MPIFinalize(void)
      {
        return ::MPI_Finalize();
      }

      int
      MPIFinalized(int *flag)
      {
        return ::MPI_Finalized(flag);
      }

      double
      MPIWtime(void)
      {
        return ::MPI_Wtime();
      }

      bool
      MPIErrIsSuccess(int errCode)
      {
        return (errCode == MPISuccess);
      }

      std::string
      MPIErrMsg(int errCode)
      {
        char errString[MPI_MAX_ERROR_STRING];
        int  N;
        ::MPI_Error_string(errCode, errString, &N);
        std::string returnValue(errString, 0, N);
        return returnValue;
      }

      std::pair<bool, std::string>
      MPIErrIsSuccessAndMsg(int errCode)
      {
        return std::make_pair(MPIErrIsSuccess(errCode), MPIErrMsg(errCode));
      }

#else // DFTEFE_WITH_MPI
      int
      MPITypeContiguous(int count, MPIDatatype oldtype, MPIDatatype *newtype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "Use of MPITypeContiguous() is not allowed when not linking to an MPI library");
      }


      int
      MPIBarrier(MPIComm comm)
      {
        return MPISuccess;
      }

      int
      MPIIbarrier(MPIComm comm, MPIRequest *request)
      {
        return MPISuccess;
      }


      int
      MPICommCreateGroup(MPIComm  comm,
                         MPIGroup group,
                         int      tag,
                         MPIComm *newcomm)
      {
        return MPISuccess;
      }

      int
      MPICommFree(MPIComm *comm)
      {
        return MPISuccess;
      }

      int
      MPICommGroup(MPIComm comm, MPIGroup *group)
      {
        return MPISuccess;
      }

      int
      MPICommRank(MPIComm comm, int *rank)
      {
        *rank = 0;
        return MPISuccess;
      }

      int
      MPICommSize(MPIComm comm, int *size)
      {
        *size = 1;
        return MPISuccess;
      }

      int
      MPIGroupFree(MPIGroup *group)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIGroup_free is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIGroupIncl(MPIGroup group, int n, const int ranks[], MPIGroup *newgroup)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIGroupIncl is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIGroupTranslateRanks(MPIGroup  group1,
                             int       n,
                             const int ranks1[],
                             MPIGroup  group2,
                             int       ranks2[])
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIGroup_translate_ranks is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIInit(int *argc, char ***argv)
      {
        return MPISuccess;
      }

      int
      MPIInitialized(int *flag)
      {
        *flag = 1;
        return MPISuccess;
      }

      int
      MPIIprobe(int source, int tag, MPIComm comm, int *flag, MPIStatus *status)
      {
        *flag = 1;
        return MPISuccess;
      }

      int
      MPITest(MPIRequest *request, int *flag, MPIStatus *status)
      {
        *flag = 1;
        return MPISuccess;
      }

      int
      MPITestall(int         count,
                 MPIRequest *requests,
                 int *       flag,
                 MPIStatus * statuses)
      {
        *flag = 1;
        return MPISuccess;
      }

      int
      MPIInitThread(int *argc, char ***argv, int required, int *provided)
      {
        *provided = MPIThreadMultiple;
        return MPISuccess;
      }


      int
      MPIOpCreate(MPIUserFunction *user_fn, int commute, MPIOp *op)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIOp_create is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIOpFree(MPIOp *op)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIOp_free is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }


      int
      MPIRequestFree(MPIRequest *request)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIRequest_free is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }


      int
      MPITypeCommit(MPIDatatype *datatype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIType_commit is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPITypeFree(MPIDatatype *datatype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIType_free is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPITypeVector(int          count,
                    int          blocklength,
                    int          stride,
                    MPIDatatype  oldtype,
                    MPIDatatype *newtype)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIType_vector is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIWait(MPIRequest *request, MPIStatus *status)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIWait is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIWaitall(int count, MPIRequest requests[], MPIStatus statuses[])
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIWait_All is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIErrorString(int errorcode, char *string, int *resultlen)
      {
        DFTEFE_AssertWithMsg(
          false,
          "MPIError_string is not implemented when not linking with an MPI library.");
        return MPISuccess;
      }

      int
      MPIFinalize(void)
      {
        return MPISuccess;
      }

      int
      MPIFinalized(int *flag)
      {
        *flag = 1;
        return MPISuccess;
      }

      double
      MPIWtime(void)
      {
        return std::time(nullptr);
      }

      bool
      MPIErrIsSuccess(int errCode)
      {
        return (errCode == MPISuccess);
      }

      std::string
      MPIErrMsg(int errCode)
      {
        std::string returnValue =
          MPIErrIsSuccess(errCode) ? "MPI Success" : "MPI Failure";
        return returnValue;
      }

      std::pair<bool, std::string>
      MPIErrIsSuccessAndMsg(int errCode)
      {
        return std::make_pair(MPIErrIsSuccess(errCode), MPIErrMsg(errCode));
      }

#endif // DFTEFE_WITH_MPI

      unsigned int
      numMPIProcesses(const MPIComm mpi_communicator)
      {
        int       n_jobs = 1;
        int ierr = MPICommSize(mpi_communicator, &n_jobs);
        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(ierr);
        DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                                  "MPI Error:" + mpiIsSuccessAndMsg.second);
        return n_jobs;                                
      }

      unsigned int
      thisMPIProcess(const MPIComm mpi_communicator)
      {
        int       rank = 0;
        int ierr = MPICommRank(mpi_communicator, &rank);
        std::pair<bool, std::string> mpiIsSuccessAndMsg =
          utils::mpi::MPIErrIsSuccessAndMsg(ierr);
        DFTEFE_AssertWithMsg(mpiIsSuccessAndMsg.first,
                                  "MPI Error:" + mpiIsSuccessAndMsg.second);
        return rank;  
      }
    }  // end of namespace mpi
  }    // end of namespace utils
} // end of namespace dftefe
